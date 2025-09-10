#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds teams_snapshot.json using real OpenDota data for the Road to TI group stage.
- Parses team list + group W-L from input.txt
- Matches those names to OpenDota pro team IDs (aliases + fuzzy tokens)
- Pulls team matches, filters by event window, fetches match details
- Aggregates per-game K/D/A totals (team side only), computes averages + KDA
- Computes winrate over individual games in the window (recent_wr70)
- Writes JSON schema expected by ti25_bracket_predictor.py
"""

import os
import re
import json
import time
import math
import requests
from collections import defaultdict

import pandas as pd

OPENDOTA_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY", "").strip()

# -------------------------
# Event window (UTC date range)
# -------------------------
# The group stage ran Sep 4–8, 2025 (user-provided dates). Override via env if needed.
START_DATE_STR = os.environ.get("RTI_START_DATE", "2025-09-04")  # YYYY-MM-DD
END_DATE_STR   = os.environ.get("RTI_END_DATE",   "2025-09-08")  # inclusive

def date_to_epoch_utc(datestr: str) -> int:
    # interpret midnight UTC at start-of-day
    return int(time.mktime(time.strptime(datestr + " 00:00:00", "%Y-%m-%d %H:%M:%S")))

START_EPOCH = date_to_epoch_utc(START_DATE_STR)
# add 24h to make END inclusive
END_EPOCH   = date_to_epoch_utc(END_DATE_STR) + 24*3600 - 1

# -------------------------
# HTTP helpers
# -------------------------
def _get(url, params=None, sleep_s=0.0):
    if params is None:
        params = {}
    if API_KEY:
        params.setdefault("api_key", API_KEY)
    try:
        r = requests.get(url, params=params, timeout=20, headers={
            "User-Agent": "ti25-bracket/1.0 (+github-actions)"
        })
        if sleep_s:
            time.sleep(sleep_s)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[http] GET {url} failed: {e}")
        return None

# -------------------------
# Input parsing (teams + group W-L)
# -------------------------
TEAM_LINE = re.compile(r"^\s*([^:#]+?)\s*:\s*(\d+)\s*-\s*(\d+)\s*$")

def parse_input_file(path="input.txt"):
    """
    Reads lines like:
        Team Falcons: 3-2
    Returns:
        teams_order: [team names in file order]
        group_map: {team: winrate_float}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please ensure your input file exists.")

    teams_order = []
    group_map = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = TEAM_LINE.match(line)
            if m:
                team = m.group(1).strip()
                w = int(m.group(2))
                l = int(m.group(3))
                total = w + l
                wr = (w / total) if total > 0 else 0.5
                teams_order.append(team)
                group_map[team] = wr

    if not teams_order:
        raise ValueError("No 'Team: W-L' lines found in input.txt. Please include your group records.")

    return teams_order, group_map

# -------------------------
# Name normalization + aliases
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

ALIASES = {
    "xtreme gaming": ["xtreme gaming", "xg", "xtreme"],
    "betboom team": ["betboom team", "betboom", "bb"],
    "team tidebound": ["team tidebound", "tidebound", "tidebd", "tidebd team"],
    "team falcons": ["team falcons", "falcons"],
    "parivision": ["parivision", "pari", "pari vision", "pari-vision"],
    "heroic": ["heroic", "HEROIC".lower()],
    "nigma galaxy": ["nigma galaxy", "ngx"],
    "tundra esports": ["tundra esports", "tundra"],
}

def generate_name_keys(name: str):
    """
    All normalized keys to try for a given team name: self + known aliases.
    """
    n = norm(name)
    keys = {n}
    # expand using alias table (by looking up canonical that best matches the given name)
    best_canon = None
    best_overlap = 0
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    for canon, alist in ALIASES.items():
        can_tokens = set(re.findall(r"[a-z0-9]+", canon))
        overlap = len(tokens & can_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_canon = canon
    if best_canon:
        for alt in ALIASES[best_canon]:
            keys.add(norm(alt))
    return list(keys)

# -------------------------
# OpenDota team matching
# -------------------------
def fetch_pro_teams():
    print("[info] Fetching OpenDota pro teams...")
    data = _get(f"{OPENDOTA_BASE}/teams")
    return data if isinstance(data, list) else []

def build_team_index(teams):
    """
    Build dictionaries for fast matching on name/tag and fuzzy tokens.
    """
    exact = {}   # norm(name/tag) -> list of team records
    by_id = {}   # id -> team record
    token_index = defaultdict(list)  # token -> list of team records

    for t in teams:
        tid = t.get("team_id")
        name = (t.get("name") or "").strip()
        tag  = (t.get("tag")  or "").strip()
        if not tid or not (name or tag):
            continue
        rec = {
            "team_id": tid,
            "name": name,
            "tag": tag,
            "rating": t.get("rating", 0.0) or 0.0,
            "last_match_time": t.get("last_match_time", 0) or 0
        }
        by_id[tid] = rec
        for key in [name, tag]:
            if key:
                exact.setdefault(norm(key), []).append(rec)
                for tok in re.findall(r"[a-z0-9]+", key.lower()):
                    token_index[tok].append(rec)

    return exact, token_index, by_id

def match_team_to_opendota_id(name, exact_idx, token_idx):
    """
    Try exact alias matches first; then a simple token-overlap score.
    Prefer higher rating + more recent activity on ties.
    """
    keys = generate_name_keys(name)

    # 1) exact match on any alias key (name/tag normalized)
    candidates = []
    for k in keys:
        if k in exact_idx:
            candidates.extend(exact_idx[k])

    def uniq_by_id(recs):
        seen = set()
        out = []
        for r in recs:
            tid = r["team_id"]
            if tid not in seen:
                seen.add(tid)
                out.append(r)
        return out

    if candidates:
        candidates = uniq_by_id(candidates)
        candidates.sort(key=lambda r: (r["rating"], r["last_match_time"]), reverse=True)
        return candidates[0]["team_id"]

    # 2) token overlap
    name_tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    scored = {}
    for tok in name_tokens:
        for rec in token_idx.get(tok, []):
            tid = rec["team_id"]
            scored.setdefault(tid, {"rec": rec, "score": 0})
            scored[tid]["score"] += 1

    if scored:
        ranked = list(scored.values())
        ranked.sort(key=lambda x: (x["score"], x["rec"]["rating"], x["rec"]["last_match_time"]), reverse=True)
        return ranked[0]["rec"]["team_id"]

    return None

# -------------------------
# OpenDota stats per team in window
# -------------------------
def fetch_team_matches(team_id, limit=200):
    return _get(f"{OPENDOTA_BASE}/teams/{team_id}/matches", params={"limit": limit}, sleep_s=0.1) or []

def fetch_match_details(match_id):
    return _get(f"{OPENDOTA_BASE}/matches/{match_id}", sleep_s=0.1) or {}

def aggregate_team_window_stats(team_id):
    """
    Returns dict with:
      games, wins, k_sum, d_sum, a_sum  (team totals across all games)
      and derived k_avg, d_avg, a_avg, kda, recent_wr70
    Filters matches by START_EPOCH..END_EPOCH using each match start_time.
    """
    matches = fetch_team_matches(team_id)
    if not matches:
        return None

    games = 0
    wins  = 0
    k_sum = d_sum = a_sum = 0

    for m in matches:
        st = m.get("start_time")
        if not isinstance(st, int):
            continue
        if st < START_EPOCH or st > END_EPOCH:
            continue

        match_id   = m.get("match_id")
        our_radiant = bool(m.get("radiant"))  # this is the POV team's side
        # Win if our side == radiant_win
        radiant_win = bool(m.get("radiant_win"))
        won = (our_radiant == radiant_win)

        # Fetch details to sum team K/D/A from players on our side
        det = fetch_match_details(match_id)
        players = det.get("players", [])
        if not players or len(players) < 10:
            # If details missing, skip the game — don't contaminate averages
            continue

        # Identify our side from summary 'radiant'
        team_k = team_d = team_a = 0
        for p in players:
            if bool(p.get("isRadiant")) == our_radiant:
                team_k += int(p.get("kills", 0) or 0)
                team_d += int(p.get("deaths", 0) or 0)
                team_a += int(p.get("assists", 0) or 0)

        # accept only sensible rows
        if team_k == team_d == team_a == 0:
            # If all zeros, probably invalid record — skip
            continue

        games += 1
        wins  += 1 if won else 0
        k_sum += team_k
        d_sum += team_d
        a_sum += team_a

    if games == 0:
        return None

    k_avg = k_sum / games
    d_avg = d_sum / games
    a_avg = a_sum / games
    kda   = (k_avg + a_avg) / max(1.0, d_avg)
    wr    = wins / games

    return {
        "games": games,
        "wins": wins,
        "k_avg": round(k_avg, 6),
        "d_avg": round(d_avg, 6),
        "a_avg": round(a_avg, 6),
        "kda": round(kda, 6),
        "recent_wr70": round(wr, 6),
    }

# -------------------------
# TI pressure/meta placeholders
# -------------------------
def ti_pressure_placeholder():
    return 0.5

def meta_align_placeholder():
    return 0.5

# -------------------------
# Main build
# -------------------------
def main():
    teams_order, group_map = parse_input_file("input.txt")

    pro_teams = fetch_pro_teams()
    exact_idx, token_idx, _by_id = build_team_index(pro_teams)

    out_rows = []
    for team in teams_order:
        print(f"[info] Resolving team: {team} ...")
        tid = match_team_to_opendota_id(team, exact_idx, token_idx)
        if not tid:
            print(f"[warn] Could not match '{team}' to an OpenDota team id. Skipping this team.")
            continue

        print(f"[info] Fetching window stats for {team} (team_id={tid}) ...")
        agg = aggregate_team_window_stats(tid)

        if not agg:
            print(f"[warn] No valid matches in window for {team}. Skipping.")
            continue

        row = {
            "team": team,
            "recent_wr70": agg["recent_wr70"],   # window winrate over games
            "group": group_map.get(team, 0.5),   # from input.txt W-L
            "pen": 0.0,                          # keep at 0 unless you have roster penalties
            "k_avg": agg["k_avg"],
            "d_avg": agg["d_avg"],
            "a_avg": agg["a_avg"],
            "kda": agg["kda"],
            "ti_pressure": ti_pressure_placeholder(),
            "meta_align": meta_align_placeholder(),
        }
        out_rows.append(row)

    if not out_rows:
        raise RuntimeError("No teams produced any stats. Check team names and the date window ENV (RTI_START_DATE / RTI_END_DATE).")

    # Persist
    with open("teams_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    # Also print a quick table for logs
    df = pd.DataFrame(out_rows)
    print(df.to_string(index=False))
    print(f"[ok] Wrote teams_snapshot.json with {len(out_rows)} teams.")

if __name__ == "__main__":
    main()

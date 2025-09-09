#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor

Fixes & improvements:
- Shows component breakdown (so Falcons +0.06 is visible).
- Correctly computes K/D/A by fetching match DETAILS for a subset of recent games.
- Robust TI-pressure scrape: fetch HTML with headers, then pandas.read_html on the HTML text.
- Logs when TI-pressure map is empty and falls back to neutral 0.5 (but now should fill for most well-known nicknames).
- Still uses last-50 recent matches for WR; uses last-20 for KDA detail to limit API load.
- Keeps momentum bonus (+0.05% per Main Event win) and correct double-elim bracket.

Run: python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time, logging, re
from collections import defaultdict
from typing import Dict, List, Tuple

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------- Config ----------
API_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS_API = {'Authorization': f"Bearer {API_KEY}"} if API_KEY else {}

# Be polite to public APIs
SLEEP_LIST = 0.20        # seconds between list requests
SLEEP_DETAIL = 0.25      # seconds between match detail requests

TEAM_IDS = {
    "Tundra Esports": 8291895,
    "Xtreme Gaming": 8261500,
    "PARIVISION": 9572001,
    "Heroic": 9303484,
    "Team Tidebound": 9640842,
    "Team Falcons": 9247354,
    "BetBoom Team": 14124,
    "Nigma Galaxy": 7554697,
}

# Upper bracket quarterfinals (per Liquipedia main event)
UB_QF = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

# Weights & bonuses
W_RECENT      = 0.35   # recent WR (50)
W_GROUP       = 0.20   # group WR from input.txt
W_ROSTER      = 0.10   # roster penalty
W_FORM50      = 0.20   # recent KDA-ish form
W_TI_PRESSURE = 0.15   # prior TIs performance

ROSTER_ISSUES   = {"Tundra Esports": True}
ROSTER_PENALTY  = 0.08
H2H_BOOST       = 0.04
FALCONS_BONUS   = 0.06      # your personal boost
MAIN_WIN_BONUS  = 0.0005    # +0.05% per simulated Main Event win

TI_STATS_URLS = [
    "https://liquipedia.net/dota2/The_International/2024/Statistics",
    "https://liquipedia.net/dota2/The_International/2023/Statistics",
    "https://liquipedia.net/dota2/The_International/2022/Statistics",
]
HTTP_HEADERS = {
    "User-Agent": "TI25-bracket-bot/1.0 (+https://github.com/ejpdtd-beep)",
    "Accept": "text/html,application/xhtml+xml",
}

# ---------- Input parsing ----------
def parse_input():
    group_wr, h2h = {}, []
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ">" in line:
            a, b = [x.strip() for x in line.split(">")]
            h2h.append((a, b, 1))
        elif "<" in line:
            a, b = [x.strip() for x in line.split("<")]
            h2h.append((a, b, -1))
        elif ":" in line:
            name, rec = line.split(":", 1)
            w, l = map(int, rec.strip().split("-"))
            group_wr[name.strip()] = w / (w + l) if (w + l) > 0 else 0.5
    return group_wr, h2h

# ---------- OpenDota helpers ----------
def get_team_matches(team_id: int, limit: int = 50) -> List[dict]:
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/matches?limit={limit}", headers=HEADERS_API, timeout=25)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        return r.json() or []
    except Exception as e:
        logging.info(f"[warn] get_team_matches({team_id}) failed: {e}")
        return []

def get_match_detail(match_id: int) -> dict:
    try:
        r = requests.get(f"{API_BASE}/matches/{match_id}", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_DETAIL)
        return r.json() or {}
    except Exception as e:
        logging.info(f"[warn] get_match_detail({match_id}) failed: {e}")
        return {}

def recent_wr(matches: List[dict]) -> float:
    wins = tot = 0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            tot += 1
            team_won = (m.get('radiant') and m.get('radiant_win')) or ((not m.get('radiant')) and (not m.get('radiant_win')))
            wins += 1 if team_won else 0
    return wins / tot if tot else 0.5

def recent_kda_via_details(team_id: int, matches: List[dict], detail_cap: int = 20) -> Tuple[float,float,float,float]:
    """
    Compute team K/D/A by fetching detailed match data for up to `detail_cap` recent matches.
    Returns (k_avg, d_avg, a_avg, kda).
    """
    # Take the newest `detail_cap` matches that have a match_id
    mids = [m.get("match_id") for m in matches if m.get("match_id")][:detail_cap]
    if not mids:
        return 0.0, 0.0, 0.0, 1.0

    k_sum = d_sum = a_sum = tot = 0
    for mid in mids:
        d = get_match_detail(mid)
        if not d or "players" not in d:
            continue

        radiant_team_id = d.get("radiant_team_id")
        dire_team_id = d.get("dire_team_id")
        # Which side was our team?
        side = None
        if radiant_team_id == team_id:
            side = "radiant"
        elif dire_team_id == team_id:
            side = "dire"
        else:
            # Fallback: infer from players if possible
            side = None
            for p in d.get("players", []):
                if p.get("team") in (0, 1):  # 0=radiant, 1=dire in some dumps
                    pass

        rk = rd = ra = 0
        dk = dd = da = 0
        # Sum players by side
        for p in d.get("players", []):
            is_radiant = p.get("isRadiant")
            kills = p.get("kills", 0) or 0
            deaths = p.get("deaths", 0) or 0
            assists = p.get("assists", 0) or 0
            if is_radiant:
                rk += kills; rd += deaths; ra += assists
            else:
                dk += kills; dd += deaths; da += assists

        if side == "radiant":
            tk, td, ta = rk, dd, ra  # deaths taken as enemy kills? No — deaths for own side:
            td = rd  # use radiant deaths for radiant side
        elif side == "dire":
            tk, td, ta = dk, rd, da
            td = dd
        else:
            # If we couldn't identify side, skip
            continue

        k_sum += tk; d_sum += td; a_sum += ta; tot += 1

    if not tot:
        return 0.0, 0.0, 0.0, 1.0

    k_avg = k_sum / tot
    d_avg = d_sum / tot
    a_avg = a_sum / tot
    kda = (k_avg + a_avg) / max(d_avg, 1e-6)
    return k_avg, d_avg, a_avg, kda

def get_current_roster_names(team_id: int, max_players=5) -> List[str]:
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/players", headers=HEADERS_API, timeout=25)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        players = r.json() or []
        current = [p for p in players if p.get("is_current_team_member")]
        pool = current if current else sorted(players, key=lambda x: (x.get("games_played", 0), x.get("is_current_team_member", False)), reverse=True)
        names = []
        for p in pool:
            n = (p.get("name") or p.get("personaname") or "").strip()
            if n: names.append(n)
            if len(names) >= max_players: break
        return names
    except Exception as e:
        logging.info(f"[warn] roster fetch failed for {team_id}: {e}")
        return []

# ---------- TI pressure scoring ----------
def fetch_ti_stats_tables() -> Dict[str, float]:
    """
    Build nickname -> score map by fetching Liquipedia TI stats HTML and then using read_html on the content.
    If we still get no tables, we return an empty map and log a warning.
    """
    nick_score = defaultdict(float)
    weight_per_year = [1.0, 0.8, 0.6]  # 2024 > 2023 > 2022

    for idx, url in enumerate(TI_STATS_URLS):
        w_year = weight_per_year[idx] if idx < len(weight_per_year) else 0.5
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=30)
            resp.raise_for_status()
            html = resp.text
            tables = pd.read_html(html)  # parse on fetched HTML
        except Exception as e:
            logging.info(f"[warn] TI stats fetch failed for {url}: {e}")
            continue

        # heuristics to award points for leaderboard presence
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            candidate_cols = [c for c in table.columns if any(s in str(c).lower() for s in ["player", "nickname", "name"])]
            rank_cols = [c for c in table.columns if any(s in str(c).lower() for s in ["#", "rank", "place", "pos"])]

            if not candidate_cols:
                continue

            for _, row in table.iterrows():
                for pc in candidate_cols:
                    pname = str(row.get(pc, "")).strip()
                    if not pname or pname.lower() in ("nan", "—", "-", "none"):
                        continue
                    # strip team tags and decorations
                    nickname = re.sub(r"[^A-Za-z0-9_\-\.]", "", pname.split()[0])
                    base = 1.0
                    if rank_cols:
                        try:
                            r = row.get(rank_cols[0])
                            r = int(str(r).strip().replace(".", "").split()[0])
                            base = max(0.5, 2.5 - 0.15 * (r - 1))  # rank 1 ~ 2.5
                        except Exception:
                            pass
                    nick_score[nickname] += base * w_year
                    break  # count once per row

    if not nick_score:
        logging.info("[warn] TI pressure nick-score map is empty; falling back to neutral 0.5 for all teams.")
    return nick_score

def match_nickname(raw: str, nick_map: Dict[str, float]) -> float:
    if not raw: return 0.0
    variants = {
        raw, raw.lower(), raw.upper(),
        raw.replace("-", ""), raw.replace("_", ""),
        raw.split()[0]
    }
    best = 0.0
    for v in variants:
        if v in nick_map: best = max(best, nick_map[v])
        if v.capitalize() in nick_map: best = max(best, nick_map[v.capitalize()])
    if best > 0: return best
    # last-chance case-insensitive
    rl = raw.lower()
    for k, v in nick_map.items():
        if k.lower() == rl:
            best = max(best, v)
    return best

def team_ti_pressure_score(roster_names: List[str], nick_score_map: Dict[str, float]) -> Tuple[float, float]:
    """
    Returns (ti_pressure, ti_pressure_raw_mean).
    Raw mean is handy to print in the table for debugging.
    """
    if not roster_names or not nick_score_map:
        return 0.5, 0.0

    raw_scores = [match_nickname(n, nick_score_map) for n in roster_names]
    found = [s for s in raw_scores if s > 0]
    if not found:
        return 0.5, 0.0

    raw_mean = sum(found) / len(found)
    # squash raw_mean into [~0.45, ~0.95]
    norm = 1.0 - 1.0 / (1.0 + raw_mean / 3.0)
    ti_score = 0.45 + 0.5 * norm
    return ti_score, raw_mean

# ---------- Scoring ----------
def squash(x: float, c: float = 22.0) -> float:
    try:
        return 1.0 - (1.0 / (1.0 + x / c))
    except Exception:
        return 0.5

def base_team_score(name: str, g_wr: Dict[str, float], nick_score_map: Dict[str, float]) -> dict:
    team_id = TEAM_IDS[name]
    matches50 = get_team_matches(team_id, limit=50)
    wr50 = recent_wr(matches50)

    # K/D/A via details on up to 20 most recent matches
    k_avg, d_avg, a_avg, kda = recent_kda_via_details(team_id, matches50, detail_cap=20)

    gs  = g_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if ROSTER_ISSUES.get(name, False) else 0.0
    roster = get_current_roster_names(team_id)
    ti_pressure, ti_raw = team_ti_pressure_score(roster, nick_score_map)

    form_component = 0.45 * squash(k_avg) + 0.55 * squash(kda * 10)

    comp_recent   = W_RECENT      * wr50
    comp_group    = W_GROUP       * gs
    comp_roster   = W_ROSTER      * (1 - pen)
    comp_form     = W_FORM50      * form_component
    comp_ti       = W_TI_PRESSURE * ti_pressure
    comp_falcons  = FALCONS_BONUS if name == "Team Falcons" else 0.0

    score = comp_recent + comp_group + comp_roster + comp_form + comp_ti + comp_falcons

    return {
        "team": name,
        "recent_wr50": wr50,
        "group": gs,
        "pen": pen,
        "k_avg": k_avg,
        "d_avg": d_avg,
        "a_avg": a_avg,
        "kda": kda,
        "ti_pressure": ti_pressure,
        "ti_pressure_raw": ti_raw,
        "roster": ", ".join(roster) if roster else "",
        "comp_recent": comp_recent,
        "comp_group": comp_group,
        "comp_roster": comp_roster,
        "comp_form": comp_form,
        "comp_ti": comp_ti,
        "comp_falcons": comp_falcons,
        "score": score,
    }

def h2h_adjust(a: str, b: str, h2h: List[Tuple[str,str,int]]) -> float:
    adj = 0.0
    for x, y, res in h2h:
        if x == a and y == b:
            adj = H2H_BOOST if res == 1 else -H2H_BOOST
    return adj

def predict(a: str, b: str, scores: dict, h2h: list, wins_so_far: Dict[str,int]) -> Tuple[str, float]:
    sa = scores[a]['score'] + h2h_adjust(a, b, h2h) + wins_so_far[a]*MAIN_WIN_BONUS
    sb = scores[b]['score'] + h2h_adjust(b, a, h2h) + wins_so_far[b]*MAIN_WIN_BONUS
    p_a = 1 / (1 + math.exp(-8 * (sa - sb)))
    winner = a if p_a >= 0.5 else b
    prob = p_a if winner == a else (1 - p_a)
    return winner, prob

# ---------- Bracket ----------
def simulate_bracket(scores: dict, h2h: list) -> dict:
    wins_so_far = defaultdict(int)

    # UB QF
    ubqf_winners, ubqf_losers = [], []
    for i, (a, b) in enumerate(UB_QF, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        l = b if w == a else a
        wins_so_far[w] += 1
        print(f"UB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubqf_winners.append(w); ubqf_losers.append(l)

    # UB SF
    ubsf_pairs = [(ubqf_winners[0], ubqf_winners[1]), (ubqf_winners[2], ubqf_winners[3])]
    ubsf_winners, ubsf_losers = [], []
    for i, (a, b) in enumerate(ubsf_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        l = b if w == a else a
        wins_so_far[w] += 1
        print(f"UB SF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubsf_winners.append(w); ubsf_losers.append(l)

    # UB Final
    a, b = ubsf_winners
    ubf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    ubf_loser = b if ubf_winner == a else a
    wins_so_far[ubf_winner] += 1
    print(f"UB Final: {a} vs {b} → {ubf_winner} (p={p:.2f})")

    # LB R1
    lbr1_pairs = [(ubqf_losers[0], ubqf_losers[1]), (ubqf_losers[2], ubqf_losers[3])]
    lbr1_winners = []
    for i, (a, b) in enumerate(lbr1_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        wins_so_far[w] += 1
        print(f"LB R1-{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbr1_winners.append(w)

    # LB QF (LBR1-A vs Loser UB-SF2) & (LBR1-B vs Loser UB-SF1)
    lbqf_pairs = [(lbr1_winners[0], ubsf_losers[1]), (lbr1_winners[1], ubsf_losers[0])]
    lbqf_winners = []
    for i, (a, b) in enumerate(lbqf_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        wins_so_far[w] += 1
        print(f"LB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbqf_winners.append(w)

    # LB SF
    a, b = lbqf_winners
    lbsf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    wins_so_far[lbsf_winner] += 1
    print(f"LB SF: {a} vs {b} → {lbsf_winner} (p={p:.2f})")

    # LB Final vs UB Final loser
    a, b = lbsf_winner, ubf_loser
    lbf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    wins_so_far[lbf_winner] += 1
    print(f"LB Final: {a} vs {b} → {lbf_winner} (p={p:.2f})")

    # Grand Final
    a, b = ubf_winner, lbf_winner
    gf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    print(f"Grand Final: {a} vs {b} → Champion {gf_winner} (p={p:.2f})")

    return {
        "ubqf_winners": ubqf_winners, "ubqf_losers": ubqf_losers,
        "ubsf_winners": ubsf_winners, "ubsf_losers": ubsf_losers,
        "ubf_winner": ubf_winner, "ubf_loser": ubf_loser,
        "lbr1_winners": lbr1_winners, "lbqf_winners": lbqf_winners,
        "lbsf_winner": lbsf_winner, "lbf_winner": lbf_winner,
        "gf_winner": gf_winner
    }

def main():
    logging.info("Building TI pressure map…")
    nick_score_map = fetch_ti_stats_tables()

    g_wr, h2h = parse_input()
    scores = {t: base_team_score(t, g_wr, nick_score_map) for t in TEAM_IDS}

    # Show components so you can verify effects
    cols = [
        'recent_wr50','group','pen','k_avg','d_avg','a_avg','kda',
        'ti_pressure','ti_pressure_raw',
        'comp_recent','comp_group','comp_roster','comp_form','comp_ti','comp_falcons',
        'score'
    ]
    df = (pd.DataFrame(scores).T
          .set_index('team')[cols]
          .sort_values('score', ascending=False))
    print(df)

    res = simulate_bracket(scores, h2h)

    out = "bracket_prediction.md"
    with open(out,"w", encoding="utf-8") as f:
        f.write("# TI25 Full Bracket Prediction\n\n")
        f.write(df.to_markdown() + "\n\n")
        f.write("### Rosters used for TI pressure\n")
        for t in TEAM_IDS:
            f.write(f"- {t}: {scores[t]['roster'] or '(unknown)'}\n")
        f.write("\n## Bracket Results\n")
        f.write(f"UB QF Winners: {res['ubqf_winners']}\n")
        f.write(f"UB SF Winners: {res['ubsf_winners']}\n")
        f.write(f"UB Final Winner: {res['ubf_winner']}\n")
        f.write(f"LB R1 Winners: {res['lbr1_winners']}\n")
        f.write(f"LB QF Winners: {res['lbqf_winners']}\n")
        f.write(f"LB SF Winner: {res['lbsf_winner']}\n")
        f.write(f"LB Final Winner: {res['lbf_winner']}\n")
        f.write(f"\n**Champion:** {res['gf_winner']}\n")
    print(f"[ok] Wrote {out}")

if __name__ == '__main__':
    main()

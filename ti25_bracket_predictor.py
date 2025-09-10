#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TI25 Bracket Predictor — real data + tie-breakers + close-match regen + per-round CSV + visuals
Now robust to inputs like "Team Tidebound: 4-1" or "(3-2)" — names are sanitized & fuzzily mapped.

Features kept:
- Real data (OpenDota): recent form (70), K/D/A, rosters, group-stage window WR, H2H in window
- Weights: group performance, H2H via tie-breakers (GS WR → H2H → recent → TI pressure), meta (light), roster continuity, KDA, TI pressure
- Falcons fan bonus (HALVED), upset buffer, per-win bonus = 0.00020
- Close-match auto-regeneration with light bootstrap noise
- Visuals: bar (scores), heatmap (KDA), bracket tree
- CSV: per match rounds
"""

import os, sys, re, math, time, random, itertools
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Config & constants
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- Weights (as previously, Falcons bonus halved) ----
W_RECENT   = 0.30     # recent 70 games WR
W_GROUP    = 0.30     # normalized group performance
W_ROSTER   = 0.10     # roster continuity (penalty inverted)
W_FORM     = 0.10     # squashed KDA component
W_TI       = 0.03     # TI pressure / prep
W_META     = 0.025    # meta alignment
W_FALCONS  = 0.03     # was 0.06 → 50% reduction

# Per-win bonus (your requested value)
MAIN_EVENT_WIN_BONUS = 0.00020

# Upset logic
UPSET_BUFFER = 0.55       # underdog must have p > 0.55 to upset
CLOSE_MARGIN = 0.010      # absolute score diff threshold
REGENERATIONS = 7         # number of bootstrap samples when close
NOISE_STD = 0.006         # jitter scale for close matches

OD_BASE = "https://api.opendota.com/api"
TODAY = datetime.utcnow()

def _date_env(name, default_dt):
    v = os.getenv(name, "")
    if not v:
        return default_dt
    return datetime.strptime(v, "%Y-%m-%d")

# Default to last 30 days if not pinned to Road-to-TI GS
GROUP_START = _date_env("GROUP_START_DATE", TODAY - timedelta(days=30))
GROUP_END   = _date_env("GROUP_END_DATE", TODAY)

OD_HEADERS = {}
if os.getenv("OPENDOTA_API_KEY"):
    OD_HEADERS["Authorization"] = f"Bearer {os.getenv('OPENDOTA_API_KEY')}"

def get_json(url, params=None, retries=4, backoff=0.8):
    for i in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=OD_HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                print(f"[warn] GET {url} failed: {e}", flush=True)
                return None
            time.sleep(backoff * (2 ** i))
    return None

# -----------------------------
# Input sanitization & mapping
# -----------------------------
RE_TRAILING_RECORD = re.compile(r'\s*(?:[:\-–]\s*|\(\s*)\d+\s*-\s*\d+\s*\)?\s*$', re.UNICODE)

def sanitize_team_name(s: str) -> str:
    """
    Remove trailing record annotations like ': 4-1', '- 3-2', or '(2-3)'.
    Also trims whitespace.
    """
    s = s.strip()
    s = RE_TRAILING_RECORD.sub('', s)
    return s.strip()

def canonicalize(s: str) -> str:
    s = sanitize_team_name(s)
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()
    return s

def get_team_id_map():
    data = get_json(f"{OD_BASE}/teams") or []
    return { (t.get("name") or "").strip(): int(t["team_id"]) for t in data if t.get("team_id") }

def build_canonical_maps():
    """
    Build:
      - canon_to_id: canonical name -> team_id
      - canon_to_official: canonical name -> official OpenDota name
    Adds with/without 'team ' prefix for robustness.
    """
    raw = get_team_id_map()
    canon_to_id = {}
    canon_to_official = {}
    for official, tid in raw.items():
        c = canonicalize(official)
        if c:
            canon_to_id[c] = tid
            canon_to_official[c] = official
        # also add without/with 'team ' prefix
        if c.startswith("team "):
            alt = c.replace("team ", "", 1).strip()
            if alt and alt not in canon_to_id:
                canon_to_id[alt] = tid
                canon_to_official[alt] = official
        else:
            alt = ("team " + c).strip()
            if alt not in canon_to_id:
                canon_to_id[alt] = tid
                canon_to_official[alt] = official
    return canon_to_id, canon_to_official

def best_match_id(canon: str, canon_to_id: dict) -> int | None:
    """
    Exact match first; else simple token-overlap (Jaccard) fuzzy match.
    """
    if canon in canon_to_id:
        return canon_to_id[canon]
    toks = set(canon.split())
    best, best_s = None, 0.0
    for k, tid in canon_to_id.items():
        kt = set(k.split())
        score = len(toks & kt) / max(1, len(toks | kt))
        if score > best_s:
            best, best_s = tid, score
    return best if best_s >= 0.5 else None

def pretty_official(canon: str, canon_to_official: dict, fallback: str) -> str:
    return canon_to_official.get(canon) or sanitize_team_name(fallback)

# -----------------------------
# Team data & metrics
# -----------------------------
TeamInfo = namedtuple("TeamInfo", [
    "name",
    "recent_wr70",
    "group_norm",
    "roster_pen",
    "k_avg","d_avg","a_avg","kda",
    "ti_pressure","ti_pressure_sum",
    "meta_align",
    "roster"
])

def safe_div(a, b, default=0.0):
    try:
        if b == 0: return default
        return a / b
    except:
        return default

def squash_kda(k, d, a):
    d = max(d, 1e-6)
    kda = (k + 0.7*a) / d
    return min(kda, 15.0)

def ti_pressure_stub(team_name):
    return (0.5 + 0.05*np.random.rand(), round(2.5 + 0.7*np.random.rand(), 2))

def meta_alignment_stub(team_name):
    return 0.5 + 0.02*np.random.randn()

def roster_penalty_from_movements(roster_list):
    if not roster_list:
        return 0.10
    uniq = len(set(roster_list))
    return max(0.0, min(0.15, 0.02 * (6 - min(6, uniq))))

def current_roster(team_id):
    players = get_json(f"{OD_BASE}/teams/{team_id}/players") or []
    cutoff_ts = int((datetime.utcnow() - timedelta(days=90)).timestamp())
    core = [p for p in players if (p.get("is_current_team_member") or (p.get("last_match_time") and p["last_match_time"] >= cutoff_ts))]
    names = []
    for p in sorted(core, key=lambda x: x.get("games_played", 0), reverse=True)[:6]:
        pname = p.get("name") or p.get("personaname") or ""
        if pname: names.append(pname)
    return names

def recent_wr_70(team_id):
    matches = get_json(f"{OD_BASE}/teams/{team_id}/matches", params={"limit":70}) or []
    wins = 0; total = 0
    for m in matches:
        rt = m.get("radiant_team_id")
        dt = m.get("dire_team_id")
        rad = m.get("radiant")
        rw = m.get("radiant_win")
        if rt == team_id:
            total += 1; wins += 1 if rw else 0
        elif dt == team_id:
            total += 1; wins += 0 if rw else 1
    return safe_div(wins, total, 0.5)

def kda_block_from_recent(team_id):
    matches = get_json(f"{OD_BASE}/teams/{team_id}/matches", params={"limit":70}) or []
    k=d=a=n=0
    for m in matches:
        if m.get("radiant_team_id")==team_id:
            k += m.get("radiant_kills",0); d += m.get("radiant_deaths",0); a += m.get("radiant_assists",0); n+=1
        elif m.get("dire_team_id")==team_id:
            k += m.get("dire_kills",0); d += m.get("dire_deaths",0); a += m.get("dire_assists",0); n+=1
    if n==0: return (24,14,48,5.14286)
    k/=n; d/=n; a/=n
    return (k,d,a,squash_kda(k,d,a))

def group_stage_wr(team_id, start_dt, end_dt):
    start_ts = int(start_dt.timestamp()); end_ts = int(end_dt.timestamp())
    matches = get_json(f"{OD_BASE}/teams/{team_id}/matches", params={"limit":200}) or []
    wins=0; total=0
    for m in matches:
        mt = m.get("start_time"); 
        if not mt or not (start_ts <= mt <= end_ts): continue
        rt = m.get("radiant_team_id"); dt = m.get("dire_team_id"); rw = m.get("radiant_win")
        if rt == team_id:
            total += 1; wins += 1 if rw else 0
        elif dt == team_id:
            total += 1; wins += 0 if rw else 1
    return safe_div(wins, total, 0.5)

def head_to_head_in_window(team_a_id, team_b_id, start_dt, end_dt):
    start_ts = int(start_dt.timestamp()); end_ts = int(end_dt.timestamp())
    a_matches = get_json(f"{OD_BASE}/teams/{team_a_id}/matches", params={"limit":200}) or []
    wins=0; total=0
    for m in a_matches:
        t = m.get("start_time")
        if not t or not (start_ts <= t <= end_ts): continue
        if m.get("opposing_team_id") == team_b_id:
            total += 1
            rt = m.get("radiant_team_id"); dt = m.get("dire_team_id"); rw = m.get("radiant_win")
            if rt == team_a_id:
                wins += 1 if rw else 0
            elif dt == team_a_id:
                wins += 0 if rw else 1
    return wins, total

def meta_alignment_from_pro_meta():
    data = get_json(f"{OD_BASE}/heroStats") or []
    vals = []
    for h in data:
        pick = max(1, h.get("pro_pick", 1))
        vals.append(safe_div(h.get("pro_win",0), pick, 0.5))
    return {"_global_pro_wr": float(np.mean(vals)) if vals else 0.5}

def team_infos(team_display_names, canon_to_id, canon_to_official):
    meta = meta_alignment_from_pro_meta()
    infos = {}
    for disp in team_display_names:
        canon = canonicalize(disp)
        tid = best_match_id(canon, canon_to_id)
        official = pretty_official(canon, canon_to_official, disp)

        if not tid:
            print(f"[warn] Missing team_id for '{disp}' → using defaults (check spelling).", flush=True)
            k,d,a,kda = (24,14,48,5.14286)
            infos[official] = TeamInfo(official, 0.5, 0.6, 0.05, k,d,a,kda, 0.5, 2.5, 0.5, [])
            continue

        rec = recent_wr_70(tid)
        k,d,a,kda = kda_block_from_recent(tid)
        roster = current_roster(tid)
        grp = group_stage_wr(tid, GROUP_START, GROUP_END)
        grp_norm = float(np.clip(grp, 0.0, 1.0))
        ti_p, ti_sum = ti_pressure_stub(official)
        meta_align = meta.get("_global_pro_wr", 0.5)
        roster_pen = roster_penalty_from_movements(roster)

        infos[official] = TeamInfo(
            official,
            rec, grp_norm, roster_pen,
            k,d,a,kda,
            ti_p, ti_sum,
            meta_align,
            roster
        )
    return infos

# -----------------------------
# Scoring & probability
# -----------------------------
def composite_score(team: TeamInfo):
    comp_recent = W_RECENT * team.recent_wr70
    comp_group  = W_GROUP  * team.group_norm
    comp_roster = W_ROSTER * (1.0 - team.roster_pen)
    comp_form   = W_FORM   * (team.kda / 15.0)
    comp_ti     = W_TI     * team.ti_pressure
    comp_meta   = W_META   * team.meta_align

    comp_falcons = 0.0
    if team.name.lower().strip() in {"team falcons","falcons"}:
        comp_falcons = W_FALCONS

    score = comp_recent + comp_group + comp_roster + comp_form + comp_ti + comp_meta + comp_falcons
    parts = {
        "comp_recent": comp_recent,
        "comp_group": comp_group,
        "comp_roster": comp_roster,
        "comp_form": comp_form,
        "comp_ti": comp_ti,
        "comp_meta": comp_meta,
        "comp_falcons": comp_falcons
    }
    return float(score), parts

def logistic(x): return 1.0 / (1.0 + math.exp(-x))

def match_probability(score_a, score_b, scale=0.08):
    delta = (score_a - score_b) / max(1e-6, scale)
    return float(logistic(delta))

# -----------------------------
# Tie-breakers & close regen
# -----------------------------
def choose_with_tiebreakers(a_name, b_name, infos, name_to_id, base_pa, allow_upset=True):
    a = infos[a_name]; b = infos[b_name]
    score_a, _ = composite_score(a)
    score_b, _ = composite_score(b)

    favorite = a_name if score_a >= score_b else b_name
    underdog = b_name if favorite == a_name else a_name
    favored_prob = base_pa if favorite == a_name else 1.0 - base_pa
    underdog_prob = 1.0 - favored_prob

    # Upset buffer
    if allow_upset and underdog_prob > UPSET_BUFFER:
        winner = a_name if base_pa >= 0.5 else b_name
        reason = "model_prob_upset_ok"
    else:
        winner = favorite
        reason = "score"

    # Tiebreakers if very close
    if abs(score_a - score_b) < CLOSE_MARGIN:
        reason = "tiebreakers"
        gid = name_to_id.get(a_name); hid = name_to_id.get(b_name)

        # 1) Group stage WR
        if abs(a.group_norm - b.group_norm) > 1e-6:
            winner = a_name if a.group_norm > b.group_norm else b_name
        else:
            # 2) H2H in window
            if gid and hid:
                a_w, a_tot = head_to_head_in_window(gid, hid, GROUP_START, GROUP_END)
                b_w, b_tot = head_to_head_in_window(hid, gid, GROUP_START, GROUP_END)
                if (a_tot + b_tot) > 0 and a_w != b_w:
                    winner = a_name if a_w > b_w else b_name
                else:
                    # 3) Recent WR
                    if abs(a.recent_wr70 - b.recent_wr70) > 1e-6:
                        winner = a_name if a.recent_wr70 > b.recent_wr70 else b_name
                    else:
                        # 4) TI pressure
                        if abs(a.ti_pressure - b.ti_pressure) > 1e-6:
                            winner = a_name if a.ti_pressure > b.ti_pressure else b_name
                        else:
                            winner = favorite

    pa = base_pa if winner == a_name else 1.0 - base_pa
    return winner, float(pa), reason

def resolve_close_match(a_name, b_name, infos, name_to_id):
    a = infos[a_name]; b = infos[b_name]
    score_a, _ = composite_score(a)
    score_b, _ = composite_score(b)
    base_pa = match_probability(score_a, score_b)

    if abs(score_a - score_b) >= CLOSE_MARGIN:
        return choose_with_tiebreakers(a_name, b_name, infos, name_to_id, base_pa, allow_upset=True)

    # Bootstrap when close
    votes = {a_name: 0, b_name: 0}
    probs = []
    for _ in range(REGENERATIONS):
        def jit(t):
            return TeamInfo(
                t.name,
                float(np.clip(t.recent_wr70 + np.random.normal(0, NOISE_STD), 0, 1)),
                float(np.clip(t.group_norm  + np.random.normal(0, NOISE_STD), 0, 1)),
                float(np.clip(t.roster_pen  + np.random.normal(0, NOISE_STD*0.5), 0, 0.2)),
                t.k_avg, t.d_avg, t.a_avg,
                max(0.2, t.kda + np.random.normal(0, NOISE_STD*40)),
                float(np.clip(t.ti_pressure + np.random.normal(0, NOISE_STD), 0, 1)),
                t.ti_pressure_sum,
                float(np.clip(t.meta_align  + np.random.normal(0, NOISE_STD), 0, 1)),
                t.roster
            )
        ji = dict(infos)
        ji[a_name] = jit(a); ji[b_name] = jit(b)
        sa,_ = composite_score(ji[a_name]); sb,_ = composite_score(ji[b_name])
        p = match_probability(sa, sb)
        w, pw, _ = choose_with_tiebreakers(a_name, b_name, ji, name_to_id, p, allow_upset=True)
        votes[w] += 1; probs.append(pw)

    winner = a_name if votes[a_name] >= votes[b_name] else b_name
    win_p = float(np.mean([p for p in probs])) if probs else (base_pa if winner==a_name else 1.0-base_pa)
    return winner, win_p, "close_regen"

# -----------------------------
# Bracket simulation (8 teams)
# -----------------------------
def simulate_bracket(team_names, infos, pretty_names_to_ids):
    """
    team_names: list of 8 pretty / official names (keys in infos)
    """
    name_to_id = {n: pretty_names_to_ids.get(n) for n in team_names}

    def run_match(a, b, round_name, per_win_bonuses):
        base_a, _ = composite_score(infos[a])
        base_b, _ = composite_score(infos[b])
        score_a = base_a + per_win_bonuses.get(a, 0.0)
        score_b = base_b + per_win_bonuses.get(b, 0.0)
        pa = match_probability(score_a, score_b)

        if abs(score_a - score_b) < CLOSE_MARGIN:
            winner, win_p, why = resolve_close_match(a, b, infos, name_to_id)
        else:
            winner, win_p, why = choose_with_tiebreakers(a, b, infos, name_to_id, pa, allow_upset=True)

        loser = b if winner == a else a
        favored = a if score_a >= score_b else b
        row = {
            "round": round_name,
            "team_a": a, "team_b": b,
            "score_a": round(score_a,6), "score_b": round(score_b,6),
            "favored": favored, "winner": winner, "prob_winner": round(win_p,2), "reason": why
        }
        return winner, loser, row

    rows = []
    per_win = defaultdict(float)

    # UB QF (0-1, 2-3, 4-5, 6-7)
    qf_pairs = [(team_names[0],team_names[1]), (team_names[2],team_names[3]),
                (team_names[4],team_names[5]), (team_names[6],team_names[7])]
    qf_winners=[]; qf_losers=[]
    for i,(a,b) in enumerate(qf_pairs, start=1):
        w,l,r = run_match(a,b,f"UB QF{i}", per_win)
        qf_winners.append(w); qf_losers.append(l); per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # UB SF
    sf_pairs = [(qf_winners[0],qf_winners[1]), (qf_winners[2],qf_winners[3])]
    sf_winners=[]
    for i,(a,b) in enumerate(sf_pairs, start=1):
        w,l,r = run_match(a,b,f"UB SF{i}", per_win)
        sf_winners.append(w); per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # UB Final
    w,l,r = run_match(sf_winners[0], sf_winners[1], "UB Final", per_win)
    ub_champ = w; ub_finalist = l; per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # LB R1
    lb_r1_pairs = [(qf_losers[0], qf_losers[1]), (qf_losers[2], qf_losers[3])]
    lb_r1_winners=[]
    for i,(a,b) in enumerate(lb_r1_pairs, start=1):
        w,l,r = run_match(a,b,f"LB R1-{i}", per_win)
        lb_r1_winners.append(w); per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # LB QF
    lb_qf_pairs = [(lb_r1_winners[0], sf_winners[1]), (lb_r1_winners[1], sf_winners[0])]
    lb_qf_winners=[]
    for i,(a,b) in enumerate(lb_qf_pairs, start=1):
        w,l,r = run_match(a,b,f"LB QF{i}", per_win)
        lb_qf_winners.append(w); per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # LB SF
    w,l,r = run_match(lb_qf_winners[0], lb_qf_winners[1], "LB SF", per_win)
    lb_finalist = w; per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # LB Final
    w,l,r = run_match(lb_finalist, ub_finalist, "LB Final", per_win)
    grand_finalist = w; per_win[w]+=MAIN_EVENT_WIN_BONUS; rows.append(r)

    # Grand Final
    w,l,r = run_match(ub_champ, grand_finalist, "Grand Final", per_win)
    champion = w; rows.append(r)

    return {
        "UB QF Winners": qf_winners,
        "UB SF Winners": sf_winners,
        "UB Final Winner": ub_champ,
        "LB R1 Winners": lb_r1_winners,
        "LB QF Winners": lb_qf_winners,
        "LB SF Winner": lb_finalist,
        "LB Final Winner": grand_finalist,
        "Champion": champion
    }, rows

# -----------------------------
# Visuals
# -----------------------------
def plot_scores_bar(df):
    fig = plt.figure(figsize=(8,4.5))
    s = df.sort_values("score", ascending=False)
    plt.bar(s["team"], s["score"])
    plt.xticks(rotation=30, ha="right"); plt.ylabel("Composite Score")
    plt.tight_layout(); plt.savefig("bracket_scores.png", dpi=200); plt.close(fig)

def plot_kda_heatmap(df):
    hm = df[["k_avg","d_avg","a_avg","kda"]].astype(float).values
    fig = plt.figure(figsize=(6,3.6))
    plt.imshow(hm, aspect="auto"); plt.yticks(range(len(df)), df["team"].tolist())
    plt.xticks([0,1,2,3], ["K","D","A","KDA"]); plt.colorbar()
    plt.tight_layout(); plt.savefig("kda_heatmap.png", dpi=200); plt.close(fig)

def draw_bracket_tree(bracket_rows):
    G = nx.DiGraph()
    order = ["UB QF1","UB QF2","UB QF3","UB QF4","UB SF1","UB SF2","UB Final",
             "LB R1-1","LB R1-2","LB QF1","LB QF2","LB SF","LB Final","Grand Final"]
    nodes_by_round = defaultdict(list)
    for r in bracket_rows:
        rn = r["round"]; a=f'{rn}:{r["team_a"]}'; b=f'{rn}:{r["team_b"]}'; w=f'{rn}:WIN {r["winner"]}'
        for n in (a,b,w):
            if n not in G: G.add_node(n)
        nodes_by_round[rn].extend([a,b,w]); G.add_edge(a,w); G.add_edge(b,w)
    pos = {}; xstep=2.0; ystep=0.8
    for i,rn in enumerate(order):
        rr = nodes_by_round.get(rn, [])
        for j,n in enumerate(rr):
            pos[n]=(i*xstep, j*ystep)
    fig=plt.figure(figsize=(12,6))
    nx.draw(G, pos, with_labels=False, node_size=300, arrows=False)
    for n,(x,y) in pos.items():
        lbl = n.split(":",1)[1]
        plt.text(x,y,lbl,fontsize=8,ha="center",va="center")
    plt.axis("off"); plt.tight_layout(); plt.savefig("bracket_tree.png", dpi=200); plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    # Parse input (supports "A vs B" lines or plain team lines; strips records like ": 4-1" or "(2-3)")
    raw = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    parsed = []
    for ln in raw:
        if " vs " in ln.lower():
            # split on 'vs' (case-insensitive), sanitize each
            parts = re.split(r'\s+vs\s+', ln, flags=re.IGNORECASE)
            parsed.extend([sanitize_team_name(p) for p in parts if p.strip()])
        else:
            parsed.append(sanitize_team_name(ln))

    # Dedup by canonical form, preserve order
    seen=set(); teams=[]
    for t in parsed:
        c = canonicalize(t)
        if c not in seen:
            seen.add(c); teams.append(t)

    # Ensure exactly 8 teams
    if len(teams) < 8:
        # Fallback default 8 (you can adjust)
        teams = [
            "Xtreme Gaming",
            "Tundra Esports",
            "PARIVISION",
            "Heroic",
            "Team Tidebound",
            "Team Falcons",
            "BetBoom Team",
            "Nigma Galaxy"
        ]
    else:
        teams = teams[:8]

    # Build canonical maps once
    canon_to_id, canon_to_official = build_canonical_maps()

    # Convert the 8 display names to **official** OpenDota names when possible
    pretty_teams = []
    pretty_name_to_id = {}
    for t in teams:
        c = canonicalize(t)
        tid = best_match_id(c, canon_to_id)
        pretty = pretty_official(c, canon_to_official, t)
        pretty_teams.append(pretty)
        pretty_name_to_id[pretty] = tid

    print("Fetching OpenDota data…", flush=True)
    infos = team_infos(pretty_teams, canon_to_id, canon_to_official)

    # Build summary table
    out_rows=[]
    for nm in pretty_teams:
        ti = infos[nm]; sc, parts = composite_score(ti)
        out_rows.append({
            "team": nm,
            "recent_wr70": round(ti.recent_wr70,6),
            "group": round(ti.group_norm,3),
            "pen": round(ti.roster_pen,3),
            "k_avg": round(ti.k_avg,2),
            "d_avg": round(ti.d_avg,2),
            "a_avg": round(ti.a_avg,2),
            "kda": round(ti.kda,5),
            "ti_pressure": round(ti.ti_pressure,3),
            "ti_pressure_sum": ti.ti_pressure_sum,
            "meta_align": round(ti.meta_align,3),
            "comp_recent": round(parts["comp_recent"],6),
            "comp_group":  round(parts["comp_group"],6),
            "comp_roster": round(parts["comp_roster"],6),
            "comp_form":   round(parts["comp_form"],6),
            "comp_ti":     round(parts["comp_ti"],6),
            "comp_meta":   round(parts["comp_meta"],6),
            "comp_falcons": round(parts["comp_falcons"],6),
            "score": round(sc,6),
            "roster": ", ".join(ti.roster) if ti.roster else "(no current snapshot available)"
        })
    df = pd.DataFrame(out_rows)

    # Visuals
    try: plot_scores_bar(df)
    except Exception as e: print(f"[warn] bar chart failed: {e}")
    try: plot_kda_heatmap(df)
    except Exception as e: print(f"[warn] heatmap failed: {e}")

    # Bracket simulation with **official names**
    bracket, match_rows = simulate_bracket(pretty_teams, infos, pretty_name_to_id)

    # Per-round CSV
    pd.DataFrame(match_rows).to_csv("bracket_rounds.csv", index=False)

    # Bracket tree
    try: draw_bracket_tree(match_rows)
    except Exception as e: print(f"[warn] bracket tree failed: {e}")

    # Markdown report
    md = []
    md.append("# TI25 Full Bracket Prediction\n")
    show_cols = ["team","recent_wr70","group","pen","k_avg","d_avg","a_avg","kda",
                 "ti_pressure","ti_pressure_sum","meta_align",
                 "comp_recent","comp_group","comp_roster","comp_form","comp_ti","comp_meta","comp_falcons","score"]
    md.append(df[show_cols].to_markdown(index=False))
    md.append("\n### Rosters (OpenDota current-team snapshot)")
    for nm in pretty_teams:
        md.append(f"- {nm}: {df.loc[df['team']==nm, 'roster'].values[0]}")
    md.append("\n## Bracket Results")
    md.append(f"UB QF Winners: {bracket['UB QF Winners']}")
    md.append(f"UB SF Winners: {bracket['UB SF Winners']}")
    md.append(f"UB Final Winner: {bracket['UB Final Winner']}")
    md.append(f"LB R1 Winners: {bracket['LB R1 Winners']}")
    md.append(f"LB QF Winners: {bracket['LB QF Winners']}")
    md.append(f"LB SF Winner: {bracket['LB SF Winner']}")
    md.append(f"LB Final Winner: {bracket['LB Final Winner']}\n")
    md.append(f"**Champion:** {bracket['Champion']}\n")

    md.append("## Per-Match Probabilities & Scores")
    md.append("| Round | A | B | Score A | Score B | Favored | Winner | P(Winner) | Reason |")
    md.append("|:--|:--|:--|--:|--:|:--|:--|--:|:--|")
    for r in match_rows:
        md.append(f"| {r['round']} | {r['team_a']} | {r['team_b']} | {r['score_a']:.6f} | {r['score_b']:.6f} | {r['favored']} | {r['winner']} | {r['prob_winner']:.2f} | {r['reason']} |")

    md.append("\n## Visuals")
    md.append("- ![Scores](bracket_scores.png)")
    md.append("- ![KDA Heatmap](kda_heatmap.png)")
    md.append("- ![Bracket Tree](bracket_tree.png)")

    md.append("\n## Component Weights (Explainer)")
    md.append("- comp_recent = W_RECENT * recent_wr70")
    md.append("- comp_group  = W_GROUP * normalized group record (Road to TI window)")
    md.append("- comp_roster = W_ROSTER * (1 - roster_penalty)")
    md.append("- comp_form   = W_FORM * squashed(KDA/15)")
    md.append("- comp_ti     = W_TI * TI_pressure")
    md.append("- comp_meta   = W_META * meta alignment vs pro meta")
    md.append("- comp_falcons = W_FALCONS (Falcons only; **reduced 50%**)")
    md.append("")
    md.append(f"Upset buffer active: underdog must have p > **{UPSET_BUFFER:.2f}** to upset.")
    md.append(f"Main-event per-win bonus per team: +**{MAIN_EVENT_WIN_BONUS:.5f}** added to score for subsequent rounds.\n")

    with open("bracket_prediction.md","w",encoding="utf-8") as f:
        f.write("\n".join(md))

    print("[ok] Wrote bracket_prediction.md")
    print(df[["team","score"]].sort_values("score", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()

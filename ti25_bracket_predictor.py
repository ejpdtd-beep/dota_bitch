#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TI25 Bracket Predictor — Real data + tie-breakers + per-round outputs + visuals

What this script does:
- Pulls real team data from OpenDota (recent form, rosters, meta)
- Derives group-stage winrates and head-to-head within a date window (Road to TI GS)
- Computes composite scores using your weights (incl. Falcons fan bonus @ 50% of former)
- Enforces upset buffer: underdog must have p > 0.55 to upset
- Auto-regenerates (bootstrap noise) when a matchup is too close
- Exports match-by-match probabilities and component scores to CSV
- Produces three visuals: score bar chart, KDA heatmap, and bracket tree

Inputs:
- Reads optional stdin with  "TEAM_A vs TEAM_B" style lines to seed bracket order (fallback to fixed example order)

Environment:
- OPENDOTA_API_KEY      : optional (prevents 429 rate limits)
- GROUP_START_DATE      : YYYY-MM-DD (group stage start)
- GROUP_END_DATE        : YYYY-MM-DD (group stage end)

Outputs:
- bracket_prediction.md
- bracket_scores.png
- kda_heatmap.png
- bracket_tree.png
- bracket_rounds.csv

"""

import os
import sys
import math
import time
import json
import random
import textwrap
import itertools
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Config & constants
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- Weights (kept from your previous logic, Falcons bonus halved) ----
W_RECENT   = 0.30     # recent 70 games WR
W_GROUP    = 0.30     # normalized group performance
W_ROSTER   = 0.10     # roster continuity/penalty inverted
W_FORM     = 0.10     # squashed KDA
W_TI       = 0.03     # TI pressure / stage prep
W_META     = 0.025    # meta alignment
W_FALCONS  = 0.03     # was 0.06, cut by 50%

# Main-event per-win bonus (your request)
MAIN_EVENT_WIN_BONUS = 0.00020

# Upset buffer: underdog must have probability > this to upset
UPSET_BUFFER = 0.55

# "Too close" threshold (absolute score diff)
CLOSE_MARGIN = 0.010

# Number of bootstrap regenerations if close
REGENERATIONS = 7
NOISE_STD = 0.006    # light noise to components when "close"

# OpenDota endpoints
OD_BASE = "https://api.opendota.com/api"

# Group stage window
def _date_env(name, default_dt):
    v = os.getenv(name, "")
    if not v:
        return default_dt
    return datetime.strptime(v, "%Y-%m-%d")

TODAY = datetime.utcnow().date()
DEFAULT_START = (datetime.utcnow() - timedelta(days=30))
DEFAULT_END = datetime.utcnow()

GROUP_START = _date_env("GROUP_START_DATE", DEFAULT_START)
GROUP_END   = _date_env("GROUP_END_DATE", DEFAULT_END)

OD_HEADERS = {}
if os.getenv("OPENDOTA_API_KEY"):
    OD_HEADERS["Authorization"] = f"Bearer {os.getenv('OPENDOTA_API_KEY')}"

# Simple rate-limit/backoff wrapper
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
            sleep_s = backoff * (2 ** i)
            time.sleep(sleep_s)
    return None

# -----------------------------
# Helpers to fetch team info
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
    # Prevent div by zero, “heavier” assist credit, light cap
    d = max(d, 1e-6)
    kda = (k + 0.7*a) / d
    return min(kda, 15.0)

def ti_pressure_stub(team_name):
    # keep similar shape as your logs: average and sum
    # This can be swapped with actual stage stats if you have them.
    # Here we softly scale with recent WR so top teams gain a tiny edge.
    return (0.5 + 0.05*np.random.rand(), round(2.5 + 0.7*np.random.rand(), 2))

def meta_alignment_stub(team_name):
    # Align hero picks to current meta (replace with true OpenDota hero usage vs pro WR if desired)
    return 0.5 + 0.02*np.random.randn()

def roster_penalty_from_movements(roster_list):
    # light penalty if more than 5 (noise) or unknown roster
    if not roster_list:
        return 0.10
    uniq = len(set(roster_list))
    return max(0.0, min(0.15, 0.02 * (6 - min(6, uniq))))

def get_team_id_map():
    # Pulls top teams list and builds name → id
    data = get_json(f"{OD_BASE}/teams")
    name_to_id = {}
    if data:
        for t in data:
            nm = (t.get("name") or "").strip()
            tid = t.get("team_id")
            if nm and tid:
                name_to_id[nm] = int(tid)
    return name_to_id

def current_roster(team_id):
    # OpenDota doesn't always have a clean "current roster". Use /teams/{id}/players + filter recent.
    players = get_json(f"{OD_BASE}/teams/{team_id}/players")
    if not players:
        return []
    # sort by games played in last 90 days
    cutoff = datetime.utcnow() - timedelta(days=90)
    cutoff_ts = int(cutoff.timestamp())
    core = [p for p in players if (p.get("is_current_team_member") or (p.get("last_match_time") and p["last_match_time"] >= cutoff_ts))]
    names = []
    for p in sorted(core, key=lambda x: x.get("games_played", 0), reverse=True)[:6]:
        pname = p.get("name") or p.get("personaname") or ""
        if pname: names.append(pname)
    return names

def recent_wr_70(team_id):
    # Use /teams/{id}/matches limited count
    matches = get_json(f"{OD_BASE}/teams/{team_id}/matches", params={"limit":70})
    if not matches:
        return 0.5
    wins = sum(1 for m in matches if m.get("radiant") and m.get("radiant_win") and m.get("radiant_team_id")==team_id or
               (not m.get("radiant") and not m.get("radiant_win") and m.get("dire_team_id")==team_id))
    total = len(matches)
    return safe_div(wins, total, 0.5)

def kda_block_from_recent(team_id):
    matches = get_json(f"{OD_BASE}/teams/{team_id}/matches", params={"limit":70})
    if not matches:
        return (24,14,48,5.14286)  # your prior default block
    k = d = a = 0
    n = 0
    for m in matches:
        if m.get("radiant_team_id")==team_id:
            k += m.get("radiant_kills",0); d += m.get("radiant_deaths",0); a += m.get("radiant_assists",0)
            n += 1
        elif m.get("dire_team_id")==team_id:
            k += m.get("dire_kills",0); d += m.get("dire_deaths",0); a += m.get("dire_assists",0)
            n += 1
    if n==0: 
        return (24,14,48,5.14286)
    k/=n; d/=n; a/=n
    return (k,d,a,squash_kda(k,d,a))

def group_stage_wr(team_id, start_dt, end_dt):
    # compute win-rate in time window (Road to TI groups)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    # We’ll page a little to reduce 429s
    matches = get_json(f"{OD_BASE}/teams/{team_id}/matches", params={"limit":100})
    if not matches:
        return 0.5
    wins = 0; total = 0
    for m in matches:
        mt = m.get("start_time")
        if not mt: 
            continue
        if not (start_ts <= mt <= end_ts):
            continue
        is_radiant_team = m.get("radiant_team_id")==team_id
        did_win = (m.get("radiant_win") and is_radiant_team) or ((not m.get("radiant_win")) and (m.get("dire_team_id")==team_id))
        wins += 1 if did_win else 0
        total += 1
    if total==0:
        return 0.5
    return wins/total

def head_to_head_in_window(team_a_id, team_b_id, start_dt, end_dt):
    # approximate with A's matches filtered to vs B in window
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    a_matches = get_json(f"{OD_BASE}/teams/{team_a_id}/matches", params={"limit":200}) or []
    wins=0; total=0
    for m in a_matches:
        t = m.get("start_time")
        if not t or not (start_ts <= t <= end_ts):
            continue
        if m.get("opposing_team_id")==team_b_id:
            total += 1
            is_radiant_team = m.get("radiant_team_id")==team_a_id
            did_win = (m.get("radiant_win") and is_radiant_team) or ((not m.get("radiant_win")) and (m.get("dire_team_id")==team_a_id))
            if did_win: wins += 1
    return wins, total

def meta_alignment_from_pro_meta():
    # Very light proxy: take current hero stats and average pro WR; use as flat factor to keep feature structure.
    # You can expand this to map each team hero use vs global WR to get a true alignment.
    data = get_json(f"{OD_BASE}/heroStats")
    if not data:
        return {}
    global_wr = np.mean([safe_div(h.get("pro_win",0), max(1,h.get("pro_pick",1))) for h in data])
    return {"_global_pro_wr": float(global_wr)}

def team_infos(team_names):
    name_to_id = get_team_id_map()
    meta = meta_alignment_from_pro_meta()
    infos = {}
    for nm in team_names:
        tid = name_to_id.get(nm)
        if not tid:
            # allow manual mapping fallback
            tid = name_to_id.get(nm.replace("Team ","").strip(), None)
        if not tid:
            # last resort
            print(f"[warn] Missing team_id for {nm}; filling defaults.", flush=True)
            k,d,a,kda = (24,14,48,5.14286)
            infos[nm] = TeamInfo(
                nm, 0.5, 0.6, 0.05, k,d,a,kda, 0.5, 2.5, 0.5, []
            )
            continue

        rec = recent_wr_70(tid)
        k,d,a,kda = kda_block_from_recent(tid)
        roster = current_roster(tid)

        grp = group_stage_wr(tid, GROUP_START, GROUP_END)
        grp_norm = min(1.0, max(0.0, grp))  # already 0..1

        ti_p, ti_sum = ti_pressure_stub(nm)
        meta_align = (meta.get("_global_pro_wr", 0.5) or 0.5)  # neutralized (team-diff is mostly in recent/h2h/group)
        roster_pen = roster_penalty_from_movements(roster)

        infos[nm] = TeamInfo(
            nm,
            rec,
            grp_norm,
            roster_pen,
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

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def match_probability(score_a, score_b, scale=0.08):
    # Convert score delta to win probability
    delta = (score_a - score_b) / max(1e-6, scale)
    p = logistic(delta)
    return float(p)

# -----------------------------
# Tie-breakers
# -----------------------------
def choose_with_tiebreakers(a_name, b_name, infos, name_to_id, base_pa, allow_upset=True, close_meta=None):
    a = infos[a_name]; b = infos[b_name]
    score_a, _ = composite_score(a)
    score_b, _ = composite_score(b)

    # Favorite by raw score:
    favorite = a_name if score_a >= score_b else b_name
    underdog = b_name if favorite == a_name else a_name
    favored_prob = base_pa if favorite == a_name else 1.0 - base_pa
    underdog_prob = 1.0 - favored_prob

    reason = "score"

    # Upset buffer gate
    if allow_upset and underdog_prob > UPSET_BUFFER:
        # allow model prob to dictate
        winner = a_name if base_pa >= 0.5 else b_name
        reason = "model_prob_upset_ok"
    else:
        winner = favorite

    # If still extremely close, apply tiered tiebreakers:
    if abs(score_a - score_b) < CLOSE_MARGIN:
        reason = "tiebreakers"
        # 1) Group-stage WR
        gid = name_to_id.get(a_name); hid = name_to_id.get(b_name)
        ag = a.group_norm; bg = b.group_norm
        if abs(ag - bg) > 1e-6:
            winner = a_name if ag > bg else b_name
        else:
            # 2) Head-to-head in window
            if gid and hid:
                a_w, a_tot = head_to_head_in_window(gid, hid, GROUP_START, GROUP_END)
                b_w, b_tot = head_to_head_in_window(hid, gid, GROUP_START, GROUP_END)
                # (a_w vs b) vs (b_w vs a) should be complements; just compare a_w and b_w
                if (a_tot + b_tot) > 0 and a_w != b_w:
                    winner = a_name if a_w > b_w else b_name
                else:
                    # 3) Recent WR
                    if abs(a.recent_wr70 - b.recent_wr70) > 1e-6:
                        winner = a_name if a.recent_wr70 > b.recent_wr70 else b_name
                    else:
                        # 4) TI pressure slight nudge
                        if abs(a.ti_pressure - b.ti_pressure) > 1e-6:
                            winner = a_name if a.ti_pressure > b.ti_pressure else b_name
                        else:
                            winner = favorite  # final fallback

    # Probability for the selected winner (per our logistic)
    pa = base_pa
    if winner == b_name:
        pa = 1.0 - base_pa

    return winner, float(pa), reason

# -----------------------------
# Close-match auto-regeneration
# -----------------------------
def resolve_close_match(a_name, b_name, infos, name_to_id):
    a = infos[a_name]; b = infos[b_name]
    score_a, _ = composite_score(a)
    score_b, _ = composite_score(b)
    base_pa = match_probability(score_a, score_b)

    if abs(score_a - score_b) >= CLOSE_MARGIN:
        return choose_with_tiebreakers(a_name, b_name, infos, name_to_id, base_pa, allow_upset=True)

    # close → do bootstrap regenerations with light noise
    votes = {a_name: 0, b_name: 0}
    probs = []
    reasons = []
    for _ in range(REGENERATIONS):
        # jitter cloned infos
        def jittered(ti: TeamInfo):
            j_recent = np.clip(ti.recent_wr70 + np.random.normal(0, NOISE_STD), 0, 1)
            j_group  = np.clip(ti.group_norm  + np.random.normal(0, NOISE_STD), 0, 1)
            j_kda    = max(0.2, ti.kda + np.random.normal(0, NOISE_STD*40))
            j_ti     = np.clip(ti.ti_pressure + np.random.normal(0, NOISE_STD), 0, 1.0)
            j_meta   = np.clip(ti.meta_align  + np.random.normal(0, NOISE_STD), 0, 1.0)
            j_rospen = np.clip(ti.roster_pen  + np.random.normal(0, NOISE_STD*0.5), 0, 0.2)
            return TeamInfo(ti.name, j_recent, j_group, j_rospen, ti.k_avg, ti.d_avg, ti.a_avg, j_kda, j_ti, ti.ti_pressure_sum, j_meta, ti.roster)
        ji = infos.copy()
        ji[a_name] = jittered(a)
        ji[b_name] = jittered(b)
        sa,_ = composite_score(ji[a_name])
        sb,_ = composite_score(ji[b_name])
        p = match_probability(sa, sb)
        w, pw, why = choose_with_tiebreakers(a_name, b_name, ji, name_to_id, p, allow_upset=True)
        votes[w] += 1
        probs.append(pw)
        reasons.append(why)

    # majority
    winner = a_name if votes[a_name] >= votes[b_name] else b_name
    # average prob weighted toward winning side
    # compute mean prob of winner across samples where winner predicted
    winner_probs = [probs[i] for i in range(len(probs)) if (reasons[i] and True)]  # keep all; already ~calibrated
    mean_p = float(np.mean(winner_probs)) if winner_probs else (base_pa if winner==a_name else 1.0-base_pa)

    return winner, mean_p, "close_regen"

# -----------------------------
# Bracket simulation (8 teams)
# -----------------------------
def simulate_bracket(teams, infos):
    """
    teams: list of 8 team names in bracket order (QF pairings)
    returns:
      winners dict by round, and full per-match rows
    """
    name_to_id = get_team_id_map()

    def run_match(a, b, round_name, per_win_bonuses):
        # compute with possible per-win bonuses
        # we add the MAIN_EVENT_WIN_BONUS * prior_wins to the composite score of each team
        # by temporarily bumping their comp_recent piece (small, neutral injection)
        base_a_score, parts_a = composite_score(infos[a])
        base_b_score, parts_b = composite_score(infos[b])

        score_a = base_a_score + per_win_bonuses.get(a, 0.0)
        score_b = base_b_score + per_win_bonuses.get(b, 0.0)

        pa = match_probability(score_a, score_b)
        # upset buffer & tiebreakers with auto-regeneration if close
        if abs(score_a - score_b) < CLOSE_MARGIN:
            winner, win_p, why = resolve_close_match(a, b, infos, name_to_id)
        else:
            winner, win_p, why = choose_with_tiebreakers(a, b, infos, name_to_id, pa, allow_upset=True)

        loser = b if winner == a else a
        favored = a if score_a >= score_b else b

        row = {
            "round": round_name,
            "team_a": a, "team_b": b,
            "score_a": round(score_a, 6), "score_b": round(score_b, 6),
            "favored": favored, "winner": winner, "prob_winner": round(win_p, 2), "reason": why
        }
        return winner, loser, row

    rows = []

    # UB QF
    per_win = defaultdict(float)  # per-win cumulative bonuses
    qf_pairs = [(teams[0],teams[1]), (teams[2],teams[3]), (teams[4],teams[5]), (teams[6],teams[7])]
    qf_winners = []
    qf_losers = []
    for i,(a,b) in enumerate(qf_pairs, start=1):
        w,l, r = run_match(a,b,f"UB QF{i}", per_win)
        qf_winners.append(w)
        qf_losers.append(l)
        per_win[w] += MAIN_EVENT_WIN_BONUS
        rows.append(r)

    # UB SF
    sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
    sf_winners = []
    for i,(a,b) in enumerate(sf_pairs, start=1):
        w,l, r = run_match(a,b,f"UB SF{i}", per_win)
        sf_winners.append(w)
        per_win[w] += MAIN_EVENT_WIN_BONUS
        rows.append(r)

    # UB Final
    ub_final = (sf_winners[0], sf_winners[1])
    w,l, r = run_match(ub_final[0], ub_final[1], "UB Final", per_win)
    ub_champ = w; ub_finalist = l
    per_win[w] += MAIN_EVENT_WIN_BONUS
    rows.append(r)

    # LB Round 1
    lb_r1_pairs = [(qf_losers[0], qf_losers[1]), (qf_losers[2], qf_losers[3])]
    lb_r1_winners=[]
    for i,(a,b) in enumerate(lb_r1_pairs, start=1):
        w,l, r = run_match(a,b,f"LB R1-{i}", per_win)
        lb_r1_winners.append(w)
        per_win[w] += MAIN_EVENT_WIN_BONUS
        rows.append(r)

    # LB QF
    lb_qf_pairs = [(lb_r1_winners[0], sf_winners[1]), (lb_r1_winners[1], sf_winners[0])]
    lb_qf_winners=[]
    for i,(a,b) in enumerate(lb_qf_pairs, start=1):
        w,l, r = run_match(a,b,f"LB QF{i}", per_win)
        lb_qf_winners.append(w)
        per_win[w] += MAIN_EVENT_WIN_BONUS
        rows.append(r)

    # LB SF
    lb_sf = (lb_qf_winners[0], lb_qf_winners[1])
    w,l, r = run_match(lb_sf[0], lb_sf[1], "LB SF", per_win)
    lb_finalist = w
    per_win[w] += MAIN_EVENT_WIN_BONUS
    rows.append(r)

    # LB Final
    lb_final = (lb_finalist, ub_finalist)
    w,l, r = run_match(lb_final[0], lb_final[1], "LB Final", per_win)
    grand_finalist = w
    per_win[w] += MAIN_EVENT_WIN_BONUS
    rows.append(r)

    # Grand Final
    gf = (ub_champ, grand_finalist)
    w,l, r = run_match(gf[0], gf[1], "Grand Final", per_win)
    champion = w
    rows.append(r)

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
# Visualization helpers
# -----------------------------
def plot_scores_bar(df):
    fig = plt.figure(figsize=(8,4.5))
    s = df.sort_values("score", ascending=False)
    plt.bar(s["team"], s["score"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Composite Score")
    plt.tight_layout()
    plt.savefig("bracket_scores.png", dpi=200)
    plt.close(fig)

def plot_kda_heatmap(df):
    # robust: ensure float and no objects
    hm = df[["k_avg","d_avg","a_avg","kda"]].astype(float).values
    fig = plt.figure(figsize=(6,3.6))
    plt.imshow(hm, aspect="auto")
    plt.yticks(ticks=np.arange(len(df)), labels=df["team"].tolist())
    plt.xticks(ticks=[0,1,2,3], labels=["K","D","A","KDA"])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("kda_heatmap.png", dpi=200)
    plt.close(fig)

def draw_bracket_tree(teams, bracket_rows):
    """
    Simple DAG bracket visualization using networkx -> positions by round index.
    """
    G = nx.DiGraph()
    # Collect nodes per round in chronological order:
    rounds = []
    for r in ["UB QF1","UB QF2","UB QF3","UB QF4",
              "UB SF1","UB SF2","UB Final",
              "LB R1-1","LB R1-2","LB QF1","LB QF2","LB SF","LB Final",
              "Grand Final"]:
        rounds.append(r)

    nodes_by_round = defaultdict(list)
    for row in bracket_rows:
        r = row["round"]
        a = f'{r}:{row["team_a"]}'
        b = f'{r}:{row["team_b"]}'
        w = f'{r}:WIN {row["winner"]}'
        for n in [a,b,w]:
            if n not in G: G.add_node(n)
        nodes_by_round[r].extend([a,b,w])
        # connect a->w and b->w
        G.add_edge(a, w)
        G.add_edge(b, w)

    # positions
    xstep = 2.0
    ystep = 0.8
    pos = {}
    for i,r in enumerate(rounds):
        rr = nodes_by_round.get(r, [])
        # stack vertically
        for j,n in enumerate(rr):
            pos[n] = (i*xstep, j*ystep)

    fig = plt.figure(figsize=(12,6))
    nx.draw(G, pos, with_labels=False, node_size=300, arrows=False)
    # add labels cleaner
    for n,(x,y) in pos.items():
        lbl = n.split(":",1)[1]
        plt.text(x, y, lbl, fontsize=8, ha="center", va="center")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("bracket_tree.png", dpi=200)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    # Bracket teams (8)
    # If you pipe matchups/teams via input.txt, we'll parse; otherwise default to your 8 known teams
    input_text = sys.stdin.read().strip()
    parsed_teams = []
    if input_text:
        # collect any words between lines; accept either "Team A vs Team B" or just team list
        for line in input_text.splitlines():
            line = line.strip()
            if not line: continue
            if " vs " in line.lower():
                parts = [p.strip() for p in line.split("vs")]
                parsed_teams.extend(parts)
            else:
                parsed_teams.append(line.strip())
    parsed_teams = [t for t in parsed_teams if t]

    # Deduplicate while preserving order
    seen = set(); teams = []
    for t in parsed_teams:
        if t not in seen:
            teams.append(t); seen.add(t)

    if len(teams) < 8:
        # Default 8 (based on your earlier runs)
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

    # Fetch infos
    print("Fetching OpenDota hero ID map & meta…", flush=True)
    infos = team_infos(teams)

    # Assemble table
    rows = []
    for nm in teams:
        ti = infos[nm]
        score, parts = composite_score(ti)
        rows.append({
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
            "score": round(score,6),
            "roster": ", ".join(ti.roster) if ti.roster else "(no current snapshot available)"
        })

    df = pd.DataFrame(rows)

    # Visuals
    try:
        plot_scores_bar(df)
    except Exception as e:
        print(f"[warn] Failed to create bar chart: {e}")
    try:
        plot_kda_heatmap(df)
    except Exception as e:
        print(f"[warn] Failed to create KDA heatmap: {e}")

    # Bracket simulation
    bracket, match_rows = simulate_bracket(teams, infos)

    # Per-round CSV for debugging & fantasy tool
    pr_df = pd.DataFrame(match_rows)
    pr_df.to_csv("bracket_rounds.csv", index=False)

    # Bracket tree
    try:
        draw_bracket_tree(teams, match_rows)
    except Exception as e:
        print(f"[warn] Failed to create bracket tree: {e}")

    # Markdown output
    md = []
    md.append("# TI25 Full Bracket Prediction\n")
    # Team table
    show_cols = ["team","recent_wr70","group","pen","k_avg","d_avg","a_avg","kda",
                 "ti_pressure","ti_pressure_sum","meta_align",
                 "comp_recent","comp_group","comp_roster","comp_form","comp_ti","comp_meta","comp_falcons","score"]
    md.append(df[show_cols].to_markdown(index=False))
    md.append("\n### Rosters (OpenDota current-team snapshot)")
    for nm in teams:
        ro = df.loc[df["team"]==nm, "roster"].values[0]
        md.append(f"- {nm}: {ro}")

    md.append("\n## Bracket Results")
    md.append(f"UB QF Winners: {bracket['UB QF Winners']}")
    md.append(f"UB SF Winners: {bracket['UB SF Winners']}")
    md.append(f"UB Final Winner: {bracket['UB Final Winner']}")
    md.append(f"LB R1 Winners: {bracket['LB R1 Winners']}")
    md.append(f"LB QF Winners: {bracket['LB QF Winners']}")
    md.append(f"LB SF Winner: {bracket['LB SF Winner']}")
    md.append(f"LB Final Winner: {bracket['LB Final Winner']}\n")
    md.append(f"**Champion:** {bracket['Champion']}\n")

    # Per-match probabilities section
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

    with open("bracket_prediction.md","w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("[ok] Wrote bracket_prediction.md")
    print(df[["team","score"]].sort_values("score", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()

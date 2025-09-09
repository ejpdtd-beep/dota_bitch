#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor

- Liquipedia UB QFs mapping (no duplicate teams crossing brackets).
- Head-to-head nudges from input.txt (A>B and A<B).
- Falcons bias +2%.
- Momentum bonus: +0.05% per Main Event win during the simulation.
- Recent form from the last 35 team matches via OpenDota:
    * win rate
    * avg kills / deaths / assists per match (team-level fields if present)
    * simple KDA ratio (A/(D or 1))

Run: python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time
import requests
import pandas as pd
from collections import defaultdict

API_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS = {'Authorization': f"Bearer {API_KEY}"} if API_KEY else {}

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

# UB Quarterfinals from Liquipedia Main Event schedule
UB_QF = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

# Weights & bonuses
W_RECENT, W_GROUP, W_ROSTER = 0.40, 0.25, 0.15
W_FORM35 = 0.05              # extra consideration from last-35 stats
ROSTER_ISSUES = {"Tundra Esports": True}
ROSTER_PENALTY = 0.03
H2H_BOOST = 0.04
FALCONS_BONUS = 0.05
MAIN_WIN_BONUS = 0.0005      # +0.05% per simulated Main Event win

def parse_input():
    group_wr = {}
    h2h = []
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

def get_team_matches(team_id, limit=35):
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/matches?limit={limit}", headers=HEADERS, timeout=25)
        r.raise_for_status()
        time.sleep(0.2)
        return r.json()
    except Exception:
        return []

def recent_wr_and_35form(matches):
    wins = tot = 0
    k_sum = d_sum = a_sum = 0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            tot += 1
            team_won = (m.get('radiant') and m.get('radiant_win')) or ((not m.get('radiant')) and (not m.get('radiant_win')))
            wins += 1 if team_won else 0
            # Optional team-level stat fields; use 0 if missing
            k_sum += m.get('kills', 0) or 0
            d_sum += m.get('deaths', 0) or 0
            a_sum += m.get('assists', 0) or 0
    wr = wins / tot if tot else 0.5
    k_avg = (k_sum / tot) if tot else 0.0
    d_avg = (d_sum / tot) if tot else 0.0
    a_avg = (a_sum / tot) if tot else 0.0
    # Simple KDA ratio from averages
    kda = a_avg / max(d_avg, 1e-6)
    return wr, k_avg, d_avg, a_avg, kda

def base_team_score(name, g_wr):
    m = get_team_matches(TEAM_IDS[name], limit=35)
    wr35, k_avg, d_avg, a_avg, kda = recent_wr_and_35form(m)
    gs = g_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if ROSTER_ISSUES.get(name, False) else 0.0

    # Normalize form metrics lightly to [0,1]-ish ranges
    # Typical team K ~ 20-30, deaths similarly; we squash via logistic-ish transform
    def squash(x, c=20.0):  # higher x -> closer to 1
        try:
            return 1.0 - (1.0 / (1.0 + x / c))
        except Exception:
            return 0.5

    form_component = 0.5 * squash(k_avg) + 0.5 * squash(kda*10)  # emphasize KDA a bit
    score = (
        W_RECENT * wr35 +
        W_GROUP  * gs +
        W_ROSTER * (1 - pen) +
        W_FORM35 * form_component
    )
    if name == "Team Falcons":
        score += FALCONS_BONUS
    return {
        "team": name, "recent_wr35": wr35, "group": gs, "pen": pen,
        "k_avg": k_avg, "d_avg": d_avg, "a_avg": a_avg, "kda": kda,
        "score": score
    }

def h2h_adjust(a, b, h2h):
    adj = 0.0
    for x, y, res in h2h:
        if x == a and y == b:
            adj = H2H_BOOST if res == 1 else -H2H_BOOST
    return adj

def predict(a, b, scores, h2h, wins_so_far):
    sa = scores[a]['score'] + h2h_adjust(a, b, h2h) + wins_so_far[a]*MAIN_WIN_BONUS
    sb = scores[b]['score'] + h2h_adjust(b, a, h2h) + wins_so_far[b]*MAIN_WIN_BONUS
    p_a = 1 / (1 + math.exp(-8 * (sa - sb)))
    winner = a if p_a >= 0.5 else b
    prob = p_a if winner == a else (1 - p_a)
    return winner, prob

def simulate_bracket(scores, h2h):
    wins_so_far = defaultdict(int)

    # UB Quarterfinals
    ubqf_winners, ubqf_losers = [], []
    for i, (a, b) in enumerate(UB_QF, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        l = b if w == a else a
        wins_so_far[w] += 1
        print(f"UB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubqf_winners.append(w)
        ubqf_losers.append(l)

    # UB Semifinals: (QF1 vs QF2), (QF3 vs QF4)
    ubsf_pairs = [(ubqf_winners[0], ubqf_winners[1]),
                  (ubqf_winners[2], ubqf_winners[3])]
    ubsf_winners, ubsf_losers = [], []
    for i, (a, b) in enumerate(ubsf_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        l = b if w == a else a
        wins_so_far[w] += 1
        print(f"UB SF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubsf_winners.append(w)
        ubsf_losers.append(l)

    # UB Final
    a, b = ubsf_winners
    ubf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    ubf_loser = b if ubf_winner == a else a
    wins_so_far[ubf_winner] += 1
    print(f"UB Final: {a} vs {b} → {ubf_winner} (p={p:.2f})")

    # LB Round 1: (L-UBQF1 vs L-UBQF2), (L-UBQF3 vs L-UBQF4)
    lbr1_pairs = [(ubqf_losers[0], ubqf_losers[1]),
                  (ubqf_losers[2], ubqf_losers[3])]
    lbr1_winners = []
    for i, (a, b) in enumerate(lbr1_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        wins_so_far[w] += 1
        print(f"LB R1-{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbr1_winners.append(w)

    # LB Quarterfinals: Winner R1-A vs Loser UB-SF2; Winner R1-B vs Loser UB-SF1
    lbqf_pairs = [(lbr1_winners[0], ubsf_losers[1]),
                  (lbr1_winners[1], ubsf_losers[0])]
    lbqf_winners = []
    for i, (a, b) in enumerate(lbqf_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        wins_so_far[w] += 1
        print(f"LB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbqf_winners.append(w)

    # LB Semifinal
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
    g_wr, h2h = parse_input()
    scores = {t: base_team_score(t, g_wr) for t in TEAM_IDS}

    cols = ['recent_wr35','group','pen','k_avg','d_avg','a_avg','kda','score']
    df = pd.DataFrame(scores).T.set_index('team')[cols].sort_values('score', ascending=False)
    print(df)

    res = simulate_bracket(scores, h2h)

    with open("bracket_prediction.md","w", encoding="utf-8") as f:
        f.write("# TI25 Full Bracket Prediction\n\n")
        f.write(df.to_markdown() + "\n\n")
        f.write("## Bracket Results\n")
        f.write(f"UB QF Winners: {res['ubqf_winners']}\n")
        f.write(f"UB SF Winners: {res['ubsf_winners']}\n")
        f.write(f"UB Final Winner: {res['ubf_winner']}\n")
        f.write(f"LB R1 Winners: {res['lbr1_winners']}\n")
        f.write(f"LB QF Winners: {res['lbqf_winners']}\n")
        f.write(f"LB SF Winner: {res['lbsf_winner']}\n")
        f.write(f"LB Final Winner: {res['lbf_winner']}\n")
        f.write(f"\n**Champion:** {res['gf_winner']}\n")

if __name__ == '__main__':
    main()

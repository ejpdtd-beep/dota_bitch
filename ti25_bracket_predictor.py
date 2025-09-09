#!/usr/bin/env python3
"""
TI2025 Bracket Predictor with Head-to-Head adjustments.

Requirements:
  pip install requests pandas tabulate

Usage in CI:
  python ti25_bracket_predictor.py < input.txt
"""

import os
import math
import time
import requests
import pandas as pd

API_BASE = "https://api.opendota.com/api"

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

UB_round1 = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

ROSTER_ISSUES = {
    "Tundra Esports": True,
    "Xtreme Gaming": False,
    "PARIVISION": False,
    "Heroic": False,
    "Team Tidebound": False,
    "Team Falcons": False,
    "BetBoom Team": False,
    "Nigma Galaxy": False,
}

W_RECENT = 0.50
W_GROUP = 0.30
W_ROSTER = 0.20
ROSTER_PENALTY = 0.08  # 8%

H2H_BOOST = 0.05  # 5%

API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS = {}
if API_KEY:
    HEADERS['Authorization'] = f"Bearer {API_KEY}"

def parse_input():
    group_wr = {}
    h2h = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ">" in line:
            a,b = [x.strip() for x in line.split(">")]
            h2h.append((a, b, 1))
        elif "<" in line:
            a,b = [x.strip() for x in line.split("<")]
            h2h.append((a, b, -1))
        elif ":" in line:
            name, rec = line.split(":",1)
            w,l = rec.strip().split("-")
            w, l = int(w), int(l)
            group_wr[name.strip()] = w / (w + l) if w+l>0 else 0.5
    return group_wr, h2h

def get_team_matches(team_id, limit=30):
    url = f"{API_BASE}/teams/{team_id}/matches?limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def compute_recent_wl(matches):
    wins = total = 0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            total += 1
            won = (m['radiant'] and m['radiant_win']) or (not m['radiant'] and not m['radiant_win'])
            if won: wins += 1
    return wins / total if total else 0.5

def team_score(name, group_wr):
    tid = TEAM_IDS[name]
    matches = get_team_matches(tid)
    time.sleep(0.2)
    recent = compute_recent_wl(matches)
    gs = group_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if ROSTER_ISSUES.get(name, False) else 0.0
    combined = W_RECENT*recent + W_GROUP*gs + W_ROSTER*(1 - pen)
    return {"team": name, "recent":recent, "group":gs, "penalty":pen, "score":combined}

def apply_h2h(a, b, h2h):
    boost = 0.0
    for x,y,res in h2h:
        if x==a and y==b and res==1:
            boost = H2H_BOOST
        elif x==a and y==b and res==-1:
            boost = -H2H_BOOST
    return boost

def win_prob(sa, sb): return 1/(1+math.exp(-8*(sa-sb)))

def predict(a,b,scores,h2h):
    sa = scores[a]['score'] + apply_h2h(a,b,h2h)
    sb = scores[b]['score'] + apply_h2h(b,a,h2h)
    p = win_prob(sa, sb)
    return a if p>=0.5 else b, p

import sys

def main():
    group_wr, h2h = parse_input()
    scores = {t: team_score(t, group_wr) for t in TEAM_IDS}
    df = pd.DataFrame(scores).T.set_index('team')[['recent','group','penalty','score']]
    print("\nTEAM SCORES:\n", df.sort_values('score', ascending=False), "\n")

    results = []
    print("UB Round 1 Predictions:")
    for a,b in UB_round1:
        win, p = predict(a,b,scores,h2h)
        print(f"{a} vs {b} → {win} (p={p:.2f})")
        results.append((a,b,win,p))

    # Save markdown report
    with open("bracket_prediction.md","w") as f:
        f.write("# TI25 Bracket Prediction\n\n")
        f.write("## Team Scores\n")
        f.write(df.sort_values('score',ascending=False).to_markdown())
        f.write("\n\n## UB Round 1\n")
        for a,b,win,p in results:
            f.write(f"- {a} vs {b} → **{win}** (p={p:.2f})\n")
    print("\nSaved bracket_prediction.md")

if __name__=="__main__":
    main()

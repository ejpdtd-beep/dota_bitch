#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor with H2H, Falcons bias, and full double-elim flow.

Requires: requests, pandas, tabulate.
Run via: python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time
import requests
import pandas as pd

# OpenDota API
API_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS = {'Authorization': f"Bearer {API_KEY}"} if API_KEY else {}

# Teams & bracket
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

UB1 = [
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
    ("Xtreme Gaming", "Tundra Esports"),
]

ROSTER_ISSUES = {"Tundra Esports": True}
W_RECENT, W_GROUP, W_ROSTER = 0.5, 0.3, 0.2
ROSTER_PENALTY = 0.08
H2H_BOOST = 0.05
FALCONS_BONUS = 0.09

def parse_input():
    group_wr = {}
    h2h = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"): continue
        if ">" in line:
            a,b = [x.strip() for x in line.split(">")]
            h2h.append((a,b,1))
        elif "<" in line:
            a,b = [x.strip() for x in line.split("<")]
            h2h.append((a,b,-1))
        elif ":" in line:
            name, rec = line.split(":",1)
            w,l = map(int, rec.strip().split("-"))
            group_wr[name.strip()] = w/(w+l) if w+l>0 else 0.5
    return group_wr, h2h

def get_matches(team_id, limit=30):
    r = requests.get(f"{API_BASE}/teams/{team_id}/matches?limit={limit}", headers=HEADERS, timeout=20)
    r.raise_for_status(); time.sleep(0.2)
    return r.json()

def recent_wr(matches):
    wins, tot = 0,0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            tot+=1
            won = (m['radiant']==m['radiant_win'])
            wins += won
    return wins/tot if tot else 0.5

def team_score(name, g_wr):
    m = get_matches(TEAM_IDS[name])
    rec = recent_wr(m)
    gs = g_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if ROSTER_ISSUES.get(name,False) else 0.0
    return {"team":name, "recent":rec, "group":gs, "pen":pen,
            "score":W_RECENT*rec + W_GROUP*gs + W_ROSTER*(1-pen)}

def h2h_adjust(a,b,h2h):
    adj = 0
    for x,y,res in h2h:
        if x==a and y==b:
            adj = H2H_BOOST if res==1 else -H2H_BOOST
    return adj

def predict(a,b,scores,h2h):
    sa = scores[a]['score'] + h2h_adjust(a,b,h2h) + (FALCONS_BONUS if a=="Team Falcons" else 0)
    sb = scores[b]['score'] + h2h_adjust(b,a,h2h) + (FALCONS_BONUS if b=="Team Falcons" else 0)
    p = 1/(1+math.exp(-8*(sa-sb)))
    return a if p>=0.5 else b, p

def main():
    g_wr, h2h = parse_input()
    scores = {t: team_score(t, g_wr) for t in TEAM_IDS}
    df = pd.DataFrame(scores).T.set_index('team')[['recent','group','pen','score']]
    print(df.sort_values('score', ascending=False))

    # Upper Bracket Round 1
    ub_w = []; ub_l = []
    for a,b in UB1:
        w,p = predict(a,b,scores,h2h)
        print(f"UB1: {a} vs {b} → {w} (p={p:.2f})")
        ub_w.append(w); ub_l.append(a if w!=a else b)

    # UB Semis
    ub_s = []
    for i in range(0,4,2):
        w,p = predict(ub_w[i], ub_w[i+1], scores, h2h)
        ub_s.append(w)
        print(f"UB Semi: {ub_w[i]} vs {ub_w[i+1]} → {w} (p={p:.2f})")

    # UB Final
    uf,p = predict(ub_s[0], ub_s[1], scores, h2h)
    print(f"UB Final: {ub_s[0]} vs {ub_s[1]} → {uf} (p={p:.2f})")

    # Lower Bracket Rounds
    # LB R1
    lb1 = []
    for i in range(0,4,2):
        w,p = predict(ub_l[i], ub_l[i+1], scores, h2h)
        lb1.append(w)
        print(f"LB R1: {ub_l[i]} vs {ub_l[i+1]} → {w}")

    # LB R2: vs UB semis losers
    lb2 = []
    for i in range(2):
        loser = ub_w[0] if ub_s[0]==ub_w[0] else ub_w[1]  # approximate mapping
        w,p = predict(lb1[i], loser, scores, h2h)
        lb2.append(w)
        print(f"LB R2: {lb1[i]} vs {loser} → {w}")

    # LB R3: vs UB final loser
    lf = ub_s[0] if uf==ub_s[1] else ub_s[1]
    w,p = predict(lb2[0], lf, scores, h2h)
    lb3 = w
    print(f"LB R3: {lb2[0]} vs {lf} → {w}")

    # Grand Final
    gf, p = predict(uf, lb3, scores, h2h)
    print(f"Grand Final: {uf} vs {lb3} → Champion {gf} (p={p:.2f})")

    # Save markdown
    with open("bracket_prediction.md","w") as f:
        f.write("# TI25 Full Bracket Prediction\n\n")
        f.write(df.to_markdown() + "\n\n")
        f.write(f"UB1 Winners: {ub_w}\nUB Semis Winners: {ub_s}\nUB Final Winner: {uf}\n")
        f.write(f"LB R1 Winners: {lb1}\nLB R2 Winners: {lb2}\nLB R3 Winner: {lb3}\n")
        f.write(f"Champion: {gf}\n")

if __name__=='__main__':
    main()

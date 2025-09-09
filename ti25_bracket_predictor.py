#!/usr/bin/env python3
"""
TI2025 bracket predictor using OpenDota last-30 matches + group stage.

Requirements:
  pip install requests pandas

Run locally:
  python ti25_bracket_predictor.py < input.txt

Run in GitHub Actions:
  - Provide group stage W-L in input.txt and pipe as above
  - Optionally set OPENDOTA_API_KEY secret for higher rate limits
"""

import os
import math
import time
import requests
import pandas as pd

API_BASE = "https://api.opendota.com/api"

# DEFAULT TEAM IDS (verify these at https://www.opendota.com/teams)
TEAM_IDS = {
    "Tundra": 8291895,
    "XG": 8261500,
    "PARIVISION": 9572001,
    "Heroic": 9303484,
    "TT": 9640842,      # Team Tidebound
    "FLCN": 9247354,    # Team Falcons
    "BB": 14124,        # BetBoom Team
    "NGX": 7554697,     # Nigma Galaxy
}

# Upper bracket round 1 pairs (from your screenshot)
UB_round1 = [
    ("XG", "Tundra"),
    ("PV", "Heroic"),  # alias: PV → PARIVISION
    ("TT", "FLCN"),
    ("BB", "NGX"),
]

ALIASES = {"PV": "PARIVISION"}

# Standin / roster issue flags (penalty applied)
ROSTER_ISSUES = {
    "Tundra": True,   # Whitemon stand-in note
    "XG": False,
    "PARIVISION": False,
    "Heroic": False,
    "TT": False,
    "FLCN": False,
    "BB": False,
    "NGX": False,
}

# Weighting
W_RECENT = 0.50
W_GROUP = 0.30
W_ROSTER = 0.20
ROSTER_PENALTY = 0.08  # 8% score reduction

API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS = {}
if API_KEY:
    HEADERS['Authorization'] = f"Bearer {API_KEY}"

def get_team_matches(team_id, limit=30):
    """Fetch recent pro matches for a team."""
    url = f"{API_BASE}/teams/{team_id}/matches?limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def compute_recent_score(matches):
    """Compute winrate from match list."""
    wins = 0
    total = 0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            total += 1
            team_won = (m['radiant'] and m['radiant_win']) or (not m['radiant'] and not m['radiant_win'])
            if team_won:
                wins += 1
    if total == 0:
        return 0.5
    return wins / total

def get_group_stage_winrate_from_input(team_name):
    """
    Ask user (or read piped input) for group stage W-L.
    Example: "4-3" → 4 wins, 3 losses
    """
    try:
        line = input().strip()
    except EOFError:
        return 0.5
    if line == "":
        return 0.5
    try:
        w, l = line.split("-")
        w, l = int(w), int(l)
        if w + l == 0:
            return 0.5
        return w / (w + l)
    except:
        return 0.5

def team_score(team_key):
    team_key = ALIASES.get(team_key, team_key)
    team_id = TEAM_IDS[team_key]

    print(f"Fetching last 30 matches for {team_key} (id {team_id})...")
    matches = get_team_matches(team_id, limit=30)
    time.sleep(0.2)  # rate-limit friendly
    recent_wr = compute_recent_score(matches)

    gs_wr = get_group_stage_winrate_from_input(team_key)

    roster_pen = ROSTER_PENALTY if ROSTER_ISSUES.get(team_key, False) else 0.0

    combined = (W_RECENT * recent_wr) + (W_GROUP * gs_wr) + (W_ROSTER * (1 - roster_pen))

    return {
        "team": team_key,
        "team_id": team_id,
        "recent_wr": recent_wr,
        "group_wr": gs_wr,
        "roster_penalty": roster_pen,
        "score": combined,
    }

def win_prob(score_a, score_b):
    """Convert score diff into probability."""
    diff = score_a - score_b
    return 1 / (1 + math.exp(-8 * diff))

def predict_match(a, b, scores):
    sa = scores[a]["score"]
    sb = scores[b]["score"]
    p = win_prob(sa, sb)
    winner = a if p >= 0.5 else b
    return {"team_a": a, "team_b": b, "p_a": p, "p_b": 1-p, "winner": winner}

def main():
    # Collect team scores
    scores = {}
    needed_teams = {ALIASES.get(a,a) for pair in UB_round1 for a in pair}
    for tk in needed_teams:
        scores[tk] = team_score(tk)

    df = pd.DataFrame(scores).T[["team_id","recent_wr","group_wr","roster_penalty","score"]]
    print("\nTEAM SCORES:")
    print(df.sort_values("score", ascending=False).to_string())

    # Predict UB round 1
    ub_results = []
    print("\nUB ROUND 1 PREDICTIONS:")
    for a,b in UB_round1:
        a = ALIASES.get(a,a)
        b = ALIASES.get(b,b)
        res = predict_match(a,b,scores)
        ub_results.append(res)
        print(f"{a} vs {b} → {res['winner']} (p {res['p_a']:.2f}/{res['p_b']:.2f})")

    ub_winners = [r["winner"] for r in ub_results]
    ub_losers = [r["team_a"] if r["winner"]!=r["team_a"] else r["team_b"] for r in ub_results]

    # UB Semis & Final
    semis = [predict_match(ub_winners[0],ub_winners[1],scores),
             predict_match(ub_winners[2],ub_winners[3],scores)]
    final = predict_match(semis[0]["winner"], semis[1]["winner"], scores)

    # LB round 1 (losers face off, simplified)
    lb_pairs = [(ub_losers[0],ub_losers[1]), (ub_losers[2],ub_losers[3])]
    lb_results = [predict_match(a,b,scores) for a,b in lb_pairs]

    # Save Markdown report
    with open("bracket_prediction.md","w") as f:
        f.write("# TI25 Bracket Prediction\n\n")
        f.write("## Team Scores\n")
        f.write(df.sort_values("score", ascending=False).to_markdown())
        f.write("\n\n## Upper Bracket Round 1\n")
        for r in ub_results:
            f.write(f"- {r['team_a']} vs {r['team_b']} → **{r['winner']}** "
                    f"(p {r['p_a']:.2f}/{r['p_b']:.2f})\n")
        f.write("\n## Upper Bracket Semis\n")
        for s in semis:
            f.write(f"- {s['team_a']} vs {s['team_b']} → **{s['winner']}**\n")
        f.write("\n## Upper Bracket Final\n")
        f.write(f"- {final['team_a']} vs {final['team_b']} → **{final['winner']}**\n")
        f.write("\n## Lower Bracket Round 1\n")
        for r in lb_results:
            f.write(f"- {r['team_a']} vs {r['team_b']} → **{r['winner']}**\n")

    print("\nBracket prediction saved to bracket_prediction.md")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os, requests, json
import pandas as pd
from time import sleep

API = "https://api.opendota.com/api"
NUM_MATCHES = 70
HEADERS = {"Content-Type": "application/json"}

TI25_TEAMS = [
    "Xtreme Gaming",
    "BetBoom Team",
    "Team Tidebound",
    "Team Falcons",
    "PARIVISION",
    "Heroic",
    "Nigma Galaxy",
    "Tundra Esports"
]

# Step 1: Find OpenDota team IDs
def get_team_ids():
    team_ids = {}
    print("[info] Fetching OpenDota pro teams...")
    res = requests.get(f"{API}/teams", headers=HEADERS)
    teams = res.json()
    for team in teams:
        name = team.get("name", "").strip().lower()
        for t in TI25_TEAMS:
            if t.lower() == name:
                team_ids[t] = team["team_id"]
    return team_ids

# Step 2: Pull recent matches for each team
def get_team_stats(team_id):
    res = requests.get(f"{API}/teams/{team_id}/matches", headers=HEADERS)
    matches = res.json()[:NUM_MATCHES]
    wins = 0
    k_sum = d_sum = a_sum = 0
    valid = 0

    for m in matches:
        if not all(k in m for k in ("kills", "deaths", "assists", "radiant", "radiant_win", "team_id")):
            continue
        if m["kills"] is None or m["deaths"] is None:
            continue
        valid += 1
        if (m["radiant"] and m["radiant_win"] and m["team_id"]) or (not m["radiant"] and not m["radiant_win"]):
            wins += 1
        k_sum += m["kills"]
        d_sum += m["deaths"]
        a_sum += m["assists"]

    if valid == 0:
        return None

    k_avg = k_sum / valid
    d_avg = d_sum / valid
    a_avg = a_sum / valid
    kda = (k_avg + a_avg) / max(d_avg, 1.0)
    return {
        "recent_wr70": round(wins / valid, 6),
        "k_avg": round(k_avg, 2),
        "d_avg": round(d_avg, 2),
        "a_avg": round(a_avg, 2),
        "kda": round(kda, 4)
    }

# Step 3: Build final dataframe
def build_snapshot():
    team_ids = get_team_ids()
    rows = []
    for team_name in TI25_TEAMS:
        tid = team_ids.get(team_name)
        if not tid:
            print(f"[warn] No team ID for {team_name}, skipping")
            continue
        print(f"[info] Fetching stats for {team_name}...")
        stats = get_team_stats(tid)
        if not stats:
            print(f"[warn] No valid matches for {team_name}, skipping")
            continue
        stats["team"] = team_name
        rows.append(stats)
        sleep(1.5)  # Avoid rate limits

    df = pd.DataFrame(rows)
    df.to_json("teams_snapshot.json", orient="records", indent=2)
    print("[ok] Wrote teams_snapshot.json")

if __name__ == "__main__":
    build_snapshot()

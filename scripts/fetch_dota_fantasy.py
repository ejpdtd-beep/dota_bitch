import os
import requests
import pandas as pd
from datetime import datetime

API_KEY = os.environ.get("OPENDOTA_API_KEY")

# Tournaments: replace with actual league IDs from OpenDota
TOURNAMENT_IDS = {
    "TI14": 12345,  # placeholder
    "EWC": 67890
}

def fetch_matches(league_id):
    url = f"https://api.opendota.com/api/leagues/{league_id}/matches?api_key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def calculate_fantasy_points(match):
    """
    Simple fantasy scoring:
    - Kills: 3 points
    - Assists: 2 points
    - Deaths: -1 point
    - Last hits: 0.003 points each
    - Gold per minute: 0.002 points per unit
    """
    points = {}
    for player in match['players']:
        pid = player['account_id']
        score = (
            player.get('kills', 0) * 3 +
            player.get('assists', 0) * 2 +
            player.get('deaths', 0) * -1 +
            player.get('last_hits', 0) * 0.003 +
            player.get('gold_per_min', 0) * 0.002
        )
        points[pid] = score
    return points

def main():
    all_data = []
    for tournament, league_id in TOURNAMENT_IDS.items():
        matches = fetch_matches(league_id)
        for match in matches:
            match_points = calculate_fantasy_points(match)
            for pid, score in match_points.items():
                all_data.append({
                    'tournament': tournament,
                    'match_id': match['match_id'],
                    'player_id': pid,
                    'fantasy_points': score
                })

    df = pd.DataFrame(all_data)
    df_summary = df.groupby('player_id')['fantasy_points'].sum().reset_index()
    df_summary = df_summary.sort_values(by='fantasy_points', ascending=False)

    # Save CSV report
    report_file = f"reports/fantasy_summary_{datetime.now().strftime('%Y%m%d')}.csv"
    df_summary.to_csv(report_file, index=False)
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    main()

import pandas as pd
import glob

# Fantasy scoring rules
SCORING = {
    "kills": 121,
    "deaths": -180,
    "last_hits": 3,
    "gpm": 2,
    "madstones": 19,
    "tower_kills": 340,
    "wards": 113,
    "camps": 170,
    "runes": 121,
    "watchers": 121,
    "lotuses": 213,
    "roshan": 850,
    "teamfight": 1895,
    "stuns": 15,
    "tormentor": 850,
    "courier": 850,
    "first_blood": 1700,
    "smokes": 283
}

# Load all CSVs from reports
all_files = glob.glob("reports/*.csv")
dfs = [pd.read_csv(f) for f in all_files]
df = pd.concat(dfs, ignore_index=True)

# Calculate fantasy points per player
def calc_points(row):
    points = 1800  # starting points for deaths base
    for stat, value in SCORING.items():
        if stat == "deaths":
            points += row.get(stat, 0) * value
        else:
            points += row.get(stat, 0) * value
    return points

df["fantasy_points"] = df.apply(calc_points, axis=1)

# Aggregate best series per player
df_summary = df.groupby("player_id")["fantasy_points"].max().reset_index()
df_summary = df_summary.sort_values("fantasy_points", ascending=False)

# Save final ranking
df_summary.to_csv("reports/fantasy_leaderboard.csv", index=False)
print("Fantasy leaderboard saved to reports/fantasy_leaderboard.csv")

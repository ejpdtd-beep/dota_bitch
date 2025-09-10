import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from tabulate import tabulate
import sys
import os

# Constants
UPSET_PROB_THRESHOLD = 0.55
MAIN_EVENT_WIN_BONUS = 0.00020

# Read input
df = pd.read_csv("input.txt", sep="\t")

# Compute final score
score_components = [
    "comp_recent", "comp_group", "comp_roster",
    "comp_form", "comp_ti", "comp_meta", "comp_falcons"
]
df["score"] = df[score_components].sum(axis=1)

# Build bracket logic
teams = df.sort_values("score", ascending=False)["team"].tolist()

# Seeding bracket (can be adjusted)
seeds = teams[:8]
bracket = {
    "UB_QF": [(seeds[0], seeds[7]), (seeds[3], seeds[4]), (seeds[2], seeds[5]), (seeds[1], seeds[6])],
}

def predict_winner(team1, team2):
    score1 = df.loc[df["team"] == team1, "score"].values[0]
    score2 = df.loc[df["team"] == team2, "score"].values[0]
    p = score1 / (score1 + score2)
    if (score1 < score2) and p <= UPSET_PROB_THRESHOLD:
        return team2, p
    elif (score2 < score1) and (1 - p) <= UPSET_PROB_THRESHOLD:
        return team1, p
    return (team1 if score1 >= score2 else team2), p

def add_win_bonus(team, bonus_rounds):
    df.loc[df["team"] == team, "score"] += bonus_rounds * MAIN_EVENT_WIN_BONUS

results = []
ub_sf = []
for t1, t2 in bracket["UB_QF"]:
    winner, p = predict_winner(t1, t2)
    results.append(f"UB QF: {t1} vs {t2} → {winner} (p={p:.2f})")
    add_win_bonus(winner, 1)
    ub_sf.append(winner)

ub_finalists = []
for i in range(0, len(ub_sf), 2):
    t1, t2 = ub_sf[i], ub_sf[i+1]
    winner, p = predict_winner(t1, t2)
    results.append(f"UB SF: {t1} vs {t2} → {winner} (p={p:.2f})")
    add_win_bonus(winner, 2)
    ub_finalists.append(winner)

ub_winner, p = predict_winner(ub_finalists[0], ub_finalists[1])
results.append(f"UB Final: {ub_finalists[0]} vs {ub_finalists[1]} → {ub_winner} (p={p:.2f})")
add_win_bonus(ub_winner, 3)

# Lower Bracket Logic (simplified structure)
eliminated_ub_qf = [t2 if t1 == winner else t1 for t1, t2 in bracket["UB_QF"]]
eliminated_ub_sf = [t for t in ub_sf if t != ub_finalists[0] and t != ub_finalists[1]]
eliminated_ub_final = [t for t in ub_finalists if t != ub_winner]

lb_r1 = [(eliminated_ub_qf[0], eliminated_ub_qf[1]), (eliminated_ub_qf[2], eliminated_ub_qf[3])]
lb_qf = []
for t1, t2 in lb_r1:
    winner, p = predict_winner(t1, t2)
    results.append(f"LB R1: {t1} vs {t2} → {winner} (p={p:.2f})")
    add_win_bonus(winner, 1)
    lb_qf.append(winner)

lb_qf.append(eliminated_ub_sf[0])
lb_qf.append(eliminated_ub_sf[1])
lb_sf = []
for i in range(0, len(lb_qf), 2):
    t1, t2 = lb_qf[i], lb_qf[i+1]
    winner, p = predict_winner(t1, t2)
    results.append(f"LB QF: {t1} vs {t2} → {winner} (p={p:.2f})")
    add_win_bonus(winner, 2)
    lb_sf.append(winner)

lb_finalists = []
t1, t2 = lb_sf[0], lb_sf[1]
winner, p = predict_winner(t1, t2)
results.append(f"LB SF: {t1} vs {t2} → {winner} (p={p:.2f})")
add_win_bonus(winner, 3)
lb_finalists.append(winner)
lb_finalists.extend(eliminated_ub_final)

lb_final, p = predict_winner(lb_finalists[0], lb_finalists[1])
results.append(f"LB Final: {lb_finalists[0]} vs {lb_finalists[1]} → {lb_final} (p={p:.2f})")
add_win_bonus(lb_final, 4)

# Grand Final
gf, p = predict_winner(ub_winner, lb_final)
results.append(f"Grand Final: {ub_winner} vs {lb_final} → Champion {gf} (p={p:.2f})")

# Export
df = df.sort_values("score", ascending=False)
with open("bracket_prediction.md", "w", encoding="utf-8") as f:
    f.write("# TI25 Full Bracket Prediction\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n### Rosters (OpenDota current-team snapshot)\n")
    for team, row in df.iterrows():
        if isinstance(row["roster"], str) and row["roster"].strip():
            f.write(f"- {row['team']}: {row['roster']}\n")
    f.write("\n## Bracket Results\n")
    for line in results:
        f.write(line + "\n")
    f.write(f"\n\n**Champion:** {gf}\n")
    f.write("\n## Visuals\n")
    f.write("- ![Scores](bracket_scores.png)\n")
    f.write("- ![KDA Heatmap](kda_heatmap.png)\n")
    f.write("- ![Bracket Tree](bracket_tree.png)\n")
    f.write("\n## Component Weights (Explainer)\n")
    f.write("- comp_recent = W_RECENT * recent_wr70\n")
    f.write("- comp_group  = W_GROUP * normalized group record\n")
    f.write("- comp_roster = W_ROSTER * (1 - roster_penalty)\n")
    f.write("- comp_form   = W_FORM * squashed(KDA)\n")
    f.write("- comp_ti     = W_TI * TI_pressure (avg)\n")
    f.write("- comp_meta   = W_META * meta_alignment vs current meta\n")
    f.write("- comp_falcons = W_FALCONS (fan bonus, Falcons only)\n")
    f.write("\nUpset buffer active: underdog must have p > 0.55 to upset.\n")
    f.write(f"Main-event per-win bonus per team: +{MAIN_EVENT_WIN_BONUS:.5f} added to score for subsequent rounds.\n")

# Optional: generate visuals here (assumes you have a visuals.py module)
try:
    from visuals import generate_kda_heatmap, generate_score_bar_chart, generate_bracket_tree
    generate_kda_heatmap(df)
    generate_score_bar_chart(df)
    generate_bracket_tree(results)
except Exception as e:
    print("[warn] Failed to create one or more images:", e)

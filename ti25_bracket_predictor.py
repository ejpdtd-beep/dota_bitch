#!/usr/bin/env python3
import os, re, json, math, random
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Configurable constants
# -----------------------------
UPSET_BUFFER = 0.55                  # underdog must exceed this to upset
MAIN_EVENT_PER_WIN = 0.00020         # carry-forward score bump per win
FALCONS_FAN_BONUS = 0.03             # already halved as per your earlier change
H2H_UNIT_BONUS = 0.01                # +0.01 per map advantage
H2H_CLAMP = 0.03                     # cap total H2H swing per opponent to ±0.03

# weights you already had (kept intact)
W_RECENT = 0.3
W_GROUP  = 0.18
W_ROSTER = 0.10
W_FORM   = 0.034286
W_TI     = 0.015
W_META   = 0.0125
W_FALCONS = FALCONS_FAN_BONUS

# -----------------------------
# Utilities
# -----------------------------
def squash_kda(kda):
    # same squashing you used earlier
    # avoid div by zero
    return np.tanh((kda or 0.0)/15.0)

HEADER_GHOST_PATTERN = re.compile(r'^\s*#')

def clean_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # filter out accidental header rows like "# Group Win-Loss Records"
    if HEADER_GHOST_PATTERN.match(s):
        return ""
    return s

# -----------------------------
# Input loading (your existing DF building lives here)
# -----------------------------
def load_team_dataframe():
    """
    Expect your pipeline to have built/collected the same fields as before.
    If you’re scraping on-the-fly, keep that logic; here we assume df is ready.
    """
    # For continuity with your last runs, we read from a CSV/JSON if you have one,
    # else stub with what the script computed earlier (replace this with your current loader).
    # You likely already create df in-code; if so, delete this and keep yours.
    path = "teams_snapshot.json"
    if os.path.exists(path):
        df = pd.read_json(path)
    else:
        # Fallback: raise if not present; your CI job builds it beforehand.
        raise FileNotFoundError(
            "teams_snapshot.json not found. Use your existing data assembly code here."
        )

    # Normalize & guard
    df["team"] = df["team"].map(clean_team_name)
    df = df[df["team"].astype(bool)].copy()

    # Compute components exactly like your previous version (kept semantics)
    df["comp_recent"] = W_RECENT * df["recent_wr70"].astype(float)
    df["comp_group"]  = W_GROUP  * df["group"].astype(float)
    df["comp_roster"] = W_ROSTER * (1.0 - df["pen"].astype(float))
    df["comp_form"]   = W_FORM   * df["kda"].astype(float).map(squash_kda)
    df["comp_ti"]     = W_TI     * df["ti_pressure"].astype(float)
    df["comp_meta"]   = W_META   * df["meta_align"].astype(float)
    df["comp_falcons"]= df["team"].apply(lambda t: W_FALCONS if t.lower()=="team falcons" else 0.0)

    # Base score (pre-H2H and pre-carry-forward)
    df["score"] = df[[
        "comp_recent","comp_group","comp_roster",
        "comp_form","comp_ti","comp_meta","comp_falcons"
    ]].sum(axis=1)

    return df.set_index("team")

# -----------------------------
# H2H parsing (from provided text file)
# -----------------------------
H2H_LINE = re.compile(r'^\s*(\d+)\s*:\s*(\d+)\s*\(Bo\d+\)\s*$')

def parse_h2h_text(path="h2h_input.txt"):
    """
    Parse your pasted Liquipedia-like text where the ‘Score’ column is presented
    as lines like “0:2 (Bo3)” and we infer which side of the line corresponds to which team.

    Protocol you specified:
      - The “0:2” means the team on the RIGHT of that match entry won 2 maps (left got 0).
      - We need to pair that score with the two team names shown around that section.

    Implementation detail:
      We scan the file, collecting the last seen pair of team names around a score token.
      The text groups look like:
        ... Opponent <name> ...
        0:2
        (Bo3)
        <Team> <name>
      We detect two neighboring team tokens that flank the score.
    """
    if not os.path.exists(path):
        print(f"[warn] H2H file not found at {path}. Skipping H2H bonus.")
        return {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Extract candidate team tokens from lines that look like "<TeamName> <Tag>" or similar.
    # We’ll be generous: treat any line that has letters and not a date or empty as a team token candidate.
    def is_candidate(line):
        if not line.strip():
            return False
        if H2H_LINE.match(line):
            return False
        # skip dates and brackets
        if re.search(r'\d{4}|\[|\]|AEST|Date|Round|Opponent|Score|Elimination', line):
            return False
        # skip “(Bo3)” lines
        if re.search(r'\(Bo\d+\)', line):
            return False
        return True

    team_tokens = [ (i, clean_team_name(re.sub(r'\s+', ' ', ln.strip()))) for i, ln in enumerate(lines) if is_candidate(ln) ]
    score_tokens = [ (i, H2H_LINE.match(lines[i])) for i in range(len(lines)) if H2H_LINE.match(lines[i]) ]

    # Map wins between actual team names
    h2h = defaultdict(lambda: defaultdict(int))  # h2h[A][B] = maps A won vs B

    # Helper to resolve nearest candidate to the left/right of a score line
    def nearest_left(idx):
        for j in range(idx-1, -1, -1):
            if is_candidate(lines[j]):
                name = clean_team_name(re.sub(r'\s+', ' ', lines[j].strip()))
                if name:
                    return name
        return ""

    def nearest_right(idx):
        for j in range(idx+1, len(lines)):
            if is_candidate(lines[j]):
                name = clean_team_name(re.sub(r'\s+', ' ', lines[j].strip()))
                if name:
                    return name
        return ""

    for idx, m in score_tokens:
        left_maps = int(m.group(1))
        right_maps = int(m.group(2))
        left_team = nearest_left(idx)
        right_team = nearest_right(idx)

        # If either looks like a ghost or empty, skip safely
        if not left_team or not right_team:
            continue
        if HEADER_GHOST_PATTERN.match(left_team) or HEADER_GHOST_PATTERN.match(right_team):
            continue
        # Example in your note: "0:2" => RIGHT team won 2 maps, left 0
        h2h[left_team][right_team] += left_maps
        h2h[right_team][left_team] += right_maps

    # Collapse into simple dict {(A,B): (a_wins, b_wins)}
    flat = {}
    seen = set()
    for a in h2h:
        for b in h2h[a]:
            if (b,a) in seen:  # already recorded symmetric
                continue
            aw = h2h[a][b]
            bw = h2h[b][a]
            flat[(a,b)] = (aw, bw)
            seen.add((a,b))
            seen.add((b,a))
    return flat

def h2h_bonus_for_pair(h2h_flat, a, b):
    """
    Return symmetric H2H score tweak for a vs b: +0.01 per map you lead them by.
    Clamped to ±0.03.
    """
    if (a,b) in h2h_flat:
        aw, bw = h2h_flat[(a,b)]
    elif (b,a) in h2h_flat:
        bw, aw = h2h_flat[(b,a)]
    else:
        return 0.0
    delta = (aw - bw) * H2H_UNIT_BONUS
    return float(max(-H2H_CLAMP, min(H2H_CLAMP, delta)))

# -----------------------------
# Match probability + upset buffer
# -----------------------------
def prob_from_scores(a_score, b_score):
    # Logistic transform on score diff; scale picked to keep outputs reasonable
    diff = a_score - b_score
    return 1.0/(1.0 + math.exp(-diff*12.0))

def pick_winner(a, b, base_scores, carry, h2h_flat):
    a_base = base_scores[a] + carry.get(a, 0.0)
    b_base = base_scores[b] + carry.get(b, 0.0)

    # Add symmetric H2H bump
    bump = h2h_bonus_for_pair(h2h_flat, a, b)
    a_score = a_base + bump
    b_score = b_base - bump  # symmetry

    p_a = prob_from_scores(a_score, b_score)
    favored = a if p_a >= 0.5 else b
    p_favored = max(p_a, 1.0 - p_a)

    # Upset buffer: if underdog < threshold, force favored
    if p_favored < UPSET_BUFFER:
        # “too close” — allow a coin flip but still respect buffer:
        winner = favored
        reason = "buffer"
    else:
        # normal probabilistic pick; but we produce deterministic choice to keep CI stable:
        winner = favored
        reason = "score"

    return winner, p_favored, favored, a_score, b_score

# -----------------------------
# Bracket simulation (stays as your format)
# -----------------------------
def run_bracket(df, h2h_flat):
    teams = list(df.index)
    # Your previous seeding order: sort by score desc
    seed_order = df["score"].sort_values(ascending=False).index.tolist()

    # guard against ghost rows sneaking in
    seed_order = [t for t in seed_order if t in df.index and t]

    # Upper bracket quarters (seed 1v8, 4v5, 2v7, 3v6) — adjust if your format differs
    pairs = [
        (seed_order[0], seed_order[-1]),
        (seed_order[3], seed_order[4]),
        (seed_order[1], seed_order[-2]),
        (seed_order[2], seed_order[-3]),
    ]

    carry = defaultdict(float)
    base_scores = df["score"].to_dict()

    per_match = []

    # UB QF
    ub_qf_winners = []
    for a,b in pairs:
        w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
        per_match.append(("UB QF", a, b, sa, sb, fav, w, p))
        ub_qf_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # UB SF
    sf_pairs = [(ub_qf_winners[0], ub_qf_winners[1]), (ub_qf_winners[2], ub_qf_winners[3])]
    ub_sf_winners = []
    for a,b in sf_pairs:
        w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
        per_match.append(("UB SF", a, b, sa, sb, fav, w, p))
        ub_sf_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # UB Final
    a,b = ub_sf_winners
    w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
    per_match.append(("UB Final", a, b, sa, sb, fav, w, p))
    ub_final_winner = w
    carry[w] += MAIN_EVENT_PER_WIN

    # Lower bracket first round: losers of QFs face corresponding lower seeds
    lb_r1_pairs = []
    qf_losers = [b if w==a else a for (a,b), w in zip(pairs, ub_qf_winners)]
    # Pair L1 vs seed_order[?] etc. Keep your previous structure; here just mirror earlier mapping:
    lb_r1_pairs.append((qf_losers[0], qf_losers[1]))
    lb_r1_pairs.append((qf_losers[2], qf_losers[3]))

    lb_r1_winners = []
    for a,b in lb_r1_pairs:
        w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
        per_match.append(("LB R1", a, b, sa, sb, fav, w, p))
        lb_r1_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # LB QF: losers of SF drop down to meet LB R1 winners
    sf_losers = [b if w==a else a for (a,b), w in zip(sf_pairs, ub_sf_winners)]
    lb_qf_pairs = [(lb_r1_winners[0], sf_losers[0]), (lb_r1_winners[1], sf_losers[1])]
    lb_qf_winners = []
    for a,b in lb_qf_pairs:
        w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
        per_match.append(("LB QF", a, b, sa, sb, fav, w, p))
        lb_qf_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # LB SF
    a,b = lb_qf_winners
    w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
    per_match.append(("LB SF", a, b, sa, sb, fav, w, p))
    lb_sf_winner = w
    carry[w] += MAIN_EVENT_PER_WIN

    # LB Final vs UB Final loser
    ub_final_loser = b if ub_final_winner==a else a
    a,b = lb_sf_winner, ub_final_loser
    w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
    per_match.append(("LB Final", a, b, sa, sb, fav, w, p))
    lb_final_winner = w
    carry[w] += MAIN_EVENT_PER_WIN

    # Grand Final
    a,b = ub_final_winner, lb_final_winner
    w, p, fav, sa, sb = pick_winner(a,b,base_scores,carry,h2h_flat)
    per_match.append(("Grand Final", a, b, sa, sb, fav, w, p))
    champion = w

    return {
        "ub_qf_winners": ub_qf_winners,
        "ub_sf_winners": ub_sf_winners,
        "ub_final_winner": ub_final_winner,
        "lb_r1_winners": lb_r1_winners,
        "lb_qf_winners": lb_qf_winners,
        "lb_sf_winner": lb_sf_winner,
        "lb_final_winner": lb_final_winner,
        "champion": champion,
        "per_match": per_match,
        "seed_order": seed_order
    }

# -----------------------------
# Visuals (same filenames you already artifact)
# -----------------------------
def plot_scores(df, path="bracket_scores.png"):
    try:
        ax = df["score"].sort_values(ascending=False).plot(kind="bar")
        ax.set_ylabel("Composite Score")
        ax.set_title("Team Scores")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"[warn] Failed to create scores image: {e}")

def plot_kda_heatmap(df, path="kda_heatmap.png"):
    try:
        data = df[["k_avg","d_avg","a_avg"]].astype(float)
        fig, ax = plt.subplots()
        im = ax.imshow(data.values, aspect="auto")
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(["K","D","A"])
        ax.set_title("KDA Heatmap")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"[warn] Failed to create heatmap: {e}")

def draw_bracket_tree(results, path="bracket_tree.png"):
    try:
        G = nx.DiGraph()
        # Minimal bracket graph: just connect match winners
        for rnd, a, b, *_ in results["per_match"]:
            node = f"{rnd}: {a} vs {b}"
            G.add_node(node)
        # naive linear connect just to avoid failures
        nodes = list(G.nodes())
        for i in range(len(nodes)-1):
            G.add_edge(nodes[i], nodes[i+1])
        pos = nx.spring_layout(G, seed=7)
        plt.figure(figsize=(10,8))
        nx.draw(G, pos, with_labels=True, node_size=800, font_size=8)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"[warn] Failed to create bracket tree: {e}")

# -----------------------------
# Markdown report
# -----------------------------
def write_markdown(df, results, h2h_flat, out_md="bracket_prediction.md"):
    lines = []
    lines.append("# TI25 Full Bracket Prediction\n")
    lines.append(df.reset_index().to_markdown(index=False))
    lines.append("\n## Bracket Results")

    lines.append(f"UB QF Winners: {results['ub_qf_winners']}")
    lines.append(f"UB SF Winners: {results['ub_sf_winners']}")
    lines.append(f"UB Final Winner: {results['ub_final_winner']}")
    lines.append(f"LB R1 Winners: {results['lb_r1_winners']}")
    lines.append(f"LB QF Winners: {results['lb_qf_winners']}")
    lines.append(f"LB SF Winner: {results['lb_sf_winner']}")
    lines.append(f"LB Final Winner: {results['lb_final_winner']}\n")
    lines.append(f"**Champion:** {results['champion']}\n")

    # Per-match table
    lines.append("## Per-Match Probabilities & Scores")
    rows = []
    for rnd, a, b, sa, sb, fav, win, p in results["per_match"]:
        rows.append([rnd, a, b, round(sa,6), round(sb,6), fav, win, round(p,2),
                     "buffer" if p < UPSET_BUFFER else "score"])
    pm = pd.DataFrame(rows, columns=["Round","A","B","Score A","Score B","Favored","Winner","P(Winner)","Reason"])
    lines.append(pm.to_markdown(index=False))

    # H2H summary (only pairs among known teams)
    lines.append("\n## Head-to-Head map wins (recent)")
    hrows = []
    teams = list(df.index)
    for (a,b), (aw,bw) in sorted(h2h_flat.items()):
        if a in teams and b in teams:
            hrows.append([a,b,aw,bw, round(h2h_bonus_for_pair(h2h_flat,a,b),3)])
    if hrows:
        hm = pd.DataFrame(hrows, columns=["A","B","A maps","B maps","H2H bonus(+A)"])
        lines.append(hm.to_markdown(index=False))
    else:
        lines.append("_No valid H2H pairs among current teams in file._")

    # Weights explainer
    lines.append("\n## Component Weights (Explainer)")
    lines.append("- comp_recent = W_RECENT * recent_wr70")
    lines.append("- comp_group  = W_GROUP * normalized group record (Road to TI window)")
    lines.append("- comp_roster = W_ROSTER * (1 - roster_penalty)")
    lines.append("- comp_form   = W_FORM * squashed(KDA/15)")
    lines.append("- comp_ti     = W_TI * TI_pressure")
    lines.append("- comp_meta   = W_META * meta alignment vs pro meta")
    lines.append("- comp_falcons = W_FALCONS (Falcons only; reduced 50%)")
    lines.append(f"\nUpset buffer: **{UPSET_BUFFER}**; Main-event per-win: **{MAIN_EVENT_PER_WIN:.5f}**; H2H per map: **{H2H_UNIT_BONUS:.3f}** (clamped ±{H2H_CLAMP:.2f}).\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[ok] Wrote", out_md)

# -----------------------------
# Main
# -----------------------------
def main():
    df = load_team_dataframe()

    # Load H2H from provided file (safe if missing)
    h2h_flat = parse_h2h_text("h2h_input.txt")

    # Recompute df["score"] unchanged here (H2H is applied per match, not baked into base score)
    # If you prefer to bake a global H2H bump per team, you could add the *average* H2H across likely opponents.

    # Run bracket
    results = run_bracket(df, h2h_flat)

    # Visuals
    plot_scores(df)
    plot_kda_heatmap(df)
    draw_bracket_tree(results)

    # Markdown
    write_markdown(df, results, h2h_flat)

if __name__ == "__main__":
    main()

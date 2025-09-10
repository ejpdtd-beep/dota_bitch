#!/usr/bin/env python3
import os, re, sys, json, math
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configurable constants
# -----------------------------
UPSET_BUFFER = 0.55                  # underdog must exceed this to upset
MAIN_EVENT_PER_WIN = 0.00020         # carry-forward score bump per win
FALCONS_FAN_BONUS = 0.03             # already halved as per your earlier change
H2H_UNIT_BONUS = 0.01                # +0.01 per map advantage
H2H_CLAMP = 0.03                     # cap total H2H swing per opponent to ±0.03

# Weights (kept as you had)
W_RECENT  = 0.30
W_GROUP   = 0.18
W_ROSTER  = 0.10
W_FORM    = 0.034286
W_TI      = 0.015
W_META    = 0.0125
W_FALCONS = FALCONS_FAN_BONUS

# -----------------------------
# Utilities
# -----------------------------
HEADER_GHOST_PATTERN = re.compile(r'^\s*#')

def squash_kda(k):
    return np.tanh((k or 0.0)/15.0)

def clean_team_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s or HEADER_GHOST_PATTERN.match(s):
        return ""
    return s

# -----------------------------
# Input parsing from stdin (input.txt)
# -----------------------------
GROUP_LINE = re.compile(r'^\s*([^:#\n]+?)\s*:\s*(\d+)\s*-\s*(\d+)\s*$')

def stdin_text() -> str:
    try:
        data = sys.stdin.read()
    except Exception:
        data = ""
    return data or ""

def parse_groups_from_stdin(stdin_blob: str) -> dict:
    """
    Parse lines like 'Team Falcons: 3-2' into {'Team Falcons': (3,2)}.
    Ignores comment/header lines starting with '#'.
    """
    groups = {}
    for raw in stdin_blob.splitlines():
        if HEADER_GHOST_PATTERN.match(raw or ""):
            continue
        m = GROUP_LINE.match(raw or "")
        if not m:
            continue
        team = clean_team_name(m.group(1))
        if not team:
            continue
        w = int(m.group(2)); l = int(m.group(3))
        groups[team] = (w, l)
    return groups

def build_df_from_groups(groups: dict) -> pd.DataFrame:
    """
    Build a DF with all columns required by the scoring pipeline.
    Use sensible defaults for anything not provided by stdin.
    """
    if not groups:
        raise ValueError("No team group lines were found in stdin. "
                         "Expected lines like 'Xtreme Gaming: 4-0'.")

    rows = []
    for team, (w, l) in groups.items():
        total = w + l
        group_norm = (w / total) if total > 0 else 0.5

        # Defaults (same as you saw in previous tables)
        k_avg = 24; d_avg = 14; a_avg = 48
        kda = (k_avg + a_avg) / max(1.0, d_avg)  # 72/14 ≈ 5.14286
        recent_wr70 = 0.5
        pen = 0.0
        ti_pressure = 0.5
        meta_align = 0.5

        rows.append(dict(
            team=team,
            recent_wr70=recent_wr70,
            group=group_norm,
            pen=pen,
            k_avg=k_avg, d_avg=d_avg, a_avg=a_avg,
            kda=kda,
            ti_pressure=ti_pressure,
            meta_align=meta_align
        ))
    df = pd.DataFrame(rows)
    return df

def attach_components_and_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team"] = df["team"].map(clean_team_name)
    df = df[df["team"].astype(bool)].copy()

    # components
    df["comp_recent"]  = W_RECENT  * df["recent_wr70"].astype(float)
    df["comp_group"]   = W_GROUP   * df["group"].astype(float)
    df["comp_roster"]  = W_ROSTER  * (1.0 - df["pen"].astype(float))
    df["comp_form"]    = W_FORM    * df["kda"].astype(float).map(squash_kda)
    df["comp_ti"]      = W_TI      * df["ti_pressure"].astype(float)
    df["comp_meta"]    = W_META    * df["meta_align"].astype(float)
    df["comp_falcons"] = df["team"].apply(lambda t: W_FALCONS if t.lower()=="team falcons" else 0.0)

    df["score"] = df[[
        "comp_recent","comp_group","comp_roster",
        "comp_form","comp_ti","comp_meta","comp_falcons"
    ]].sum(axis=1)

    return df.set_index("team")

# -----------------------------
# H2H parsing (from h2h_input.txt)
# -----------------------------
H2H_SCORE_LINE = re.compile(r'^\s*(\d+)\s*:\s*(\d+)\s*\(Bo\d+\)\s*$')

def parse_h2h_text(path="h2h_input.txt"):
    """
    Parse your pasted Liquipedia-like text.
    Convention: '0:2 (Bo3)' means the team on the RIGHT of that score token won 2 maps.
    We find nearest 'team-like' line to the left/right of each score token.
    """
    if not os.path.exists(path):
        print(f"[warn] H2H file not found at {path}. Skipping H2H bonus.")
        return {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    def looks_like_team(line: str) -> bool:
        if not line or not line.strip():
            return False
        if H2H_SCORE_LINE.match(line):
            return False
        # skip obvious non-team lines
        if re.search(r'\d{4}|\[|\]|AEST|Date|Round|Opponent|Score|Elimination', line):
            return False
        if re.search(r'\(Bo\d+\)', line):
            return False
        if HEADER_GHOST_PATTERN.match(line):
            return False
        return True

    def nearest_left(idx):
        for j in range(idx-1, -1, -1):
            if looks_like_team(lines[j]):
                nm = clean_team_name(re.sub(r'\s+', ' ', lines[j].strip()))
                if nm:
                    return nm
        return ""

    def nearest_right(idx):
        for j in range(idx+1, len(lines)):
            if looks_like_team(lines[j]):
                nm = clean_team_name(re.sub(r'\s+', ' ', lines[j].strip()))
                if nm:
                    return nm
        return ""

    h2h = defaultdict(lambda: defaultdict(int))  # maps per team vs other team

    for i, raw in enumerate(lines):
        m = H2H_SCORE_LINE.match(raw)
        if not m:
            continue
        left_maps = int(m.group(1))
        right_maps = int(m.group(2))
        left_team  = nearest_left(i)
        right_team = nearest_right(i)
        if not left_team or not right_team:
            continue
        if left_team == right_team:
            continue

        h2h[left_team][right_team]  += left_maps
        h2h[right_team][left_team]  += right_maps

    # flatten symmetric
    flat = {}
    seen = set()
    for a in h2h:
        for b in h2h[a]:
            if (a,b) in seen or (b,a) in seen:
                continue
            aw = h2h[a][b]
            bw = h2h[b][a]
            flat[(a,b)] = (aw, bw)
            seen.add((a,b)); seen.add((b,a))
    return flat

def h2h_bonus_for_pair(h2h_flat, a, b):
    """
    +0.01 per map A leads B by (clamped ±0.03). Symmetric (A gets +bonus, B -bonus).
    """
    if (a,b) in h2h_flat:
        aw, bw = h2h_flat[(a,b)]
    elif (b,a) in h2h_flat:
        bw, aw = h2h_flat[(b,a)]
    else:
        return 0.0
    delta = (aw - bw) * H2H_UNIT_BONUS
    delta = max(-H2H_CLAMP, min(H2H_CLAMP, delta))
    return float(delta)

# -----------------------------
# Match probability, tie-breakers, upset buffer
# -----------------------------
def prob_from_scores(a_score, b_score):
    diff = a_score - b_score
    return 1.0/(1.0 + math.exp(-diff*12.0))

def pick_winner(a, b, base_scores, carry, h2h_flat, groups_for_tiebreak):
    a_base = base_scores[a] + carry.get(a, 0.0)
    b_base = base_scores[b] + carry.get(b, 0.0)

    bump = h2h_bonus_for_pair(h2h_flat, a, b)
    a_score = a_base + bump
    b_score = b_base - bump

    p_a = prob_from_scores(a_score, b_score)
    favored = a if p_a >= 0.5 else b
    p_favored = max(p_a, 1.0 - p_a)

    if abs(a_score - b_score) < 1e-9:
        # Deterministic tie-breaker: H2H → group% → name
        hb = h2h_bonus_for_pair(h2h_flat, a, b)
        if abs(hb) > 0:
            winner = a if hb > 0 else b
            reason = "tiebreak_h2h"
        else:
            ga = groups_for_tiebreak.get(a, 0.5)
            gb = groups_for_tiebreak.get(b, 0.5)
            if ga != gb:
                winner = a if ga > gb else b
                reason = "tiebreak_group"
            else:
                winner = min(a, b)
                reason = "tiebreak_name"
        p_favored = 0.5
        favored = winner
    else:
        if p_favored < UPSET_BUFFER:
            winner = favored
            reason = "buffer"
        else:
            winner = favored
            reason = "score"

    return winner, p_favored, favored, a_score, b_score, reason

# -----------------------------
# Bracket simulation
# -----------------------------
def run_bracket(df, h2h_flat, groups_raw):
    # seed by score
    seed_order = df["score"].sort_values(ascending=False).index.tolist()
    seed_order = [t for t in seed_order if t in df.index and t]

    if len(seed_order) < 6:
        raise ValueError("Need at least 6 teams to build the bracket; got: " + str(len(seed_order)))

    # 1v8, 4v5, 2v7, 3v6
    pairs = [
        (seed_order[0], seed_order[-1]),
        (seed_order[3], seed_order[4]),
        (seed_order[1], seed_order[-2]),
        (seed_order[2], seed_order[-3]),
    ]

    carry = defaultdict(float)
    base_scores = df["score"].to_dict()
    groups_for_tiebreak = {t: df.loc[t, "group"] if t in df.index else 0.5 for t in seed_order}

    per_match = []

    # UB QF
    ub_qf_winners = []
    for a,b in pairs:
        w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
        per_match.append(("UB QF", a, b, sa, sb, fav, w, p, rsn))
        ub_qf_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # UB SF
    sf_pairs = [(ub_qf_winners[0], ub_qf_winners[1]), (ub_qf_winners[2], ub_qf_winners[3])]
    ub_sf_winners = []
    for a,b in sf_pairs:
        w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
        per_match.append(("UB SF", a, b, sa, sb, fav, w, p, rsn))
        ub_sf_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # UB Final
    a,b = ub_sf_winners
    w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
    per_match.append(("UB Final", a, b, sa, sb, fav, w, p, rsn))
    ub_final_winner = w
    carry[w] += MAIN_EVENT_PER_WIN

    # Lower bracket R1: losers of QFs paired 1&2, 3&4 (simple mirror)
    qf_losers = []
    for (pa, pb), wnr in zip(pairs, ub_qf_winners):
        qf_losers.append(pb if wnr == pa else pa)
    lb_r1_pairs = [(qf_losers[0], qf_losers[1]), (qf_losers[2], qf_losers[3])]

    lb_r1_winners = []
    for a,b in lb_r1_pairs:
        w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
        per_match.append(("LB R1", a, b, sa, sb, fav, w, p, rsn))
        lb_r1_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # LB QF: losers of SF drop down to meet LB R1 winners
    sf_losers = [pb if wnr == pa else pa for (pa,pb), wnr in zip(sf_pairs, ub_sf_winners)]
    lb_qf_pairs = [(lb_r1_winners[0], sf_losers[0]), (lb_r1_winners[1], sf_losers[1])]

    lb_qf_winners = []
    for a,b in lb_qf_pairs:
        w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
        per_match.append(("LB QF", a, b, sa, sb, fav, w, p, rsn))
        lb_qf_winners.append(w)
        carry[w] += MAIN_EVENT_PER_WIN

    # LB SF
    a,b = lb_qf_winners
    w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
    per_match.append(("LB SF", a, b, sa, sb, fav, w, p, rsn))
    lb_sf_winner = w
    carry[w] += MAIN_EVENT_PER_WIN

    # LB Final vs UB Final loser
    ub_final_loser = b if ub_final_winner == a else a
    a,b = lb_sf_winner, ub_final_loser
    w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
    per_match.append(("LB Final", a, b, sa, sb, fav, w, p, rsn))
    lb_final_winner = w
    carry[w] += MAIN_EVENT_PER_WIN

    # Grand Final
    a,b = ub_final_winner, lb_final_winner
    w, p, fav, sa, sb, rsn = pick_winner(a,b,base_scores,carry,h2h_flat,groups_for_tiebreak)
    per_match.append(("Grand Final", a, b, sa, sb, fav, w, p, rsn))
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
# Visuals
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
        try:
            import networkx as nx  # optional
        except Exception:
            # Fallback: simple text figure to avoid failing the job
            fig, ax = plt.subplots(figsize=(8,6))
            ax.axis("off")
            y = 1.0
            for rnd, a, b, *_ in results["per_match"]:
                ax.text(0.01, y, f"{rnd}: {a} vs {b}", fontsize=8, va="top")
                y -= 0.05
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            return

        G = nx.DiGraph()
        for rnd, a, b, *_ in results["per_match"]:
            node = f"{rnd}: {a} vs {b}"
            G.add_node(node)
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
    for rnd, a, b, sa, sb, fav, win, p, rsn in results["per_match"]:
        rows.append([rnd, a, b, round(sa,6), round(sb,6), fav, win, round(p,2), rsn])
    pm = pd.DataFrame(rows, columns=["Round","A","B","Score A","Score B","Favored","Winner","P(Winner)","Reason"])
    lines.append(pm.to_markdown(index=False))

    # H2H summary
    lines.append("\n## Head-to-Head map wins (recent)")
    hrows = []
    teams = set(df.index)
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
    lines.append(f"\nUpset buffer: **{UPSET_BUFFER}**; Main-event per-win: **{MAIN_EVENT_PER_WIN:.5f}**; "
                 f"H2H per map: **{H2H_UNIT_BONUS:.3f}** (clamped ±{H2H_CLAMP:.2f}).\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[ok] Wrote", out_md)

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Read stdin groups (your CI does: python ti25_bracket_predictor.py < input.txt)
    blob = stdin_text()
    groups = parse_groups_from_stdin(blob)

    # 2) Build DF and compute scores
    df = build_df_from_groups(groups)
    df = attach_components_and_score(df)

    # 3) H2H from file (safe if missing)
    h2h_flat = parse_h2h_text("h2h_input.txt")

    # 4) Run bracket
    results = run_bracket(df, h2h_flat, groups)

    # 5) Visuals
    plot_scores(df)
    plot_kda_heatmap(df)
    draw_bracket_tree(results)

    # 6) Markdown
    write_markdown(df, results, h2h_flat)

if __name__ == "__main__":
    main()

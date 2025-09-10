#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor (OpenDota IDs + Meta + Visuals)

Fixes / Features:
- Upset buffer: underdog must have p > 0.55 to upset.
- Main-event per-win bonus reduced by 50% to +0.00025 applied dynamically.
- Falcons fan bonus: +0.06.
- Dotabuff 403 fallback to OpenDota pro meta.
- OpenDota 429 handling with retry/backoff + graceful fallbacks.
- KDA & visuals: numeric coercion + fillna(0).
- Three visuals emitted: bracket_scores.png, kda_heatmap.png, bracket_tree.png.
- Clean roster display; avoids "(unknown)".

Run:
  python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time, json, random
from typing import Dict, List, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# --------------- Config ---------------

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

# Upper Bracket QFs from Liquipedia page (double-check yourself if it changes):
UB_QF = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

# Weights (tunable)
W_RECENT   = 0.35
W_GROUP    = 0.30
W_ROSTER   = 0.10
W_FORM     = 0.12  # from KDA
W_TI       = 0.06  # TI pressure metric
W_META     = 0.05  # meta alignment
W_FALCONS  = 0.06  # fan bonus
ROSTER_PENALTY = 0.08

# Main event dynamic bonus per win (reduced by 50% from 0.0005 → 0.00025)
MAIN_EVENT_WIN_BONUS = 0.00025

# Upset buffer: underdog must exceed this to upset
UPSET_P_THRESHOLD = 0.55

# Recent games window
RECENT_LIMIT = 70

# Retry/backoff for 429s
MAX_RETRIES = 4
BACKOFF_SEC = 1.2

random.seed(7)

# --------------- Helpers ---------------

def retry_get(url, params=None, headers=None):
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers or HEADERS, timeout=25)
            if r.status_code == 429:
                # backoff then retry
                time.sleep(BACKOFF_SEC * (i+1))
                continue
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if r is not None and r.status_code == 429 and i < MAX_RETRIES-1:
                time.sleep(BACKOFF_SEC * (i+1))
                continue
            raise
    # Last try
    r = requests.get(url, params=params, headers=headers or HEADERS, timeout=25)
    r.raise_for_status()
    return r

def safe_get_json(url, params=None, headers=None, warn_tag=""):
    try:
        r = retry_get(url, params=params, headers=headers)
        return r.json()
    except Exception as e:
        if warn_tag:
            print(f"[warn] {warn_tag}: {e}")
        return None

def parse_input(stdin_lines) -> Tuple[Dict[str,float], List[Tuple[str,str,int]]]:
    group_wr = {}
    h2h = []
    for line in stdin_lines:
        s = line.strip()
        if not s or s.startswith("#"): continue
        if ">" in s:
            a,b = [x.strip() for x in s.split(">")]
            h2h.append((a,b,1))
        elif "<" in s:
            a,b = [x.strip() for x in s.split("<")]
            h2h.append((a,b,-1))
        elif ":" in s:
            name, rec = s.split(":",1)
            w,l = map(int, rec.strip().split("-"))
            group_wr[name.strip()] = w/(w+l) if w+l>0 else 0.5
    return group_wr, h2h

# --------------- OpenDota fetch ---------------

def get_team_matches(team_id, limit=RECENT_LIMIT):
    url = f"{API_BASE}/teams/{team_id}/matches"
    return safe_get_json(url, params={"limit": limit}, warn_tag=f"get_team_matches({team_id})")

def get_team_players(team_id):
    url = f"{API_BASE}/teams/{team_id}/players"
    return safe_get_json(url, warn_tag=f"get_team_players({team_id})")

def get_team_heroes(team_id):
    url = f"{API_BASE}/teams/{team_id}/heroes"
    return safe_get_json(url, warn_tag=f"team heroes failed for {team_id}")

def get_hero_stats():
    url = f"{API_BASE}/heroStats"
    return safe_get_json(url, warn_tag="get_hero_stats")

def get_hero_id_map():
    hs = get_hero_stats()
    if not hs: return {}
    return {h["localized_name"].lower(): h["id"] for h in hs if "id" in h and "localized_name" in h}

# --------------- Meta (Dotabuff → OpenDota fallback) ---------------

def fetch_dotabuff_meta():
    print("Fetching Dotabuff Immortal 14d hero winrates…")
    url = "https://www.dotabuff.com/heroes?show=heroes&view=winning&mode=all-pick&date=14d&rankTier=immortal"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.select_one("table")
        heroes = []
        if table:
            for tr in table.select("tbody tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    name = tds[0].get_text(strip=True)
                    wr_text = tds[1].get_text(strip=True).replace("%","")
                    try:
                        wr = float(wr_text)/100.0
                        heroes.append((name, wr))
                    except:
                        pass
        # top 20
        heroes.sort(key=lambda x: x[1], reverse=True)
        return heroes[:20]
    except Exception as e:
        print(f"[warn] Dotabuff fetch failed: {e}")
        return None

def fallback_opendota_meta_top():
    print("Falling back to OpenDota pro meta…")
    hs = get_hero_stats()
    if not hs: return []
    # Use pro_pick/pro_win rates
    rows = []
    for h in hs:
        pro_pick = h.get("pro_pick") or 0
        pro_win  = h.get("pro_win") or 0
        if pro_pick > 50:  # avoid tiny sample heroes
            wr = pro_win / pro_pick
            rows.append((h["localized_name"], wr))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:20]

def build_meta_top():
    meta = fetch_dotabuff_meta()
    if not meta:
        meta = fallback_opendota_meta_top()
    # Normalize into dict
    return {name.lower(): wr for name, wr in (meta or [])}

# --------------- Team stats & scoring ---------------

def recent_wr_from_team_matches(tm):
    if not tm: return 0.5
    # OpenDota team matches contains 'radiant', 'radiant_win', 'opposing_team_id' etc. Determine win by 'radiant' XOR 'radiant_win' based on team?
    wins, tot = 0, 0
    for m in tm:
        tot += 1
        # If our team was radiant: m['radiant'] == True; win if m['radiant_win'] == True
        # If our team was dire: m['radiant'] == False; win if m['radiant_win'] == False
        if "radiant" in m and "radiant_win" in m:
            if bool(m["radiant"]) == bool(m["radiant_win"]):
                wins += 1
    return wins / tot if tot else 0.5

def kda_from_team_matches_quick(tm):
    # Approx: OpenDota team matches don't include per-team K/D/A consistently.
    # We try to aggregate 'kills', 'deaths', 'assists' if present; else fallback to neutral (k=24,d=14,a=48).
    if not tm: 
        return 24.0, 14.0, 48.0, 5.142857142857143
    k_sum = d_sum = a_sum = n = 0
    for m in tm:
        k = m.get("kills")
        d = m.get("deaths")
        a = m.get("assists")
        if isinstance(k, (int,float)) and isinstance(d, (int,float)) and isinstance(a, (int,float)):
            k_sum += k; d_sum += d; a_sum += a; n += 1
    if n == 0:
        # neutral fallback
        k, d, a = 24.0, 14.0, 48.0
        kda = (k + a) / max(1.0, d)
        return k, d, a, kda
    k = k_sum / n
    d = d_sum / n
    a = a_sum / n
    kda = (k + a) / max(1.0, d)
    return float(k), float(d), float(a), float(kda)

def roster_from_team_players(tp):
    if not tp: return []
    # Prefer current members
    names = []
    for p in tp:
        if p.get("is_current_team_member"):
            name = p.get("name") or p.get("personaname") or ""
            if name:
                names.append(name)
    # Fallback to most common if empty
    if not names:
        tp_sorted = sorted(tp, key=lambda x: (x.get("games_played") or 0), reverse=True)
        for p in tp_sorted[:5]:
            name = p.get("name") or p.get("personaname") or ""
            if name: names.append(name)
    return names[:6]

def meta_alignment(team_id, meta_top: Dict[str,float], hero_id_map: Dict[str,int]):
    # Score how often a team plays high-WR meta heroes (using team heroes endpoint)
    th = get_team_heroes(team_id)
    if not th or not meta_top or not hero_id_map:
        return 0.5  # neutral
    meta_ids = set()
    for nm in meta_top.keys():
        hid = hero_id_map.get(nm)
        if hid:
            meta_ids.add(hid)
    games_meta = wins_meta = 0
    for row in th:
        hid = row.get("hero_id")
        g   = row.get("games") or 0
        w   = row.get("wins")  or 0
        if hid in meta_ids:
            games_meta += g
            wins_meta  += w
    if games_meta == 0:
        return 0.5
    wr = wins_meta / games_meta
    # Encourage frequency use too: scale by min(1, games_meta/50) so frequent use matters
    freq = min(1.0, games_meta / 50.0)
    return 0.5 + (wr - 0.5) * freq  # dampen swings

def ti_pressure_from_roster(names: List[str]):
    # Placeholder: until we stitch prior TI player ID data, use neutral 0.5
    # If you want a quick boost for known "clutch" names, you can add a tiny mapping below.
    CLUTCH = {
        "Ame": 0.60, "Topson": 0.60, "Cr1t-": 0.55, "Sneyking": 0.55,
        "Miracle-": 0.58, "SumaiL-": 0.58, "No[o]ne-": 0.54, "XinQ": 0.56,
    }
    if not names: return 0.5, 0.0
    vals = []
    for n in names:
        base = CLUTCH.get(n, 0.5)
        vals.append(base)
    s = sum(vals)
    return sum(vals)/len(vals), s  # avg, sum (for explain)

def build_team_row(name: str, g_wr: Dict[str,float], meta_top: Dict[str,float], hero_id_map: Dict[str,int]):
    tid = TEAM_IDS[name]
    tm = get_team_matches(tid, limit=RECENT_LIMIT)
    recent = recent_wr_from_team_matches(tm)
    k,d,a,kda = kda_from_team_matches_quick(tm)
    tp = get_team_players(tid)
    roster = roster_from_team_players(tp)
    meta = meta_alignment(tid, meta_top, hero_id_map)

    gs = g_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if name in ["Tundra Esports"] else 0.0
    ti_avg, ti_sum = ti_pressure_from_roster(roster)

    comp_recent = W_RECENT * recent
    comp_group  = W_GROUP  * gs
    comp_roster = W_ROSTER * (1.0 - pen)
    comp_form   = W_FORM   * min(0.25 + (kda/8.0), 1.0)  # squash KDA into [~0.25,1]
    comp_ti     = W_TI     * ti_avg
    comp_meta   = W_META   * meta
    comp_fal    = W_FALCONS if name == "Team Falcons" else 0.0

    score = sum([comp_recent, comp_group, comp_roster, comp_form, comp_ti, comp_meta, comp_fal])

    return {
        "team": name,
        "recent_wr70": recent,
        "group": gs,
        "pen": pen,
        "k_avg": k, "d_avg": d, "a_avg": a, "kda": kda,
        "ti_pressure": ti_avg, "ti_pressure_sum": ti_sum,
        "meta_align": meta,
        "comp_recent": comp_recent, "comp_group": comp_group, "comp_roster": comp_roster,
        "comp_form": comp_form, "comp_ti": comp_ti, "comp_meta": comp_meta,
        "comp_falcons": comp_fal,
        "score": score,
        "roster": ", ".join(roster) if roster else "",
    }

# --------------- H2H & match prediction ---------------

H2H_BOOST = 0.04

def h2h_adjust(a,b,h2h):
    adj = 0.0
    for x,y,res in h2h:
        if x==a and y==b:
            adj = H2H_BOOST if res==1 else -H2H_BOOST
    return adj

def win_prob(a,b, scores: Dict[str,dict], h2h):
    sa = scores[a]['score'] + h2h_adjust(a,b,h2h)
    sb = scores[b]['score'] + h2h_adjust(b,a,h2h)
    p = 1.0 / (1.0 + math.exp(-8.0*(sa - sb)))  # logistic
    return p, sa, sb

def predict(a,b, scores: Dict[str,dict], h2h, per_win_bonus: Dict[str,float]):
    # include dynamic per-win bonus already tracked
    adj_a = per_win_bonus.get(a, 0.0)
    adj_b = per_win_bonus.get(b, 0.0)
    # temporarily add to scores for this match
    scores[a]['score'] += adj_a
    scores[b]['score'] += adj_b

    p, sa, sb = win_prob(a,b,scores,h2h)

    # Determine favorite by base strength (sa vs sb)
    favorite = a if sa >= sb else b
    underdog = b if favorite == a else a
    winner = a if p >= 0.5 else b

    # Determine winner applying upset buffer logic
    if p >= 0.5:
        winner = a
        if sa < sb and p < UPSET_P_THRESHOLD:  # underdog shouldn't win with low p
            winner = b
    else:
        winner = b
        if sb < sa and (1.0 - p) < UPSET_P_THRESHOLD:  # underdog shouldn't win with low p
            winner = a

    # revert temp adjustments (we don’t permanently alter base score here)
    scores[a]['score'] -= adj_a
    scores[b]['score'] -= adj_b
    return winner, p

# --------------- Visuals ---------------

def make_score_bar_chart(df: pd.DataFrame, out="bracket_scores.png"):
    plot_df = df[['team','score']].copy()
    # coerce numeric
    plot_df['score'] = pd.to_numeric(plot_df['score'], errors='coerce').fillna(0.0)
    plot_df = plot_df.sort_values('score', ascending=True)

    plt.figure(figsize=(8,4.5))
    plt.barh(plot_df['team'], plot_df['score'])
    plt.title("Team Scores")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def make_kda_heatmap(df: pd.DataFrame, out="kda_heatmap.png"):
    cols = ["k_avg","d_avg","a_avg","kda"]
    plot_df = df[['team']+cols].copy()
    for c in cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce').fillna(0.0)
    mat = plot_df[cols].values.astype(float)

    plt.figure(figsize=(6,3.2))
    plt.imshow(mat, aspect='auto')
    plt.xticks(range(len(cols)), cols, rotation=0)
    plt.yticks(range(len(plot_df['team'])), plot_df['team'])
    plt.title("KDA Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def make_bracket_tree(ub_qf, ub_sf, ub_final, lb_flow, champion, out="bracket_tree.png"):
    # Lightweight bracket diagram using text boxes
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.axis('off')

    # Positions
    x0, x1, x2, x3 = 0.05, 0.30, 0.55, 0.80
    y_qf = [0.8, 0.6, 0.4, 0.2]
    y_sf = [0.7, 0.3]
    y_f  = 0.5
    y_lb = [0.35, 0.25, 0.15, 0.05]

    # QFs
    for i,(a,b,winner) in enumerate(ub_qf):
        ax.text(x0, y_qf[i], f"{a} vs {b}\n→ {winner}", ha='left', va='center')

    # SFs
    for i,(a,b,winner) in enumerate(ub_sf):
        ax.text(x1, y_sf[i], f"{a} vs {b}\n→ {winner}", ha='left', va='center')

    # UB Final
    a,b,uf = ub_final
    ax.text(x2, y_f, f"UB Final:\n{a} vs {b}\n→ {uf}", ha='left', va='center')

    # LB flow
    for i, text in enumerate(lb_flow):
        ax.text(x1, y_lb[i], text, ha='left', va='center')

    # Champion
    ax.text(x3, 0.5, f"Champion:\n{champion}", ha='left', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

# --------------- Main ---------------

def main():
    # meta & hero map
    meta_top = build_meta_top()
    hero_map = get_hero_id_map()

    # parse input
    g_wr, h2h = parse_input(sys.stdin.readlines())

    # build scores
    rows = []
    for t in TEAM_IDS:
        rows.append(build_team_row(t, g_wr, meta_top, hero_map))
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    # pretty order of columns
    cols = ["team","recent_wr70","group","pen","k_avg","d_avg","a_avg","kda",
            "ti_pressure","ti_pressure_sum","meta_align",
            "comp_recent","comp_group","comp_roster","comp_form","comp_ti","comp_meta","comp_falcons","score","roster"]
    df = df[cols]
    print(df.set_index('team')[["recent_wr70","group","pen","k_avg","d_avg","a_avg","kda","ti_pressure","ti_pressure_sum","meta_align","comp_recent","comp_group","comp_roster","comp_form","comp_ti","comp_meta","comp_falcons","score","roster"]])

    # dictionary for quick access
    scores = {r["team"]: r for _,r in df.iterrows()}

    # dynamic per-win bonus tracker
    per_win_bonus = {t:0.0 for t in TEAM_IDS}

    # ---- Upper Bracket QFs
    ub_qf_winners = []
    ub_qf_log = []
    for a,b in UB_QF:
        w,p = predict(a,b, scores, h2h, per_win_bonus)
        ub_qf_winners.append(w)
        ub_qf_log.append((a,b,w))
        print(f"UB QF: {a} vs {b} → {w} (p={p:.2f})")
        per_win_bonus[w] += MAIN_EVENT_WIN_BONUS

    # ---- Upper Bracket SFs
    ub_sf_pairs = [(ub_qf_winners[0], ub_qf_winners[1]), (ub_qf_winners[2], ub_qf_winners[3])]
    ub_sf_winners = []
    ub_sf_log = []
    for a,b in ub_sf_pairs:
        w,p = predict(a,b, scores, h2h, per_win_bonus)
        ub_sf_winners.append(w)
        ub_sf_log.append((a,b,w))
        print(f"UB SF: {a} vs {b} → {w} (p={p:.2f})")
        per_win_bonus[w] += MAIN_EVENT_WIN_BONUS

    # ---- Upper Bracket Final
    a,b = ub_sf_winners
    uf,p = predict(a,b, scores, h2h, per_win_bonus)
    print(f"UB Final: {a} vs {b} → {uf} (p={p:.2f})")
    per_win_bonus[uf] += MAIN_EVENT_WIN_BONUS

    # ---- Lower Bracket
    # LB R1: two matches among UB QF losers (pair 1 losers & pair 2 losers)
    ub_qf_losers = []
    for i,(a,b,w) in enumerate(ub_qf_log):
        ub_qf_losers.append(a if w!=a else b)
    lb_r1_pairs = [(ub_qf_losers[0], ub_qf_losers[1]), (ub_qf_losers[2], ub_qf_losers[3])]
    lb_r1_winners = []
    for a,b in lb_r1_pairs:
        w,p = predict(a,b, scores, h2h, per_win_bonus)
        lb_r1_winners.append(w)
        print(f"LB R1: {a} vs {b} → {w} (p={p:.2f})")
        per_win_bonus[w] += MAIN_EVENT_WIN_BONUS

    # LB QF: vs UB SF losers
    ub_sf_losers = []
    for (a,b,w) in ub_sf_log:
        ub_sf_losers.append(a if w!=a else b)
    lb_qf_pairs = [(lb_r1_winners[0], ub_sf_losers[0]), (lb_r1_winners[1], ub_sf_losers[1])]
    lb_qf_winners = []
    for a,b in lb_qf_pairs:
        w,p = predict(a,b, scores, h2h, per_win_bonus)
        lb_qf_winners.append(w)
        print(f"LB QF: {a} vs {b} → {w} (p={p:.2f})")
        per_win_bonus[w] += MAIN_EVENT_WIN_BONUS

    # LB SF: winners face each other
    a,b = lb_qf_winners
    lb_sf_winner, p = predict(a,b, scores, h2h, per_win_bonus)
    print(f"LB SF: {a} vs {b} → {lb_sf_winner} (p={p:.2f})")
    per_win_bonus[lb_sf_winner] += MAIN_EVENT_WIN_BONUS

    # LB Final: vs UB Final loser
    ub_final_loser = (a if uf!=a else b) if uf in (a,b) else (ub_sf_winners[0] if uf==ub_sf_winners[1] else ub_sf_winners[1])
    a,b = lb_sf_winner, ub_final_loser
    lb_final_winner, p = predict(a,b, scores, h2h, per_win_bonus)
    print(f"LB Final: {a} vs {b} → {lb_final_winner} (p={p:.2f})")
    per_win_bonus[lb_final_winner] += MAIN_EVENT_WIN_BONUS

    # Grand Final
    a,b = uf, lb_final_winner
    champion, p = predict(a,b, scores, h2h, per_win_bonus)
    print(f"Grand Final: {a} vs {b} → Champion {champion} (p={p:.2f})")

    # ---- Visuals
    try:
        make_score_bar_chart(df[['team','score']].copy())
        make_kda_heatmap(df[['team','k_avg','d_avg','a_avg','kda']].copy())
        # Build LB flow strings for compact bracket image
        lb_flow = [
            f"LB R1: {lb_r1_pairs[0][0]} vs {lb_r1_pairs[0][1]} → {lb_r1_winners[0]}",
            f"LB R1: {lb_r1_pairs[1][0]} vs {lb_r1_pairs[1][1]} → {lb_r1_winners[1]}",
            f"LB QF1: {lb_qf_pairs[0][0]} vs {lb_qf_pairs[0][1]} → {lb_qf_winners[0]}",
            f"LB QF2: {lb_qf_pairs[1][0]} vs {lb_qf_pairs[1][1]} → {lb_qf_winners[1]}",
            f"LB SF: {a} vs {b} → {lb_sf_winner}",
            f"LB Final: {lb_sf_winner} vs {ub_final_loser} → {lb_final_winner}",
        ]
        make_bracket_tree(ub_qf_log, ub_sf_log, (ub_sf_winners[0], ub_sf_winners[1], uf), lb_flow, champion)
    except Exception as e:
        print(f"[warn] Failed to create one or more images: {e}")

    # ---- Markdown output
    with open("bracket_prediction.md","w") as f:
        f.write("# TI25 Full Bracket Prediction\n\n")
        f.write(df.drop(columns=['roster']).to_markdown(index=False))
        f.write("\n\n### Rosters (OpenDota current-team snapshot)\n")
        for _,r in df.iterrows():
            roster = r['roster']
            if roster:
                f.write(f"- {r['team']}: {roster}\n")
            else:
                f.write(f"- {r['team']}: (no current snapshot available)\n")
        f.write("\n## Bracket Results\n")
        f.write(f"UB QF Winners: {[w for _,_,w in ub_qf_log]}\n")
        f.write(f"UB SF Winners: {[w for _,_,w in ub_sf_log]}\n")
        f.write(f"UB Final Winner: {uf}\n")
        f.write(f"LB R1 Winners: {lb_r1_winners}\n")
        f.write(f"LB QF Winners: {lb_qf_winners}\n")
        f.write(f"LB SF Winner: {lb_sf_winner}\n")
        f.write(f"LB Final Winner: {lb_final_winner}\n\n")
        f.write(f"**Champion:** {champion}\n\n")
        f.write("## Visuals\n")
        f.write("- ![Scores](bracket_scores.png)\n")
        f.write("- ![KDA Heatmap](kda_heatmap.png)\n")
        f.write("- ![Bracket Tree](bracket_tree.png)\n\n")
        f.write("## Component Weights (Explainer)\n")
        f.write("- comp_recent = W_RECENT * recent_wr70\n")
        f.write("- comp_group  = W_GROUP * normalized group record\n")
        f.write("- comp_roster = W_ROSTER * (1 - roster_penalty)\n")
        f.write("- comp_form   = W_FORM * squashed(KDA)\n")
        f.write("- comp_ti     = W_TI * TI_pressure (avg)\n")
        f.write("- comp_meta   = W_META * meta_alignment vs current meta\n")
        f.write("- comp_falcons = W_FALCONS (fan bonus, Falcons only)\n")
        f.write(f"\nUpset buffer active: underdog must have p > {UPSET_P_THRESHOLD:.2f} to upset.\n")
        f.write(f"Main-event per-win bonus per team: +{MAIN_EVENT_WIN_BONUS} added to score for subsequent rounds.\n")
    print("[ok] Wrote bracket_prediction.md")

if __name__ == "__main__":
    main()

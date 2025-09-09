#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor

- Recent form from last 50 matches (OpenDota).
- TI "pressure" metric from player performance across the last 3 Internationals (Liquipedia stats tables).
- Falcons personal bias +0.06.
- Momentum bonus +0.05% per Main Event win.
- Correct double-elim mapping.

Run: python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time, logging
from collections import defaultdict
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

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

UB_QF = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

# Weights & bonuses
W_RECENT      = 0.35
W_GROUP       = 0.20
W_ROSTER      = 0.10
W_FORM50      = 0.20
W_TI_PRESSURE = 0.15

ROSTER_ISSUES   = {"Tundra Esports": True}
ROSTER_PENALTY  = 0.08
H2H_BOOST       = 0.04
FALCONS_BONUS   = 0.06
MAIN_WIN_BONUS  = 0.0005

def parse_input():
    group_wr, h2h = {}, []
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if ">" in line:
            a, b = [x.strip() for x in line.split(">")]
            h2h.append((a, b, 1))
        elif "<" in line:
            a, b = [x.strip() for x in line.split("<")]
            h2h.append((a, b, -1))
        elif ":" in line:
            name, rec = line.split(":", 1)
            w, l = map(int, rec.strip().split("-"))
            group_wr[name.strip()] = w / (w + l) if (w + l) > 0 else 0.5
    return group_wr, h2h

def get_team_matches(team_id, limit=50):
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/matches?limit={limit}", headers=HEADERS, timeout=25)
        r.raise_for_status()
        time.sleep(0.2)
        return r.json()
    except Exception as e:
        logging.info(f"[warn] get_team_matches({team_id}) failed: {e}")
        return []

def recent_wr_and_form(matches):
    wins = tot = 0
    k_sum = d_sum = a_sum = 0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            tot += 1
            team_won = (m.get('radiant') and m.get('radiant_win')) or ((not m.get('radiant')) and (not m.get('radiant_win')))
            wins += 1 if team_won else 0
            k_sum += m.get('kills', 0) or 0
            d_sum += m.get('deaths', 0) or 0
            a_sum += m.get('assists', 0) or 0
    wr  = wins / tot if tot else 0.5
    k   = (k_sum / tot) if tot else 0.0
    d   = (d_sum / tot) if tot else 0.0
    a   = (a_sum / tot) if tot else 0.0
    kda = (k + a) / max(d, 1e-6) if (k or a or d) else 1.0
    return wr, k, d, a, kda

def get_current_roster_names(team_id, max_players=5):
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/players", headers=HEADERS, timeout=25)
        r.raise_for_status()
        time.sleep(0.2)
        players = r.json() or []
        current = [p for p in players if p.get("is_current_team_member")]
        pool = current if current else sorted(players, key=lambda x: (x.get("games_played", 0), x.get("is_current_team_member", False)), reverse=True)
        names = []
        for p in pool:
            n = (p.get("name") or p.get("personaname") or "").strip()
            if n: names.append(n)
            if len(names) >= max_players: break
        return names
    except Exception as e:
        logging.info(f"[warn] roster fetch failed for {team_id}: {e}")
        return []

def fetch_ti_stats_tables():
    urls = [
        "https://liquipedia.net/dota2/The_International/2024/Statistics",
        "https://liquipedia.net/dota2/The_International/2023/Statistics",
        "https://liquipedia.net/dota2/The_International/2022/Statistics",
    ]
    weight_per_year = [1.0, 0.8, 0.6]
    nick_score = defaultdict(float)
    for idx, url in enumerate(urls):
        w_year = weight_per_year[idx] if idx < len(weight_per_year) else 0.5
        try:
            tables = pd.read_html(url)
        except Exception:
            continue
        for table in tables:
            cols = [str(c).lower() for c in table.columns]
            cand = [c for c in table.columns if any(s in str(c).lower() for s in ["player", "nickname", "name"])]
            ranks = [c for c in table.columns if any(s in str(c).lower() for s in ["#", "rank", "place", "pos"])]
            if not cand: continue
            for _, row in table.iterrows():
                for pc in cand:
                    pname = str(row.get(pc, "")).strip()
                    if not pname or pname.lower() in ("nan", "—", "-"): continue
                    nickname = pname.split()[0].strip("*•-—")
                    base = 1.0
                    if ranks:
                        try:
                            r = row.get(ranks[0])
                            r = int(str(r).strip().replace(".", "").split()[0])
                            base = max(0.5, 2.5 - 0.15 * (r - 1))
                        except Exception:
                            pass
                    nick_score[nickname] += base * w_year
                    break
    return nick_score

def team_ti_pressure_score(roster_names, nick_score_map):
    if not roster_names: return 0.5
    scores = []
    for raw in roster_names:
        candidates = {raw, raw.replace("-", ""), raw.replace("_", ""), raw.split()[0]}
        best = 0.0
        for c in candidates:
            best = max(best, nick_score_map.get(c, 0.0), nick_score_map.get(c.capitalize(), 0.0))
            for k, v in nick_score_map.items():
                if k.lower() == c.lower():
                    best = max(best, v)
        scores.append(best)
    if sum(scores) <= 0: return 0.5
    mean = sum(scores) / len(scores)
    norm = 1.0 - 1.0 / (1.0 + mean / 3.0)
    return 0.45 + 0.5 * norm

def base_team_score(name, g_wr, nick_score_map):
    matches = get_team_matches(TEAM_IDS[name], limit=50)
    wr50, k_avg, d_avg, a_avg, kda = recent_wr_and_form(matches)
    gs  = g_wr.get(name, 0.5)
    pen = 0.0 if not ROSTER_ISSUES.get(name) else ROSTER_PENALTY
    roster = get_current_roster_names(TEAM_IDS[name])
    ti_pressure = team_ti_pressure_score(roster, nick_score_map)

    def squash(x, c=22.0):
        try: return 1.0 - (1.0 / (1.0 + x / c))
        except Exception: return 0.5

    form_component = 0.45 * squash(k_avg) + 0.55 * squash(kda * 10)
    score = (
        W_RECENT      * wr50 +
        W_GROUP       * gs +
        W_ROSTER      * (1 - pen) +
        W_FORM50      * form_component +
        W_TI_PRESSURE * ti_pressure
    )
    if name == "Team Falcons":
        score += FALCONS_BONUS

    return {
        "team": name, "recent_wr50": wr50, "group": gs, "pen": pen,
        "k_avg": k_avg, "d_avg": d_avg, "a_avg": a_avg, "kda": kda,
        "ti_pressure": ti_pressure, "roster": ", ".join(roster) if roster else "",
        "score": score
    }

def h2h_adjust(a, b, h2h):
    adj = 0.0
    for x, y, res in h2h:
        if x == a and y == b:
            adj = H2H_BOOST if res == 1 else -H2H_BOOST
    return adj

def predict(a, b, scores, h2h, wins_so_far):
    sa = scores[a]['score'] + h2h_adjust(a, b, h2h) + wins_so_far[a]*MAIN_WIN_BONUS
    sb = scores[b]['score'] + h2h_adjust(b, a, h2h) + wins_so_far[b]*MAIN_WIN_BONUS
    p_a = 1 / (1 + math.exp(-8 * (sa - sb)))
    winner = a if p_a >= 0.5 else b
    prob = p_a if winner == a else (1 - p_a)
    return winner, prob

def simulate_bracket(scores, h2h):
    wins_so_far = defaultdict(int)
    ubqf_winners, ubqf_losers = [], []
    for i, (a, b) in enumerate(UB_QF, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        l = b if w == a else a
        wins_so_far[w] += 1
        print(f"UB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubqf_winners.append(w); ubqf_losers.append(l)

    ubsf_pairs = [(ubqf_winners[0], ubqf_winners[1]), (ubqf_winners[2], ubqf_winners[3])]
    ubsf_winners, ubsf_losers = [], []
    for i, (a, b) in enumerate(ubsf_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        l = b if w == a else a
        wins_so_far[w] += 1
        print(f"UB SF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubsf_winners.append(w); ubsf_losers.append(l)

    a, b = ubsf_winners
    ubf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    ubf_loser = b if ubf_winner == a else a
    wins_so_far[ubf_winner] += 1
    print(f"UB Final: {a} vs {b} → {ubf_winner} (p={p:.2f})")

    lbr1_pairs = [(ubqf_losers[0], ubqf_losers[1]), (ubqf_losers[2], ubqf_losers[3])]
    lbr1_winners = []
    for i, (a, b) in enumerate(lbr1_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        wins_so_far[w] += 1
        print(f"LB R1-{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbr1_winners.append(w)

    lbqf_pairs = [(lbr1_winners[0], ubsf_losers[1]), (lbr1_winners[1], ubsf_losers[0])]
    lbqf_winners = []
    for i, (a, b) in enumerate(lbqf_pairs, start=1):
        w, p = predict(a, b, scores, h2h, wins_so_far)
        wins_so_far[w] += 1
        print(f"LB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbqf_winners.append(w)

    a, b = lbqf_winners
    lbsf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    wins_so_far[lbsf_winner] += 1
    print(f"LB SF: {a} vs {b} → {lbsf_winner} (p={p:.2f})")

    a, b = lbsf_winner, ubf_loser
    lbf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    wins_so_far[lbf_winner] += 1
    print(f"LB Final: {a} vs {b} → {lbf_winner} (p={p:.2f})")

    a, b = ubf_winner, lbf_winner
    gf_winner, p = predict(a, b, scores, h2h, wins_so_far)
    print(f"Grand Final: {a} vs {b} → Champion {gf_winner} (p={p:.2f})")

    return {
        "ubqf_winners": ubqf_winners, "ubqf_losers": ubqf_losers,
        "ubsf_winners": ubsf_winners, "ubsf_losers": ubsf_losers,
        "ubf_winner": ubf_winner, "ubf_loser": ubf_loser,
        "lbr1_winners": lbr1_winners, "lbqf_winners": lbqf_winners,
        "lbsf_winner": lbsf_winner, "lbf_winner": lbf_winner,
        "gf_winner": gf_winner
    }

def main():
    logging.info("Fetching TI statistics for player pressure metric…")
    nick_score_map = fetch_ti_stats_tables()

    g_wr, h2h = parse_input()
    scores = {t: base_team_score(t, g_wr, nick_score_map) for t in TEAM_IDS}

    cols = ['recent_wr50','group','pen','k_avg','d_avg','a_avg','kda','ti_pressure','score','roster']
    df = (pd.DataFrame(scores).T
          .set_index('team')[cols]
          .sort_values('score', ascending=False))
    print(df)

    res = simulate_bracket(scores, h2h)

    out = "bracket_prediction.md"
    with open(out,"w", encoding="utf-8") as f:
        f.write("# TI25 Full Bracket Prediction\n\n")
        f.write(df.drop(columns=['roster']).to_markdown() + "\n\n")
        f.write("### Rosters used for TI pressure\n")
        for t in TEAM_IDS:
            f.write(f"- {t}: {scores[t]['roster'] or '(unknown)'}\n")
        f.write("\n## Bracket Results\n")
        f.write(f"UB QF Winners: {res['ubqf_winners']}\n")
        f.write(f"UB SF Winners: {res['ubsf_winners']}\n")
        f.write(f"UB Final Winner: {res['ubf_winner']}\n")
        f.write(f"LB R1 Winners: {res['lbr1_winners']}\n")
        f.write(f"LB QF Winners: {res['lbqf_winners']}\n")
        f.write(f"LB SF Winner: {res['lbsf_winner']}\n")
        f.write(f"LB Final Winner: {res['lbf_winner']}\n")
        f.write(f"\n**Champion:** {res['gf_winner']}\n")
    print(f"[ok] Wrote {out}")

if __name__ == '__main__':
    main()

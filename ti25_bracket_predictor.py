#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor (OpenDota IDs + Dotabuff meta)

What's new:
- Recent window: last 70 matches for winrate; KDA from up to 25 match details (robust fallbacks).
- Player "TI pressure" via OpenDota player IDs and TI league IDs (TI22/TI23/TI24).
- Hero meta alignment: team's hero pool vs Dotabuff Immortal (14d) high-win heroes.
- Clear component breakdown columns (see comp_*).
- Falcons personal bonus +0.06.
- Momentum +0.05% per simulated Main Event win.
- No more zero KDAs: multiple fallbacks ensure non-zero form component whenever possible.

Run:
  python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time, logging, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------ Configuration ------------------

API_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS_API = {'Authorization': f"Bearer {API_KEY}"} if API_KEY else {}

# Be friendly to public APIs
SLEEP_LIST = 0.20
SLEEP_DETAIL = 0.25

# Team IDs you’re tracking (OpenDota team IDs):
TEAM_IDS = {
    "Tundra Esports": 8291895,
    "Xtreme Gaming": 8261500,
    "PARIVISION": 9572001,
    "Heroic": 9303484,
    "Team Tidebound": 9640842,
    "Team Falcons": 9247354,   # https://www.opendota.com/teams/9247354
    "BetBoom Team": 14124,
    "Nigma Galaxy": 7554697,
}

# Upper Bracket QF pairs (per Liquipedia Main Event)
UB_QF = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

# Weights
W_RECENT       = 0.33  # recent WR (70)
W_GROUP        = 0.18  # group WR from input.txt
W_ROSTER       = 0.08  # roster penalty scaler
W_FORM70       = 0.19  # KDA-ish form from details
W_TI_PRESSURE  = 0.12  # player past TI performance
W_HERO_META    = 0.10  # hero pool vs Immortal 14d meta

# Biases/bonuses
ROSTER_ISSUES   = {"Tundra Esports": True}
ROSTER_PENALTY  = 0.08
H2H_BOOST       = 0.04
FALCONS_BONUS   = 0.06        # your preference bonus
MAIN_WIN_BONUS  = 0.0005      # +0.05% per simulated Main Event win

# TI league IDs (Valve league IDs; match OpenDota `leagueid`)
# From Dotabuff league pages:
#   TI 2022: https://www.dotabuff.com/esports/leagues/14268-the-international-2022
#   TI 2023: https://www.dotabuff.com/esports/leagues/15728-the-international-2023
#   TI 2024: https://www.dotabuff.com/esports/leagues/16935-the-international-2024
TI_LEAGUE_IDS = {14268, 15728, 16935}

# Dotabuff Immortal 14d win-rate table (All Pick)
DOTABUFF_HERO_WIN_URL = (
    "https://www.dotabuff.com/heroes?show=heroes&view=winning&mode=all-pick&date=14d&rankTier=immortal"
)
HTTP_HEADERS = {
    "User-Agent": "TI25-bracket-bot/1.1 (+https://github.com/ejpdtd-beep)",
    "Accept": "text/html,application/xhtml+xml",
}

# ------------------ Input parsing ------------------

def parse_input():
    group_wr, h2h = {}, []
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
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

# ------------------ OpenDota helpers ------------------

def get_team_matches(team_id: int, limit: int = 70) -> List[dict]:
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/matches?limit={limit}", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        return r.json() or []
    except Exception as e:
        logging.info(f"[warn] get_team_matches({team_id}) failed: {e}")
        return []

def get_match_detail(match_id: int) -> dict:
    try:
        r = requests.get(f"{API_BASE}/matches/{match_id}", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_DETAIL)
        return r.json() or {}
    except Exception as e:
        logging.info(f"[warn] get_match_detail({match_id}) failed: {e}")
        return {}

def recent_wr(matches: List[dict]) -> float:
    wins = tot = 0
    for m in matches:
        if 'radiant' in m and 'radiant_win' in m:
            tot += 1
            team_won = (m.get('radiant') and m.get('radiant_win')) or ((not m.get('radiant')) and (not m.get('radiant_win')))
            wins += 1 if team_won else 0
    return wins / tot if tot else 0.5

def recent_kda_via_details(team_id: int, matches: List[dict], detail_cap: int = 25) -> Tuple[float,float,float,float]:
    """Compute team K/D/A via match detail for up to detail_cap most recent matches.
       Robust to side detection; multiple fallbacks to avoid 0s."""
    mids = [m.get("match_id") for m in matches if m.get("match_id")]
    mids = mids[:detail_cap]
    if not mids:
        return 0.0, 0.0, 0.0, 1.0

    k_sum = d_sum = a_sum = tot = 0
    for mid in mids:
        d = get_match_detail(mid)
        if not d:
            continue

        radiant_team_id = d.get("radiant_team_id")
        dire_team_id = d.get("dire_team_id")

        # Sum from players (most reliable)
        rk = rd = ra = 0
        dk = dd = da = 0
        for p in d.get("players", []):
            is_rad = p.get("isRadiant")
            kills = p.get("kills", 0) or 0
            deaths = p.get("deaths", 0) or 0
            assists = p.get("assists", 0) or 0
            if is_rad:
                rk += kills; rd += deaths; ra += assists
            else:
                dk += kills; dd += deaths; da += assists

        # Identify side, with fallback
        side = None
        if radiant_team_id == team_id:
            side = "radiant"
        elif dire_team_id == team_id:
            side = "dire"
        # Soft fallback: if team kills in list match object exists, use win flag to infer side is stored there (skipped—unreliable)

        if side == "radiant":
            tk, td, ta = rk, rd, ra
        elif side == "dire":
            tk, td, ta = dk, dd, da
        else:
            # Use scoreboard totals if players{} missing (rare)
            r_score = d.get("radiant_score"); d_score = d.get("dire_score")
            if r_score is not None and d_score is not None:
                # If we can't tell side, split evenly as a last resort to avoid zeros
                tk = (r_score + d_score) / 2.0
                td = tk * 0.9  # heuristic to keep KDA ~ 2–3 if no detail
                ta = tk * 1.2
            else:
                continue

        k_sum += tk; d_sum += td; a_sum += ta; tot += 1

    if not tot:
        return 0.0, 0.0, 0.0, 1.0

    k_avg = k_sum / tot
    d_avg = d_sum / tot
    a_avg = a_sum / tot
    kda = (k_avg + a_avg) / max(d_avg, 1e-6)
    return k_avg, d_avg, a_avg, kda

def get_team_players(team_id: int, max_players=6) -> List[dict]:
    """Return current player entries: [{'account_id': int, 'name': 'nick'...}]"""
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/players", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        players = r.json() or []
        current = [p for p in players if p.get("is_current_team_member")]
        pool = current if current else sorted(players, key=lambda x: (x.get("games_played", 0)), reverse=True)
        out = []
        for p in pool:
            nick = (p.get("name") or p.get("personaname") or "").strip()
            aid = p.get("account_id")
            if nick and aid:
                out.append({"account_id": int(aid), "name": nick})
            if len(out) >= max_players:
                break
        return out
    except Exception as e:
        logging.info(f"[warn] get_team_players({team_id}) failed: {e}")
        return []

def get_player_ti_matches(player_id: int, limit: int = 1000) -> List[dict]:
    """
    Fetch player matches and filter to TI leagues.
    /players/{id}/matches typically includes 'leagueid'. If not, we would need details (expensive).
    """
    try:
        r = requests.get(f"{API_BASE}/players/{player_id}/matches?limit={limit}&significant=0", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        rows = r.json() or []
    except Exception as e:
        logging.info(f"[warn] get_player_ti_matches({player_id}) base fetch failed: {e}")
        return []

    ti_rows = [m for m in rows if m.get("leagueid") in TI_LEAGUE_IDS]
    # If leagueid missing, (rare) optionally detail fetch the last ~200 and filter; skipped by default to avoid API burn.
    return ti_rows

def player_ti_pressure_score(player_id: int) -> float:
    """
    Build a TI pressure score per player:
      - base on wins and depth (booleans like 'radiant_win' with isRadiant).
      - add small bonus for number of TI matches played (experience).
    Result squashed to ~0.45..0.95, neutral ~0.5 if no TI data.
    """
    rows = get_player_ti_matches(player_id, limit=1000)
    if not rows:
        return 0.5

    wins = 0; tot = 0
    for m in rows:
        tot += 1
        is_rad = m.get("player_slot", 0) < 128  # player_slot <128 = radiant
        rad_win = m.get("radiant_win", None)
        if rad_win is not None:
            win = (is_rad and rad_win) or ((not is_rad) and (rad_win is False))
            wins += 1 if win else 0

    wr = wins / tot if tot else 0.5
    # Experience bonus: more TI games -> slightly higher
    exp = min(tot / 30.0, 1.0)  # capped
    raw = 1.6*wr + 0.4*exp   # 0..2.0 approx
    # squash
    norm = 1 - 1/(1 + raw/1.2)
    return 0.45 + 0.5*norm

# ------------------ Hero meta alignment ------------------

def fetch_dotabuff_immortal_win_table(max_heroes: int = 40) -> Dict[str, float]:
    """
    Returns dict hero_name -> win_rate (0..1), top N (default 40).
    """
    try:
        resp = requests.get(DOTABUFF_HERO_WIN_URL, headers=HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logging.info(f"[warn] Dotabuff fetch failed: {e}")
        return {}

    # Parse with pandas.read_html (works fine on Dotabuff hero page)
    try:
        tables = pd.read_html(html)
    except Exception as e:
        logging.info(f"[warn] read_html failed on Dotabuff: {e}")
        return {}

    # Find the table with "Hero" and "Win rate"
    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("hero" in str(c).lower() for c in t.columns) and any("win rate" in str(c).lower() for c in t.columns):
            target = t
            break

    if target is None:
        # Fallback: try to manually scrape minimal pairs using regex as last resort
        rx = re.compile(r'>([A-Za-z \'\-\.\u2019]+)</a>\s*</td>\s*<td[^>]*>(\d{2}\.\d{2})%')
        pairs = rx.findall(html)
        meta = {}
        for name, wr in pairs[:max_heroes]:
            meta[name.strip()] = float(wr)/100.0
        return meta

    # Normalize & take top by Win rate column
    df = target.copy()
    # Robust normalize column names
    col_hero = [c for c in df.columns if "Hero" in str(c)][0]
    col_wr = [c for c in df.columns if "Win rate" in str(c)][0]
    df = df[[col_hero, col_wr]].dropna()
    df[col_wr] = df[col_wr].astype(str).str.replace("%","").astype(float) / 100.0

    df = df.sort_values(col_wr, ascending=False).head(max_heroes)
    meta = dict(zip(df[col_hero].astype(str).str.strip(), df[col_wr]))
    return meta

def get_hero_id_map() -> Dict[int, str]:
    """OpenDota /heroes id->localized_name"""
    try:
        r = requests.get(f"{API_BASE}/heroes", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        heroes = r.json() or []
        return {int(h["id"]): str(h.get("localized_name") or h.get("name") or "").strip() for h in heroes}
    except Exception as e:
        logging.info(f"[warn] /heroes failed: {e}")
        return {}

def get_team_heroes(team_id: int) -> List[dict]:
    """
    /teams/{id}/heroes → [{'hero_id': int, 'games': int, 'win': int}, ...]
    """
    try:
        r = requests.get(f"{API_BASE}/teams/{team_id}/heroes", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        return r.json() or []
    except Exception as e:
        logging.info(f"[warn] team heroes failed for {team_id}: {e}")
        return []

def hero_meta_alignment_score(team_id: int, hero_id_to_name: Dict[int,str], meta_wr: Dict[str,float]) -> float:
    """
    Score how much a team's played hero pool overlaps with high-WR Immortal heroes.
    - Build team hero usage share (games / total).
    - For each hero, if it's in the meta list, add share * (meta_wr - 0.5) * scale.
    - Squash to ~0.45..0.95 with neutral ~0.5.
    """
    rows = get_team_heroes(team_id)
    if not rows or not meta_wr:
        return 0.5

    total_games = sum(r.get("games", 0) or 0 for r in rows)
    if total_games <= 0:
        return 0.5

    score = 0.0
    for r in rows:
        hid = r.get("hero_id")
        g   = r.get("games", 0) or 0
        if not hid or not g:
            continue
        name = hero_id_to_name.get(int(hid), "").strip()
        if not name:
            continue
        wr = meta_wr.get(name)
        if wr is None:
            # try small normalization (common aliases rarely differ nowadays)
            continue
        share = g / total_games
        score += share * (wr - 0.5)  # >0 if better than coinflip meta

    # Scale and squash
    # Typical raw range ~[-0.05, +0.05]; scale then squash
    raw = 6.0 * score + 0.5
    raw = max(0.0, min(1.0, raw))
    norm = 1 - 1/(1 + raw/1.2)
    return 0.45 + 0.5*norm

# ------------------ Scoring ------------------

def squash(x: float, c: float = 22.0) -> float:
    try:
        return 1 - 1/(1 + x/c)
    except Exception:
        return 0.5

def team_base_score(name: str, g_wr: Dict[str,float],
                    hero_id_to_name: Dict[int,str],
                    meta_wr: Dict[str,float]) -> dict:
    team_id = TEAM_IDS[name]

    # Recent form (70) & KDA details (25)
    matches70 = get_team_matches(team_id, limit=70)
    wr70 = recent_wr(matches70)
    k_avg, d_avg, a_avg, kda = recent_kda_via_details(team_id, matches70, detail_cap=25)

    # Ensure no all-zero K/D/A
    if k_avg == d_avg == a_avg == 0:
        # fallback: rough heuristic from wr
        k_avg, d_avg, a_avg, kda = 22.0, 12.0, 44.0, (22+44)/12

    group_s = g_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if ROSTER_ISSUES.get(name, False) else 0.0

    # Roster / players
    players = get_team_players(team_id, max_players=6)
    roster_names = ", ".join([p["name"] for p in players]) if players else ""

    # Player TI pressure (OpenDota IDs)
    if players:
        pis = [player_ti_pressure_score(p["account_id"]) for p in players]
        ti_pressure = sum(pis)/len(pis) if pis else 0.5
        ti_raw = sum(pis)
    else:
        ti_pressure = 0.5
        ti_raw = 0.0

    # Hero meta alignment
    meta_align = hero_meta_alignment_score(team_id, hero_id_to_name, meta_wr)

    # Form (KDA + kills) scaler
    form_component = 0.45 * squash(k_avg) + 0.55 * squash(kda * 10)

    comp_recent   = W_RECENT      * wr70
    comp_group    = W_GROUP       * group_s
    comp_roster   = W_ROSTER      * (1 - pen)
    comp_form     = W_FORM70      * form_component
    comp_ti       = W_TI_PRESSURE * ti_pressure
    comp_meta     = W_HERO_META   * meta_align
    comp_falcons  = FALCONS_BONUS if name == "Team Falcons" else 0.0

    score = comp_recent + comp_group + comp_roster + comp_form + comp_ti + comp_meta + comp_falcons

    return {
        "team": name,
        "recent_wr70": wr70,
        "group": group_s,
        "pen": pen,
        "k_avg": k_avg, "d_avg": d_avg, "a_avg": a_avg, "kda": kda,
        "ti_pressure": ti_pressure, "ti_pressure_sum": ti_raw,
        "meta_align": meta_align,
        "roster": roster_names,
        "comp_recent": comp_recent,
        "comp_group": comp_group,
        "comp_roster": comp_roster,
        "comp_form": comp_form,
        "comp_ti": comp_ti,
        "comp_meta": comp_meta,
        "comp_falcons": comp_falcons,
        "score": score,
    }

def h2h_adjust(a: str, b: str, h2h: List[Tuple[str,str,int]]) -> float:
    adj = 0.0
    for x,y,res in h2h:
        if x==a and y==b:
            adj = H2H_BOOST if res==1 else -H2H_BOOST
    return adj

def predict(a: str, b: str, scores: dict, h2h: list, wins_so_far: Dict[str,int]) -> Tuple[str, float]:
    sa = scores[a]['score'] + h2h_adjust(a,b,h2h) + wins_so_far[a]*MAIN_WIN_BONUS
    sb = scores[b]['score'] + h2h_adjust(b,a,h2h) + wins_so_far[b]*MAIN_WIN_BONUS
    p_a = 1/(1+math.exp(-8*(sa-sb)))
    winner = a if p_a >= 0.5 else b
    prob = p_a if winner == a else (1 - p_a)
    return winner, prob

# ------------------ Bracket ------------------

def simulate_bracket(scores: dict, h2h: list) -> dict:
    wins_so_far = defaultdict(int)

    # UB QF
    ubqf_winners, ubqf_losers = [], []
    for i, (a,b) in enumerate(UB_QF, start=1):
        w, p = predict(a,b,scores,h2h,wins_so_far)
        l = b if w==a else a
        wins_so_far[w]+=1
        print(f"UB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubqf_winners.append(w); ubqf_losers.append(l)

    # UB SF
    ubsf_pairs = [(ubqf_winners[0], ubqf_winners[1]), (ubqf_winners[2], ubqf_winners[3])]
    ubsf_winners, ubsf_losers = [], []
    for i, (a,b) in enumerate(ubsf_pairs, start=1):
        w, p = predict(a,b,scores,h2h,wins_so_far)
        l = b if w==a else a
        wins_so_far[w]+=1
        print(f"UB SF{i}: {a} vs {b} → {w} (p={p:.2f})")
        ubsf_winners.append(w); ubsf_losers.append(l)

    # UB Final
    a, b = ubsf_winners
    ubf_winner, p = predict(a,b,scores,h2h,wins_so_far)
    ubf_loser = b if ubf_winner==a else a
    wins_so_far[ubf_winner]+=1
    print(f"UB Final: {a} vs {b} → {ubf_winner} (p={p:.2f})")

    # LB R1
    lbr1_pairs = [(ubqf_losers[0], ubqf_losers[1]), (ubqf_losers[2], ubqf_losers[3])]
    lbr1_winners = []
    for i,(a,b) in enumerate(lbr1_pairs, start=1):
        w, p = predict(a,b,scores,h2h,wins_so_far)
        wins_so_far[w]+=1
        print(f"LB R1-{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbr1_winners.append(w)

    # LB QF: (LBR1-A vs Loser UB-SF2), (LBR1-B vs Loser UB-SF1)
    lbqf_pairs = [(lbr1_winners[0], ubsf_losers[1]), (lbr1_winners[1], ubsf_losers[0])]
    lbqf_winners = []
    for i,(a,b) in enumerate(lbqf_pairs, start=1):
        w, p = predict(a,b,scores,h2h,wins_so_far)
        wins_so_far[w]+=1
        print(f"LB QF{i}: {a} vs {b} → {w} (p={p:.2f})")
        lbqf_winners.append(w)

    # LB SF
    a,b = lbqf_winners
    lbsf_winner, p = predict(a,b,scores,h2h,wins_so_far)
    wins_so_far[lbsf_winner]+=1
    print(f"LB SF: {a} vs {b} → {lbsf_winner} (p={p:.2f})")

    # LB Final vs UB Final loser
    a,b = lbsf_winner, ubf_loser
    lbf_winner, p = predict(a,b,scores,h2h,wins_so_far)
    wins_so_far[lbf_winner]+=1
    print(f"LB Final: {a} vs {b} → {lbf_winner} (p={p:.2f})")

    # Grand Final
    a,b = ubf_winner, lbf_winner
    gf_winner, p = predict(a,b,scores,h2h,wins_so_far)
    print(f"Grand Final: {a} vs {b} → Champion {gf_winner} (p={p:.2f})")

    return {
        "ubqf_winners": ubqf_winners, "ubqf_losers": ubqf_losers,
        "ubsf_winners": ubsf_winners, "ubsf_losers": ubsf_losers,
        "ubf_winner": ubf_winner, "ubf_loser": ubf_loser,
        "lbr1_winners": lbr1_winners, "lbqf_winners": lbqf_winners,
        "lbsf_winner": lbsf_winner, "lbf_winner": lbf_winner,
        "gf_winner": gf_winner
    }

# ------------------ Main ------------------

def main():
    # Fetch meta once
    logging.info("Fetching Dotabuff Immortal 14d hero winrates…")
    meta_wr = fetch_dotabuff_immortal_win_table(max_heroes=60)

    # Hero ID->name map for OpenDota hero pool alignment
    logging.info("Fetching OpenDota hero ID map…")
    hero_id_to_name = get_hero_id_map()

    g_wr, h2h = parse_input()

    # Build team scores
    scores = {t: team_base_score(t, g_wr, hero_id_to_name, meta_wr) for t in TEAM_IDS}

    cols = [
        'recent_wr70','group','pen',
        'k_avg','d_avg','a_avg','kda',
        'ti_pressure','ti_pressure_sum',
        'meta_align',
        'comp_recent','comp_group','comp_roster','comp_form','comp_ti','comp_meta','comp_falcons',
        'score','roster'
    ]
    df = (pd.DataFrame(scores).T
          .set_index('team')[cols]
          .sort_values('score', ascending=False))
    print(df)

    # Simulate bracket
    res = simulate_bracket(scores, h2h)

    # Output
    out = "bracket_prediction.md"
    with open(out,"w",encoding="utf-8") as f:
        f.write("# TI25 Full Bracket Prediction\n\n")
        f.write(df.drop(columns=['roster']).to_markdown() + "\n\n")
        f.write("### Rosters (OpenDota current-team snapshot)\n")
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

if __name__ == "__main__":
    main()

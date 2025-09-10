#!/usr/bin/env python3
"""
TI2025 Main Event Bracket Predictor (OpenDota IDs + Dotabuff meta + Visuals)

Features:
- Recent window: last 70 matches (WR); KDA via up to 25 match details (safe fallback).
- Player TI pressure via OpenDota player IDs (TI22/TI23/TI24 leagues).
- Hero meta alignment: Team hero usage & WR vs Dotabuff Immortal 14d meta (realistic headers, with OpenDota pro meta fallback).
- Falcons bonus +0.06; momentum +0.05% per Main Event win.
- Visual outputs:
  - bracket_scores.png  (bar chart)
  - kda_heatmap.png     (K/D/A heatmap)
  - bracket_tree.png    (bracket flow diagram)
- Clear component breakdown so each contribution is visible.

Run:
  python ti25_bracket_predictor.py < input.txt
"""

import os, sys, math, time, logging, re
from collections import defaultdict
from typing import Dict, List, Tuple

import requests
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------ Configuration ------------------

API_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY")
HEADERS_API = {'Authorization': f"Bearer {API_KEY}"} if API_KEY else {}

# polite delays for public APIs
SLEEP_LIST = 0.20
SLEEP_DETAIL = 0.25

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

# Upper Bracket QFs (per Liquipedia Main Event bracket)
UB_QF = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("PARIVISION", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BetBoom Team", "Nigma Galaxy"),
]

# Scoring weights
W_RECENT       = 0.33  # recent WR (70)
W_GROUP        = 0.18  # group WR from input.txt
W_ROSTER       = 0.08  # roster penalty scaler
W_FORM70       = 0.19  # KDA-ish form
W_TI_PRESSURE  = 0.12  # player past TI performance
W_HERO_META    = 0.10  # hero pool vs Immortal 14d meta

# Biases/bonuses
ROSTER_ISSUES   = {"Tundra Esports": True}
ROSTER_PENALTY  = 0.08
H2H_BOOST       = 0.04
FALCONS_BONUS   = 0.06       # personal bias
MAIN_WIN_BONUS  = 0.0005     # +0.05% per Main Event win

# TI league IDs (Valve league IDs)
TI_LEAGUE_IDS = {14268, 15728, 16935}  # TI 2022, 2023, 2024

# Dotabuff Immortal 14d win-rate table (All Pick)
DOTABUFF_HERO_WIN_URL = (
    "https://www.dotabuff.com/heroes?show=heroes&view=winning&mode=all-pick&date=14d&rankTier=immortal"
)
HTTP_HEADERS = {
    # Realistic browser headers to reduce 403
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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
    mids = [m.get("match_id") for m in matches if m.get("match_id")]
    mids = mids[:detail_cap]
    if not mids:
        return 0.0, 0.0, 0.0, 1.0
    k_sum = d_sum = a_sum = tot = 0
    for mid in mids:
        d = get_match_detail(mid)
        if not d:
            continue
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

        side = None
        if d.get("radiant_team_id") == team_id:
            side = "radiant"
        elif d.get("dire_team_id") == team_id:
            side = "dire"

        if side == "radiant":
            tk, td, ta = rk, rd, ra
        elif side == "dire":
            tk, td, ta = dk, dd, da
        else:
            # last-resort: avoid all-zeroes if side cannot be inferred
            r_score = d.get("radiant_score"); d_score = d.get("dire_score")
            if r_score is not None and d_score is not None:
                tk = (r_score + d_score) / 2.0
                td = max(1.0, tk * 0.9)
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
    """Return current team players as [{'account_id': int, 'name': 'nick'}]"""
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
    """Fetch player matches and filter to TI leagues by leagueid."""
    try:
        r = requests.get(f"{API_BASE}/players/{player_id}/matches?limit={limit}&significant=0", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        rows = r.json() or []
    except Exception as e:
        logging.info(f"[warn] get_player_ti_matches({player_id}) failed: {e}")
        return []
    return [m for m in rows if m.get("leagueid") in TI_LEAGUE_IDS]

def player_ti_pressure_score(player_id: int) -> float:
    rows = get_player_ti_matches(player_id, limit=1000)
    if not rows:
        return 0.5
    wins = tot = 0
    for m in rows:
        tot += 1
        is_rad = m.get("player_slot", 0) < 128
        rad_win = m.get("radiant_win")
        if rad_win is not None:
            win = (is_rad and rad_win) or ((not is_rad) and (rad_win is False))
            wins += 1 if win else 0
    wr = wins / tot if tot else 0.5
    exp = min(tot / 30.0, 1.0)   # experience cap
    raw = 1.6*wr + 0.4*exp       # 0..2 approx
    norm = 1 - 1/(1 + raw/1.2)
    return 0.45 + 0.5*norm

# ------------------ Hero meta alignment ------------------

def fetch_dotabuff_immortal_win_table(max_heroes: int = 60) -> Dict[str, float]:
    """hero name -> wr (0..1). Returns {} on failure."""
    try:
        resp = requests.get(DOTABUFF_HERO_WIN_URL, headers=HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logging.info(f"[warn] Dotabuff fetch failed: {e}")
        return {}
    try:
        tables = pd.read_html(html)
    except Exception as e:
        logging.info(f"[warn] read_html failed on Dotabuff: {e}")
        return {}
    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("hero" in str(c).lower() for c in t.columns) and any("win rate" in str(c).lower() for c in t.columns):
            target = t
            break
    if target is None:
        return {}
    col_hero = [c for c in target.columns if "Hero" in str(c)][0]
    col_wr = [c for c in target.columns if "Win rate" in str(c)][0]
    df = target[[col_hero, col_wr]].dropna()
    df[col_wr] = df[col_wr].astype(str).str.replace("%","", regex=False).astype(float) / 100.0
    df = df.sort_values(col_wr, ascending=False).head(max_heroes)
    return dict(zip(df[col_hero].astype(str).str.strip(), df[col_wr]))

def fetch_opendota_pro_meta_wr() -> Dict[str, float]:
    """
    Fallback meta if Dotabuff blocks us:
    Use OpenDota /heroStats pro winrate (approximation, not Immortal AP).
    """
    try:
        r = requests.get(f"{API_BASE}/heroStats", headers=HEADERS_API, timeout=30)
        r.raise_for_status()
        time.sleep(SLEEP_LIST)
        arr = r.json() or []
    except Exception as e:
        logging.info(f"[warn] OpenDota /heroStats failed: {e}")
        return {}
    meta = {}
    for h in arr:
        name = h.get("localized_name") or h.get("name") or ""
        pro_pick = h.get("pro_pick", 0) or 0
        pro_win = h.get("pro_win", 0) or 0
        wr = (pro_win / pro_pick) if pro_pick else 0.5
        meta[str(name).strip()] = wr
    return meta

def get_hero_id_map() -> Dict[int, str]:
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
    """ /teams/{id}/heroes → [{'hero_id': int, 'games': int, 'win': int}, ...] """
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
    Score team alignment with meta heroes:
      Sum over heroes: usage_share * team_hero_wr * (meta_wr - 0.5).
      (meta_wr - 0.5) centers at zero; positive if hero is strong in the meta.
      team_hero_wr is win/games from OpenDota team heroes endpoint.
    Squash result to ~0.45..0.95 (neutral ~0.5).
    """
    rows = get_team_heroes(team_id)
    if not rows:
        return 0.5
    total_games = sum(r.get("games", 0) or 0 for r in rows)
    if total_games <= 0:
        return 0.5
    if not meta_wr:
        # if meta empty, neutral
        return 0.5

    score = 0.0
    for r in rows:
        hid = r.get("hero_id")
        g   = r.get("games", 0) or 0
        w   = r.get("win", 0) or 0
        if not hid or g <= 0:
            continue
        name = hero_id_to_name.get(int(hid), "").strip()
        if not name:
            continue
        mwr = meta_wr.get(name)
        if mwr is None:
            continue
        share = g / total_games
        team_wr = w / g
        score += share * team_wr * (mwr - 0.5)  # core formulation

    # Typical raw is small; scale then squash
    raw = 8.0 * score + 0.5
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

    matches70 = get_team_matches(team_id, limit=70)
    wr70 = recent_wr(matches70)
    k_avg, d_avg, a_avg, kda = recent_kda_via_details(team_id, matches70, detail_cap=25)

    # robust fallback so we never get all-zero K/D/A
    if k_avg == 0 and d_avg == 0 and a_avg == 0:
        # heuristic from WR
        k_avg, d_avg, a_avg, kda = 24.0, 14.0, 48.0, (24+48)/max(14,1)

    group_s = g_wr.get(name, 0.5)
    pen = ROSTER_PENALTY if ROSTER_ISSUES.get(name, False) else 0.0

    players = get_team_players(team_id, max_players=6)
    roster_names = ", ".join([p["name"] for p in players]) if players else ""

    # Player TI pressure (OpenDota IDs)
    if players:
        ti_list = [player_ti_pressure_score(p["account_id"]) for p in players]
        ti_pressure = sum(ti_list)/len(ti_list) if ti_list else 0.5
        ti_raw_sum = sum(ti_list)
    else:
        ti_pressure = 0.5
        ti_raw_sum = 0.0

    # Hero meta alignment
    meta_align = hero_meta_alignment_score(team_id, hero_id_to_name, meta_wr)

    # Form component from KDA & kills
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
        "ti_pressure": ti_pressure, "ti_pressure_sum": ti_raw_sum,
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

# ------------------ Visuals ------------------

def save_score_bar_chart(df_scores: pd.DataFrame, out="bracket_scores.png"):
    # sorted ascending for nicer horizontal bars
    sub = df_scores[['score']].sort_values('score', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sub.index, sub['score'])
    ax.set_xlabel('Final Score')
    ax.set_title('TI25 Team Bracket Predictor Scores')
    for bar in bars:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.3f}', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def save_kda_heatmap(df_scores: pd.DataFrame, out="kda_heatmap.png"):
    # Build K/D/A matrix
    mat = df_scores[['k_avg','d_avg','a_avg']].copy()
    # Normalize for display range (optional)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat.values, aspect='auto')
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['K','D','A'])
    ax.set_title('TI25 Team K/D/A Heatmap (last 70)')
    # annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i,j]:.1f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def save_bracket_tree(res: dict, out="bracket_tree.png"):
    """
    Simple bracket tree visualization using matplotlib text and lines.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    # Layout coordinates
    # QFs (left), SFs (mid), UB Final (right top), LB path (bottom), GF (far right)
    # UB QF
    qf_y = [5, 4, 2.5, 1.5]
    for i, y in enumerate(qf_y):
        ax.text(0.05, y, f"UB QF{i+1}: {UB_QF[i][0]} vs {UB_QF[i][1]}", fontsize=9)

    # Results extraction
    ubqf_w = res['ubqf_winners']; ubqf_l = res['ubqf_losers']
    ubsf_w = res['ubsf_winners']; ubsf_l = res['ubsf_losers']
    ubf_w  = res['ubf_winner']; ubf_l = res['ubf_loser']
    lbr1_w = res['lbr1_winners']; lbqf_w = res['lbqf_winners']
    lbsf_w = res['lbsf_winner']; lbf_w  = res['lbf_winner']
    gf_w   = res['gf_winner']

    # UB SFs
    ax.text(0.40, 4.5, f"UB SF1: {ubqf_w[0]} vs {ubqf_w[1]}  →  {ubsf_w[0]}", fontsize=10)
    ax.text(0.40, 2.0, f"UB SF2: {ubqf_w[2]} vs {ubqf_w[3]}  →  {ubsf_w[1]}", fontsize=10)

    # UB Final
    ax.text(0.70, 3.25, f"UB Final: {ubsf_w[0]} vs {ubsf_w[1]}  →  {ubf_w}", fontsize=11, fontweight='bold')

    # LB R1
    ax.text(0.05, -0.3, f"LB R1-1: {ubqf_l[0]} vs {ubqf_l[1]}  →  {lbr1_w[0]}", fontsize=9)
    ax.text(0.05, -1.1, f"LB R1-2: {ubqf_l[2]} vs {ubqf_l[3]}  →  {lbr1_w[1]}", fontsize=9)

    # LB QF
    ax.text(0.40, -0.7, f"LB QF1: {lbr1_w[0]} vs {ubsf_l[1]}  →  {lbqf_w[0]}", fontsize=9)
    ax.text(0.40, -1.5, f"LB QF2: {lbr1_w[1]} vs {ubsf_l[0]}  →  {lbqf_w[1]}", fontsize=9)

    # LB SF
    ax.text(0.70, -1.1, f"LB SF: {lbqf_w[0]} vs {lbqf_w[1]}  →  {lbsf_w}", fontsize=10)

    # LB Final
    ax.text(0.85, 1.5, f"LB Final: {lbsf_w} vs {ubf_l}  →  {lbf_w}", fontsize=10)

    # Grand Final
    ax.text(0.93, 0.6, f"Grand Final: {ubf_w} vs {lbf_w}  →  Champion {gf_w}", fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

# ------------------ Main ------------------

def main():
    # Try Dotabuff meta first; fallback to OpenDota pro meta
    logging.info("Fetching Dotabuff Immortal 14d hero winrates…")
    meta_wr = fetch_dotabuff_immortal_win_table(max_heroes=60)
    if not meta_wr:
        logging.info("Falling back to OpenDota pro meta…")
        meta_wr = fetch_opendota_pro_meta_wr()

    logging.info("Fetching OpenDota hero ID map…")
    hero_id_to_name = get_hero_id_map()

    g_wr, h2h = parse_input()

    # Compute team scores
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

    # Save visuals
    try:
        save_score_bar_chart(df, out="bracket_scores.png")
        save_kda_heatmap(df, out="kda_heatmap.png")
        save_bracket_tree(res, out="bracket_tree.png")
    except Exception as e:
        logging.info(f"[warn] Failed to create one or more images: {e}")

    # Write markdown
    out = "bracket_prediction.md"
    with open(out, "w", encoding="utf-8") as f:
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
        f.write("\n## Visuals\n")
        f.write("- ![Scores](bracket_scores.png)\n")
        f.write("- ![KDA Heatmap](kda_heatmap.png)\n")
        f.write("- ![Bracket Tree](bracket_tree.png)\n")
    print(f"[ok] Wrote {out}")

if __name__ == "__main__":
    main()

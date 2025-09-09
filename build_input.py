#!/usr/bin/env python3
# dota_bitch/build_input.py
import re, sys, time
import requests
from bs4 import BeautifulSoup

LP_URL = "https://liquipedia.net/dota2/The_International/2025/Group_Stage"

# Map Liquipedia display names -> your canonical keys (adjust if needed)
NAME_MAP = {
    "Xtreme Gaming": "Xtreme Gaming",
    "BetBoom Team": "BetBoom Team",
    "Team Tidebound": "Team Tidebound",
    "Team Falcons": "Team Falcons",
    "PARIVISION": "PARIVISION",
    "HEROIC": "Heroic",
    "HEROIC": "Heroic",
    "Nigma Galaxy": "Nigma Galaxy",
    "Tundra Esports": "Tundra Esports",
    # common others (we keep them in case they appeared in GS lists)
    "Team Spirit": "Team Spirit",
    "Team Liquid": "Team Liquid",
    "Aurora Gaming": "Aurora Gaming",
    "Natus Vincere": "Natus Vincere",
    "BOOM Esports": "BOOM Esports",
    "Wildcard": "Wildcard",
    "Yakutou Brothers": "Yakutou Brothers",
    "Team Nemesis": "Team Nemesis",
}

# Put your 8 playoff teams here so we can optionally filter later if desired
FINAL_EIGHT = {
    "Xtreme Gaming", "BetBoom Team", "Team Tidebound", "Team Falcons",
    "PARIVISION", "Heroic", "Nigma Galaxy", "Tundra Esports"
}

# Your preferred W-L block at the top (edit these if you want to lock them)
GROUP_RECORDS = {
    "Xtreme Gaming": "4-0",
    "BetBoom Team": "4-1",
    "Team Tidebound": "4-1",
    "Team Falcons": "3-2",
    "PARIVISION": "3-2",
    "Heroic": "3-2",
    "Nigma Galaxy": "2-3",
    "Tundra Esports": "2-3",
}

def canon(name):
    return NAME_MAP.get(name.strip(), name.strip())

def fetch_html(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "TI25-bracket-bot"})
    r.raise_for_status()
    return r.text

def parse_matches(html):
    """
    Parse the 'Matches' section. We look for rows with two team links and a score like '2:1'.
    Returns list of tuples: (winner, loser, a_score, b_score)
    """
    soup = BeautifulSoup(html, "lxml")
    text_blocks = soup.select("#mw-content-text .mw-parser-output")[0]

    results = []
    # Heuristic: find all patterns like 'TeamA ... 2:1 ... TeamB' in same line/container
    # We’ll search over link-text + neighboring text.
    # Liquipedia markup changes occasionally; this is forgiving and de-dupes.
    rows = text_blocks.find_all(["p", "li", "tr", "div"], recursive=True)
    seen = set()
    score_re = re.compile(r"(\d+)\s*[:\-]\s*(\d+)")
    for row in rows:
        txt = " ".join(row.stripped_strings)
        m = score_re.search(txt)
        if not m:
            continue
        a_score, b_score = int(m.group(1)), int(m.group(2))
        teams = [a.get_text(" ", strip=True) for a in row.find_all("a") if a.get("href", "").startswith("/dota2/")]
        # Filter obvious non-team anchors
        teams = [t for t in teams if t and t.lower() not in ("view match details", "statistics", "main event", "group stage")]
        if len(teams) < 2:
            continue

        # Take first two team-like anchors around the score
        A = canon(teams[0])
        B = canon(teams[1])
        if (A, B, a_score, b_score) in seen or (B, A, b_score, a_score) in seen:
            continue
        seen.add((A, B, a_score, b_score))

        # Winner/loser
        if a_score == b_score:
            # Shouldn't happen in Bo3; skip draws defensively
            continue
        winner, loser = (A, B) if a_score > b_score else (B, A)
        results.append((winner, loser, a_score, b_score))
    return results

def write_input(path, records, h2h_pairs):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Group Win-Loss Records\n")
        for team, rec in records.items():
            f.write(f"{team}: {rec}\n")
        f.write("\n# Head-to-head results (Group Stage)\n")
        for w, l, a, b in h2h_pairs:
            f.write(f"{w} > {l}\n")

def main():
    html = fetch_html(LP_URL)
    matches = parse_matches(html)

    # If you ONLY want matchups among the final 8, uncomment this filter:
    # matches = [(w,l,a,b) for (w,l,a,b) in matches if w in FINAL_EIGHT and l in FINAL_EIGHT]

    write_input("input.txt", GROUP_RECORDS, matches)
    print(f"Wrote input.txt with {len(matches)} GS matchups.", file=sys.stderr)

if __name__ == "__main__":
    main()

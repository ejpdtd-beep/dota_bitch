#!/usr/bin/env python3
# dota_bitch/build_input.py
import re, sys
import requests
from bs4 import BeautifulSoup

LP_URL = "https://liquipedia.net/dota2/The_International/2025/Group_Stage"

NAME_MAP = {
    "Xtreme Gaming": "Xtreme Gaming",
    "BetBoom Team": "BetBoom Team",
    "Team Tidebound": "Team Tidebound",
    "Team Falcons": "Team Falcons",
    "PARIVISION": "PARIVISION",
    "HEROIC": "Heroic",
    "Heroic": "Heroic",
    "Nigma Galaxy": "Nigma Galaxy",
    "Tundra Esports": "Tundra Esports",
    "Team Spirit": "Team Spirit",
    "Team Liquid": "Team Liquid",
    "Aurora Gaming": "Aurora Gaming",
    "Natus Vincere": "Natus Vincere",
    "BOOM Esports": "BOOM Esports",
    "Wildcard": "Wildcard",
    "Yakutou Brothers": "Yakutou Brothers",
    "Team Nemesis": "Team Nemesis",
}

FINAL_EIGHT = {
    "Xtreme Gaming", "BetBoom Team", "Team Tidebound", "Team Falcons",
    "PARIVISION", "Heroic", "Nigma Galaxy", "Tundra Esports"
}

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

def canon(name): return NAME_MAP.get(name.strip(), name.strip())

def fetch_html(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "TI25-bracket-bot"})
    r.raise_for_status()
    return r.text

def parse_matches(html):
    from bs4 import BeautifulSoup
    import re
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("#mw-content-text .mw-parser-output")
    results, seen = [], set()
    score_re = re.compile(r"(\d+)\s*[:\-]\s*(\d+)")
    rows = root.find_all(["p", "li", "tr", "div"], recursive=True)
    for row in rows:
        txt = " ".join(row.stripped_strings)
        m = score_re.search(txt)
        if not m: continue
        a_score, b_score = int(m.group(1)), int(m.group(2))
        teams = [a.get_text(" ", strip=True) for a in row.find_all("a") if a.get("href", "").startswith("/dota2/")]
        teams = [t for t in teams if t and t.lower() not in ("view match details", "statistics", "main event", "group stage")]
        if len(teams) < 2: continue
        A, B = canon(teams[0]), canon(teams[1])
        if (A,B,a_score,b_score) in seen or (B,A,b_score,a_score) in seen: continue
        seen.add((A,B,a_score,b_score))
        if a_score == b_score: continue
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
    # To limit to final 8 only, uncomment next line:
    # matches = [(w,l,a,b) for (w,l,a,b) in matches if w in FINAL_EIGHT and l in FINAL_EIGHT]
    write_input("input.txt", GROUP_RECORDS, matches)
    print(f"Wrote input.txt with {len(matches)} GS matchups.", file=sys.stderr)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import itertools
import random

TEAMS = [
    "Xtreme Gaming",
    "PARIVISION",
    "Heroic",
    "Team Tidebound",
    "Team Falcons",
    "BetBoom Team",
    "Nigma Galaxy",
    "Tundra Esports",
]

# Example group records (adjust freely)
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

random.seed(7)

def emit_h2h():
    pairs = list(itertools.combinations(TEAMS, 2))
    lines = []
    for a,b in pairs:
        # Create one directed H2H for density
        winner = random.choice([a,b])
        loser  = b if winner == a else a
        lines.append(f"{winner} > {loser}")
    return lines

def main():
    lines = []
    lines.append("# Group Win-Loss Records")
    for t, rec in GROUP_RECORDS.items():
        lines.append(f"{t}: {rec}")
    lines.append("")
    lines.append("# Head-to-head results (Group Stage)")
    lines.extend(emit_h2h())
    with open("input.txt","w") as f:
        f.write("\n".join(lines))
    print(f"Wrote input.txt with {len(lines)} lines.")

if __name__ == "__main__":
    main()

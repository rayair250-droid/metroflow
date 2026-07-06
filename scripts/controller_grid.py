#!/usr/bin/env python
"""Regenerate the README "all four controllers" table.

Runs every controller on each shipped real-line scenario under the same
rush_incident stress overlay (the hero-GIF setup) and prints total denied
boardings per (line, controller). This is the exact procedure behind the
honest-limits table in the README, kept as a script so the numbers are
reproducible rather than hand-copied.

    python scripts/controller_grid.py [--seed 42]
"""

from __future__ import annotations

import argparse
import os

from metroflow.config import load_config
from metroflow.controllers import available_controllers
from metroflow.simulation import run_simulation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (label, scenario stem) for the real-line scenarios, any order; the table is
#: sorted by station count at render time.
LINES = [
    ("Intercités P-Tours", "intercites_paris_tours"),
    ("Lyon A", "lyon_line_a"),
    ("Rennes a", "rennes_line_a"),
    ("Rennes b", "rennes_line_b"),
    ("Lyon D", "lyon_line_d"),
    ("TER Marseille-Hyères", "ter_marseille_hyeres"),
    ("Lille 1", "lille_line_1"),
    ("Toulouse A", "toulouse_line_a"),
    ("Toulouse B", "toulouse_line_b"),
    ("Paris 14", "paris_line_14"),
    ("RER A", "rer_a"),
    ("RER B", "rer_b"),
    ("Lille 2", "lille_line_2"),
    ("HK Tramway", "hongkong_tramway"),
]


def _stressed(scenario_stem: str):
    """Load a scenario and apply the rush_incident stress overlay in place."""
    cfg = load_config(os.path.join(ROOT, "scenarios", f"{scenario_stem}.yaml"))
    stress = load_config(os.path.join(ROOT, "scenarios", "rush_incident.yaml"))
    cfg.demand.arrival_scale = stress.demand.arrival_scale
    cfg.demand.baseline_frac = stress.demand.baseline_frac
    cfg.demand.peaks = stress.demand.peaks
    cfg.incidents = stress.incidents
    cfg.depot_reserve = stress.depot_reserve
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    controllers = available_controllers()
    rows = []
    for label, stem in LINES:
        cfg = _stressed(stem)
        denied = {
            c: run_simulation(cfg, c, args.seed).summary()["total_denied_boardings"]
            for c in controllers
        }
        rows.append((cfg.line.n_stations, label, denied))
    rows.sort(key=lambda r: r[0])

    header = f"| Line | St | {' | '.join(controllers)} |"
    print(header)
    print("|---|--:|" + "|".join(["--:"] * len(controllers)) + "|")
    for n, label, denied in rows:
        best = min(denied.values())
        cells = []
        for c in controllers:
            v = f"{denied[c]:,}".replace(",", " ")
            cells.append(f"**{v}**" if denied[c] == best else v)
        print(f"| {label} | {n} | {' | '.join(cells)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

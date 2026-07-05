#!/usr/bin/env python
"""Generate the demo figures and comparison/experiment JSON for the README.

Runs the four controllers on the heavy ``rush_incident`` scenario, plus a small
Monte-Carlo experiment (confidence intervals), a sensitivity sweep, and the
bunching validation, and writes the PNGs + JSON into ``examples/``.

Usage::

    python examples/generate_examples.py [--seed 42] [--replications 12]
"""

from __future__ import annotations

import argparse
import os

from metroflow import plots
from metroflow.config import load_config
from metroflow.experiment import (
    experiment_payload,
    run_experiment,
    significance_vs_baseline,
    summarize,
)
from metroflow.line import UP
from metroflow.report import comparison_payload, format_comparison, write_json
from metroflow.simulation import run_simulation
from metroflow.validation import bunching_config, bunching_reproduction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCENARIO = os.path.join(ROOT, "scenarios", "rush_incident.yaml")
DEFAULT_SCENARIO = os.path.join(ROOT, "scenarios", "default.yaml")
CONTROLLERS = ["baseline", "reactive", "predictive", "optimizer"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replications", type=int, default=12)
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    cfg = load_config(args.scenario)
    cfg.seed = args.seed
    os.makedirs(args.outdir, exist_ok=True)
    written = []

    # --- single-seed comparison (all four controllers) --------------------- #
    sims = {name: run_simulation(cfg, name, args.seed) for name in CONTROLLERS}
    summaries = [sims[name].summary() for name in CONTROLLERS]
    write_json(os.path.join(args.outdir, "comparison.json"), comparison_payload(summaries))
    written.append(os.path.join(args.outdir, "comparison.json"))

    written.append(
        plots.plot_denied_comparison(summaries, os.path.join(args.outdir, "denied_comparison.png"))
    )
    written.append(
        plots.plot_headway_comparison(sims, os.path.join(args.outdir, "headway_comparison.png"))
    )
    written.append(
        plots.plot_queue_over_time(
            sims["baseline"], os.path.join(args.outdir, "queues_baseline.png")
        )
    )
    written.append(
        plots.plot_queue_over_time(
            sims["predictive"], os.path.join(args.outdir, "queues_predictive.png")
        )
    )
    written.append(
        plots.plot_load_heatmap(
            sims["predictive"], os.path.join(args.outdir, "load_heatmap_predictive.png")
        )
    )
    written.append(
        plots.dashboard(sims, os.path.join(args.outdir, "dashboard.png"), scenario=cfg.name)
    )

    # --- Monte-Carlo experiment with confidence intervals ------------------ #
    results = run_experiment(cfg, CONTROLLERS, args.replications, args.seed)
    stats = summarize(results)
    sig = significance_vs_baseline(results, stats, baseline="baseline")
    write_json(
        os.path.join(args.outdir, "experiment.json"),
        experiment_payload(cfg.name, CONTROLLERS, args.replications, args.seed, stats, sig),
    )
    written.append(os.path.join(args.outdir, "experiment.json"))
    written.append(
        plots.plot_experiment_ci(
            stats, "total_denied_boardings", os.path.join(args.outdir, "experiment_ci.png")
        )
    )

    # --- sensitivity sweep ------------------------------------------------- #
    from metroflow.validation import sensitivity_sweep

    sweep = sensitivity_sweep(
        cfg,
        param="demand.arrival_scale",
        values=[0.05, 0.06, 0.07, 0.08, 0.09],
        controller="predictive",
        seed=args.seed,
    )
    written.append(
        plots.plot_sensitivity(sweep, os.path.join(args.outdir, "sensitivity_arrival.png"))
    )

    # --- bunching validation ---------------------------------------------- #
    bcfg = bunching_config()
    hcfg = bunching_config()
    hcfg.holding_control = True
    b_base = bunching_reproduction(run_simulation(bcfg, "baseline", args.seed), UP)
    b_hold = bunching_reproduction(run_simulation(hcfg, "baseline", args.seed), UP)
    written.append(
        plots.plot_bunching(
            [("no control", b_base), ("headway holding", b_hold)],
            os.path.join(args.outdir, "bunching_growth.png"),
        )
    )

    # --- sober line animation (GIF) --------------------------------------- #
    try:
        from metroflow.animate import render_animation

        acfg = load_config(DEFAULT_SCENARIO)
        acfg.seed = args.seed
        gif = os.path.join(args.outdir, "line_animation.gif")
        render_animation(acfg, "predictive", args.seed, gif, seconds=8, fps=10)
        written.append(gif)
    except Exception as exc:  # pragma: no cover - pillow optional
        print(f"(animation skipped: {exc})")

    print(format_comparison(summaries))
    print("\nWrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

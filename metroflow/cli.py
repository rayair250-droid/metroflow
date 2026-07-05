"""Command-line interface: ``simulate``, ``compare`` and ``plot``."""

from __future__ import annotations

import argparse
import os
import sys

from metroflow import __version__
from metroflow.config import load_config
from metroflow.controllers import available_controllers
from metroflow.errors import MetroFlowError
from metroflow.report import (
    comparison_payload,
    format_comparison,
    format_summary,
    write_json,
)
from metroflow.simulation import run_simulation

# Default controller set for `compare` (kept to the three v1 heuristics for
# backwards compatibility; add the optimiser with `--controllers`).
_ALL = ["baseline", "reactive", "predictive"]
_ALL4 = ["baseline", "reactive", "predictive", "optimizer"]


def _parse_controllers(spec, default):
    if not spec:
        return list(default)
    names = [c.strip() for c in spec.split(",") if c.strip()]
    valid = set(available_controllers())
    for c in names:
        if c not in valid:
            raise SystemExit(f"Unknown controller '{c}'. Available: {sorted(valid)}")
    return names


def _load(scenario: str | None, seed: int | None, name: str | None = None):
    cfg = load_config(scenario)
    if seed is not None:
        cfg.seed = seed
    if name is not None:
        cfg.name = name
    elif scenario is not None:
        cfg.name = os.path.splitext(os.path.basename(scenario))[0]
    return cfg


def _maybe_apply_gtfs(cfg, args) -> None:
    """If ``--gtfs`` was given, override the line from the GTFS feed in place."""
    gtfs_dir = getattr(args, "gtfs", None)
    if not gtfs_dir:
        return
    if not getattr(args, "route", None):
        raise SystemExit("--route is required when using --gtfs")
    from metroflow.gtfs import apply_gtfs

    built = apply_gtfs(cfg, gtfs_dir, args.route, getattr(args, "direction", 0))
    print(
        f"Built line from GTFS: route={built.route_id} ({built.route_label}) "
        f"direction={built.direction_id}  {built.line.n_stations} stations: "
        + " -> ".join(built.station_names)
    )


def _cmd_simulate(args) -> int:
    cfg = _load(args.scenario, args.seed)
    _maybe_apply_gtfs(cfg, args)
    sim = run_simulation(cfg, args.controller, cfg.seed)
    summary = sim.summary()
    print(format_summary(summary))

    if args.json:
        write_json(args.json, summary)
        print(f"\nWrote JSON: {args.json}")

    if args.plots:
        from metroflow import plots

        os.makedirs(args.plots, exist_ok=True)
        p1 = plots.plot_load_heatmap(sim, os.path.join(args.plots, "load_heatmap.png"))
        p2 = plots.plot_queue_over_time(sim, os.path.join(args.plots, "queues.png"))
        print(f"Wrote plots: {p1}, {p2}")
    return 0


def _run_all(cfg, seed: int, controllers):
    sims = {name: run_simulation(cfg, name, seed) for name in controllers}
    summaries = [sims[name].summary() for name in controllers]
    return sims, summaries


def _cmd_compare(args) -> int:
    cfg = _load(args.scenario, args.seed)
    controllers = _parse_controllers(getattr(args, "controllers", None), _ALL)
    sims, summaries = _run_all(cfg, cfg.seed, controllers)
    print(format_comparison(summaries))

    if args.json:
        write_json(args.json, comparison_payload(summaries))
        print(f"\nWrote JSON: {args.json}")

    if args.plots:
        from metroflow import plots

        os.makedirs(args.plots, exist_ok=True)
        outs: list[str] = []
        outs.append(
            plots.plot_denied_comparison(
                summaries, os.path.join(args.plots, "denied_comparison.png")
            )
        )
        outs.append(
            plots.plot_headway_comparison(sims, os.path.join(args.plots, "headway_comparison.png"))
        )
        first = controllers[0]
        focus = "predictive" if "predictive" in sims else controllers[-1]
        outs.append(
            plots.plot_queue_over_time(sims[first], os.path.join(args.plots, f"queues_{first}.png"))
        )
        outs.append(
            plots.plot_queue_over_time(sims[focus], os.path.join(args.plots, f"queues_{focus}.png"))
        )
        outs.append(
            plots.plot_load_heatmap(
                sims[focus],
                os.path.join(args.plots, f"load_heatmap_{focus}.png"),
            )
        )
        print("Wrote plots:\n  " + "\n  ".join(outs))
    return 0


def _cmd_experiment(args) -> int:
    from metroflow.experiment import (
        experiment_payload,
        format_experiment_report,
        run_experiment,
        significance_vs_baseline,
        summarize,
    )

    cfg = _load(args.scenario, args.seed)
    controllers = _parse_controllers(args.controllers, _ALL4)
    seed = args.seed if args.seed is not None else cfg.seed
    results = run_experiment(cfg, controllers, args.replications, seed)
    stats = summarize(results)
    baseline = "baseline" if "baseline" in controllers else controllers[0]
    sig = significance_vs_baseline(results, stats, baseline=baseline)
    print(
        format_experiment_report(
            cfg.name, controllers, args.replications, seed, stats, sig, baseline
        )
    )

    if args.json:
        write_json(
            args.json,
            experiment_payload(cfg.name, controllers, args.replications, seed, stats, sig),
        )
        print(f"\nWrote JSON: {args.json}")

    if args.plots:
        from metroflow import plots

        os.makedirs(args.plots, exist_ok=True)
        p = plots.plot_experiment_ci(
            stats,
            "total_denied_boardings",
            os.path.join(args.plots, "experiment_ci.png"),
        )
        print(f"Wrote plot: {p}")
    return 0


def _cmd_plot(args) -> int:
    """Convenience: run one controller and dump its figures."""
    cfg = _load(args.scenario, args.seed)
    sim = run_simulation(cfg, args.controller, cfg.seed)
    from metroflow import plots

    outdir = args.plots or "out"
    os.makedirs(outdir, exist_ok=True)
    p1 = plots.plot_load_heatmap(sim, os.path.join(outdir, "load_heatmap.png"))
    p2 = plots.plot_queue_over_time(sim, os.path.join(outdir, "queues.png"))
    print(f"Wrote plots: {p1}, {p2}")
    return 0


def _cmd_animate(args) -> int:
    cfg = _load(args.scenario, args.seed)
    _maybe_apply_gtfs(cfg, args)
    try:
        from metroflow.animate import (
            PillowMissingError,
            animate_from_config,
            comparison_from_config,
        )
    except Exception as exc:  # pragma: no cover - import guard
        print(f"Animation unavailable: {exc}")
        return 1
    try:
        if args.compare:
            out = comparison_from_config(cfg, args.seed, args.out, args.seconds, args.fps)
        else:
            out = animate_from_config(
                cfg, args.controller, args.seed, args.out, args.seconds, args.fps
            )
    except PillowMissingError as exc:
        print(str(exc))
        return 1
    size = os.path.getsize(out)
    kind = "comparison animation" if args.compare else "animation"
    print(f"Wrote {kind}: {out} ({size} bytes, {size / 1e6:.2f} MB)")
    return 0


def _cmd_gtfs_info(args) -> int:
    from metroflow.gtfs import describe_feed

    print(describe_feed(args.directory))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metroflow",
        description="Discrete-event metro-line simulator with predictive reserve-train injection.",
    )
    parser.add_argument("--version", action="version", version=f"metroflow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="run one controller over one scenario")
    p_sim.add_argument("--scenario", default=None, help="scenario YAML (defaults built-in)")
    p_sim.add_argument("--controller", default="predictive", choices=available_controllers())
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.add_argument("--json", default=None, help="write per-run JSON here")
    p_sim.add_argument("--plots", default=None, help="write PNG plots to this dir")
    p_sim.add_argument(
        "--gtfs",
        default=None,
        help="build the line from a GTFS feed directory instead of a YAML line block",
    )
    p_sim.add_argument("--route", default=None, help="GTFS route_id (with --gtfs)")
    p_sim.add_argument("--direction", type=int, default=0, help="GTFS direction_id (with --gtfs)")
    p_sim.set_defaults(func=_cmd_simulate)

    p_cmp = sub.add_parser("compare", help="run several controllers on the same seed and scenario")
    p_cmp.add_argument("--scenario", default=None)
    p_cmp.add_argument("--seed", type=int, default=None)
    p_cmp.add_argument(
        "--controllers",
        default=None,
        help="comma-separated controllers (default: baseline,reactive,predictive)",
    )
    p_cmp.add_argument("--json", default=None)
    p_cmp.add_argument("--plots", default=None)
    p_cmp.set_defaults(func=_cmd_compare)

    p_exp = sub.add_parser(
        "experiment",
        help="Monte-Carlo: N seeded replications per controller with 95%% CIs",
    )
    p_exp.add_argument("--scenario", default=None)
    p_exp.add_argument("--seed", type=int, default=None, help="base seed")
    p_exp.add_argument(
        "--controllers",
        default=None,
        help="comma-separated controllers (default: baseline,reactive,predictive,optimizer)",
    )
    p_exp.add_argument("--replications", type=int, default=30)
    p_exp.add_argument("--json", default=None)
    p_exp.add_argument("--plots", default=None)
    p_exp.set_defaults(func=_cmd_experiment)

    p_plot = sub.add_parser("plot", help="run one controller and write its figures")
    p_plot.add_argument("--scenario", default=None)
    p_plot.add_argument("--controller", default="predictive", choices=available_controllers())
    p_plot.add_argument("--seed", type=int, default=None)
    p_plot.add_argument("--plots", default=None)
    p_plot.set_defaults(func=_cmd_plot)

    p_anim = sub.add_parser("animate", help="render a short, sober GIF of one simulation run")
    p_anim.add_argument("--scenario", default=None)
    p_anim.add_argument("--controller", default="predictive", choices=available_controllers())
    p_anim.add_argument("--seed", type=int, default=None)
    p_anim.add_argument("--out", default="out/run.gif", help="output GIF path")
    p_anim.add_argument(
        "--compare",
        action="store_true",
        help="split-screen baseline vs predictive on the same seed (two stacked "
        "panels showing reserve-train injection cutting denied boardings); "
        "ignores --controller",
    )
    p_anim.add_argument("--seconds", type=float, default=8.0, help="target GIF playback length (s)")
    p_anim.add_argument("--fps", type=int, default=10, help="frames per second")
    p_anim.add_argument("--gtfs", default=None, help="build the line from a GTFS feed")
    p_anim.add_argument("--route", default=None, help="GTFS route_id (with --gtfs)")
    p_anim.add_argument("--direction", type=int, default=0, help="GTFS direction_id")
    p_anim.set_defaults(func=_cmd_animate)

    p_gtfs = sub.add_parser(
        "gtfs-info", help="print routes/stops detected in a GTFS feed directory"
    )
    p_gtfs.add_argument("directory", help="path to a GTFS feed directory")
    p_gtfs.set_defaults(func=_cmd_gtfs_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MetroFlowError as exc:
        # Turn expected, user-facing errors (bad scenario/GTFS/config) into a
        # clean one-line message on stderr with a non-zero exit, never a raw
        # Python traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Smoke tests for the figure-generation module.

These run every plotting function on a *tiny* simulation (short horizon, few
stations) so the fast test path stays quick, and assert only that a non-empty
PNG is produced -- pixel-level appearance is not asserted, but the code paths
(shared style, dashboard layout, colour mapping) are all exercised.
"""

from __future__ import annotations

from metroflow import plots
from metroflow.config import DemandConfig, LineConfig, SimConfig
from metroflow.line import UP
from metroflow.simulation import run_simulation
from metroflow.validation import bunching_reproduction, sensitivity_sweep


def _tiny_config() -> SimConfig:
    cfg = SimConfig(name="tiny")
    cfg.line = LineConfig(n_stations=6)
    cfg.horizon = 1800
    cfg.demand = DemandConfig(arrival_scale=0.09)
    cfg.demand.peaks = [{"center": 900.0, "width": 400.0, "amplitude": 1.5}]
    return cfg


def _png_ok(path) -> bool:
    from pathlib import Path

    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def test_single_run_plots(tmp_path):
    sim = run_simulation(_tiny_config(), "predictive", 42)
    assert _png_ok(plots.plot_load_heatmap(sim, str(tmp_path / "heat.png")))
    assert _png_ok(plots.plot_queue_over_time(sim, str(tmp_path / "queue.png")))


def test_comparison_and_dashboard(tmp_path):
    cfg = _tiny_config()
    sims = {c: run_simulation(cfg, c, 42) for c in ("baseline", "predictive", "optimizer")}
    summaries = [sims[c].summary() for c in sims]
    assert _png_ok(plots.plot_denied_comparison(summaries, str(tmp_path / "denied.png")))
    assert _png_ok(plots.plot_headway_comparison(sims, str(tmp_path / "hw.png")))
    dash = plots.dashboard(sims, str(tmp_path / "dashboard.png"), scenario="tiny")
    assert _png_ok(dash)


def test_validation_plots(tmp_path):
    cfg = _tiny_config()
    b = bunching_reproduction(run_simulation(cfg, "baseline", 42), UP)
    assert _png_ok(plots.plot_bunching([("no control", b)], str(tmp_path / "bunch.png")))
    sweep = sensitivity_sweep(
        cfg,
        param="demand.arrival_scale",
        values=[0.05, 0.07],
        controller="baseline",
        seed=42,
        metric="total_denied_boardings",
    )
    assert _png_ok(plots.plot_sensitivity(sweep, str(tmp_path / "sens.png")))


def test_apply_style_is_idempotent():
    # Calling the shared style repeatedly must not raise.
    plots.apply_style()
    plots.apply_style()

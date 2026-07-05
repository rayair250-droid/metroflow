"""Tests for the Monte-Carlo experiment harness."""

import numpy as np

from metroflow.config import SimConfig
from metroflow.experiment import (
    confidence_interval,
    run_experiment,
    significance_vs_baseline,
    summarize,
)


def _fast_cfg():
    cfg = SimConfig(name="exp")
    cfg.horizon = 2400
    cfg.demand.arrival_scale = 0.09
    cfg.demand.peaks = [{"center": 1200.0, "width": 600.0, "amplitude": 1.5}]
    cfg.incidents.enabled = False
    return cfg


def test_confidence_interval_known_values():
    st = confidence_interval(np.array([10.0, 12.0, 14.0, 16.0, 18.0]))
    assert st.mean == 14.0
    assert st.ci_low < st.mean < st.ci_high
    assert st.n == 5
    # Symmetric interval.
    assert abs((st.ci_high - st.mean) - (st.mean - st.ci_low)) < 1e-9


def test_confidence_interval_single_value():
    st = confidence_interval(np.array([5.0]))
    assert st.mean == 5.0
    assert st.ci_low == st.ci_high == 5.0


def test_run_experiment_shapes_and_ci():
    cfg = _fast_cfg()
    results = run_experiment(cfg, ["baseline", "reactive"], replications=5, seed=42)
    assert set(results) == {"baseline", "reactive"}
    assert results["baseline"]["total_denied_boardings"].shape == (5,)
    stats = summarize(results)
    st = stats["baseline"]["total_denied_boardings"]
    assert st.ci_low <= st.mean <= st.ci_high


def test_run_experiment_deterministic():
    cfg = _fast_cfg()
    a = run_experiment(cfg, ["reactive"], replications=4, seed=7)
    b = run_experiment(cfg, ["reactive"], replications=4, seed=7)
    assert np.array_equal(
        a["reactive"]["total_denied_boardings"],
        b["reactive"]["total_denied_boardings"],
    )


def test_significance_reactive_beats_baseline():
    """Reactive control should significantly cut denied boardings here."""
    cfg = _fast_cfg()
    results = run_experiment(cfg, ["baseline", "reactive"], replications=8, seed=42)
    stats = summarize(results)
    sig = significance_vs_baseline(results, stats, baseline="baseline")
    assert len(sig) == 1
    s = sig[0]
    assert s.controller == "reactive"
    assert s.mean_delta < 0  # fewer denied boardings
    assert s.significant  # Welch's t-test rejects equality

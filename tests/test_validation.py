"""Model-validation tests: Little's Law, bunching, sensitivity."""

from metroflow.line import UP
from metroflow.simulation import run_simulation
from metroflow.validation import (
    bunching_config,
    bunching_reproduction,
    littles_law_check,
    sensitivity_sweep,
    stable_config,
)


def test_littles_law_holds_in_steady_state():
    """On a stationary, unsaturated line, L ~= lambda * W."""
    sim = run_simulation(stable_config(), "baseline", 42)
    r = littles_law_check(sim)
    # The check is only meaningful when essentially nobody is denied.
    assert r.denial_rate < 0.01
    assert r.L > 0 and r.L_predicted > 0
    # Discrete-event Little's Law agreement within 20%.
    assert r.rel_error < 0.20


def test_littles_law_robust_across_seeds():
    for seed in (1, 7, 99):
        r = littles_law_check(run_simulation(stable_config(), "baseline", seed))
        assert r.rel_error < 0.25


def test_bunching_grows_without_control():
    """A small perturbation amplifies into bunching over time (headway CV up)."""
    sim = run_simulation(bunching_config(), "baseline", 42)
    b = bunching_reproduction(sim, UP)
    assert b.early_cv < 0.6  # trains start close to evenly spaced
    assert b.grew  # variability increases over the run
    assert b.late_cv > 2.0 * b.early_cv  # a large, unambiguous amplification


def test_holding_control_suppresses_bunching():
    """Even-headway holding (the classic anti-bunching control) suppresses it.

    Reserve injection targets crowding, not headway evenness, so the faithful
    suppression demonstration uses forward-headway holding.
    """
    base = bunching_reproduction(run_simulation(bunching_config(), "baseline", 42), UP)
    hcfg = bunching_config()
    hcfg.holding_control = True
    held = bunching_reproduction(run_simulation(hcfg, "baseline", 42), UP)
    assert held.late_cv < base.late_cv
    # Holding brings late-run variability back down toward the regular start.
    assert held.late_cv < 0.5 * base.late_cv


def test_sensitivity_sweep_monotone_in_demand():
    """Denied boardings should rise as arrival intensity rises."""
    cfg = bunching_config()
    res = sensitivity_sweep(
        cfg,
        param="demand.arrival_scale",
        values=[0.03, 0.045, 0.06],
        controller="baseline",
        seed=42,
        metric="total_denied_boardings",
    )
    assert res.outcomes[0] <= res.outcomes[-1]
    assert len(res.outcomes) == 3

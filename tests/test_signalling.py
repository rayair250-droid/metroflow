"""Unit tests for the safe-separation (signalling) models."""

import math

import pytest

from metroflow.config import SignallingConfig, SimConfig
from metroflow.signalling import (
    block_index,
    braking_distance,
    fixed_block_clear,
    fixed_block_min_separation,
    make_signalling_model,
    min_moving_block_headway,
    min_safe_separation,
)
from metroflow.simulation import run_simulation


# -- pure kinematics -------------------------------------------------------- #
def test_braking_distance_formula():
    # v^2 / (2a): 20 m/s, 1 m/s^2 -> 200 m.
    assert braking_distance(20.0, 1.0) == pytest.approx(200.0)
    assert braking_distance(0.0, 1.0) == 0.0
    # Halving deceleration doubles the braking distance.
    assert braking_distance(20.0, 0.5) == pytest.approx(400.0)


def test_braking_distance_degenerate():
    assert braking_distance(10.0, 0.0) == math.inf
    assert braking_distance(-5.0, 1.0) == 0.0


def test_min_safe_separation_components():
    # 20*2 (reaction) + 200 (braking) + 50 (margin) + 90 (length) = 380.
    sep = min_safe_separation(20.0, 1.0, 50.0, 2.0, 90.0)
    assert sep == pytest.approx(380.0)


def test_separation_grows_with_speed():
    slow = min_safe_separation(10.0, 1.0, 50.0, 2.0, 90.0)
    fast = min_safe_separation(20.0, 1.0, 50.0, 2.0, 90.0)
    assert fast > slow  # speed-dependent, unlike a fixed constant


def test_moving_block_headway_speed_dependent():
    h_slow = min_moving_block_headway(10.0, 1.0, 50.0, 2.0, 90.0)
    h_fast = min_moving_block_headway(20.0, 1.0, 50.0, 2.0, 90.0)
    # The implied minimum *time* headway is not constant across speeds.
    assert h_slow != pytest.approx(h_fast)
    assert h_slow > 0 and h_fast > 0


# -- fixed block ------------------------------------------------------------ #
def test_block_index():
    assert block_index(0.0, 400.0) == 0
    assert block_index(399.9, 400.0) == 0
    assert block_index(400.0, 400.0) == 1
    assert block_index(950.0, 400.0) == 2
    with pytest.raises(ValueError):
        block_index(100.0, 0.0)


def test_fixed_block_min_separation():
    assert fixed_block_min_separation(500.0, 1, 90.0) == pytest.approx(590.0)
    assert fixed_block_min_separation(500.0, 2, 0.0) == pytest.approx(1000.0)


def test_fixed_block_clear_occupancy():
    # Follower in block 0 (pos 100), one clear block required, +1 direction.
    # A leader in block 1 (pos 600) occupies the block immediately ahead -> blocked.
    assert not fixed_block_clear(100.0, [600.0], 500.0, +1, n_clear=1)
    # Leader far away in block 3 -> clear.
    assert fixed_block_clear(100.0, [1600.0], 500.0, +1, n_clear=1)
    # A leader in the follower's own block also blocks entry.
    assert not fixed_block_clear(100.0, [200.0], 500.0, +1, n_clear=1)


# -- configured model ------------------------------------------------------- #
def test_model_moving_block_clear_decision():
    cfg = SignallingConfig(mode="moving_block", train_length_m=90.0)
    m = make_signalling_model(cfg)
    req = m.required_separation(20.0)
    assert m.is_clear(0.0, req + 1.0, 20.0) is True
    assert m.is_clear(0.0, req - 1.0, 20.0) is False
    # No leader -> always clear.
    assert m.is_clear(0.0, None, 20.0) is True


def test_model_enforce_off_is_permissive():
    cfg = SignallingConfig(mode="moving_block", enforce=False)
    m = make_signalling_model(cfg)
    assert m.is_clear(0.0, 0.0, 20.0) is True


def test_model_invalid_mode():
    with pytest.raises(ValueError):
        make_signalling_model(SignallingConfig(mode="nonsense"))


# -- engine integration ----------------------------------------------------- #
def _fast_cfg(mode):
    cfg = SimConfig(name="sig")
    cfg.horizon = 3600
    cfg.demand.arrival_scale = 0.09
    cfg.signalling.mode = mode
    return cfg


def test_engine_records_holds_moving_block():
    cfg = _fast_cfg("moving_block")
    sim = run_simulation(cfg, "predictive", 42)
    # Enforcing separation should produce at least some platform holds and no
    # forced (deadlock-cap) releases in a well-posed scenario.
    assert sim.metrics.signal_holds > 0
    assert sim.metrics.forced_holds == 0


def test_fixed_block_more_restrictive_than_moving():
    """Fixed block quantises separation to whole sections -> more holds."""
    mb = run_simulation(_fast_cfg("moving_block"), "baseline", 7).metrics
    fb = run_simulation(_fast_cfg("fixed_block"), "baseline", 7).metrics
    assert fb.signal_hold_time_s >= mb.signal_hold_time_s


def test_enforce_off_matches_no_holds():
    cfg = _fast_cfg("moving_block")
    cfg.signalling.enforce = False
    sim = run_simulation(cfg, "baseline", 3)
    assert sim.metrics.signal_holds == 0

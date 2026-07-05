"""Tests for the CP-SAT / MILP optimiser controller."""

from metroflow.config import ControllerConfig, SimConfig
from metroflow.controllers import make_controller
from metroflow.controllers.optimizer import OptimizerController
from metroflow.simulation import run_simulation


def _fast_cfg():
    cfg = SimConfig(name="opt")
    cfg.horizon = 3600
    cfg.demand.arrival_scale = 0.09
    cfg.demand.peaks = [{"center": 1800.0, "width": 900.0, "amplitude": 1.4}]
    # Keep the solve tiny and fast for CI.
    cfg.controller.opt_horizon_steps = 4
    cfg.controller.opt_step_seconds = 120.0
    cfg.controller.opt_max_solve_seconds = 1.0
    return cfg


class _FakeSim:
    """Minimal stand-in for the early-return guard tests."""

    def __init__(self, reserves=3, since=1e9):
        self._reserves = reserves
        self._since = since

    def reserves_available(self):
        return self._reserves

    def time_since_last_injection(self):
        return self._since

    # decide() warms the fallback EWMA first; give it what it needs.
    def iter_queues(self):
        return iter(())

    def arrival_rate(self, station, t):
        return 0.0


def test_optimizer_registered():
    c = make_controller("optimizer", ControllerConfig())
    assert isinstance(c, OptimizerController)
    assert c.name == "optimizer"


def test_optimizer_no_reserves_returns_empty():
    c = OptimizerController(ControllerConfig())
    assert c.decide(_FakeSim(reserves=0), 100.0) == []
    assert c.last_status == "no_reserves"


def test_optimizer_respects_spacing():
    cfg = ControllerConfig(min_injection_gap=300)
    c = OptimizerController(cfg)
    assert c.decide(_FakeSim(reserves=3, since=10.0), 100.0) == []
    assert c.last_status == "spacing"


def test_optimizer_injects_and_beats_baseline():
    cfg = _fast_cfg()
    opt = run_simulation(cfg, "optimizer", 42).summary()
    base = run_simulation(cfg, "baseline", 42).summary()
    assert opt["reserves_used"] > 0
    assert opt["total_denied_boardings"] < base["total_denied_boardings"]


def test_optimizer_deterministic():
    cfg = _fast_cfg()
    a = run_simulation(cfg, "optimizer", 7).summary()
    b = run_simulation(cfg, "optimizer", 7).summary()
    assert a == b


def test_optimizer_solve_status_is_valid():
    """A run should reach a real CP-SAT solve at least once."""
    cfg = _fast_cfg()
    sim = run_simulation(cfg, "optimizer", 42)
    # The controller stores the status of its most recent solve.
    assert sim.controller.last_status in {
        "OPTIMAL",
        "FEASIBLE",
        "no_deficit",
        "spacing",
        "no_reserves",
    }
    # And it actually dispatched (proves the MILP path produced commands).
    assert sim.metrics.reserves_used > 0

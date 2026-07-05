"""Controller unit tests using a lightweight fake simulation."""

from metroflow.config import ControllerConfig
from metroflow.controllers import available_controllers, make_controller
from metroflow.controllers.baseline import BaselineController
from metroflow.controllers.predictive import PredictiveController
from metroflow.controllers.reactive import ReactiveController
from metroflow.line import UP


class _FakeLine:
    def __init__(self, n):
        self.n_stations = n


class _FakeSim:
    def __init__(self, queues, reserves=3, since=1e9, rates=None, n=12):
        self._queues = queues
        self._reserves = reserves
        self._since = since
        self._rates = rates or {}
        self.line = _FakeLine(n)

    def iter_queues(self):
        for (s, d), length in self._queues.items():
            yield s, d, length

    def reserves_available(self):
        return self._reserves

    def time_since_last_injection(self):
        return self._since

    def arrival_rate(self, station, t):
        return self._rates.get(station, 0.0)


def test_registry():
    # v2 adds the MILP optimiser alongside the three v1 heuristics.
    assert set(available_controllers()) == {
        "baseline",
        "reactive",
        "predictive",
        "optimizer",
    }
    assert isinstance(make_controller("reactive", ControllerConfig()), ReactiveController)


def test_baseline_never_injects():
    c = BaselineController(ControllerConfig())
    assert c.decide(_FakeSim({(5, UP): 500}), 100.0) == []


def test_reactive_fires_on_threshold():
    cfg = ControllerConfig(queue_threshold=90)
    c = ReactiveController(cfg)
    # Below threshold: no action.
    assert c.decide(_FakeSim({(5, UP): 50}), 100.0) == []
    # Above threshold: inject at the worst platform, in the loaded direction.
    cmds = c.decide(_FakeSim({(3, UP): 40, (5, UP): 120}), 100.0)
    assert len(cmds) == 1
    assert cmds[0].station == 5
    assert cmds[0].direction == UP


def test_reactive_respects_reserves_and_gap():
    cfg = ControllerConfig(queue_threshold=90, min_injection_gap=300)
    c = ReactiveController(cfg)
    assert c.decide(_FakeSim({(5, UP): 500}, reserves=0), 1000.0) == []
    assert c.decide(_FakeSim({(5, UP): 500}, since=10.0), 1000.0) == []


def test_predictive_injects_preemptively_upstream():
    cfg = ControllerConfig(queue_threshold=90, horizon=300, predictive_fraction=0.6)
    c = PredictiveController(cfg)
    # First poll establishes the baseline queue; forecast is still low.
    assert c.decide(_FakeSim({(5, UP): 20}), 0.0) == []
    # Second poll sees a rising queue; the horizon forecast crosses the trigger
    # (0.6 * 90 = 54) even though the current queue (50) is below it.
    cmds = c.decide(_FakeSim({(5, UP): 50}), 60.0)
    assert len(cmds) == 1
    assert cmds[0].direction == UP
    # Inserted upstream of the forecast hotspot (S05 - 2 = S03).
    assert cmds[0].station == 3


def test_predictive_respects_reserves():
    cfg = ControllerConfig(queue_threshold=90, horizon=300, predictive_fraction=0.6)
    c = PredictiveController(cfg)
    c.decide(_FakeSim({(5, UP): 20}), 0.0)
    assert c.decide(_FakeSim({(5, UP): 50}, reserves=0), 60.0) == []

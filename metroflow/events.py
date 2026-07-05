"""Stochastic incident generator.

Runs as a SimPy process that, on a fixed cadence, rolls independent dice for
each incident class and applies its effect to the live simulation. Every
incident is timestamped and logged to the metrics collector.

Incident classes
----------------
breakdown
    An in-service train fails. It stops taking passengers and runs empty to the
    depot, becoming a reserve again after a repair delay.
signal_failure
    A segment suffers a temporary speed restriction: travel time over it is
    multiplied by ``signal_slowdown`` for ``signal_duration`` seconds.
dwell_event
    A door/boarding problem adds ``dwell_event_extra`` seconds to the next dwell
    at an affected station.
surge
    A sudden demand spike (event let-out) multiplies a station's arrival rate for
    ``surge_duration`` seconds.
"""

from __future__ import annotations

from metroflow.config import IncidentConfig
from metroflow.train import TrainState


class IncidentManager:
    def __init__(self, sim, cfg: IncidentConfig):
        self.sim = sim
        self.cfg = cfg

    def run(self, env):
        cfg = self.cfg
        if not cfg.enabled:
            return
        while True:
            yield env.timeout(cfg.check_interval)
            t = env.now
            rng = self.sim.rng_incident
            if rng.random() < cfg.breakdown_prob:
                self._breakdown(t)
            if rng.random() < cfg.signal_prob:
                self._signal_failure(t)
            if rng.random() < cfg.dwell_event_prob:
                self._dwell_event(t)
            if rng.random() < cfg.surge_prob:
                self._surge(t)

    # -- individual incidents ------------------------------------------------ #
    def _breakdown(self, t: float) -> None:
        candidates = [tr for tr in self.sim.trains if tr.state == TrainState.IN_SERVICE]
        # Keep at least one train running.
        if len(candidates) <= 1:
            return
        idx = int(self.sim.rng_incident.integers(0, len(candidates)))
        train = candidates[idx]
        train.state = TrainState.BROKEN
        train.available_at = t + self.cfg.breakdown_repair
        self._log(t, "breakdown", {"train": train.id, "at_station": train.position})

    def _signal_failure(self, t: float) -> None:
        n_seg = self.sim.line.n_stations - 1
        seg = int(self.sim.rng_incident.integers(0, n_seg))
        until = t + self.cfg.signal_duration
        self.sim.segment_slowdown[seg] = (self.cfg.signal_slowdown, until)
        self._log(
            t,
            "signal_failure",
            {"segment": seg, "slowdown": self.cfg.signal_slowdown, "until": until},
        )

    def _dwell_event(self, t: float) -> None:
        station = int(self.sim.rng_incident.integers(0, self.sim.line.n_stations))
        until = t + 300.0
        self.sim.dwell_penalty[station] = (self.cfg.dwell_event_extra, until)
        self._log(
            t,
            "dwell_event",
            {"station": station, "extra_s": self.cfg.dwell_event_extra},
        )

    def _surge(self, t: float) -> None:
        station = int(self.sim.rng_incident.integers(0, self.sim.line.n_stations))
        until = t + self.cfg.surge_duration
        self.sim.demand.add_surge(station, until, self.cfg.surge_multiplier)
        self._log(
            t,
            "surge",
            {
                "station": station,
                "multiplier": self.cfg.surge_multiplier,
                "until": until,
            },
        )

    def _log(self, t: float, kind: str, details: dict) -> None:
        self.sim.metrics.record_incident({"t": round(float(t), 1), "type": kind, **details})

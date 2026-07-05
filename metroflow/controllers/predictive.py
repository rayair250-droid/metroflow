"""Predictive controller: forecast saturation and inject preemptively."""

from __future__ import annotations

from metroflow.controllers.base import Controller, InjectionCommand


class PredictiveController(Controller):
    """Short-horizon forecasting controller.

    For every ``(station, direction)`` platform it keeps an EWMA estimate of the
    queue growth rate (passengers/second) and combines it with the live arrival
    intensity from the demand model. It projects each queue ``horizon`` seconds
    ahead::

        q_hat = q_now + max(ewma_slope, arrival_rate) * horizon

    If any projected queue is expected to cross ``predictive_fraction`` of the
    saturation threshold, it injects a reserve train *upstream* of the forecast
    hotspot so the train sweeps up the building queue before it saturates -- i.e.
    it acts before the reactive controller would ever fire.
    """

    name = "predictive"

    #: How many stops upstream of the hotspot the reserve is inserted.
    UPSTREAM_OFFSET = 2

    def __init__(self, cfg):
        super().__init__(cfg)
        self._prev_len: dict[tuple[int, int], int] = {}
        self._ewma: dict[tuple[int, int], float] = {}
        self._last_t: float = 0.0

    def _update_slopes(self, sim, t: float) -> dict[tuple[int, int], float]:
        dt = max(t - self._last_t, 1e-9)
        alpha = self.cfg.ewma_alpha
        slopes: dict[tuple[int, int], float] = {}
        for station, direction, length in sim.iter_queues():
            key = (station, direction)
            prev = self._prev_len.get(key, length)
            inst = (length - prev) / dt
            ewma = self._ewma.get(key)
            ewma = inst if ewma is None else alpha * inst + (1 - alpha) * ewma
            self._ewma[key] = ewma
            self._prev_len[key] = length
            slopes[key] = ewma
        self._last_t = t
        return slopes

    def decide(self, sim, t: float) -> list[InjectionCommand]:
        slopes = self._update_slopes(sim, t)

        if sim.reserves_available() <= 0:
            return []
        if sim.time_since_last_injection() < self.cfg.min_injection_gap:
            return []

        trigger = self.cfg.predictive_fraction * self.cfg.queue_threshold
        horizon = self.cfg.horizon

        best = None
        best_hat = trigger
        for station, direction, length in sim.iter_queues():
            key = (station, direction)
            slope = max(slopes.get(key, 0.0), sim.arrival_rate(station, t))
            q_hat = length + slope * horizon
            if q_hat > best_hat:
                best_hat = q_hat
                best = (station, direction, length, slope)

        if best is None:
            return []

        station, direction, length, slope = best
        # Insert upstream of the forecast hotspot (bounded to the line).
        n = sim.line.n_stations
        insert = station - direction * self.UPSTREAM_OFFSET
        insert = max(0, min(n - 1, insert))
        return [
            InjectionCommand(
                station=insert,
                direction=direction,
                reason=(
                    f"predictive: q_hat={best_hat:.0f} over {int(horizon)}s at "
                    f"S{station:02d} dir={direction:+d} (slope={slope:.3f}/s)"
                ),
            )
        ]

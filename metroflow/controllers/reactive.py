"""Reactive controller: inject when a queue crosses the saturation threshold."""

from __future__ import annotations

from metroflow.controllers.base import Controller, InjectionCommand


class ReactiveController(Controller):
    """Threshold controller.

    Every poll it finds the most saturated ``(station, direction)`` platform. If
    its current queue is at or above ``queue_threshold`` it injects a reserve
    train at that station heading in the loaded direction, providing immediate
    relief. It acts only on what is happening *now* -- there is no forecasting,
    so it necessarily reacts after saturation has already begun.
    """

    name = "reactive"

    def decide(self, sim, t: float) -> list[InjectionCommand]:
        if sim.reserves_available() <= 0:
            return []
        if sim.time_since_last_injection() < self.cfg.min_injection_gap:
            return []

        worst = None
        worst_len = -1
        for station, direction, length in sim.iter_queues():
            if length > worst_len:
                worst_len = length
                worst = (station, direction)

        if worst is None or worst_len < self.cfg.queue_threshold:
            return []

        station, direction = worst
        return [
            InjectionCommand(
                station=station,
                direction=direction,
                reason=f"reactive: queue={worst_len} >= {self.cfg.queue_threshold}",
            )
        ]

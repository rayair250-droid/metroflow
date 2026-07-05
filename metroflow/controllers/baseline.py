"""Baseline controller: fixed schedule, never injects (control group)."""

from __future__ import annotations

from metroflow.controllers.base import Controller, InjectionCommand


class BaselineController(Controller):
    """Runs the initial fleet on a fixed headway and never injects reserves.

    This is the control group the other strategies are measured against.
    """

    name = "baseline"

    def decide(self, sim, t: float) -> list[InjectionCommand]:
        return []

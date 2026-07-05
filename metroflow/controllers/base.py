"""Controller interface and the injection command type."""

from __future__ import annotations

from dataclasses import dataclass

from metroflow.config import ControllerConfig


@dataclass
class InjectionCommand:
    """A request to inject one reserve train.

    Attributes
    ----------
    station:
        Where the reserve train enters revenue service.
    direction:
        The direction (``+1`` / ``-1``) it starts running.
    reason:
        Human-readable justification, stored in the metrics log.
    """

    station: int
    direction: int
    reason: str


class Controller:
    """Base class for dispatch strategies.

    A controller is polled every ``control_interval`` seconds. It inspects the
    live simulation state and returns zero or more :class:`InjectionCommand`
    objects. The simulation is responsible for enforcing depot-reserve limits and
    the minimum spacing between injections, so a controller may propose freely.
    """

    name = "base"

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg

    def decide(self, sim, t: float) -> list[InjectionCommand]:
        raise NotImplementedError

    # Convenience hook, overridable, invoked once before the run starts.
    def setup(self, sim) -> None:  # pragma: no cover - default no-op
        pass

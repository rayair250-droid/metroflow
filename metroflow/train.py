"""Train entity and state."""

from __future__ import annotations

from collections import Counter
from enum import Enum


class TrainState(Enum):
    RESERVE = "reserve"  # parked in the depot, available for injection
    IN_SERVICE = "in_service"  # running revenue service
    BROKEN = "broken"  # failed; running empty back to the depot
    RETURNING = "returning"  # repaired-in-transit / repositioning to depot


class Train:
    """A single train unit.

    ``onboard`` maps destination-station index -> passenger count, which makes
    alighting O(1) and keeps the current load exact.
    """

    def __init__(self, train_id: int, capacity: int, state: TrainState):
        self.id = train_id
        self.capacity = capacity
        self.state = state
        self.position: int = 0
        self.direction: int = 1
        self.onboard: Counter[int] = Counter()
        #: Simulation time at which a broken train becomes reusable.
        self.available_at: float = 0.0
        #: True once the train has been dispatched at least once (reserve stat).
        self.was_injected: bool = False
        #: Continuous-position tracking (physical coordinate in metres) used by
        #: the signalling model. While dwelling/holding, ``_mv_from == _mv_to``
        #: so :meth:`position_m` returns the platform coordinate.
        self._mv_from: float = 0.0
        self._mv_to: float = 0.0
        self._mv_t0: float = 0.0
        self._mv_t1: float = 0.0

    def set_stationary(self, coord_m: float, now: float) -> None:
        """Pin the train's tracked position to a platform coordinate."""
        self._mv_from = coord_m
        self._mv_to = coord_m
        self._mv_t0 = now
        self._mv_t1 = now

    def set_moving(self, from_m: float, to_m: float, t0: float, t1: float) -> None:
        """Start a linear move between two coordinates over ``[t0, t1]``."""
        self._mv_from = from_m
        self._mv_to = to_m
        self._mv_t0 = t0
        self._mv_t1 = t1

    def position_m(self, now: float) -> float:
        """Estimated physical coordinate (m) at time ``now`` (linear interp)."""
        if self._mv_t1 <= self._mv_t0 or now <= self._mv_t0:
            return self._mv_from
        if now >= self._mv_t1:
            return self._mv_to
        frac = (now - self._mv_t0) / (self._mv_t1 - self._mv_t0)
        return self._mv_from + frac * (self._mv_to - self._mv_from)

    @property
    def load(self) -> int:
        return sum(self.onboard.values())

    @property
    def free_capacity(self) -> int:
        return max(0, self.capacity - self.load)

    def alight(self, station: int) -> int:
        """Remove and count passengers whose destination is ``station``."""
        n = self.onboard.pop(station, 0)
        return n

    def board(self, destination: int) -> None:
        self.onboard[destination] += 1

    def clear(self) -> None:
        self.onboard.clear()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Train(id={self.id}, state={self.state.value}, pos={self.position}, "
            f"dir={self.direction:+d}, load={self.load}/{self.capacity})"
        )

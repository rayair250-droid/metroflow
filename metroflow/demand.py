"""Time-varying, origin/destination-weighted passenger demand.

Arrivals form an inhomogeneous Poisson process per station. The intensity is a
smooth baseline-plus-peaks profile in time, scaled by a per-station origin
weight. Each arriving passenger draws a destination from an attraction-weighted
distribution over the reachable stations (which fixes their travel direction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from metroflow.config import DemandConfig
from metroflow.errors import ConfigError


@dataclass(slots=True)
class Passenger:
    arrival: float
    origin: int
    dest: int
    #: Set True the first time this passenger is passed up by a full train.
    denied: bool = False


class DemandModel:
    def __init__(self, cfg: DemandConfig, n_stations: int):
        self.cfg = cfg
        self.n = n_stations

        prof_origin, prof_attract = self._profile_shapes(cfg.profile, n_stations)

        # Origin weights: explicit list > named profile > default bulge.
        if cfg.origin_weights is not None:
            if len(cfg.origin_weights) != n_stations:
                raise ConfigError(
                    f"origin_weights length ({len(cfg.origin_weights)}) must equal "
                    f"n_stations ({n_stations})"
                )
            self.origin_w = np.asarray(cfg.origin_weights, dtype=float)
        else:
            self.origin_w = prof_origin

        # Attraction weights: explicit list > named profile > default bulge.
        if cfg.attraction_weights is not None:
            if len(cfg.attraction_weights) != n_stations:
                raise ConfigError(
                    f"attraction_weights length ({len(cfg.attraction_weights)}) must "
                    f"equal n_stations ({n_stations})"
                )
            self.attract = np.asarray(cfg.attraction_weights, dtype=float)
        else:
            self.attract = prof_attract

        # Transient surge state (station, end_time, multiplier); set by events.
        self._surges: list[tuple[int, float, float]] = []

    # -- shape helpers ------------------------------------------------------- #
    @staticmethod
    def _bulge(n: int, sharpness: float) -> np.ndarray:
        mid = (n - 1) / 2.0
        xs = np.arange(n, dtype=float)
        # Cosine bulge peaking at the centre; sharpness controls contrast.
        base = 0.5 * (1 + np.cos((xs - mid) / max(mid, 1e-9) * math.pi))
        return 1.0 + sharpness * base

    @staticmethod
    def _ushape(n: int, sharpness: float) -> np.ndarray:
        mid = (n - 1) / 2.0
        xs = np.arange(n, dtype=float)
        # Inverse of the bulge: 0 at the centre, 1 at both termini.
        base = 0.5 * (1 - np.cos((xs - mid) / max(mid, 1e-9) * math.pi))
        return 1.0 + sharpness * base

    @classmethod
    def _profile_shapes(cls, profile: str | None, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(origin, attraction) weight shapes for a named demand profile.

        Plausible *shapes* only — none of these is measured ridership:

        - ``metro_commuter`` (and unset, the historical default): mild origin
          bulge, strong central attraction — trips converge on the city core.
        - ``rer_bidirectional``: origins concentrated in the outer suburbs on
          BOTH sides, attraction at the central trunk — the classic morning
          flood arriving at the centre from both directions at once.
        - ``intercity_endpoint``: origins and attraction both at the termini —
          most passengers ride end to end (Intercités / TER pattern), so
          intermediate stations matter less.
        """
        if profile is None or profile == "metro_commuter":
            return cls._bulge(n, sharpness=0.6), cls._bulge(n, sharpness=1.4)
        if profile == "rer_bidirectional":
            return cls._ushape(n, sharpness=1.6), cls._bulge(n, sharpness=2.2)
        if profile == "intercity_endpoint":
            return cls._ushape(n, sharpness=2.5), cls._ushape(n, sharpness=2.5)
        raise ConfigError(
            f"unknown demand.profile {profile!r} "
            "(use metro_commuter, rer_bidirectional or intercity_endpoint)"
        )

    def temporal_profile(self, t: float) -> float:
        """Dimensionless demand multiplier in time (baseline + gaussian peaks)."""
        val = self.cfg.baseline_frac
        for peak in self.cfg.peaks:
            c = peak["center"]
            w = peak["width"]
            a = peak["amplitude"]
            val += a * math.exp(-((t - c) ** 2) / (2.0 * w * w))
        return val

    # -- surge handling (driven by events.py) -------------------------------- #
    def add_surge(self, station: int, until: float, multiplier: float) -> None:
        self._surges.append((station, until, multiplier))

    def _surge_factor(self, station: int, t: float) -> float:
        factor = 1.0
        for st, until, mult in self._surges:
            if st == station and t < until:
                factor *= mult
        return factor

    def rate(self, station: int, t: float) -> float:
        """Arrival intensity (passengers/second) at ``station`` and time ``t``."""
        base = self.cfg.arrival_scale * self.origin_w[station]
        return base * self.temporal_profile(t) * self._surge_factor(station, t)

    def direction_split(self, station: int) -> tuple[float, float]:
        """Fraction of demand at ``station`` heading (up, down).

        Derived from the destination-attraction weights: trips whose destination
        lies above the station travel up, those below travel down. Used by the
        MILP controller to forecast per-platform (directional) demand.
        """
        w = self.attract.copy()
        w[station] = 0.0
        up = float(w[station + 1 :].sum())
        down = float(w[:station].sum())
        total = up + down
        if total <= 0:
            return 0.5, 0.5
        return up / total, down / total

    # -- arrival generation -------------------------------------------------- #
    def generate_bin(
        self, station: int, t0: float, t1: float, rng: np.random.Generator
    ) -> list[Passenger]:
        """Generate arrivals for ``station`` in the interval ``[t0, t1)``."""
        mid = 0.5 * (t0 + t1)
        expected = self.rate(station, mid) * (t1 - t0)
        if expected <= 0:
            return []
        count = int(rng.poisson(expected))
        if count == 0:
            return []
        # Destination distribution excludes the origin.
        w = self.attract.copy()
        w[station] = 0.0
        total = w.sum()
        if total <= 0:
            return []
        probs = w / total
        dests = rng.choice(self.n, size=count, p=probs)
        # Spread arrival instants uniformly within the bin (deterministic order).
        times = np.sort(rng.uniform(t0, t1, size=count))
        return [Passenger(float(times[i]), station, int(dests[i])) for i in range(count)]

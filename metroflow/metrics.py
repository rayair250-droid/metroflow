"""Metric collection and summarisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepartureRecord:
    t: float
    station: int
    direction: int
    load: int
    boarded: int
    alighted: int
    denied: int


@dataclass
class QueueSample:
    t: float
    station: int
    direction: int
    length: int


@dataclass
class InjectionRecord:
    t: float
    station: int
    direction: int
    reason: str


class MetricsCollector:
    """Accumulates raw events and computes summary statistics."""

    def __init__(self, n_stations: int):
        self.n_stations = n_stations
        self.denied_boardings: int = 0
        self.passengers_generated: int = 0
        self.passengers_boarded: int = 0
        self.wait_times: list[float] = []
        #: (boarding_time, wait) pairs, for windowed Little's-Law validation.
        self.boardings: list[tuple[float, float]] = []
        self.departures: list[DepartureRecord] = []
        self.queue_samples: list[QueueSample] = []
        self.injections: list[InjectionRecord] = []
        self.reserves_used: int = 0
        self.incidents: list[dict] = []
        # departure timestamps per (station, direction) for headway analysis
        self._dep_times: dict[tuple[int, int], list[float]] = {}
        # signalling holds (safe-separation enforcement)
        self.signal_holds: int = 0
        self.signal_hold_time_s: float = 0.0
        self.forced_holds: int = 0
        # Optional per-sample animation frames (only populated when the engine's
        # frame recording is enabled; empty and cost-free for normal runs).
        self.frames: list[dict] = []

    # -- recording ----------------------------------------------------------- #
    def record_generated(self, n: int) -> None:
        self.passengers_generated += n

    def record_boarding(self, wait: float, t: float = 0.0) -> None:
        self.passengers_boarded += 1
        self.wait_times.append(wait)
        self.boardings.append((t, wait))

    def record_denied(self, n: int) -> None:
        self.denied_boardings += n

    def record_departure(self, rec: DepartureRecord) -> None:
        self.departures.append(rec)
        self._dep_times.setdefault((rec.station, rec.direction), []).append(rec.t)

    def record_queue(self, sample: QueueSample) -> None:
        self.queue_samples.append(sample)

    def record_injection(self, rec: InjectionRecord) -> None:
        self.injections.append(rec)
        self.reserves_used += 1

    def record_incident(self, incident: dict) -> None:
        self.incidents.append(incident)

    def record_hold(self, seconds: float, forced: bool = False) -> None:
        self.signal_holds += 1
        self.signal_hold_time_s += seconds
        if forced:
            self.forced_holds += 1

    def record_frame(self, t: float, trains: list, queues: dict) -> None:
        """Store one animation frame: time, in-service train snapshots and the
        per-station total queue length. ``trains`` is a list of
        ``(coord_m, direction, injected)`` tuples."""
        self.frames.append({"t": t, "trains": trains, "queues": queues})

    # -- derived statistics -------------------------------------------------- #
    def headway_stats(self) -> tuple[float, float, float]:
        """Return (mean, std, coefficient-of-variation) of headways line-wide."""
        headways: list[float] = []
        for times in self._dep_times.values():
            ts = sorted(times)
            headways.extend(np.diff(ts).tolist())
        if not headways:
            return 0.0, 0.0, 0.0
        arr = np.asarray(headways, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        cv = std / mean if mean > 0 else 0.0
        return mean, std, cv

    def headway_cv_by_station(self, direction: int) -> dict[int, float]:
        """Coefficient of variation of departure headways at each station.

        Used by the bunching-reproduction validation check: without control the
        CV should tend to grow along the running direction.
        """
        out: dict[int, float] = {}
        for (station, d), times in self._dep_times.items():
            if d != direction:
                continue
            ts = sorted(times)
            if len(ts) < 3:
                continue
            hw = np.diff(ts)
            mean = float(hw.mean())
            if mean <= 0:
                continue
            out[station] = float(hw.std()) / mean
        return out

    def regularity(self, target_headway: float, tolerance: float) -> float:
        """Fraction of headways within +/- ``tolerance`` of ``target_headway``.

        This is the high-frequency counterpart to punctuality: on a turn-up-and-go
        metro passengers do not consult a timetable, so service quality is the
        evenness of the gaps, not adherence to a scheduled minute.
        """
        headways: list[float] = []
        for times in self._dep_times.values():
            ts = sorted(times)
            headways.extend(np.diff(ts).tolist())
        if not headways or target_headway <= 0:
            return 0.0
        arr = np.asarray(headways, dtype=float)
        lo = (1.0 - tolerance) * target_headway
        hi = (1.0 + tolerance) * target_headway
        within = np.count_nonzero((arr >= lo) & (arr <= hi))
        return float(within) / len(arr)

    def capacity_utilization(self, capacity: int) -> float:
        """Mean train load as a fraction of capacity across all departures."""
        if not self.departures or capacity <= 0:
            return 0.0
        loads = np.asarray([r.load for r in self.departures], dtype=float)
        return float(loads.mean()) / float(capacity)

    def queue_stats(self) -> tuple[float, float]:
        """Return (max, mean) platform queue length across all samples."""
        if not self.queue_samples:
            return 0.0, 0.0
        lengths = np.asarray([s.length for s in self.queue_samples], dtype=float)
        return float(lengths.max()), float(lengths.mean())

    def wait_stats(self) -> tuple[float, float, float]:
        """Return (mean, p90, max) boarding wait time in seconds."""
        if not self.wait_times:
            return 0.0, 0.0, 0.0
        arr = np.asarray(self.wait_times, dtype=float)
        return float(arr.mean()), float(np.percentile(arr, 90)), float(arr.max())

    def summary(
        self,
        controller: str,
        seed: int,
        scenario: str,
        target_headway: float | None = None,
        regularity_tolerance: float = 0.5,
        train_capacity: int | None = None,
    ) -> dict:
        hw_mean, hw_std, hw_cv = self.headway_stats()
        q_max, q_mean = self.queue_stats()
        w_mean, w_p90, w_max = self.wait_stats()

        # Operator KPIs. If no timetable target headway is supplied we fall back
        # to the observed mean headway as the reference (self-normalising).
        target = target_headway if target_headway else hw_mean
        expected_wait = target / 2.0 if target > 0 else 0.0
        # Excess Journey Time proxy: platform wait beyond the value a perfectly
        # even service would give (half the headway). Modelled on the "excess"
        # family of high-frequency KPIs used by metro operators (e.g. TfL's
        # Excess Journey Time). This is a wait-only proxy, not the full network
        # EJT, and is labelled as such.
        if self.wait_times:
            waits = np.asarray(self.wait_times, dtype=float)
            excess = np.clip(waits - expected_wait, 0.0, None)
            ejt_mean = float(excess.mean())
            excess_total_s = float(excess.sum())
        else:
            ejt_mean = 0.0
            excess_total_s = 0.0
        # Lost Customer Hours proxy: excess wait of served passengers plus a
        # one-headway penalty per denied boarding, expressed in customer-hours.
        lch = (excess_total_s + self.denied_boardings * max(target, 0.0)) / 3600.0
        reg = self.regularity(target, regularity_tolerance)
        util = (
            self.capacity_utilization(train_capacity)
            if train_capacity
            else self.capacity_utilization(1)
        )

        return {
            "scenario": scenario,
            "controller": controller,
            "seed": seed,
            "total_denied_boardings": int(self.denied_boardings),
            "passengers_generated": int(self.passengers_generated),
            "passengers_boarded": int(self.passengers_boarded),
            "mean_wait_s": round(w_mean, 2),
            "p90_wait_s": round(w_p90, 2),
            "max_wait_s": round(w_max, 2),
            "max_queue": round(q_max, 2),
            "mean_queue": round(q_mean, 2),
            "headway_mean_s": round(hw_mean, 2),
            "headway_std_s": round(hw_std, 2),
            "bunching_index": round(hw_cv, 4),
            "reserves_used": int(self.reserves_used),
            "incident_count": int(len(self.incidents)),
            # -- operator KPIs (Axis 4) --------------------------------------- #
            "ejt_proxy_s": round(ejt_mean, 2),
            "lost_customer_hours": round(lch, 2),
            "regularity_pct": round(100.0 * reg, 2),
            "capacity_utilization": round(util, 4),
            # -- signalling (Axis 1) ------------------------------------------ #
            "signal_holds": int(self.signal_holds),
            "signal_hold_time_s": round(self.signal_hold_time_s, 1),
        }

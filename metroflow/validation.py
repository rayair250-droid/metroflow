"""Model-validation checks: Little's Law, bunching, and sensitivity sweeps.

These are the checks a simulation engineer runs to argue a model is *right*, not
merely that it runs: a conservation identity (Little's Law), reproduction of a
known emergent phenomenon (bus/train bunching), and a controlled parameter
sweep. See ``docs/VALIDATION.md`` for the write-up and figures.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from metroflow.config import SimConfig
from metroflow.line import UP
from metroflow.simulation import Simulation, run_simulation


# --------------------------------------------------------------------------- #
# Little's Law
# --------------------------------------------------------------------------- #
@dataclass
class LittlesLawResult:
    L: float  # time-average number of passengers waiting (platform queues)
    lam: float  # effective arrival/throughput rate (pax/s)
    W: float  # mean waiting time (s)
    L_predicted: float  # lam * W
    rel_error: float
    denial_rate: float  # share of demand denied (must be ~0 for a clean check)


def stable_config() -> SimConfig:
    """A stationary, lightly-loaded, incident-free line for Little's Law.

    Flat demand (no rush peak) makes arrivals time-homogeneous, and generous
    frequency/capacity keeps queues bounded with essentially no denied
    boardings, which is the regime in which ``L = lambda * W`` should hold
    tightly.
    """
    cfg = SimConfig(name="stable")
    cfg.horizon = 14400.0
    cfg.n_initial_trains = 8
    cfg.depot_reserve = 0
    cfg.target_headway = 180.0
    cfg.train_capacity = 600
    cfg.line.n_stations = 8
    cfg.line.segment_time = 70.0
    cfg.line.dwell_per_pax = 0.3
    cfg.demand.arrival_scale = 0.012
    cfg.demand.baseline_frac = 1.0
    cfg.demand.peaks = []  # flat -> stationary arrivals
    cfg.incidents.enabled = False
    return cfg


def littles_law_check(sim: Simulation, warmup_frac: float = 0.25) -> LittlesLawResult:
    """Check ``L ~= lambda * W`` on the platform queues of a finished run.

    All three quantities are measured over the *same* post-warmup window so the
    identity is tested at steady state (no fill-up transient contamination):
    ``lambda`` is the boarding throughput in the window, ``W`` the mean wait of
    those boardings, and ``L`` the time-average total platform occupancy.
    """
    horizon = sim.cfg.horizon
    t0 = horizon * warmup_frac
    t1 = horizon

    # L: average total number waiting across all platforms, post-warmup.
    by_time: dict[float, float] = {}
    for smp in sim.metrics.queue_samples:
        if t0 <= smp.t <= t1:
            by_time[smp.t] = by_time.get(smp.t, 0.0) + smp.length
    L = float(np.mean(list(by_time.values()))) if by_time else 0.0

    # lambda and W: throughput and mean wait of boardings within the window.
    window = [(bt, w) for (bt, w) in sim.metrics.boardings if t0 <= bt <= t1]
    span = t1 - t0
    lam = len(window) / span if span > 0 else 0.0
    W = float(np.mean([w for _, w in window])) if window else 0.0

    L_pred = lam * W
    rel_error = abs(L - L_pred) / L_pred if L_pred > 0 else float("inf")
    generated = max(1, sim.metrics.passengers_generated)
    denial_rate = sim.metrics.denied_boardings / generated
    return LittlesLawResult(L, lam, W, L_pred, rel_error, denial_rate)


# --------------------------------------------------------------------------- #
# Bunching reproduction
# --------------------------------------------------------------------------- #
@dataclass
class BunchingResult:
    early_cv: float  # headway CV in the first third of the run
    late_cv: float  # headway CV in the last third of the run
    grew: bool  # True if headway variability grew over time (bunching)
    growth_ratio: float  # late_cv / early_cv


def bunching_config() -> SimConfig:
    """A moderately loaded, incident-free line prone to bunching.

    Trains are dispatched almost perfectly evenly at ``t=0`` (headway CV near
    zero). Load-dependent dwell is the positive-feedback mechanism: a slightly
    late train collects a bigger crowd, dwells longer, falls further behind and
    the gap behind it shrinks. With no control, this small perturbation grows
    until the service is severely bunched -- so headway variability increases
    strongly from the start of the run to the end. Flat demand isolates the
    instability from the demand profile.
    """
    cfg = SimConfig(name="bunching")
    cfg.horizon = 10800.0
    # 8 trains at a 351 s headway fill the ~2808 s nominal cycle almost exactly,
    # so the fleet starts evenly spaced around the loop (near-regular headways).
    cfg.n_initial_trains = 8
    cfg.depot_reserve = 0
    cfg.target_headway = 351.0
    cfg.train_capacity = 240
    cfg.line.n_stations = 14
    cfg.line.dwell_per_pax = 0.7  # strong load->dwell coupling
    cfg.line.runtime_noise = 0.06
    cfg.line.dwell_noise = 2.0
    cfg.demand.arrival_scale = 0.045
    cfg.demand.baseline_frac = 1.0
    cfg.demand.peaks = []  # flat -> the only driver of divergence is instability
    cfg.incidents.enabled = False
    return cfg


def _headways_in_window(sim: Simulation, direction: int, t_lo: float, t_hi: float):
    hws: list[float] = []
    for (_station, d), times in sim.metrics._dep_times.items():
        if d != direction:
            continue
        ts = sorted(times)
        for i in range(1, len(ts)):
            if t_lo <= ts[i] < t_hi:
                hws.append(ts[i] - ts[i - 1])
    return hws


def _cv(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=float)
    m = arr.mean()
    return float(arr.std() / m) if m > 0 else 0.0


def bunching_reproduction(sim: Simulation, direction: int = UP) -> BunchingResult:
    """Compare headway variability early vs late in the run.

    Bunching is the temporal amplification of an initial perturbation: trains
    start evenly spaced (low headway CV) and, absent control, drift into clumps
    (high headway CV). We therefore pool one direction's headways into the first
    and last third of the horizon and compare their coefficients of variation.
    """
    H = sim.cfg.horizon
    early = _headways_in_window(sim, direction, 0.0, H / 3.0)
    late = _headways_in_window(sim, direction, 2.0 * H / 3.0, H + 1.0)
    early_cv = _cv(early)
    late_cv = _cv(late)
    ratio = late_cv / early_cv if early_cv > 0 else float("inf")
    return BunchingResult(early_cv, late_cv, late_cv > early_cv, ratio)


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #
def _set_by_path(cfg: SimConfig, path: str, value) -> None:
    """Set a dotted attribute path (e.g. ``demand.arrival_scale``) on a config."""
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


@dataclass
class SensitivityResult:
    param: str
    values: list[float]
    metric: str
    controller: str
    outcomes: list[float]


def sensitivity_sweep(
    base_cfg: SimConfig,
    param: str,
    values: list[float],
    controller: str = "predictive",
    seed: int = 42,
    metric: str = "total_denied_boardings",
) -> SensitivityResult:
    """Sweep one parameter and record a metric for each value (fixed seed)."""
    outcomes: list[float] = []
    for v in values:
        cfg = copy.deepcopy(base_cfg)
        _set_by_path(cfg, param, v)
        summary = run_simulation(cfg, controller, seed).summary()
        outcomes.append(float(summary[metric]))
    return SensitivityResult(param, list(values), metric, controller, outcomes)

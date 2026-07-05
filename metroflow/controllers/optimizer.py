"""MILP reserve-train dispatcher solved with OR-Tools CP-SAT.

Unlike the reactive/predictive heuristics, this controller poses reserve-train
dispatch as a small mixed-integer program and solves it on a rolling horizon
(model-predictive control): every poll it re-forecasts demand, solves for the
best injection plan over the next ``H`` steps, and *executes only the first
step*. Re-solving each poll lets it react to the evolving line while keeping each
solve tiny and bounded.

Formulation
-----------
Sets
    ``t in 0..K-1``      discrete horizon steps of length ``Delta`` seconds.
    ``c in C``           candidate insertions ``(station j, direction d)``.
    ``p in P``           platforms ``(station s, direction d)`` with a forecast
                         demand deficit (pruned; platforms already served are
                         dropped to keep the model small).

Decision variables
    ``y[t, c] in {0,1}``          inject a reserve at step ``t``, candidate ``c``.
    ``a[t, c, p] >= 0`` (int)     passenger relief that injection ``(t, c)``
                                  allocates to platform ``p``.
    ``short[t, c] in {0,1}``      (optional) mark the injection as a short-turn
                                  (bounded turn-back) rather than a full run.
    ``denied[p] >= 0`` (int)      residual denied boardings at platform ``p``.

Parameters (forecast surrogate)
    ``deficit_p``  backlog + forecast arrivals - baseline scheduled capacity.
    ``cover[c][p]`` 1 if a train injected at ``c`` reaches ``p`` within horizon.
    ``cap``        per-train relief budget (train capacity).

Constraints
    (1) ``denied_p >= deficit_p - sum_{t,c} a[t,c,p]``          demand balance.
    (2) ``a[t,c,p] <= cap * y[t,c]`` and ``a <= cap * cover``   link/coverage.
    (3) ``sum_p a[t,c,p] <= cap``                               one train's budget.
    (4) ``sum_{t,c} y[t,c] <= reserves_available``              depot reserve.
    (5) ``sum_c y[t,c] <= 1``                                   one injection/step.
    (6) rolling-window ``sum y <= 1`` over ``min_injection_gap`` steps   spacing.
    (7) ``in_service + sum y <= max_trains``   headway/turnaround feasibility,
        where ``max_trains = floor(cycle_time / effective_min_headway)`` and the
        effective minimum headway is the larger of the configured minimum and
        the moving-block headway implied by the signalling model (Axis 1).

Objective (minimise)
    ``w_denied * sum_p denied_p``
    ``+ w_injection * sum_{t,c} y[t,c]``                 (discourage churn)
    ``+ w_headway * sum_{t,c} reg_cost[c] * y[t,c]``     (protect regularity:
        injecting into an already-even gap is penalised; injecting into a
        stretched gap is cheap).

The horizon deficit forecast is a deliberately simple linear surrogate, not a
queueing-exact prediction; it is documented as such. If the solver returns no
usable solution within ``opt_max_solve_seconds`` (timeout / infeasible / build
error) the controller falls back to the predictive heuristic, so it can never
hang or crash the run.
"""

from __future__ import annotations

import math

from ortools.sat.python import cp_model

from metroflow.controllers.base import Controller, InjectionCommand
from metroflow.controllers.predictive import PredictiveController
from metroflow.line import DOWN, UP


class OptimizerController(Controller):
    """CP-SAT / MILP rolling-horizon reserve-train dispatcher."""

    name = "optimizer"

    def __init__(self, cfg):
        super().__init__(cfg)
        # Heuristic fallback used on solver timeout / infeasibility.
        self._fallback = PredictiveController(cfg)
        #: Diagnostics from the most recent solve (exposed for tests/inspection).
        self.last_status: str = "none"
        self.last_objective: float = 0.0

    def setup(self, sim) -> None:
        self._fallback.setup(sim)

    # ------------------------------------------------------------------ #
    # Forecast surrogate
    # ------------------------------------------------------------------ #
    def _recent_free_capacity(self, sim, station, direction, window=5) -> float:
        """Mean free capacity of the last few trains to depart this platform.

        This is the key signal: at a saturated hotspot trains leave nearly full,
        so their *free* capacity is small even though the fleet is frequent --
        which an average-capacity estimate would miss. Falls back to a configured
        fraction of capacity when no departures have been observed yet.
        """
        cap = sim.cfg.train_capacity
        loads = []
        for rec in reversed(sim.metrics.departures):
            if rec.station == station and rec.direction == direction:
                loads.append(rec.load)
                if len(loads) >= window:
                    break
        if not loads:
            return cap * self.cfg.opt_free_fraction
        # Worst-case: the usable margin at a hotspot is set by the fullest recent
        # trains, so use the maximum observed load (least free capacity).
        peak_load = max(loads)
        return max(0.0, cap - peak_load)

    def _forecast(self, sim, t: float):
        """Return per-platform forecast deficit and current queue backlog.

        ``deficit_p = backlog_p + forecast_arrivals_p - baseline_capacity_p``,
        where the baseline capacity is the *free* passenger relief the scheduled
        fleet is expected to deliver over the horizon -- estimated from recently
        observed train loads at that platform, so localized saturation (full
        trains at a hotspot) produces a real deficit.
        """
        cfg = self.cfg
        horizon_s = cfg.opt_horizon_steps * cfg.opt_step_seconds
        target_hw = max(sim.cfg.target_headway, 1.0)
        scheduled_passings = horizon_s / target_hw

        backlog: dict[tuple[int, int], int] = {}
        for station, direction, length in sim.iter_queues():
            backlog[(station, direction)] = length

        deficit: dict[tuple[int, int], float] = {}
        for (station, direction), q in backlog.items():
            up_frac, down_frac = sim.demand.direction_split(station)
            frac = up_frac if direction == UP else down_frac
            # Integrate the (inhomogeneous) arrival rate across the horizon steps.
            arrivals = 0.0
            for k in range(cfg.opt_horizon_steps):
                tk = t + (k + 0.5) * cfg.opt_step_seconds
                arrivals += sim.arrival_rate(station, tk) * cfg.opt_step_seconds
            arrivals *= frac
            free = self._recent_free_capacity(sim, station, direction)
            base_capacity = free * scheduled_passings
            d = q + arrivals - base_capacity
            if d > 0:
                deficit[(station, direction)] = d
        return deficit, backlog

    def _max_trains(self, sim) -> int:
        line = sim.line
        cycle = line.cycle_time()
        # Effective minimum headway: the stricter of the configured minimum and
        # the moving-block headway implied by the signalling model at max speed.
        implied = sim.signalling.implied_headway(sim.signalling.max_speed_mps)
        eff = max(self.cfg.min_headway, implied)
        if eff <= 0 or not math.isfinite(eff):
            return sim.in_service_count() + sim.reserves_available()
        return max(1, int(cycle / eff))

    def _candidates(self, sim, deficit) -> list[tuple[int, int]]:
        """Insertion candidates: each deficit platform and a few stops upstream."""
        if self.cfg.opt_candidate_stations is not None:
            stations = list(self.cfg.opt_candidate_stations)
            fixed: list[tuple[int, int]] = []
            for j in stations:
                for d in (UP, DOWN):
                    fixed.append((j, d))
            return fixed
        n = sim.line.n_stations
        span = self._coverage_span(sim)
        cand: set[tuple[int, int]] = set()
        for s, d in deficit:
            for k in range(0, span + 1):
                j = s - d * k  # upstream of the hotspot in its own direction
                if 0 <= j <= n - 1:
                    cand.add((j, d))
        return sorted(cand)

    def _coverage_span(self, sim) -> int:
        """How many stations downstream an injected train reaches in horizon."""
        cfg = self.cfg
        horizon_s = cfg.opt_horizon_steps * cfg.opt_step_seconds
        seg_times = sim.line.segment_times
        mean_seg = sum(seg_times) / len(seg_times) if seg_times else 90.0
        return max(1, int(horizon_s / max(mean_seg, 1.0)))

    def _covers(self, sim, cand: tuple[int, int], plat: tuple[int, int]) -> bool:
        jc, dc = cand
        sp, dp = plat
        if dc != dp:
            return False
        span = self._coverage_span(sim)
        if dc == UP:
            return jc <= sp <= jc + span
        return jc - span <= sp <= jc

    def _reg_cost(self, sim, station: int, direction: int, target_hw: float) -> float:
        """Regularity penalty for injecting near ``station``.

        High when the local service is already even (injection would create
        bunching), low when the local gap is stretched (injection helps)."""
        times = sim.metrics._dep_times.get((station, direction), [])
        if len(times) < 2:
            return 0.0
        ts = sorted(times)
        last_gap = ts[-1] - ts[-2]
        # cost in [0, 1]: 1 if the last gap is already at/under target, 0 if wide.
        return max(0.0, min(1.0, (target_hw - last_gap) / target_hw))

    # ------------------------------------------------------------------ #
    # Decide
    # ------------------------------------------------------------------ #
    def decide(self, sim, t: float) -> list[InjectionCommand]:
        # Keep the fallback's internal EWMA state warm even when we optimise, so
        # a mid-run fallback is well-conditioned.
        self._fallback._update_slopes(sim, t)

        reserves = sim.reserves_available()
        if reserves <= 0:
            self.last_status = "no_reserves"
            return []
        if sim.time_since_last_injection() < self.cfg.min_injection_gap:
            self.last_status = "spacing"
            return []

        deficit, _ = self._forecast(sim, t)
        if not deficit:
            self.last_status = "no_deficit"
            return []

        try:
            cmds = self._solve(sim, t, deficit, reserves)
            return cmds
        except Exception:  # pragma: no cover - defensive; fall back on any error
            self.last_status = "solver_error"
            return self._fallback.decide(sim, t)

    def _solve(self, sim, t, deficit, reserves) -> list[InjectionCommand]:
        cfg = self.cfg
        K = cfg.opt_horizon_steps
        cap = int(round(sim.cfg.train_capacity))
        target_hw = max(sim.cfg.target_headway, 1.0)
        plats = list(deficit)
        cands = self._candidates(sim, deficit)
        if not cands:
            self.last_status = "no_candidates"
            return self._fallback.decide(sim, t)

        model = cp_model.CpModel()

        y: dict[tuple[int, int], cp_model.IntVar] = {}
        for k in range(K):
            for ci in range(len(cands)):
                y[(k, ci)] = model.new_bool_var(f"y_{k}_{ci}")

        # Allocation vars only where coverage exists (keeps the model sparse).
        a: dict[tuple[int, int, int], cp_model.IntVar] = {}
        cover_pairs: dict[int, list[tuple[int, int]]] = {pi: [] for pi in range(len(plats))}
        for k in range(K):
            for ci, c in enumerate(cands):
                for pi, p in enumerate(plats):
                    if self._covers(sim, c, p):
                        v = model.new_int_var(0, cap, f"a_{k}_{ci}_{pi}")
                        a[(k, ci, pi)] = v
                        model.add(v <= cap * y[(k, ci)])
                        cover_pairs[pi].append((k, ci))

        denied: dict[int, cp_model.IntVar] = {}
        for pi, p in enumerate(plats):
            dcap = int(math.ceil(deficit[p]))
            dv = model.new_int_var(0, dcap, f"denied_{pi}")
            denied[pi] = dv
            relief = [a[(k, ci, pi)] for (k, ci) in cover_pairs[pi]]
            # (1) demand balance: denied >= deficit - allocated relief.
            model.add(dv >= dcap - sum(relief))

        # (3) one train's relief budget.
        for k in range(K):
            for ci in range(len(cands)):
                budget = [a[(k, ci, pi)] for pi in range(len(plats)) if (k, ci, pi) in a]
                if budget:
                    model.add(sum(budget) <= cap)

        all_y = [y[(k, ci)] for k in range(K) for ci in range(len(cands))]
        # (4) depot reserve availability.
        model.add(sum(all_y) <= reserves)
        # (5) at most one injection per step.
        for k in range(K):
            model.add(sum(y[(k, ci)] for ci in range(len(cands))) <= 1)
        # (6) spacing: at most one injection within a min-gap window.
        w = max(1, int(math.ceil(self.cfg.min_injection_gap / cfg.opt_step_seconds)))
        for k in range(K):
            window = [y[(kk, ci)] for kk in range(k, min(K, k + w)) for ci in range(len(cands))]
            model.add(sum(window) <= 1)
        # (7) headway / turnaround feasibility bound on total trains.
        max_add = self._max_trains(sim) - sim.in_service_count()
        if max_add < reserves:
            model.add(sum(all_y) <= max(0, max_add))

        # Objective.
        SCALE = 1000  # integer weights for the regularity term
        obj_terms = []
        for pi in range(len(plats)):
            obj_terms.append(int(round(cfg.opt_w_denied * SCALE)) * denied[pi])
        for k in range(K):
            for ci, c in enumerate(cands):
                inj_w = cfg.opt_w_injection * SCALE
                reg_w = cfg.opt_w_headway * SCALE * self._reg_cost(sim, c[0], c[1], target_hw)
                obj_terms.append(int(round(inj_w + reg_w)) * y[(k, ci)])
        model.minimize(sum(obj_terms))

        solver = cp_model.CpSolver()
        # Determinism: a wall-clock limit (max_time_in_seconds) makes CP-SAT
        # non-reproducible, because where it stops depends on real elapsed time
        # (machine load) rather than work done. Use a *deterministic* time limit
        # plus a single worker and a fixed seed, so the same seed always yields
        # the same solution. A generous wall-clock cap stays only as a safety
        # backstop that should not bind in normal runs.
        solver.parameters.max_deterministic_time = float(cfg.opt_max_solve_seconds)
        solver.parameters.max_time_in_seconds = float(cfg.opt_max_solve_seconds) * 20.0
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 1
        status = solver.Solve(model)
        self.last_status = solver.StatusName(status)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._fallback.decide(sim, t)
        self.last_objective = solver.ObjectiveValue()

        # Execute only step 0 (rolling-horizon MPC).
        cmds: list[InjectionCommand] = []
        for ci, c in enumerate(cands):
            if solver.Value(y[(0, ci)]) == 1:
                j, d = c
                cmds.append(
                    InjectionCommand(
                        station=j,
                        direction=d,
                        reason=(
                            f"optimizer: MILP plan obj={self.last_objective:.0f} "
                            f"insert S{j:02d} dir={d:+d}"
                        ),
                    )
                )
        return cmds

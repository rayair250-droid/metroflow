"""The SimPy discrete-event engine that ties everything together."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import numpy as np
import simpy

from metroflow.config import SimConfig
from metroflow.controllers import Controller, make_controller
from metroflow.controllers.base import InjectionCommand
from metroflow.demand import DemandModel, Passenger
from metroflow.events import IncidentManager
from metroflow.line import DOWN, UP, Line
from metroflow.metrics import (
    DepartureRecord,
    InjectionRecord,
    MetricsCollector,
    QueueSample,
)
from metroflow.signalling import make_signalling_model
from metroflow.train import Train, TrainState


class Simulation:
    """A single run of one controller over one scenario and seed."""

    def __init__(self, cfg: SimConfig, controller: Controller, seed: int):
        self.cfg = cfg
        self.controller = controller
        self.seed = seed
        # Independent RNG streams so that, for a fixed seed, passenger arrivals
        # and the incident schedule are IDENTICAL across controllers -- the only
        # thing that changes between runs is the dispatch strategy. Operational
        # run-time/dwell jitter has its own stream.
        demand_ss, incident_ss, ops_ss = np.random.SeedSequence(seed).spawn(3)
        self.rng_demand = np.random.default_rng(demand_ss)
        self.rng_incident = np.random.default_rng(incident_ss)
        self.rng_ops = np.random.default_rng(ops_ss)

        self.env = simpy.Environment()
        self.line = Line(cfg.line)
        self.demand = DemandModel(cfg.demand, self.line.n_stations)
        self.metrics = MetricsCollector(self.line.n_stations)
        self.signalling = make_signalling_model(cfg.signalling)

        # Platform queues, keyed by (station, direction).
        self.queues: dict[tuple[int, int], deque[Passenger]] = {}
        for s in range(self.line.n_stations):
            if s < self.line.n_stations - 1:
                self.queues[(s, UP)] = deque()
            if s > 0:
                self.queues[(s, DOWN)] = deque()

        # Transient incident state.
        self.segment_slowdown: dict[int, tuple[float, float]] = {}
        self.dwell_penalty: dict[int, tuple[float, float]] = {}

        # Fleet: in-service trains plus a depot reserve pool.
        self.trains: list[Train] = []
        tid = 0
        for _ in range(cfg.n_initial_trains):
            self.trains.append(Train(tid, cfg.train_capacity, TrainState.IN_SERVICE))
            tid += 1
        for _ in range(cfg.depot_reserve):
            self.trains.append(Train(tid, cfg.train_capacity, TrainState.RESERVE))
            tid += 1

        self._last_injection_time: float = -1e18
        #: Last departure time per (station, direction), for headway holding.
        self._last_dep: dict[tuple[int, int], float] = {}
        #: When True the sampler also records per-frame train/queue snapshots for
        #: animation. Off by default so normal runs pay no cost (see animate.py).
        self._record_frames: bool = False

    # ------------------------------------------------------------------ #
    # State accessors used by controllers
    # ------------------------------------------------------------------ #
    @property
    def now(self) -> float:
        return self.env.now

    def iter_queues(self) -> Iterator[tuple[int, int, int]]:
        for (station, direction), q in self.queues.items():
            yield station, direction, len(q)

    def queue_length(self, station: int, direction: int) -> int:
        q = self.queues.get((station, direction))
        return len(q) if q is not None else 0

    def arrival_rate(self, station: int, t: float) -> float:
        return self.demand.rate(station, t)

    def reserves_available(self) -> int:
        return sum(
            1
            for tr in self.trains
            if tr.state == TrainState.RESERVE and tr.available_at <= self.env.now
        )

    def in_service_count(self) -> int:
        return sum(1 for tr in self.trains if tr.state == TrainState.IN_SERVICE)

    def time_since_last_injection(self) -> float:
        return self.env.now - self._last_injection_time

    # ------------------------------------------------------------------ #
    # Passenger service
    # ------------------------------------------------------------------ #
    def _serve(self, train: Train, station: int, direction: int) -> tuple[int, int, int]:
        now = self.env.now
        alighted = train.alight(station)
        q = self.queues.get((station, direction))
        boarded = 0
        if q is not None:
            while train.free_capacity > 0 and q:
                pax = q.popleft()
                train.board(pax.dest)
                self.metrics.record_boarding(now - pax.arrival, now)
                boarded += 1
        # Count each passenger's pass-up only once: a passenger who is left
        # behind by a full train is a "denied boarding". Newly-arrived (unflagged)
        # passengers sit at the back of the FIFO queue, so we scan from the back
        # until we reach one already flagged.
        denied = 0
        if q is not None and train.free_capacity == 0 and q:
            for pax in reversed(q):
                if pax.denied:
                    break
                pax.denied = True
                denied += 1
            if denied:
                self.metrics.record_denied(denied)
        return boarded, alighted, denied

    def _dwell_time(self, station: int, boarded: int, alighted: int) -> float:
        line = self.cfg.line
        base = line.dwell_base + line.dwell_per_pax * (boarded + alighted)
        noise = float(self.rng_ops.normal(0.0, line.dwell_noise))
        extra = 0.0
        pen = self.dwell_penalty.get(station)
        if pen is not None and self.env.now < pen[1]:
            extra = pen[0]
            del self.dwell_penalty[station]  # a door event delays one departure
        return max(5.0, base + noise + extra)

    def _travel_time(self, station: int, direction: int) -> float:
        seg = self.line.segment_index(station, direction)
        tt = self.line.travel_time(station, direction)
        factor = 1.0
        if seg is not None:
            sl = self.segment_slowdown.get(seg)
            if sl is not None and self.env.now < sl[1]:
                factor = sl[0]
        noise = max(0.5, float(self.rng_ops.normal(1.0, self.cfg.line.runtime_noise)))
        return max(5.0, tt * factor * noise)

    # ------------------------------------------------------------------ #
    # Signalling: safe-separation enforcement
    # ------------------------------------------------------------------ #
    def _gap_ahead(self, train: Train, direction: int, at_coord: float):
        """Distance (m) to the nearest in-service leader ahead, or ``None``.

        The gap is measured leader-rear to follower-front, so the leader's
        physical length is subtracted. Only IN_SERVICE trains travelling the same
        direction are considered leaders; a broken/returning train is treated as
        being cleared from the running line (a modelling simplification).
        """
        now = self.env.now
        best_gap = None
        for other in self.trains:
            if other is train or other.state != TrainState.IN_SERVICE:
                continue
            if other.direction != direction:
                continue
            opos = other.position_m(now)
            if direction > 0:
                raw = opos - at_coord
            else:
                raw = at_coord - opos
            if raw <= 0:
                continue  # not ahead
            gap = raw - self.cfg.signalling.train_length_m
            if best_gap is None or gap < best_gap:
                best_gap = gap
        return best_gap

    def _leaders_positions(self, train: Train, direction: int):
        now = self.env.now
        return [
            o.position_m(now)
            for o in self.trains
            if o is not train and o.state == TrainState.IN_SERVICE and o.direction == direction
        ]

    def _hold_for_separation(self, train: Train, station: int, direction: int):
        """Hold at the platform until it is safe to enter the next segment."""
        sig = self.signalling
        if not sig.enforce:
            return
        at_coord = self.line.station_coord[station]
        speed = self.line.nominal_speed(station, direction)
        hold_dt = self.cfg.signalling.hold_poll_s
        max_hold = self.cfg.signalling.max_hold_s
        held = 0.0
        while True:
            if sig.mode == "fixed_block":
                clear = sig.is_clear(
                    at_coord,
                    None,
                    speed,
                    leaders_pos_m=self._leaders_positions(train, direction),
                    direction=direction,
                )
            else:
                gap = self._gap_ahead(train, direction, at_coord)
                clear = sig.is_clear(at_coord, gap, speed, direction=direction)
            if clear or held >= max_hold:
                break
            yield self.env.timeout(hold_dt)
            held += hold_dt
        if held > 0:
            self.metrics.record_hold(held, forced=held >= max_hold)

    # ------------------------------------------------------------------ #
    # SimPy processes
    # ------------------------------------------------------------------ #
    def _train_process(self, train: Train, start_delay: float):
        env = self.env
        if start_delay > 0:
            yield env.timeout(start_delay)
        while True:
            if train.state != TrainState.IN_SERVICE:
                return
            s = train.position
            d = train.direction
            # Determine the *departure* direction first: at a terminus the train
            # reverses, and it must serve the queue it will actually carry away
            # (the origin queue in the new direction), not the empty arrival-side
            # queue. Serving in the departure direction keeps the terminus origin
            # platforms -- (0, UP) and (N-1, DOWN) -- from starving.
            if self.line.is_terminus(s, d):
                d = -d
                train.direction = d
            train.set_stationary(self.line.station_coord[s], env.now)
            boarded, alighted, denied = self._serve(train, s, d)
            dwell = self._dwell_time(s, boarded, alighted)
            yield env.timeout(dwell)

            # Even-headway holding: if this train would depart too soon after the
            # previous departure from this platform, hold it to restore the
            # target headway (forward-headway regularisation).
            if self.cfg.holding_control:
                last = self._last_dep.get((s, d))
                if last is not None:
                    gap = env.now - last
                    target = self.cfg.target_headway
                    if gap < target:
                        hold = min(target - gap, self.cfg.holding_max_s)
                        if hold > 0:
                            yield env.timeout(hold)
                            self.metrics.record_hold(hold)
            self._last_dep[(s, d)] = env.now

            # Departure event (used for load heatmap and headway analysis).
            self.metrics.record_departure(
                DepartureRecord(
                    t=env.now,
                    station=s,
                    direction=d,
                    load=train.load,
                    boarded=boarded,
                    alighted=alighted,
                    denied=denied,
                )
            )

            if train.state == TrainState.BROKEN:
                yield from self._return_to_depot(train)
                return

            nxt = self.line.next_station(s, d)
            # By construction ``d`` is the departure direction (reversed at a
            # terminus above), so the train always has a next station here.
            assert nxt is not None, "departure direction must point into the line"

            # Signalling: do not close on the leader below the safe separation.
            yield from self._hold_for_separation(train, s, d)

            tt = self._travel_time(s, d)
            train.set_moving(
                self.line.station_coord[s],
                self.line.station_coord[nxt],
                env.now,
                env.now + tt,
            )
            yield env.timeout(tt)
            if train.state == TrainState.BROKEN:
                yield from self._return_to_depot(train)
                return
            train.position = nxt
            train.set_stationary(self.line.station_coord[nxt], env.now)

    def _return_to_depot(self, train: Train):
        """A failed train runs empty back to the depot and re-enters the pool."""
        env = self.env
        train.clear()
        mean_seg = sum(self.line.segment_times) / len(self.line.segment_times)
        dist = abs(train.position - self.line.depot_station)
        yield env.timeout(max(30.0, dist * mean_seg))
        train.position = self.line.depot_station
        train.state = TrainState.RESERVE
        train.available_at = max(train.available_at, env.now)

    def _arrivals_process(self):
        env = self.env
        bin_s = self.cfg.demand.bin_seconds
        n = self.line.n_stations
        while True:
            yield env.timeout(bin_s)
            t1 = env.now
            t0 = t1 - bin_s
            for s in range(n):
                pax_list = self.demand.generate_bin(s, t0, t1, self.rng_demand)
                if not pax_list:
                    continue
                for p in pax_list:
                    direction = UP if p.dest > p.origin else DOWN
                    self.queues[(s, direction)].append(p)
                self.metrics.record_generated(len(pax_list))

    def _controller_process(self):
        env = self.env
        interval = self.cfg.control_interval
        self.controller.setup(self)
        while True:
            yield env.timeout(interval)
            cmds = self.controller.decide(self, env.now)
            for cmd in cmds:
                if self.time_since_last_injection() < self.cfg.controller.min_injection_gap:
                    break
                if self.reserves_available() <= 0:
                    break
                self._do_inject(cmd)

    def _sampler_process(self):
        env = self.env
        interval = self.cfg.sample_interval
        while True:
            yield env.timeout(interval)
            t = env.now
            for (station, direction), q in self.queues.items():
                self.metrics.record_queue(QueueSample(t, station, direction, len(q)))
            if self._record_frames:
                trains = [
                    (tr.position_m(t), tr.direction, tr.was_injected)
                    for tr in self.trains
                    if tr.state == TrainState.IN_SERVICE
                ]
                qsum: dict[int, int] = {}
                for (station, _direction), q in self.queues.items():
                    qsum[station] = qsum.get(station, 0) + len(q)
                self.metrics.record_frame(t, trains, qsum)

    # ------------------------------------------------------------------ #
    # Injection
    # ------------------------------------------------------------------ #
    def _do_inject(self, cmd: InjectionCommand) -> bool:
        train = next(
            (
                tr
                for tr in self.trains
                if tr.state == TrainState.RESERVE and tr.available_at <= self.env.now
            ),
            None,
        )
        if train is None:
            return False
        train.state = TrainState.IN_SERVICE
        train.position = cmd.station
        train.direction = cmd.direction
        train.clear()
        train.set_stationary(self.line.station_coord[cmd.station], self.env.now)
        train.was_injected = True
        self._last_injection_time = self.env.now
        self.metrics.record_injection(
            InjectionRecord(
                t=self.env.now,
                station=cmd.station,
                direction=cmd.direction,
                reason=cmd.reason,
            )
        )
        self.env.process(self._train_process(train, start_delay=0.0))
        return True

    def inject_now(self, station: int, direction: int, reason: str = "manual") -> bool:
        """Public helper (tests / scripting) to force an injection."""
        return self._do_inject(InjectionCommand(station, direction, reason))

    # ------------------------------------------------------------------ #
    # Initial fleet placement
    # ------------------------------------------------------------------ #
    def _loop_checkpoints(self):
        """Nominal serve-points around one full round trip.

        Returns a list of ``(cumulative_time, station, direction, leg_time)`` for
        each ``(station, departure_direction)`` the schedule visits in one loop,
        plus the total cycle time. Used to pre-position the initial fleet evenly
        around the line at ``t=0`` (as a real service starts), instead of
        launching every train sequentially from one terminus.
        """
        line = self.cfg.line
        n = self.line.n_stations
        points = []  # (station, direction)
        for s in range(0, n - 1):
            points.append((s, UP))  # (0,UP)..(n-2,UP)
        for s in range(n - 1, 0, -1):
            points.append((s, DOWN))  # (n-1,DOWN)..(1,DOWN)
        checkpoints = []
        t = 0.0
        for s, d in points:
            seg = self.line.travel_time(s, d)
            leg = line.dwell_base + seg
            checkpoints.append((t, s, d, leg))
            t += leg
        return checkpoints, t

    def _initial_placements(self, n_trains: int):
        """Return ``[(station, direction, start_delay)]`` for ``n_trains``."""
        checkpoints, cycle = self._loop_checkpoints()
        headway = self.cfg.target_headway
        placements = []
        for i in range(n_trains):
            offset = (i * headway) % cycle if cycle > 0 else 0.0
            # Find the checkpoint the train is currently within.
            chosen = checkpoints[0]
            for cp in checkpoints:
                if cp[0] <= offset:
                    chosen = cp
                else:
                    break
            _, s, d, _ = chosen
            residual = offset - chosen[0]
            placements.append((s, d, residual))
        return placements

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        env = self.env
        in_service = [t for t in self.trains if t.state == TrainState.IN_SERVICE]
        placements = self._initial_placements(len(in_service))
        for train, (s, d, delay) in zip(in_service, placements, strict=False):
            train.position = s
            train.direction = d
            train.set_stationary(self.line.station_coord[s], 0.0)
            env.process(self._train_process(train, start_delay=delay))

        env.process(self._arrivals_process())
        env.process(self._controller_process())
        env.process(self._sampler_process())
        env.process(IncidentManager(self, self.cfg.incidents).run(env))

        env.run(until=self.cfg.horizon)
        return self.summary()

    def summary(self) -> dict:
        """Metrics summary for this run, including operator KPIs."""
        return self.metrics.summary(
            self.controller.name,
            self.seed,
            self.cfg.name,
            target_headway=self.cfg.target_headway,
            regularity_tolerance=self.cfg.regularity_tolerance,
            train_capacity=self.cfg.train_capacity,
        )


def run_simulation(cfg: SimConfig, controller_name: str, seed: int | None = None) -> Simulation:
    """Build and run one simulation; returns the finished :class:`Simulation`."""
    use_seed = cfg.seed if seed is None else seed
    controller = make_controller(controller_name, cfg.controller)
    sim = Simulation(cfg, controller, use_seed)
    sim.run()
    return sim

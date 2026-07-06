"""Configuration dataclasses and YAML loader for MetroFlow.

All simulation parameters live here. Calling :func:`load_config` with ``None``
returns a fully-populated, runnable default configuration, so the simulator
works with zero external files. A YAML file may override any subset of fields.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml

from metroflow.errors import ConfigError, ScenarioFileError


# --------------------------------------------------------------------------- #
# Sub-configurations
# --------------------------------------------------------------------------- #
@dataclass
class LineConfig:
    """Physical line topology and rolling-stock timing."""

    n_stations: int = 12
    #: Nominal travel time (s) for each inter-station segment. If a scalar is
    #: given it is broadcast to every segment; a list gives a per-segment value.
    segment_time: float | list[float] = 90.0
    #: Physical length (m) of each inter-station segment (scalar broadcast or a
    #: per-segment list). Together with ``segment_time`` this fixes the nominal
    #: running speed used by the signalling model. Default 800 m gives a ~8.9 m/s
    #: (~32 km/h) average, typical of an urban metro including stops.
    segment_length: float | list[float] = 800.0
    #: Standard deviation of multiplicative run-time noise (fraction of nominal).
    runtime_noise: float = 0.08
    #: Base dwell time at a station (s), before boarding/alighting load.
    dwell_base: float = 18.0
    #: Extra dwell per boarding+alighting passenger (s).
    dwell_per_pax: float = 0.45
    #: Standard deviation of additive dwell noise (s).
    dwell_noise: float = 3.0
    #: Station index where the depot connects to the line.
    depot_station: int = 0
    #: Optional explicit list of station names.
    station_names: list[str] | None = None


@dataclass
class SignallingConfig:
    """Safe-separation (signalling) parameters.

    Selects the separation regime and its physical parameters. See
    :mod:`metroflow.signalling` for the models. All lengths in metres, speeds in
    m/s, deceleration in m/s^2, times in seconds.
    """

    #: "moving_block" (CBTC, speed-dependent separation) or "fixed_block".
    mode: str = "moving_block"
    #: Maximum running speed (m/s). 20 m/s ~= 72 km/h.
    max_speed_mps: float = 20.0
    #: Service (comfort) deceleration used for the braking curve (m/s^2).
    service_decel_mps2: float = 1.0
    #: Fixed safety margin added to the braking distance (m).
    safety_margin_m: float = 50.0
    #: Reaction / control-loop communication delay (s).
    reaction_time_s: float = 2.0
    #: Train length (m); adds to the required centre-to-rear separation.
    train_length_m: float = 90.0
    #: Fixed-block section length (m); only used when ``mode == 'fixed_block'``.
    block_length_m: float = 500.0
    #: Number of empty blocks required between trains (fixed-block only).
    n_clear_blocks: int = 1
    #: If False the separation constraint is disabled (reproduces v1 behaviour).
    enforce: bool = True
    #: Poll interval (s) used while a train holds for separation at a platform.
    hold_poll_s: float = 5.0
    #: Safety cap (s) on how long a single train may be held before it is
    #: released regardless (guards against pathological deadlocks; logged).
    max_hold_s: float = 600.0


#: Valid values for :attr:`DemandConfig.profile` (shapes live in demand.py).
DEMAND_PROFILES = ("metro_commuter", "rer_bidirectional", "intercity_endpoint")


@dataclass
class DemandConfig:
    """Time-varying, origin/destination-weighted passenger demand."""

    #: Peak arrival scale (passengers/second) shared across the line.
    arrival_scale: float = 0.06
    #: Named spatial profile shaping origin/attraction weights when they are
    #: not given explicitly: ``metro_commuter`` (central bulge — also the
    #: behaviour when unset), ``rer_bidirectional`` (origins in the outer
    #: suburbs on both sides, attraction at the centre) or
    #: ``intercity_endpoint`` (terminus-to-terminus travel). Shapes adapt to
    #: any station count, so GTFS-built lines can use them directly. Still
    #: synthetic: a profile is a plausible *shape*, not measured ridership.
    profile: str | None = None
    #: Per-station origin weight. If ``None`` the profile decides.
    origin_weights: list[float] | None = None
    #: Per-station destination attraction (central stations pull more trips).
    attraction_weights: list[float] | None = None
    #: Off-peak baseline as a fraction of peak.
    baseline_frac: float = 0.28
    #: Demand peaks as ``{center, width, amplitude}`` (seconds / multiplier).
    peaks: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"center": 7200.0, "width": 2400.0, "amplitude": 1.25},
        ]
    )
    #: Time step (s) used to discretise the inhomogeneous Poisson arrivals.
    bin_seconds: float = 15.0


@dataclass
class IncidentConfig:
    """Stochastic incident generator settings."""

    enabled: bool = True
    #: How often (s) the incident manager rolls the dice.
    check_interval: float = 120.0
    #: Per-check probability of a train breakdown.
    breakdown_prob: float = 0.010
    #: Seconds a broken train needs at the depot before it is reusable.
    breakdown_repair: float = 900.0
    #: Per-check probability of a segment signal failure / speed restriction.
    signal_prob: float = 0.020
    #: Multiplicative travel-time penalty applied during a signal failure.
    signal_slowdown: float = 1.8
    #: Duration window (s) of a signal failure.
    signal_duration: float = 600.0
    #: Per-check probability of an extended-dwell (door) event.
    dwell_event_prob: float = 0.020
    #: Extra dwell (s) added by a door event.
    dwell_event_extra: float = 60.0
    #: Per-check probability of a station demand surge.
    surge_prob: float = 0.010
    #: Multiplicative demand boost during a surge.
    surge_multiplier: float = 6.0
    #: Duration window (s) of a surge.
    surge_duration: float = 900.0


@dataclass
class ControllerConfig:
    """Controller thresholds and limits (shared by all strategies)."""

    #: Queue length that counts as saturation for reactive control.
    queue_threshold: int = 90
    #: Predictive horizon (s): how far ahead saturation is forecast.
    horizon: float = 300.0
    #: EWMA smoothing factor for the predictive queue-growth estimate.
    ewma_alpha: float = 0.35
    #: Predictive controllers act at this fraction of ``queue_threshold``.
    predictive_fraction: float = 0.6
    #: Minimum time (s) between two injections (protects headway regularity).
    min_injection_gap: float = 300.0
    #: Minimum headway (s) the controller must respect.
    min_headway: float = 120.0

    # -- MILP / CP-SAT optimiser (controllers/optimizer.py) ------------------ #
    #: Rolling-horizon length in discrete steps.
    opt_horizon_steps: int = 6
    #: Duration (s) of one optimiser time step.
    opt_step_seconds: float = 120.0
    #: Hard wall-clock bound (s) on the CP-SAT solve; on timeout the controller
    #: falls back to the predictive heuristic.
    opt_max_solve_seconds: float = 2.0
    #: Objective weight on predicted denied boardings.
    opt_w_denied: float = 1.0
    #: Objective weight (per injection) discouraging unnecessary dispatch.
    opt_w_injection: float = 5.0
    #: Objective weight on the headway-regularity penalty.
    opt_w_headway: float = 2.0
    #: Candidate insertion stations (None => every station is a candidate).
    opt_candidate_stations: list[int] | None = None
    #: Assumed free capacity fraction of a scheduled train when estimating the
    #: baseline served demand in the forecast surrogate.
    opt_free_fraction: float = 0.35


@dataclass
class SimConfig:
    """Top-level simulation configuration."""

    name: str = "default"
    horizon: float = 10800.0  # 3 hours
    seed: int = 42
    #: Trains in service at t=0.
    n_initial_trains: int = 6
    #: Reserve trains available in the depot for injection.
    depot_reserve: int = 4
    #: Target headway (s) used to stagger the initial fleet.
    target_headway: float = 300.0
    #: Train capacity (passengers).
    train_capacity: int = 200
    #: How often (s) the controller is polled.
    control_interval: float = 60.0
    #: How often (s) queue lengths are sampled for metrics/plots.
    sample_interval: float = 60.0
    #: Regularity tolerance (fraction of target headway): a gap counts as
    #: "regular" if it is within +/- this fraction of the target headway.
    regularity_tolerance: float = 0.5
    #: Even-headway holding control: if True, a train ready to depart a station
    #: earlier than the target headway behind the previous departure is held to
    #: restore the headway. This is the classic real-world anti-bunching
    #: regularisation (forward-headway holding at control points).
    holding_control: bool = False
    #: Cap (s) on the extra dwell a single holding action may add.
    holding_max_s: float = 90.0

    line: LineConfig = field(default_factory=LineConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    incidents: IncidentConfig = field(default_factory=IncidentConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    signalling: SignallingConfig = field(default_factory=SignallingConfig)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _check_scalar_type(cls_name: str, key: str, value: Any, ftype: Any) -> None:
    """Reject a YAML value whose type is incompatible with the field annotation.

    Only the simple leaf types MetroFlow uses (``int``/``float``/``bool``/``str``
    and unions/optionals of these, possibly with ``list``) are checked; anything
    else is left to the dataclass/engine. Catches the common mistake of writing
    a list or string where a number is expected, so it surfaces as a clean
    :class:`ConfigError` instead of a ``TypeError`` deep in validation.
    """
    if ftype is None or value is None:
        return
    args = typing.get_args(ftype)
    allowed = set(args) if args else {ftype}
    # If a list is permitted by the annotation, accept list values as-is.
    if list in allowed or any(typing.get_origin(a) is list for a in allowed):
        if isinstance(value, list):
            return
    # Numeric fields accept int/float (but not bool, which is a stricter case);
    # bool fields accept only bool; str fields accept only str.
    if float in allowed or int in allowed:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"Config field '{key}' for {cls_name} expects a number, "
                f"got {type(value).__name__}: {value!r}"
            )
    elif bool in allowed:
        if not isinstance(value, bool):
            raise ConfigError(
                f"Config field '{key}' for {cls_name} expects true/false, "
                f"got {type(value).__name__}: {value!r}"
            )
    elif str in allowed:
        if not isinstance(value, str):
            raise ConfigError(
                f"Config field '{key}' for {cls_name} expects a string, "
                f"got {type(value).__name__}: {value!r}"
            )


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a plain dict, keeping defaults."""
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")
    # Resolve string annotations (from `__future__ annotations`) to real types.
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(
                f"Unknown config key '{key}' for {cls.__name__}. Valid keys: {sorted(known)}"
            )
        ftype = hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)  # type: ignore[arg-type]
        else:
            _check_scalar_type(cls.__name__, key, value, ftype)
            kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Invalid configuration for {cls.__name__}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigError` with ``message`` when ``condition`` is false."""
    if not condition:
        raise ConfigError(message)


def validate_config(cfg: SimConfig) -> SimConfig:
    """Validate a :class:`SimConfig`, raising :class:`ConfigError` on bad input.

    Checks the ranges that would otherwise surface as an opaque crash deep in the
    engine (non-positive station counts, capacities or headways, negative
    reserves, an unknown signalling mode, ...) and reports them with an
    actionable message. Returns ``cfg`` unchanged so it can be used inline.
    """
    _require(cfg.horizon > 0, f"horizon must be positive, got {cfg.horizon}")
    _require(
        cfg.n_initial_trains >= 1,
        f"n_initial_trains must be >= 1, got {cfg.n_initial_trains}",
    )
    _require(
        cfg.depot_reserve >= 0,
        f"depot_reserve must be >= 0, got {cfg.depot_reserve}",
    )
    _require(
        cfg.train_capacity > 0,
        f"train_capacity must be positive, got {cfg.train_capacity}",
    )
    _require(
        cfg.target_headway > 0,
        f"target_headway must be positive, got {cfg.target_headway}",
    )
    _require(
        cfg.control_interval > 0,
        f"control_interval must be positive, got {cfg.control_interval}",
    )
    _require(
        cfg.sample_interval > 0,
        f"sample_interval must be positive, got {cfg.sample_interval}",
    )

    line = cfg.line
    _require(
        line.n_stations >= 3,
        f"a metro line needs at least 3 stations, got {line.n_stations}",
    )
    _require(
        line.dwell_base >= 0,
        f"line.dwell_base must be >= 0, got {line.dwell_base}",
    )
    _require(
        0 <= line.depot_station < line.n_stations,
        f"line.depot_station {line.depot_station} out of range [0, {line.n_stations - 1}]",
    )

    sig = cfg.signalling
    _require(
        sig.mode in ("moving_block", "fixed_block"),
        f"signalling.mode must be 'moving_block' or 'fixed_block', got {sig.mode!r}",
    )
    _require(
        sig.max_speed_mps > 0,
        f"signalling.max_speed_mps must be positive, got {sig.max_speed_mps}",
    )
    _require(
        sig.service_decel_mps2 > 0,
        f"signalling.service_decel_mps2 must be positive, got {sig.service_decel_mps2}",
    )
    _require(
        sig.safety_margin_m >= 0,
        f"signalling.safety_margin_m must be >= 0, got {sig.safety_margin_m}",
    )
    _require(
        sig.block_length_m > 0,
        f"signalling.block_length_m must be positive, got {sig.block_length_m}",
    )
    _require(
        sig.n_clear_blocks >= 1,
        f"signalling.n_clear_blocks must be >= 1, got {sig.n_clear_blocks}",
    )

    dem = cfg.demand
    _require(
        dem.arrival_scale >= 0,
        f"demand.arrival_scale must be >= 0, got {dem.arrival_scale}",
    )
    _require(
        dem.baseline_frac >= 0,
        f"demand.baseline_frac must be >= 0, got {dem.baseline_frac}",
    )
    _require(
        dem.bin_seconds > 0,
        f"demand.bin_seconds must be positive, got {dem.bin_seconds}",
    )
    _require(
        dem.profile is None or dem.profile in DEMAND_PROFILES,
        f"demand.profile must be one of {sorted(DEMAND_PROFILES)}, got {dem.profile!r}",
    )

    ctl = cfg.controller
    _require(
        ctl.queue_threshold > 0,
        f"controller.queue_threshold must be positive, got {ctl.queue_threshold}",
    )
    _require(
        ctl.min_injection_gap >= 0,
        f"controller.min_injection_gap must be >= 0, got {ctl.min_injection_gap}",
    )
    _require(
        ctl.min_headway >= 0,
        f"controller.min_headway must be >= 0, got {ctl.min_headway}",
    )
    _require(
        ctl.opt_horizon_steps >= 1,
        f"controller.opt_horizon_steps must be >= 1, got {ctl.opt_horizon_steps}",
    )
    _require(
        ctl.opt_step_seconds > 0,
        f"controller.opt_step_seconds must be positive, got {ctl.opt_step_seconds}",
    )
    _require(
        ctl.opt_max_solve_seconds > 0,
        f"controller.opt_max_solve_seconds must be positive, got {ctl.opt_max_solve_seconds}",
    )
    return cfg


def load_config(path: str | None = None) -> SimConfig:
    """Load a :class:`SimConfig` from ``path``, or return defaults if ``None``.

    Raises :class:`ScenarioFileError` if the file is missing or is not valid
    YAML, and :class:`ConfigError` if its contents are structurally invalid or
    out of range.
    """
    if path is None:
        return validate_config(SimConfig())
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ScenarioFileError(f"Scenario file not found: {path}") from exc
    except OSError as exc:
        raise ScenarioFileError(f"Could not read scenario file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioFileError(f"Could not parse YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Scenario YAML must be a mapping at the top level (in {path})")
    cfg = _from_dict(SimConfig, raw)
    assert isinstance(cfg, SimConfig)
    return validate_config(cfg)


def config_to_dict(cfg: Any) -> Any:
    """Serialise a (possibly nested) dataclass config back to plain data."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, list):
        return [config_to_dict(v) for v in cfg]
    return cfg

"""Safe-separation (signalling) models.

This module replaces the v1 "trains never collide" assumption with a proper
minimum-safe-separation model. It provides two regimes, selectable from the
scenario configuration:

moving_block (CBTC)
    The minimum safe separation between a following train and its leader is
    *derived from kinematics*: it is the distance the follower needs to stop
    (braking distance at its current speed) plus the distance covered during a
    reaction/communication delay plus a fixed safety margin plus the train
    length. Because it scales with speed, the implied minimum time headway is
    not a constant -- it shrinks at low speed, which is exactly why CBTC / moving
    block yields shorter headways than fixed block.

fixed_block
    The line is divided into fixed physical block sections of length
    ``block_length_m``. A block may be occupied by at most one train, and a
    following train must keep ``n_clear`` empty block(s) between itself and its
    leader. Separation is therefore quantised to block boundaries and is
    independent of speed, which is the classic limitation of fixed-block
    signalling.

All functions here are pure (no simulation state) and unit-tested in
``tests/test_signalling.py``. SI units throughout: metres, seconds, m/s, m/s^2.

References (background, not novel claims)
----------------------------------------
- Moving-block / CBTC braking-curve separation is standard in IEEE 1474 CBTC
  performance specifications and in railway operations textbooks
  (e.g. Profillidis, "Railway Management and Engineering").
- The kinematic safe-braking distance ``v^2 / (2a)`` is elementary mechanics.
These are textbook relations reproduced here for a demonstration model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from metroflow.config import SignallingConfig
from metroflow.errors import ConfigError


def braking_distance(speed_mps: float, decel_mps2: float) -> float:
    """Distance to stop from ``speed_mps`` under constant service deceleration.

    ``d = v^2 / (2a)``. A non-positive deceleration is treated as "cannot stop"
    and returns infinity.

    >>> braking_distance(20.0, 1.0)
    200.0
    >>> braking_distance(0.0, 1.0)
    0.0
    """
    if speed_mps <= 0.0:
        return 0.0
    if decel_mps2 <= 0.0:
        return math.inf
    return (speed_mps * speed_mps) / (2.0 * decel_mps2)


def min_safe_separation(
    speed_mps: float,
    decel_mps2: float,
    safety_margin_m: float,
    reaction_time_s: float,
    train_length_m: float = 0.0,
) -> float:
    """Minimum safe centre-to-rear separation for a moving-block follower.

    ``sep = v * t_react + v^2 / (2a) + margin + train_length``.

    The three physical contributions are: distance travelled before the brake
    takes effect (reaction/communication lag), the braking distance itself, and
    a fixed safety margin; the train length accounts for the leader's physical
    extent.

    >>> round(min_safe_separation(20.0, 1.0, 50.0, 2.0, 90.0), 1)
    380.0
    """
    return (
        speed_mps * reaction_time_s
        + braking_distance(speed_mps, decel_mps2)
        + safety_margin_m
        + train_length_m
    )


def min_moving_block_headway(
    speed_mps: float,
    decel_mps2: float,
    safety_margin_m: float,
    reaction_time_s: float,
    train_length_m: float = 0.0,
) -> float:
    """Minimum time headway (s) implied by the moving-block separation.

    ``headway = safe_separation / speed``. Note the speed dependence: unlike a
    fixed-block constant, this headway varies with running speed.

    >>> h = min_moving_block_headway(20.0, 1.0, 50.0, 2.0, 90.0)
    >>> round(h, 1)
    19.0
    """
    if speed_mps <= 0.0:
        return math.inf
    sep = min_safe_separation(
        speed_mps, decel_mps2, safety_margin_m, reaction_time_s, train_length_m
    )
    return sep / speed_mps


def block_index(pos_m: float, block_length_m: float) -> int:
    """Index of the fixed block containing coordinate ``pos_m``.

    >>> block_index(950.0, 400.0)
    2
    >>> block_index(0.0, 400.0)
    0
    """
    if block_length_m <= 0.0:
        raise ValueError("block_length_m must be positive")
    return int(pos_m // block_length_m)


def fixed_block_min_separation(
    block_length_m: float, n_clear: int = 1, train_length_m: float = 0.0
) -> float:
    """Worst-case minimum separation guaranteed by an ``n_clear``-block rule.

    With ``n_clear`` empty blocks required between trains, two trains are at
    least ``n_clear * block_length`` apart in the best case and up to one extra
    block in the worst case. This helper returns the guaranteed lower bound used
    for reporting/comparison with the moving-block separation.
    """
    return n_clear * block_length_m + train_length_m


def fixed_block_clear(
    follower_pos_m: float,
    leaders_pos_m: Iterable[float],
    block_length_m: float,
    direction: int,
    n_clear: int = 1,
) -> bool:
    """True if the ``n_clear`` blocks ahead of the follower are unoccupied.

    ``direction`` is +1 (increasing coordinate) or -1. A leader sharing the
    follower's own block also blocks entry.
    """
    fb = block_index(follower_pos_m, block_length_m)
    forbidden = {fb + direction * k for k in range(0, n_clear + 1)}
    for lp in leaders_pos_m:
        if block_index(lp, block_length_m) in forbidden:
            return False
    return True


@dataclass(frozen=True)
class SignallingModel:
    """Configured safe-separation evaluator used by the engine.

    Wraps the pure functions above with the scenario parameters so the
    simulation can ask a single question: *is it safe for this follower to
    proceed given the gap to its leader?*
    """

    mode: str
    max_speed_mps: float
    service_decel_mps2: float
    safety_margin_m: float
    reaction_time_s: float
    train_length_m: float
    block_length_m: float
    n_clear_blocks: int
    enforce: bool

    def required_separation(self, speed_mps: float) -> float:
        """Separation (m) the follower must keep at ``speed_mps``."""
        if self.mode == "fixed_block":
            return fixed_block_min_separation(
                self.block_length_m, self.n_clear_blocks, self.train_length_m
            )
        return min_safe_separation(
            speed_mps,
            self.service_decel_mps2,
            self.safety_margin_m,
            self.reaction_time_s,
            self.train_length_m,
        )

    def implied_headway(self, speed_mps: float) -> float:
        """Minimum time headway (s) implied at ``speed_mps``."""
        v = min(speed_mps, self.max_speed_mps)
        if self.mode == "fixed_block":
            sep = self.required_separation(v)
            return sep / v if v > 0 else math.inf
        return min_moving_block_headway(
            v,
            self.service_decel_mps2,
            self.safety_margin_m,
            self.reaction_time_s,
            self.train_length_m,
        )

    def is_clear(
        self,
        follower_pos_m: float,
        gap_ahead_m: float | None,
        speed_mps: float,
        leaders_pos_m: Iterable[float] | None = None,
        direction: int = 1,
    ) -> bool:
        """Decide whether the follower may proceed.

        moving_block: compare ``gap_ahead_m`` (leader distance) with the
        speed-dependent required separation. fixed_block: check block occupancy
        from ``leaders_pos_m``.
        """
        if not self.enforce:
            return True
        if self.mode == "fixed_block":
            if leaders_pos_m is None:
                return True
            return fixed_block_clear(
                follower_pos_m,
                leaders_pos_m,
                self.block_length_m,
                direction,
                self.n_clear_blocks,
            )
        # moving block
        if gap_ahead_m is None:
            return True
        v = min(speed_mps, self.max_speed_mps)
        return gap_ahead_m >= self.required_separation(v)


def make_signalling_model(cfg: SignallingConfig) -> SignallingModel:
    """Build a :class:`SignallingModel` from a ``SignallingConfig``."""
    mode = cfg.mode
    if mode not in ("moving_block", "fixed_block"):
        raise ConfigError(f"signalling.mode must be 'moving_block' or 'fixed_block', got {mode!r}")
    return SignallingModel(
        mode=mode,
        max_speed_mps=float(cfg.max_speed_mps),
        service_decel_mps2=float(cfg.service_decel_mps2),
        safety_margin_m=float(cfg.safety_margin_m),
        reaction_time_s=float(cfg.reaction_time_s),
        train_length_m=float(cfg.train_length_m),
        block_length_m=float(cfg.block_length_m),
        n_clear_blocks=int(cfg.n_clear_blocks),
        enforce=bool(cfg.enforce),
    )

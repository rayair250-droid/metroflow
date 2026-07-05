"""Line topology: stations, inter-station segments and the depot."""

from __future__ import annotations

from dataclasses import dataclass

from metroflow.config import LineConfig
from metroflow.errors import ConfigError

# Direction constants: +1 travels toward higher station indices, -1 the reverse.
UP = 1
DOWN = -1


@dataclass(frozen=True)
class Station:
    index: int
    name: str


class Line:
    """A linear metro line with two running directions and a depot.

    Segment ``i`` connects station ``i`` and station ``i + 1``. Trains run to a
    terminus, reverse, and run back. A depot is attached at ``depot_station`` and
    holds reserve trains that the controller can inject.
    """

    def __init__(self, cfg: LineConfig):
        n = cfg.n_stations
        if n < 3:
            raise ConfigError(f"A line needs at least 3 stations, got {n}")
        names = cfg.station_names
        if names is not None and len(names) != n:
            raise ConfigError(f"station_names length ({len(names)}) must equal n_stations ({n})")
        self.stations: list[Station] = [
            Station(i, names[i] if names else f"S{i:02d}") for i in range(n)
        ]

        seg = cfg.segment_time
        if isinstance(seg, (list, tuple)):
            if len(seg) != n - 1:
                raise ConfigError(
                    f"segment_time list must have n_stations-1 ({n - 1}) entries, got {len(seg)}"
                )
            self.segment_times: list[float] = [float(x) for x in seg]
        else:
            self.segment_times = [float(seg)] * (n - 1)
        if any(t <= 0 for t in self.segment_times):
            raise ConfigError("all segment_time entries must be positive")

        seg_len = cfg.segment_length
        if isinstance(seg_len, (list, tuple)):
            if len(seg_len) != n - 1:
                raise ConfigError(
                    f"segment_length list must have n_stations-1 ({n - 1}) entries, "
                    f"got {len(seg_len)}"
                )
            self.segment_lengths: list[float] = [float(x) for x in seg_len]
        else:
            self.segment_lengths = [float(seg_len)] * (n - 1)
        if any(length <= 0 for length in self.segment_lengths):
            raise ConfigError("all segment_length entries must be positive")

        # Cumulative physical coordinate (m) of each station from station 0.
        self.station_coord: list[float] = [0.0]
        for length in self.segment_lengths:
            self.station_coord.append(self.station_coord[-1] + length)
        self.length_m: float = self.station_coord[-1]

        if not 0 <= cfg.depot_station < n:
            raise ConfigError(f"depot_station {cfg.depot_station} out of range [0, {n - 1}]")
        self.depot_station = cfg.depot_station

    # -- topology helpers ---------------------------------------------------- #
    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def is_terminus(self, station: int, direction: int) -> bool:
        """True if a train at ``station`` heading ``direction`` must reverse."""
        if direction == UP:
            return station == self.n_stations - 1
        return station == 0

    def next_station(self, station: int, direction: int) -> int | None:
        """Next station index in ``direction``, or ``None`` at a terminus."""
        if self.is_terminus(station, direction):
            return None
        return station + direction

    def segment_index(self, station: int, direction: int) -> int | None:
        """Index of the segment a train enters when leaving ``station``."""
        nxt = self.next_station(station, direction)
        if nxt is None:
            return None
        return min(station, nxt)

    def travel_time(self, station: int, direction: int) -> float:
        """Nominal travel time for the segment leaving ``station``."""
        seg = self.segment_index(station, direction)
        if seg is None:
            return 0.0
        return self.segment_times[seg]

    def cycle_time(self) -> float:
        """Rough nominal round-trip travel time (excludes dwell)."""
        return 2.0 * sum(self.segment_times)

    def nominal_speed(self, station: int, direction: int) -> float:
        """Nominal running speed (m/s) for the segment leaving ``station``."""
        seg = self.segment_index(station, direction)
        if seg is None:
            return 0.0
        t = self.segment_times[seg]
        return self.segment_lengths[seg] / t if t > 0 else 0.0

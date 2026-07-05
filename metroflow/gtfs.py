"""Build a MetroFlow line from standard GTFS open-transit data.

GTFS (General Transit Feed Specification) is the open format transit agencies
publish their networks in. This module parses the four core files -- ``stops.txt``,
``routes.txt``, ``trips.txt`` and ``stop_times.txt`` -- with the Python standard
library only (no heavy dependencies) and turns one route/direction into a
:class:`~metroflow.config.LineConfig`: the ordered stations and approximate
inter-station run-times inferred from the timetable in ``stop_times``.

Data sources (cited, not bundled)
---------------------------------
Real feeds are large and are intentionally NOT vendored into this repository.
Point MetroFlow at a feed you download yourself:

* Île-de-France Mobilités (IDFM), via transport.data.gouv.fr:
  https://transport.data.gouv.fr/datasets/reseau-urbain-et-interurbain-dile-de-france-mobilites
* RATP open data:
  https://www.ratp.fr/en/ratp-and-open-data

A tiny, hand-authored, clearly-illustrative GTFS sample ships under
``examples/gtfs_sample/`` so the tests and demo run fully offline. See
``scripts/fetch_gtfs.py`` for how to obtain and use a real feed.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from metroflow.config import LineConfig, SimConfig
from metroflow.errors import GtfsError

_CORE_FILES = ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")

#: Columns each core file must contain for MetroFlow to build a line.
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "stops.txt": ("stop_id",),
    "routes.txt": ("route_id",),
    "trips.txt": ("route_id", "trip_id"),
    "stop_times.txt": ("trip_id", "stop_sequence", "stop_id"),
}

# Sensible floor/ceiling (s) for inferred inter-station run-times, guarding
# against dirty timetable rows (zero or absurd gaps).
_MIN_SEGMENT_S = 20.0
_MAX_SEGMENT_S = 600.0
_DEFAULT_SEGMENT_S = 90.0


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _check_columns(fname: str, rows: list[dict]) -> None:
    """Raise :class:`GtfsError` if ``rows`` lack a required column of ``fname``."""
    required = _REQUIRED_COLUMNS.get(fname, ())
    if not required:
        return
    present = set(rows[0].keys()) if rows else set()
    missing = [c for c in required if c not in present]
    if missing:
        raise GtfsError(f"GTFS file {fname} is missing required column(s): {', '.join(missing)}")


def parse_gtfs_time(value: str) -> float | None:
    """Parse a GTFS ``HH:MM:SS`` time (hours may exceed 24) into seconds."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


@dataclass
class GtfsFeed:
    """Parsed core GTFS tables plus a few convenience indexes."""

    stops: dict[str, str]  # stop_id -> stop_name
    routes: list[dict]  # raw route rows
    trips: list[dict]  # raw trip rows
    # trip_id -> ordered [(seq, stop_id, arrival_s, departure_s)]
    stop_times: dict[str, list[tuple[int, str, float | None, float | None]]] = field(
        default_factory=dict
    )

    def route_ids(self) -> list[str]:
        return [r["route_id"] for r in self.routes]

    def route_label(self, route_id: str) -> str:
        for r in self.routes:
            if r["route_id"] == route_id:
                short = r.get("route_short_name") or ""
                long = r.get("route_long_name") or ""
                label = " ".join(x for x in (short, long) if x).strip()
                return label or route_id
        return route_id

    def trips_for(self, route_id: str, direction_id: int | None) -> list[str]:
        out = []
        for tr in self.trips:
            if tr["route_id"] != route_id:
                continue
            if direction_id is not None:
                d = tr.get("direction_id", "")
                if str(d).strip() not in ("", str(direction_id)):
                    continue
            out.append(tr["trip_id"])
        return out

    def directions_for(self, route_id: str) -> list[int]:
        seen = set()
        for tr in self.trips:
            if tr["route_id"] != route_id:
                continue
            d = str(tr.get("direction_id", "")).strip()
            seen.add(int(d) if d.isdigit() else 0)
        return sorted(seen)


def load_feed(directory: str) -> GtfsFeed:
    """Parse the four core GTFS files from ``directory`` into a :class:`GtfsFeed`."""
    if not os.path.isdir(directory):
        raise GtfsError(f"GTFS directory not found: {directory}")
    for fname in _CORE_FILES:
        p = os.path.join(directory, fname)
        if not os.path.exists(p):
            raise GtfsError(f"GTFS file not found: {p}")

    stops_rows = _read_csv(os.path.join(directory, "stops.txt"))
    _check_columns("stops.txt", stops_rows)
    stops = {r["stop_id"]: (r.get("stop_name") or r["stop_id"]) for r in stops_rows}
    routes = _read_csv(os.path.join(directory, "routes.txt"))
    _check_columns("routes.txt", routes)
    trips = _read_csv(os.path.join(directory, "trips.txt"))
    _check_columns("trips.txt", trips)

    stop_times_rows = _read_csv(os.path.join(directory, "stop_times.txt"))
    _check_columns("stop_times.txt", stop_times_rows)
    stop_times: dict[str, list[tuple[int, str, float | None, float | None]]] = {}
    for r in stop_times_rows:
        tid = r["trip_id"]
        try:
            seq = int(r["stop_sequence"])
        except (KeyError, ValueError):
            continue
        arr = parse_gtfs_time(r.get("arrival_time", ""))
        dep = parse_gtfs_time(r.get("departure_time", ""))
        stop_times.setdefault(tid, []).append((seq, r["stop_id"], arr, dep))
    for tid in stop_times:
        stop_times[tid].sort(key=lambda x: x[0])

    return GtfsFeed(stops=stops, routes=routes, trips=trips, stop_times=stop_times)


_StopTimeRow = tuple[int, str, float | None, float | None]


def _representative_trip(feed: GtfsFeed, trip_ids: list[str]) -> list[_StopTimeRow]:
    """Pick the trip with the most stops (the most complete run of the route)."""
    best: list[_StopTimeRow] = []
    for tid in trip_ids:
        seq = feed.stop_times.get(tid, [])
        if len(seq) > len(best):
            best = seq
    return best


def _segment_times(feed: GtfsFeed, trip_ids: list[str], ordered_stops: list[str]) -> list[float]:
    """Average inter-station run-time (s) for each consecutive stop pair.

    For every trip we accumulate observed ``arrival[next] - departure[cur]`` for
    each adjacent (from_stop, to_stop) pair, then read them off along the chosen
    station order, clamping to a sane range and defaulting where unseen.
    """
    pair_runs: dict[tuple[str, str], list[float]] = {}
    for tid in trip_ids:
        seq = feed.stop_times.get(tid, [])
        for (_, s0, _, dep0), (_, s1, arr1, _) in zip(seq, seq[1:], strict=False):
            if dep0 is None or arr1 is None:
                continue
            run = arr1 - dep0
            if run <= 0:
                continue
            pair_runs.setdefault((s0, s1), []).append(run)

    seg_times: list[float] = []
    for a, b in zip(ordered_stops, ordered_stops[1:], strict=False):
        runs = pair_runs.get((a, b))
        if runs:
            val = sum(runs) / len(runs)
        else:
            val = _DEFAULT_SEGMENT_S
        val = min(_MAX_SEGMENT_S, max(_MIN_SEGMENT_S, val))
        seg_times.append(round(val, 1))
    return seg_times


@dataclass
class LineFromGtfs:
    """Result of building a line: the config plus human-readable metadata."""

    line: LineConfig
    route_id: str
    route_label: str
    direction_id: int
    station_ids: list[str]
    station_names: list[str]
    segment_times: list[float]


def build_line_config(directory: str, route_id: str, direction_id: int = 0) -> LineFromGtfs:
    """Build a :class:`LineConfig` for one ``route_id``/``direction_id``."""
    feed = load_feed(directory)
    if route_id not in feed.route_ids():
        raise GtfsError(f"route_id '{route_id}' not in feed. Available: {feed.route_ids()}")
    trip_ids = feed.trips_for(route_id, direction_id)
    if not trip_ids:
        raise GtfsError(
            f"no trips for route '{route_id}' direction {direction_id}. "
            f"Directions present: {feed.directions_for(route_id)}"
        )
    rep = _representative_trip(feed, trip_ids)
    if len(rep) < 3:
        raise GtfsError(
            f"route '{route_id}' direction {direction_id} has < 3 stops; "
            "MetroFlow needs at least 3 stations"
        )
    ordered_stops = [stop_id for (_, stop_id, _, _) in rep]
    names = [feed.stops.get(sid, sid) for sid in ordered_stops]
    seg_times = _segment_times(feed, trip_ids, ordered_stops)

    line = LineConfig(
        n_stations=len(ordered_stops),
        segment_time=list(seg_times),
        station_names=names,
        depot_station=0,
    )
    return LineFromGtfs(
        line=line,
        route_id=route_id,
        route_label=feed.route_label(route_id),
        direction_id=direction_id,
        station_ids=ordered_stops,
        station_names=names,
        segment_times=seg_times,
    )


def apply_gtfs(
    cfg: SimConfig, directory: str, route_id: str, direction_id: int = 0
) -> LineFromGtfs:
    """Override ``cfg.line`` from GTFS and reset demand weights to auto-size.

    Demand origin/attraction weights (if any) are cleared so they regenerate for
    the new station count; passenger intensity keeps the scenario's ``arrival_scale``.
    Returns the built :class:`LineFromGtfs` metadata.
    """
    built = build_line_config(directory, route_id, direction_id)
    cfg.line = built.line
    cfg.demand.origin_weights = None
    cfg.demand.attraction_weights = None
    # Keep a modest fleet consistent with the (usually small) sample line.
    n = built.line.n_stations
    cfg.n_initial_trains = min(cfg.n_initial_trains, max(2, n - 1))
    cfg.name = f"gtfs:{route_id}:dir{direction_id}"
    return built


def describe_feed(directory: str) -> str:
    """Human-readable summary of routes/stops for the ``gtfs-info`` command."""
    feed = load_feed(directory)
    lines = [
        f"GTFS feed: {os.path.abspath(directory)}",
        f"  stops: {len(feed.stops)}   routes: {len(feed.routes)}   trips: {len(feed.trips)}",
        "",
    ]
    for rid in feed.route_ids():
        label = feed.route_label(rid)
        dirs = feed.directions_for(rid)
        lines.append(f"  route_id={rid}  ({label})")
        for d in dirs:
            trip_ids = feed.trips_for(rid, d)
            rep = _representative_trip(feed, trip_ids)
            names = [feed.stops.get(sid, sid) for (_, sid, _, _) in rep]
            head = " -> ".join(names[:4])
            more = " -> ..." if len(names) > 4 else ""
            lines.append(
                f"    direction {d}: {len(trip_ids)} trip(s), {len(rep)} stops: {head}{more}"
            )
    return "\n".join(lines)

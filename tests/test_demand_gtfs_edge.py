"""Edge-case coverage for the demand model and GTFS ingestion error paths.

These exercise the validation branches and degenerate cases that the happy-path
tests skip: mismatched custom weight vectors, empty arrival bins, terminus
direction splits, and malformed GTFS feeds (missing files/columns, wrong
direction, too-few stops).
"""

from __future__ import annotations

import numpy as np
import pytest

from metroflow.config import DemandConfig
from metroflow.demand import DemandModel
from metroflow.errors import ConfigError, GtfsError
from metroflow.gtfs import (
    build_line_config,
    load_feed,
    parse_gtfs_time,
)

# --------------------------------------------------------------------------- #
# demand.py
# --------------------------------------------------------------------------- #


def test_origin_weights_length_mismatch_raises():
    cfg = DemandConfig(origin_weights=[1.0, 2.0])  # only 2 for 5 stations
    with pytest.raises(ConfigError, match="origin_weights"):
        DemandModel(cfg, n_stations=5)


def test_attraction_weights_length_mismatch_raises():
    cfg = DemandConfig(attraction_weights=[1.0, 2.0, 3.0])
    with pytest.raises(ConfigError, match="attraction_weights"):
        DemandModel(cfg, n_stations=6)


def test_custom_weights_are_used_verbatim():
    ow = [1.0, 3.0, 1.0, 1.0]
    aw = [2.0, 2.0, 2.0, 2.0]
    dm = DemandModel(DemandConfig(origin_weights=ow, attraction_weights=aw), n_stations=4)
    assert np.allclose(dm.origin_w, ow)
    assert np.allclose(dm.attract, aw)


def test_direction_split_at_termini():
    dm = DemandModel(DemandConfig(), n_stations=6)
    up0, down0 = dm.direction_split(0)  # first station: nothing below it
    assert down0 == pytest.approx(0.0)
    assert up0 == pytest.approx(1.0)
    up_last, down_last = dm.direction_split(5)  # last: nothing above it
    assert up_last == pytest.approx(0.0)
    assert down_last == pytest.approx(1.0)


def test_direction_split_degenerate_all_weight_on_station():
    # If every unit of attraction sits on the station itself, the split has no
    # information and falls back to an even 50/50.
    aw = [0.0, 0.0, 5.0, 0.0, 0.0]
    dm = DemandModel(DemandConfig(attraction_weights=aw), n_stations=5)
    up, down = dm.direction_split(2)
    assert (up, down) == (0.5, 0.5)


def test_generate_bin_empty_when_rate_zero():
    # arrival_scale = 0 -> expected arrivals 0 -> empty list (no RNG draw).
    dm = DemandModel(DemandConfig(arrival_scale=0.0), n_stations=5)
    rng = np.random.default_rng(0)
    assert dm.generate_bin(2, 0.0, 60.0, rng) == []


def test_generate_bin_empty_when_no_valid_destination():
    # All attraction mass on the origin station -> after zeroing it, no valid
    # destination remains, so the bin yields nobody even with positive rate.
    aw = [0.0, 0.0, 4.0, 0.0]
    dm = DemandModel(
        DemandConfig(arrival_scale=0.5, attraction_weights=aw, baseline_frac=1.0),
        n_stations=4,
    )
    rng = np.random.default_rng(1)
    assert dm.generate_bin(2, 0.0, 120.0, rng) == []


def test_generate_bin_produces_passengers():
    dm = DemandModel(DemandConfig(arrival_scale=0.2, baseline_frac=1.0), n_stations=6)
    rng = np.random.default_rng(2)
    pax = dm.generate_bin(1, 0.0, 300.0, rng)
    assert pax  # non-empty
    assert all(p.origin == 1 and p.dest != 1 for p in pax)
    # Arrival instants are within the bin and sorted.
    times = [p.arrival for p in pax]
    assert times == sorted(times)
    assert all(0.0 <= t < 300.0 for t in times)


def test_surge_multiplies_rate():
    dm = DemandModel(DemandConfig(arrival_scale=0.1, baseline_frac=1.0), n_stations=6)
    # A pristine copy with no surge gives the surge-free reference at each t
    # (the temporal profile varies with t, so we can't reuse one instant).
    ref = DemandModel(DemandConfig(arrival_scale=0.1, baseline_frac=1.0), n_stations=6)

    dm.add_surge(3, until=200.0, multiplier=4.0)
    # During the surge window, station 3's rate is exactly 4x the reference.
    assert dm.rate(3, 100.0) == pytest.approx(4.0 * ref.rate(3, 100.0))
    # Surge does not leak to other stations...
    assert dm.rate(4, 100.0) == pytest.approx(ref.rate(4, 100.0))
    # ...nor past its end time.
    assert dm.rate(3, 250.0) == pytest.approx(ref.rate(3, 250.0))


# --------------------------------------------------------------------------- #
# gtfs.py — malformed feeds
# --------------------------------------------------------------------------- #


def _write_feed(d, *, stops, routes, trips, stop_times) -> str:
    (d / "stops.txt").write_text(stops)
    (d / "routes.txt").write_text(routes)
    (d / "trips.txt").write_text(trips)
    (d / "stop_times.txt").write_text(stop_times)
    return str(d)


_GOOD_STOPS = "stop_id,stop_name\nA,Alpha\nB,Bravo\nC,Charlie\nD,Delta\n"
_GOOD_ROUTES = "route_id,route_short_name,route_long_name\nR,1,The Line\n"


def test_parse_gtfs_time_none_returns_none():
    assert parse_gtfs_time(None) is None


def test_parse_gtfs_time_non_numeric_parts_returns_none():
    # Three colon-separated but non-integer parts -> ValueError branch -> None.
    assert parse_gtfs_time("aa:bb:cc") is None


def test_route_label_falls_back_to_id_and_multi_route_feed(tmp_path):
    from metroflow.gtfs import load_feed as _lf

    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes="route_id,route_short_name,route_long_name\nR,1,The Line\nS,,\n",
        trips="route_id,trip_id,direction_id\nR,t1,0\nS,t2,0\n",
        stop_times=(
            "trip_id,stop_sequence,stop_id,arrival_time,departure_time\n"
            "t1,1,A,08:00:00,08:00:30\n"
            "t1,2,B,08:02:00,08:02:30\n"
            "t1,3,C,08:04:00,08:04:30\n"
            "t2,1,A,08:00:00,08:00:30\n"
        ),
    )
    feed = _lf(d)
    # Route S has no names -> label falls back to the id; route R uses names.
    assert feed.route_label("S") == "S"
    assert feed.route_label("R") == "1 The Line"
    assert feed.route_label("UNKNOWN") == "UNKNOWN"
    # trips_for / directions_for skip the other route (exercises the filters).
    assert feed.trips_for("R", 0) == ["t1"]
    assert feed.directions_for("R") == [0]


def test_load_feed_skips_rows_with_bad_stop_sequence(tmp_path):
    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes=_GOOD_ROUTES,
        trips="route_id,trip_id,direction_id\nR,t1,0\n",
        stop_times=(
            "trip_id,stop_sequence,stop_id,arrival_time,departure_time\n"
            "t1,x,A,08:00:00,08:00:30\n"  # non-integer sequence -> skipped
            "t1,1,A,08:00:00,08:00:30\n"
            "t1,2,B,08:02:00,08:02:30\n"
            "t1,3,C,08:04:00,08:04:30\n"
        ),
    )
    built = build_line_config(d, "R", direction_id=0)
    # The malformed row is dropped; the three valid stops remain.
    assert built.line.n_stations == 3


def test_missing_file_raises(tmp_path):
    (tmp_path / "stops.txt").write_text(_GOOD_STOPS)
    # routes/trips/stop_times absent
    with pytest.raises(GtfsError, match="not found"):
        load_feed(str(tmp_path))


def test_missing_required_column_raises(tmp_path):
    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes="route_id\nR\n",
        # trips.txt without the required trip_id column
        trips="route_id,direction_id\nR,0\n",
        stop_times="trip_id,stop_sequence,stop_id\nt1,1,A\n",
    )
    with pytest.raises(GtfsError, match="missing required column"):
        load_feed(d)


def test_build_line_config_wrong_direction_raises(tmp_path):
    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes=_GOOD_ROUTES,
        trips="route_id,trip_id,direction_id\nR,t1,0\n",
        stop_times=(
            "trip_id,stop_sequence,stop_id,arrival_time,departure_time\n"
            "t1,1,A,08:00:00,08:00:30\n"
            "t1,2,B,08:02:00,08:02:30\n"
            "t1,3,C,08:04:00,08:04:30\n"
        ),
    )
    # Direction 1 has no trips.
    with pytest.raises(GtfsError, match="no trips"):
        build_line_config(d, "R", direction_id=1)


def test_build_line_config_too_few_stops_raises(tmp_path):
    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes=_GOOD_ROUTES,
        trips="route_id,trip_id,direction_id\nR,t1,0\n",
        stop_times=(
            "trip_id,stop_sequence,stop_id,arrival_time,departure_time\n"
            "t1,1,A,08:00:00,08:00:30\n"
            "t1,2,B,08:02:00,08:02:30\n"
        ),
    )
    with pytest.raises(GtfsError, match="at least 3 stations|< 3 stops"):
        build_line_config(d, "R", direction_id=0)


def test_segment_times_fall_back_to_default_when_times_missing(tmp_path):
    # No arrival/departure times -> run-times can't be inferred -> default 90 s,
    # clamped into range, one per segment.
    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes=_GOOD_ROUTES,
        trips="route_id,trip_id,direction_id\nR,t1,0\n",
        stop_times=("trip_id,stop_sequence,stop_id\nt1,1,A\nt1,2,B\nt1,3,C\nt1,4,D\n"),
    )
    built = build_line_config(d, "R", direction_id=0)
    assert built.line.n_stations == 4
    assert len(built.segment_times) == 3
    assert all(t == pytest.approx(90.0) for t in built.segment_times)


def test_segment_times_clamped_from_dirty_rows(tmp_path):
    # A negative/zero gap is skipped; a huge gap is clamped to the ceiling.
    d = _write_feed(
        tmp_path,
        stops=_GOOD_STOPS,
        routes=_GOOD_ROUTES,
        trips="route_id,trip_id,direction_id\nR,t1,0\n",
        stop_times=(
            "trip_id,stop_sequence,stop_id,arrival_time,departure_time\n"
            "t1,1,A,08:00:00,08:00:00\n"  # A->B gap huge (clamped to 600)
            "t1,2,B,09:00:00,09:00:00\n"  # B->C zero/negative gap -> default
            "t1,3,C,09:00:00,09:00:00\n"
            "t1,4,D,09:03:00,09:03:00\n"  # C->D = 180 s, kept
        ),
    )
    built = build_line_config(d, "R", direction_id=0)
    segs = built.segment_times
    assert len(segs) == 3
    assert segs[0] == pytest.approx(600.0)  # clamped ceiling
    assert segs[1] == pytest.approx(90.0)  # zero gap -> default
    assert segs[2] == pytest.approx(180.0)  # kept as-is

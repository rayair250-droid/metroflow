"""Tests for GTFS ingestion (metroflow/gtfs.py) against the committed sample."""

from pathlib import Path

import pytest

from metroflow.config import load_config
from metroflow.gtfs import (
    apply_gtfs,
    build_line_config,
    describe_feed,
    load_feed,
    parse_gtfs_time,
)
from metroflow.line import Line
from metroflow.simulation import run_simulation

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "gtfs_sample"

EXPECTED = ["Riverside", "Market", "Central", "University", "Parkway", "Airport"]


def test_parse_gtfs_time():
    assert parse_gtfs_time("00:01:30") == 90
    assert parse_gtfs_time("25:00:00") == 90000  # GTFS allows hours >= 24
    assert parse_gtfs_time("") is None
    assert parse_gtfs_time("bad") is None


def test_load_feed_counts():
    feed = load_feed(str(SAMPLE))
    assert len(feed.stops) == 6
    assert feed.route_ids() == ["M1"]
    assert feed.directions_for("M1") == [0, 1]
    assert len(feed.trips_for("M1", 0)) == 2


def test_station_order_and_count_direction0():
    built = build_line_config(str(SAMPLE), "M1", direction_id=0)
    assert built.line.n_stations == 6
    assert built.station_names == EXPECTED
    # inter-station run-times inferred from the timetable, one per segment.
    assert len(built.segment_times) == 5
    assert built.segment_times[0] == pytest.approx(90.0)
    assert all(t > 0 for t in built.segment_times)


def test_direction1_is_reversed():
    built = build_line_config(str(SAMPLE), "M1", direction_id=1)
    assert built.station_names == list(reversed(EXPECTED))


def test_built_line_config_is_runnable():
    built = build_line_config(str(SAMPLE), "M1", 0)
    line = Line(built.line)  # must not raise
    assert line.n_stations == 6
    assert [s.name for s in line.stations] == EXPECTED


def test_unknown_route_raises():
    with pytest.raises(ValueError):
        build_line_config(str(SAMPLE), "NOPE", 0)


def test_describe_feed_mentions_route_and_stops():
    text = describe_feed(str(SAMPLE))
    assert "route_id=M1" in text
    assert "Riverside" in text
    assert "direction 0" in text and "direction 1" in text


def _run_cli(args):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "metroflow", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def test_gtfs_info_cli():
    proc = _run_cli(["gtfs-info", str(SAMPLE)])
    assert proc.returncode == 0, proc.stderr
    assert "route_id=M1" in proc.stdout
    assert "Airport" in proc.stdout


def test_simulate_gtfs_cli(tmp_path):
    out = tmp_path / "run.json"
    proc = _run_cli(
        [
            "simulate",
            "--gtfs",
            str(SAMPLE),
            "--route",
            "M1",
            "--direction",
            "0",
            "--seed",
            "42",
            "--json",
            str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "Built line from GTFS" in proc.stdout
    import json

    payload = json.loads(out.read_text())
    assert payload["controller"] == "predictive"


def test_simulate_gtfs_requires_route():
    proc = _run_cli(["simulate", "--gtfs", str(SAMPLE), "--seed", "42"])
    assert proc.returncode != 0
    assert "--route is required" in proc.stderr


def test_simulate_end_to_end_from_gtfs():
    cfg = load_config(None)
    cfg.horizon = 1200.0  # keep the test fast
    cfg.seed = 42
    built = apply_gtfs(cfg, str(SAMPLE), "M1", 0)
    assert cfg.line.n_stations == 6
    assert built.line.station_names == EXPECTED
    sim = run_simulation(cfg, "predictive", 42)
    summary = sim.summary()
    assert summary["controller"] == "predictive"
    assert summary["passengers_generated"] > 0
    assert "total_denied_boardings" in summary


# ---------------------------------------------------------------------------
# gtfs-export: scenario files generated from a feed
# ---------------------------------------------------------------------------


def test_export_scenario_roundtrip(tmp_path):
    """An exported scenario must reload into the exact same line."""
    from metroflow.gtfs import export_scenario

    text = export_scenario(str(SAMPLE), "M1", 0, name="sample_m1", source_note="unit test")
    out = tmp_path / "sample_m1.yaml"
    out.write_text(text, encoding="utf-8")

    cfg = load_config(str(out))
    assert cfg.name == "sample_m1"
    assert cfg.line.n_stations == 6
    assert cfg.line.station_names == EXPECTED
    # The honesty header must state what is real vs synthetic.
    assert "REAL (timetable-derived)" in text
    assert "SYNTHETIC" in text
    assert "unit test" in text


def test_export_scenario_simulates(tmp_path):
    """A generated scenario must run end-to-end like any hand-written one."""
    from metroflow.gtfs import export_scenario

    out = tmp_path / "m1.yaml"
    out.write_text(export_scenario(str(SAMPLE), "M1", 0), encoding="utf-8")
    cfg = load_config(str(out))
    cfg.horizon = 1200.0
    sim = run_simulation(cfg, "predictive", 42)
    assert sim.summary()["passengers_generated"] > 0


def test_cli_gtfs_export_writes_file(tmp_path):
    out = tmp_path / "exported.yaml"
    proc = _run_cli(
        ["gtfs-export", str(SAMPLE), "--route", "M1", "--direction", "0", "--out", str(out)]
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    cfg = load_config(str(out))
    assert cfg.line.n_stations == 6


def test_cli_gtfs_export_unknown_route_fails_cleanly():
    proc = _run_cli(["gtfs-export", str(SAMPLE), "--route", "NOPE"])
    assert proc.returncode == 2
    assert "error:" in proc.stderr


def test_load_feed_route_filter_matches_full_load():
    """The big-feed push-down must not change what gets built."""
    full = load_feed(str(SAMPLE))
    filtered = load_feed(str(SAMPLE), route_ids={"M1"})
    # Trips outside the filter are gone, M1 trips identical.
    assert {t["route_id"] for t in filtered.trips} == {"M1"}
    m1_full = full.trips_for("M1", 0)
    assert filtered.trips_for("M1", 0) == m1_full
    for tid in m1_full:
        assert filtered.stop_times[tid] == full.stop_times[tid]
    # Stops/routes stay complete (labels, gtfs-info).
    assert filtered.stops == full.stops
    assert [r["route_id"] for r in filtered.routes] == [r["route_id"] for r in full.routes]

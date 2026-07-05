"""Edge-case, input-validation and property-based (hypothesis) tests.

Two concerns are covered here:

1. **Clean typed errors.** Bad user input (missing/invalid scenario, unknown
   controller, malformed GTFS, structurally invalid config) must surface as a
   typed :class:`~metroflow.errors.MetroFlowError` and, through the CLI, as a
   one-line message with a non-zero exit code -- never a raw Python traceback.

2. **Invariants.** Property-based tests assert structural invariants of the
   demand model and line topology across a wide range of randomised inputs.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from metroflow.cli import main
from metroflow.config import (
    DemandConfig,
    LineConfig,
    SimConfig,
    validate_config,
)
from metroflow.demand import DemandModel
from metroflow.errors import ConfigError, GtfsError, MetroFlowError, ScenarioFileError
from metroflow.line import DOWN, UP, Line

# --------------------------------------------------------------------------- #
# Clean typed errors (library level)
# --------------------------------------------------------------------------- #


def test_missing_scenario_file_raises_scenario_error():
    from metroflow.config import load_config

    with pytest.raises(ScenarioFileError):
        load_config("/nonexistent/does_not_exist.yaml")


def test_malformed_yaml_raises_scenario_error(tmp_path):
    from metroflow.config import load_config

    bad = tmp_path / "bad.yaml"
    bad.write_text("this: : : not valid yaml\n  - broken")
    with pytest.raises(ScenarioFileError):
        load_config(str(bad))


def test_non_mapping_yaml_raises_config_error(tmp_path):
    from metroflow.config import load_config

    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n- 3\n")
    with pytest.raises(ConfigError):
        load_config(str(bad))


def test_wrong_scalar_type_raises_config_error(tmp_path):
    from metroflow.config import load_config

    # A list where a number is expected must be caught cleanly, not crash later
    # in validation with a raw TypeError.
    bad = tmp_path / "typed.yaml"
    bad.write_text("name: x\nhorizon: [1, 2, 3]\n")
    with pytest.raises(ConfigError):
        load_config(str(bad))

    bad2 = tmp_path / "typed2.yaml"
    bad2.write_text('name: x\nn_initial_trains: "lots"\n')
    with pytest.raises(ConfigError):
        load_config(str(bad2))


def test_list_valued_segment_time_still_accepted(tmp_path):
    from metroflow.config import load_config

    ok = tmp_path / "ok.yaml"
    ok.write_text("name: y\nline:\n  n_stations: 4\n  segment_time: [80, 90, 100]\n")
    cfg = load_config(str(ok))
    assert cfg.line.n_stations == 4


def test_invalid_line_config_raises_config_error():
    # Only one station: no segments -> structurally invalid.
    cfg = SimConfig(name="x", line=LineConfig(n_stations=1))
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_typed_errors_are_metroflow_errors():
    # Both are catchable under the common base, so the CLI can handle them
    # uniformly, and both are also ValueErrors for interop.
    assert issubclass(ConfigError, MetroFlowError)
    assert issubclass(GtfsError, MetroFlowError)
    assert issubclass(ConfigError, ValueError)


def test_bad_gtfs_directory_raises_gtfs_error():
    from metroflow.gtfs import load_feed

    with pytest.raises(GtfsError):
        load_feed("/nonexistent/gtfs/dir")


# --------------------------------------------------------------------------- #
# Clean typed errors (CLI level) -- no traceback, non-zero exit
# --------------------------------------------------------------------------- #


def _run_main(argv):
    """Invoke the CLI in-process, capturing stderr and the exit code."""
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = main(argv)
    except SystemExit as exc:  # argparse-style exits (unknown controller, etc.)
        code = exc.code if isinstance(exc.code, int) else 1
        if exc.code and not isinstance(exc.code, int):
            err.write(str(exc.code))
    return code, err.getvalue()


def test_cli_missing_scenario_is_clean_error():
    code, err = _run_main(["simulate", "--scenario", "/no/such/file.yaml"])
    assert code != 0
    assert "error:" in err.lower() or "not found" in err.lower()
    assert "Traceback" not in err


def test_cli_unknown_controller_is_clean_error():
    # argparse rejects an out-of-choices controller before our code runs.
    code, err = _run_main(["simulate", "--controller", "does_not_exist"])
    assert code != 0
    assert "Traceback" not in err


def test_cli_gtfs_without_route_is_clean_error(tmp_path):
    code, err = _run_main(["simulate", "--gtfs", str(tmp_path), "--controller", "baseline"])
    assert code != 0
    assert "--route" in err or "route" in err.lower()
    assert "Traceback" not in err


def test_cli_bad_gtfs_dir_is_clean_error(tmp_path):
    code, err = _run_main(
        [
            "simulate",
            "--gtfs",
            str(tmp_path / "missing"),
            "--route",
            "R1",
            "--controller",
            "baseline",
        ]
    )
    assert code != 0
    assert "Traceback" not in err


# --------------------------------------------------------------------------- #
# Property-based invariants
# --------------------------------------------------------------------------- #


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_stations=st.integers(min_value=3, max_value=40),
    station=st.integers(min_value=0, max_value=39),
)
def test_direction_split_is_a_valid_probability(n_stations, station):
    station = min(station, n_stations - 1)
    dm = DemandModel(DemandConfig(), n_stations)
    up, down = dm.direction_split(station)
    assert 0.0 <= up <= 1.0
    assert 0.0 <= down <= 1.0
    assert up + down == pytest.approx(1.0, abs=1e-9)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_stations=st.integers(min_value=3, max_value=30),
    t=st.floats(min_value=0.0, max_value=86400.0, allow_nan=False),
)
def test_arrival_rate_is_non_negative(n_stations, t):
    dm = DemandModel(DemandConfig(), n_stations)
    for s in range(n_stations):
        assert dm.rate(s, t) >= 0.0


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_stations=st.integers(min_value=3, max_value=40))
def test_next_station_stays_in_bounds_and_reverses_at_termini(n_stations):
    line = Line(LineConfig(n_stations=n_stations))
    # Interior stations advance by exactly one in the travel direction.
    for s in range(n_stations):
        for d in (UP, DOWN):
            nxt = line.next_station(s, d)
            if line.is_terminus(s, d):
                assert nxt is None
            else:
                assert nxt == s + d
                assert 0 <= nxt <= n_stations - 1


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_stations=st.integers(min_value=3, max_value=25),
    seg=st.floats(min_value=10.0, max_value=300.0, allow_nan=False),
)
def test_station_coords_are_monotone_increasing(n_stations, seg):
    line = Line(LineConfig(n_stations=n_stations, segment_time=seg, segment_length=seg * 8))
    coords = line.station_coord
    assert coords[0] == 0.0
    assert all(coords[i] < coords[i + 1] for i in range(len(coords) - 1))
    assert line.length_m == pytest.approx(coords[-1])

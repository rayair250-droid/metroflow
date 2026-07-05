import pytest

from metroflow.config import LineConfig
from metroflow.line import DOWN, UP, Line


def test_construction_and_names():
    line = Line(LineConfig(n_stations=12, segment_time=90))
    assert line.n_stations == 12
    assert [s.index for s in line.stations] == list(range(12))
    assert line.stations[0].name == "S00"
    assert len(line.segment_times) == 11
    assert all(t == 90 for t in line.segment_times)


def test_termini_and_next_station():
    line = Line(LineConfig(n_stations=6))
    assert line.is_terminus(5, UP) is True
    assert line.is_terminus(0, DOWN) is True
    assert line.is_terminus(3, UP) is False
    assert line.next_station(2, UP) == 3
    assert line.next_station(2, DOWN) == 1
    # A terminus in the running direction has no next station (reversal needed).
    assert line.next_station(5, UP) is None
    assert line.next_station(0, DOWN) is None


def test_segment_index_and_travel_time():
    line = Line(LineConfig(n_stations=5, segment_time=[10, 20, 30, 40]))
    assert line.segment_index(0, UP) == 0
    assert line.segment_index(3, UP) == 3
    assert line.segment_index(2, DOWN) == 1
    assert line.travel_time(0, UP) == 10
    assert line.travel_time(2, DOWN) == 20
    # At a terminus there is no outbound segment.
    assert line.segment_index(4, UP) is None
    assert line.travel_time(4, UP) == 0.0


def test_cycle_time():
    line = Line(LineConfig(n_stations=4, segment_time=[10, 20, 30]))
    assert line.cycle_time() == pytest.approx(2 * 60)


def test_invalid_configs():
    with pytest.raises(ValueError):
        Line(LineConfig(n_stations=2))
    with pytest.raises(ValueError):
        Line(LineConfig(n_stations=5, segment_time=[1, 2]))
    with pytest.raises(ValueError):
        Line(LineConfig(n_stations=5, depot_station=9))

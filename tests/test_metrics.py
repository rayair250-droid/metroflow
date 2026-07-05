from metroflow.line import UP
from metroflow.metrics import (
    DepartureRecord,
    InjectionRecord,
    MetricsCollector,
    QueueSample,
)


def test_headway_stats_regular():
    m = MetricsCollector(4)
    for t in (0.0, 100.0, 200.0, 300.0):
        m.record_departure(DepartureRecord(t, 0, UP, load=0, boarded=0, alighted=0, denied=0))
    mean, std, cv = m.headway_stats()
    assert mean == 100.0
    assert std == 0.0
    assert cv == 0.0


def test_headway_stats_irregular():
    m = MetricsCollector(4)
    for t in (0.0, 50.0, 250.0):  # headways 50, 200
        m.record_departure(DepartureRecord(t, 0, UP, load=0, boarded=0, alighted=0, denied=0))
    mean, std, cv = m.headway_stats()
    assert mean == 125.0
    assert std == 75.0
    assert cv == 0.6


def test_queue_stats():
    m = MetricsCollector(4)
    for length in (10, 20, 30):
        m.record_queue(QueueSample(0.0, 0, UP, length))
    q_max, q_mean = m.queue_stats()
    assert q_max == 30.0
    assert q_mean == 20.0


def test_wait_stats():
    m = MetricsCollector(4)
    for w in range(0, 101, 10):  # 0..100
        m.record_boarding(float(w))
    mean, p90, mx = m.wait_stats()
    assert mean == 50.0
    assert mx == 100.0
    assert 88.0 <= p90 <= 92.0


def test_denied_and_injection_recording():
    m = MetricsCollector(4)
    m.record_denied(5)
    m.record_denied(3)
    assert m.denied_boardings == 8
    m.record_injection(InjectionRecord(10.0, 2, UP, "test"))
    assert m.reserves_used == 1
    assert len(m.injections) == 1


def test_summary_keys():
    m = MetricsCollector(4)
    s = m.summary("baseline", 42, "unit")
    for key in (
        "total_denied_boardings",
        "mean_wait_s",
        "p90_wait_s",
        "max_queue",
        "headway_std_s",
        "bunching_index",
        "reserves_used",
        "incident_count",
    ):
        assert key in s

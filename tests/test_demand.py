import numpy as np

from metroflow.config import DemandConfig
from metroflow.demand import DemandModel


def _model(n=12, **kw):
    return DemandModel(DemandConfig(**kw), n)


def test_rush_exceeds_offpeak():
    m = _model(peaks=[{"center": 7200.0, "width": 2400.0, "amplitude": 1.25}])
    peak = m.temporal_profile(7200.0)
    offpeak = m.temporal_profile(0.0)
    assert peak > offpeak
    # Peak should be materially higher than the baseline.
    assert peak > 1.5 * offpeak


def test_central_attraction_higher_than_edge():
    m = _model()
    # Default attraction weights bulge toward the centre.
    assert m.attract[6] > m.attract[0]
    assert m.attract[6] > m.attract[11]


def test_rate_scales_with_time_and_position():
    m = _model()
    r_peak = m.rate(6, 7200.0)
    r_off = m.rate(6, 0.0)
    assert r_peak > r_off > 0


def test_generate_bin_deterministic_and_valid():
    m1 = _model()
    m2 = _model()
    out1 = m1.generate_bin(4, 7000.0, 7015.0, np.random.default_rng(123))
    out2 = m2.generate_bin(4, 7000.0, 7015.0, np.random.default_rng(123))
    assert [(p.arrival, p.dest) for p in out1] == [(p.arrival, p.dest) for p in out2]
    for p in out1:
        assert p.origin == 4
        assert p.dest != 4
        assert 0 <= p.dest < 12
        assert 7000.0 <= p.arrival < 7015.0


def test_surge_increases_rate():
    m = _model()
    t = 3000.0
    base = m.rate(3, t)
    m.add_surge(3, until=4000.0, multiplier=5.0)
    assert m.rate(3, t) == base * 5.0
    # Surge has expired by t=4500 -> rate matches an un-surged model at the same time.
    assert m.rate(3, 4500.0) == _model().rate(3, 4500.0)
    # A different station is unaffected by the surge.
    assert m.rate(4, t) == _model().rate(4, t)

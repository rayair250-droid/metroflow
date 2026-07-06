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


# ---------------------------------------------------------------------------
# Named spatial demand profiles
# ---------------------------------------------------------------------------


def _model_with_profile(profile, n=21):
    from metroflow.config import DemandConfig
    from metroflow.demand import DemandModel

    return DemandModel(DemandConfig(profile=profile), n)


def test_profile_default_equals_metro_commuter():
    import numpy as np

    a = _model_with_profile(None)
    b = _model_with_profile("metro_commuter")
    assert np.allclose(a.origin_w, b.origin_w)
    assert np.allclose(a.attract, b.attract)


def test_profile_rer_bidirectional_shape():
    m = _model_with_profile("rer_bidirectional")
    mid, end = len(m.origin_w) // 2, 0
    # Origins: suburbs (both termini) generate more than the centre.
    assert m.origin_w[end] > m.origin_w[mid]
    assert m.origin_w[-1] > m.origin_w[mid]
    # Attraction: the central trunk pulls hardest.
    assert m.attract[mid] > m.attract[end]


def test_profile_intercity_endpoint_shape():
    m = _model_with_profile("intercity_endpoint")
    mid = len(m.origin_w) // 2
    assert m.origin_w[0] > m.origin_w[mid]
    assert m.attract[0] > m.attract[mid]
    assert m.attract[-1] > m.attract[mid]


def test_profile_unknown_rejected():
    import pytest

    from metroflow.errors import ConfigError

    with pytest.raises(ConfigError):
        _model_with_profile("teleportation")


def test_profile_explicit_weights_win():
    import numpy as np

    from metroflow.config import DemandConfig
    from metroflow.demand import DemandModel

    cfg = DemandConfig(profile="rer_bidirectional", origin_weights=[1.0] * 5)
    m = DemandModel(cfg, 5)
    assert np.allclose(m.origin_w, 1.0)  # explicit list overrides the profile
    assert m.attract[2] > m.attract[0]  # attraction still from the profile

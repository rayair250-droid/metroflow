from metroflow.simulation import run_simulation


def _summary(cfg, controller, seed):
    sim = run_simulation(cfg, controller, seed)
    return sim.metrics.summary(controller, seed, cfg.name)


def test_determinism_same_seed(fast_config):
    a = _summary(fast_config, "predictive", 42)
    b = _summary(fast_config, "predictive", 42)
    assert a == b


def test_different_seeds_differ(fast_config):
    a = _summary(fast_config, "predictive", 1)
    b = _summary(fast_config, "predictive", 2)
    assert a != b


def test_baseline_never_injects(fast_config):
    s = _summary(fast_config, "baseline", 42)
    assert s["reserves_used"] == 0


def test_reactive_and_predictive_inject(fast_config):
    assert _summary(fast_config, "reactive", 42)["reserves_used"] > 0
    assert _summary(fast_config, "predictive", 42)["reserves_used"] > 0


def test_basic_sanity(fast_config):
    s = _summary(fast_config, "predictive", 42)
    assert s["passengers_boarded"] > 0
    assert s["headway_mean_s"] > 0
    assert s["passengers_generated"] >= s["passengers_boarded"]


def test_incidents_identical_across_controllers(stress_config):
    """Fixed seed => same arrivals & incidents; only the controller changes."""
    base = _summary(stress_config, "baseline", stress_config.seed)
    pred = _summary(stress_config, "predictive", stress_config.seed)
    assert base["incident_count"] == pred["incident_count"]


def test_value_predictive_beats_baseline(stress_config):
    """The money shot: predictive strictly reduces denied boardings."""
    seed = stress_config.seed
    base = _summary(stress_config, "baseline", seed)
    pred = _summary(stress_config, "predictive", seed)
    assert pred["total_denied_boardings"] < base["total_denied_boardings"]

"""Shared fixtures for the MetroFlow test-suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from metroflow.config import SimConfig, load_config

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "scenarios"


@pytest.fixture
def scenarios_dir() -> Path:
    return SCENARIOS


@pytest.fixture
def fast_config() -> SimConfig:
    """A small, quick configuration that still forces injections."""
    cfg = SimConfig(name="fast")
    cfg.horizon = 3600
    cfg.demand.arrival_scale = 0.09
    cfg.demand.peaks = [{"center": 1800.0, "width": 900.0, "amplitude": 1.4}]
    return cfg


@pytest.fixture
def stress_config() -> SimConfig:
    """The committed heavy scenario, used for the value assertion.

    Loading the YAML also exercises the config loader end-to-end.
    """
    return load_config(str(SCENARIOS / "rush_incident.yaml"))

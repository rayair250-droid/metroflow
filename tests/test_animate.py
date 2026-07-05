"""Tests for the sober GIF animation (metroflow/animate.py).

Kept fast and deterministic: a tiny line, a short horizon and only a handful of
frames. We assert a valid, non-empty GIF is produced -- never on pixel content.
"""

from pathlib import Path

import pytest

from metroflow.config import LineConfig, SimConfig

pytest.importorskip("PIL", reason="Pillow is required to write the animation GIF")

from metroflow.animate import render_animation, render_comparison_animation  # noqa: E402

GIF_MAGIC = (b"GIF87a", b"GIF89a")


def _tiny_config() -> SimConfig:
    cfg = SimConfig()
    cfg.name = "tiny"
    cfg.horizon = 600.0
    cfg.n_initial_trains = 2
    cfg.depot_reserve = 1
    cfg.line = LineConfig(n_stations=4, segment_time=60.0)
    cfg.incidents.enabled = False
    return cfg


def test_render_produces_valid_gif(tmp_path):
    out = tmp_path / "run.gif"
    cfg = _tiny_config()
    path = render_animation(cfg, "predictive", seed=42, out_path=str(out), seconds=1.0, fps=4)
    p = Path(path)
    assert p.exists()
    data = p.read_bytes()
    assert len(data) > 0
    assert data[:6] in GIF_MAGIC  # valid GIF header
    assert len(data) < 5 * 1024 * 1024  # comfortably under 5 MB


def test_render_is_deterministic(tmp_path):
    cfg = _tiny_config()
    a = tmp_path / "a.gif"
    b = tmp_path / "b.gif"
    render_animation(cfg, "baseline", 7, str(a), seconds=1.0, fps=4)
    render_animation(_tiny_config(), "baseline", 7, str(b), seconds=1.0, fps=4)
    assert a.read_bytes() == b.read_bytes()


def test_comparison_produces_valid_gif(tmp_path):
    out = tmp_path / "compare.gif"
    cfg = _tiny_config()
    path = render_comparison_animation(cfg, seed=42, out_path=str(out), seconds=1.0, fps=4)
    p = Path(path)
    assert p.exists()
    data = p.read_bytes()
    assert len(data) > 0
    assert data[:6] in GIF_MAGIC  # valid GIF header
    assert len(data) < 5 * 1024 * 1024  # comfortably under 5 MB


def test_comparison_is_deterministic(tmp_path):
    a = tmp_path / "a.gif"
    b = tmp_path / "b.gif"
    render_comparison_animation(_tiny_config(), 7, str(a), seconds=1.0, fps=4)
    render_comparison_animation(_tiny_config(), 7, str(b), seconds=1.0, fps=4)
    assert a.read_bytes() == b.read_bytes()


def test_animate_cli(tmp_path):
    """The `animate` CLI produces a valid GIF from a small scenario file."""
    import subprocess
    import sys

    scenario = tmp_path / "tiny.yaml"
    scenario.write_text(
        "name: tiny\nhorizon: 600\nn_initial_trains: 2\ndepot_reserve: 1\n"
        "line:\n  n_stations: 4\n  segment_time: 60\n"
        "incidents:\n  enabled: false\n"
    )
    out = tmp_path / "cli.gif"
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "metroflow",
            "animate",
            "--scenario",
            str(scenario),
            "--controller",
            "baseline",
            "--seed",
            "42",
            "--out",
            str(out),
            "--seconds",
            "1",
            "--fps",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert out.read_bytes()[:6] in GIF_MAGIC

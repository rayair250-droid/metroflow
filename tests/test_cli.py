import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIO = ROOT / "scenarios" / "rush_incident.yaml"


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "metroflow", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def test_compare_cli_emits_valid_json(tmp_path):
    out = tmp_path / "cmp.json"
    proc = _run(
        [
            "compare",
            "--scenario",
            str(SCENARIO),
            "--seed",
            "42",
            "--json",
            str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert set(payload["results"]) == {"baseline", "reactive", "predictive"}
    for name in ("baseline", "reactive", "predictive"):
        assert "total_denied_boardings" in payload["results"][name]
    assert "denied_boardings_delta" in payload
    assert "predictive" in payload["denied_boardings_delta"]


def test_simulate_cli(tmp_path):
    out = tmp_path / "run.json"
    proc = _run(
        [
            "simulate",
            "--scenario",
            str(SCENARIO),
            "--controller",
            "predictive",
            "--seed",
            "42",
            "--json",
            str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["controller"] == "predictive"
    assert payload["seed"] == 42


def test_plots_written(tmp_path):
    outdir = tmp_path / "figs"
    proc = _run(
        [
            "simulate",
            "--scenario",
            str(SCENARIO),
            "--controller",
            "baseline",
            "--seed",
            "42",
            "--plots",
            str(outdir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "load_heatmap.png").exists()
    assert (outdir / "queues.png").exists()


def test_help_and_version():
    assert _run(["--version"]).returncode == 0
    assert _run(["--help"]).returncode == 0

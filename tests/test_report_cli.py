"""Coverage-focused tests for the text/JSON reporting and the CLI commands.

The CLI is driven **in-process** (calling ``metroflow.cli.main`` directly rather
than through a subprocess) so these paths count toward coverage and so the exact
return codes and side effects (JSON files, PNGs) can be asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

from metroflow.cli import main
from metroflow.report import (
    comparison_payload,
    format_comparison,
    format_summary,
)
from metroflow.simulation import run_simulation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = str(ROOT / "scenarios" / "default.yaml")


def _tiny_summaries():
    """Fast three-controller summaries on a short-horizon default scenario."""
    from metroflow.config import load_config

    cfg = load_config(DEFAULT)
    cfg.horizon = 1800
    return [run_simulation(cfg, c, 42).summary() for c in ("baseline", "reactive", "predictive")]


# --------------------------------------------------------------------------- #
# report.py
# --------------------------------------------------------------------------- #


def test_format_summary_contains_all_rows():
    s = _tiny_summaries()[0]
    text = format_summary(s)
    assert "MetroFlow run:" in text
    assert "Denied boardings" in text
    assert "Passengers boarded" in text
    # One header line + separator + one line per reported row.
    assert text.count("\n") >= 15


def test_format_comparison_aligned_table_and_deltas():
    summaries = _tiny_summaries()
    text = format_comparison(summaries)
    assert "MetroFlow comparison" in text
    for c in ("baseline", "reactive", "predictive"):
        assert c in text
    # Headline delta block is present and reports a percentage.
    assert "Change in denied boardings vs baseline" in text
    assert "%" in text


def test_format_comparison_custom_baseline():
    summaries = _tiny_summaries()
    text = format_comparison(summaries, baseline="reactive")
    assert "vs reactive" in text


def test_comparison_payload_structure_and_deltas():
    summaries = _tiny_summaries()
    payload = comparison_payload(summaries)
    assert set(payload["results"]) == {"baseline", "reactive", "predictive"}
    assert "denied_boardings_delta" in payload
    d = payload["denied_boardings_delta"]["reactive"]
    assert "delta_vs_baseline" in d and "pct_vs_baseline" in d
    # Delta is consistent with the two absolute values.
    base = payload["results"]["baseline"]["total_denied_boardings"]
    react = payload["results"]["reactive"]["total_denied_boardings"]
    assert d["delta_vs_baseline"] == react - base


def test_comparison_payload_without_baseline_has_no_delta():
    summaries = [s for s in _tiny_summaries() if s["controller"] != "baseline"]
    payload = comparison_payload(summaries)
    assert "denied_boardings_delta" not in payload


# --------------------------------------------------------------------------- #
# cli.py — happy paths, in-process
# --------------------------------------------------------------------------- #


def _short_scenario(tmp_path: Path) -> str:
    """Write a short-horizon copy of the default scenario for quick CLI runs."""
    import yaml

    from metroflow.config import load_config

    load_config(DEFAULT)  # ensure the source parses
    text = (ROOT / "scenarios" / "default.yaml").read_text()
    data = yaml.safe_load(text)
    data["horizon"] = 1800
    p = tmp_path / "short.yaml"
    p.write_text(yaml.safe_dump(data))
    return str(p)


def test_cli_simulate_writes_json_and_plots(tmp_path):
    scen = _short_scenario(tmp_path)
    jpath = tmp_path / "run.json"
    plots = tmp_path / "figs"
    code = main(
        [
            "simulate",
            "--scenario",
            scen,
            "--controller",
            "baseline",
            "--seed",
            "7",
            "--json",
            str(jpath),
            "--plots",
            str(plots),
        ]
    )
    assert code == 0
    payload = json.loads(jpath.read_text())
    assert payload["controller"] == "baseline"
    assert payload["seed"] == 7
    assert (plots / "load_heatmap.png").exists()
    assert (plots / "queues.png").exists()


def test_cli_compare_writes_json_and_plots(tmp_path):
    scen = _short_scenario(tmp_path)
    jpath = tmp_path / "cmp.json"
    plots = tmp_path / "cmpfigs"
    code = main(
        [
            "compare",
            "--scenario",
            scen,
            "--seed",
            "7",
            "--controllers",
            "baseline,predictive",
            "--json",
            str(jpath),
            "--plots",
            str(plots),
        ]
    )
    assert code == 0
    payload = json.loads(jpath.read_text())
    assert set(payload["results"]) == {"baseline", "predictive"}
    assert (plots / "denied_comparison.png").exists()
    assert (plots / "headway_comparison.png").exists()


def test_cli_experiment_writes_json_and_plot(tmp_path):
    scen = _short_scenario(tmp_path)
    jpath = tmp_path / "exp.json"
    plots = tmp_path / "expfigs"
    code = main(
        [
            "experiment",
            "--scenario",
            scen,
            "--seed",
            "7",
            "--controllers",
            "baseline,reactive",
            "--replications",
            "2",
            "--json",
            str(jpath),
            "--plots",
            str(plots),
        ]
    )
    assert code == 0
    payload = json.loads(jpath.read_text())
    assert payload["replications"] == 2
    assert (plots / "experiment_ci.png").exists()


def test_cli_plot_command(tmp_path):
    scen = _short_scenario(tmp_path)
    outdir = tmp_path / "plotout"
    code = main(["plot", "--scenario", scen, "--controller", "baseline", "--plots", str(outdir)])
    assert code == 0
    assert (outdir / "load_heatmap.png").exists()
    assert (outdir / "queues.png").exists()

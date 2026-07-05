"""Monte-Carlo experiment harness with confidence intervals.

Runs ``N`` seeded replications of each controller over one scenario and reports
the mean of each metric with a 95% confidence interval, plus a pairwise
significance test (Welch's t-test) of the primary metric against the baseline.
This turns a single-seed anecdote into a statistically defensible comparison.

Common random numbers
----------------------
The same set of seeds is used for every controller, so replication ``i`` faces
the identical demand realisation and incident schedule regardless of the
dispatch strategy (variance-reduction by common random numbers). The reported
Welch's t-test treats the two samples as independent, which is *conservative*
here because CRN induces positive correlation; a significant Welch result is
therefore trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from metroflow.config import SimConfig
from metroflow.simulation import run_simulation

#: Numeric metrics collected per replication (order = report row order).
METRIC_KEYS: list[str] = [
    "total_denied_boardings",
    "mean_wait_s",
    "p90_wait_s",
    "lost_customer_hours",
    "ejt_proxy_s",
    "regularity_pct",
    "bunching_index",
    "reserves_used",
]

#: Metric the significance test is run on.
PRIMARY_METRIC = "total_denied_boardings"


@dataclass
class MetricStat:
    mean: float
    ci_low: float
    ci_high: float
    std: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> MetricStat:
    """Mean and two-sided t-based confidence interval of ``values``."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return MetricStat(mean, mean, mean, 0.0, n)
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half = tcrit * se
    return MetricStat(mean, mean - half, mean + half, sd, n)


def run_experiment(
    cfg: SimConfig,
    controllers: list[str],
    replications: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Run replications and return ``{controller: {metric: value_array}}``.

    Replication ``i`` uses ``seed + i`` for every controller (common random
    numbers).
    """
    seeds = [seed + i for i in range(replications)]
    raw: dict[str, dict[str, list[float]]] = {c: {k: [] for k in METRIC_KEYS} for c in controllers}
    for c in controllers:
        for sd in seeds:
            summary = run_simulation(cfg, c, sd).summary()
            for k in METRIC_KEYS:
                raw[c][k].append(float(summary[k]))
    return {
        c: {k: np.asarray(v, dtype=float) for k, v in metrics.items()} for c, metrics in raw.items()
    }


def summarize(results: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, MetricStat]]:
    return {
        c: {k: confidence_interval(v) for k, v in metrics.items()} for c, metrics in results.items()
    }


@dataclass
class SignificanceResult:
    controller: str
    baseline: str
    metric: str
    mean_delta: float
    pct_delta: float
    t_stat: float
    p_value: float
    significant: bool
    ci_overlap: bool


def significance_vs_baseline(
    results: dict[str, dict[str, np.ndarray]],
    stats_by_ctrl: dict[str, dict[str, MetricStat]],
    baseline: str,
    metric: str = PRIMARY_METRIC,
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Welch's t-test of ``metric`` for each controller against ``baseline``."""
    out: list[SignificanceResult] = []
    if baseline not in results:
        return out
    base_vals = results[baseline][metric]
    base_stat = stats_by_ctrl[baseline][metric]
    for c, metrics in results.items():
        if c == baseline:
            continue
        vals = metrics[metric]
        res = stats.ttest_ind(vals, base_vals, equal_var=False)
        t_stat = float(res.statistic)
        p = float(res.pvalue)
        cstat = stats_by_ctrl[c][metric]
        overlap = not (cstat.ci_high < base_stat.ci_low or base_stat.ci_high < cstat.ci_low)
        mean_delta = cstat.mean - base_stat.mean
        pct = 100.0 * mean_delta / base_stat.mean if base_stat.mean else 0.0
        out.append(
            SignificanceResult(
                controller=c,
                baseline=baseline,
                metric=metric,
                mean_delta=mean_delta,
                pct_delta=pct,
                t_stat=t_stat,
                p_value=p,
                significant=(not math.isnan(p)) and p < alpha,
                ci_overlap=overlap,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_LABELS = {
    "total_denied_boardings": "Denied boardings",
    "mean_wait_s": "Mean wait (s)",
    "p90_wait_s": "P90 wait (s)",
    "lost_customer_hours": "Lost cust. hours",
    "ejt_proxy_s": "EJT proxy (s)",
    "regularity_pct": "Regularity (%)",
    "bunching_index": "Bunching index",
    "reserves_used": "Reserves used",
}


def format_experiment_report(
    cfg_name: str,
    controllers: list[str],
    replications: int,
    seed: int,
    stats_by_ctrl: dict[str, dict[str, MetricStat]],
    significance: list[SignificanceResult],
    baseline: str,
) -> str:
    order = ["baseline", "reactive", "predictive", "optimizer"]
    cols = [c for c in order if c in controllers] + [c for c in controllers if c not in order]
    label_w = 18
    col_w = max(20, max((len(c) for c in cols), default=20))

    def cell(text: str, w: int) -> str:
        return str(text).rjust(w)

    lines = [
        f"MetroFlow Monte-Carlo experiment  scenario={cfg_name}  "
        f"replications={replications}  base_seed={seed}",
        "Values are mean +/- 95% CI over replications (common random numbers).",
        "",
    ]
    header = "Metric".ljust(label_w) + "".join(cell(c, col_w) for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for k in METRIC_KEYS:
        row = _LABELS.get(k, k).ljust(label_w)
        for c in cols:
            st = stats_by_ctrl[c][k]
            row += cell(f"{st.mean:.1f}+/-{st.half_width:.1f}", col_w)
        lines.append(row)

    lines.append("")
    lines.append(
        f"Significance of {_LABELS[PRIMARY_METRIC]} vs {baseline} (Welch's t-test, alpha=0.05):"
    )
    if not significance:
        lines.append("  (no baseline in the controller set)")
    for s in significance:
        verdict = "SIGNIFICANT" if s.significant else "not significant"
        overlap = "CIs overlap" if s.ci_overlap else "CIs disjoint"
        lines.append(
            f"  {s.controller:<11} delta={s.mean_delta:+.1f} "
            f"({s.pct_delta:+.1f}%)  t={s.t_stat:+.2f}  p={s.p_value:.4g}  "
            f"-> {verdict}; {overlap}"
        )
    return "\n".join(lines)


def experiment_payload(
    cfg_name: str,
    controllers: list[str],
    replications: int,
    seed: int,
    stats_by_ctrl: dict[str, dict[str, MetricStat]],
    significance: list[SignificanceResult],
) -> dict:
    return {
        "scenario": cfg_name,
        "replications": replications,
        "base_seed": seed,
        "controllers": controllers,
        "metrics": {
            c: {
                k: {
                    "mean": st.mean,
                    "ci_low": st.ci_low,
                    "ci_high": st.ci_high,
                    "std": st.std,
                    "n": st.n,
                }
                for k, st in metrics.items()
            }
            for c, metrics in stats_by_ctrl.items()
        },
        "significance": [
            {
                "controller": s.controller,
                "baseline": s.baseline,
                "metric": s.metric,
                "mean_delta": s.mean_delta,
                "pct_delta": s.pct_delta,
                "t_stat": s.t_stat,
                "p_value": s.p_value,
                "significant": s.significant,
                "ci_overlap": s.ci_overlap,
            }
            for s in significance
        ],
    }

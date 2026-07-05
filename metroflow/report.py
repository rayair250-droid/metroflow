"""JSON and text-table reporting."""

from __future__ import annotations

import json

# Rows shown in the comparison table: (json_key, label, lower_is_better).
_TABLE_ROWS = [
    ("total_denied_boardings", "Denied boardings", True),
    ("mean_wait_s", "Mean wait (s)", True),
    ("p90_wait_s", "P90 wait (s)", True),
    ("max_queue", "Max queue", True),
    ("mean_queue", "Mean queue", True),
    ("headway_std_s", "Headway std (s)", True),
    ("bunching_index", "Bunching index", True),
    # -- operator KPIs (Axis 4) --------------------------------------------- #
    ("ejt_proxy_s", "EJT proxy (s)", True),
    ("lost_customer_hours", "Lost cust. hours", True),
    ("regularity_pct", "Regularity (%)", False),
    ("capacity_utilization", "Capacity util.", False),
    # -- signalling (Axis 1) ------------------------------------------------ #
    ("signal_holds", "Signal holds", False),
    ("reserves_used", "Reserves used", False),
    ("incident_count", "Incidents", False),
    ("passengers_boarded", "Passengers boarded", False),
]


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def format_summary(summary: dict) -> str:
    lines = [
        f"MetroFlow run: scenario={summary['scenario']} "
        f"controller={summary['controller']} seed={summary['seed']}",
        "-" * 56,
    ]
    for key, label, _ in _TABLE_ROWS:
        lines.append(f"  {label:<22} {summary[key]:>14}")
    return "\n".join(lines)


def format_comparison(summaries: list[dict], baseline: str = "baseline") -> str:
    """Render an aligned table comparing controllers on the same seed/scenario."""
    order = ["baseline", "reactive", "predictive", "optimizer"]
    by_name: dict[str, dict] = {s["controller"]: s for s in summaries}
    cols = [c for c in order if c in by_name] + [c for c in by_name if c not in order]

    scenario = summaries[0]["scenario"]
    seed = summaries[0]["seed"]
    header = ["Metric"] + cols
    widths = [24] + [max(12, len(c)) for c in cols]

    def row(cells: list[str]) -> str:
        return "  ".join(
            str(c).rjust(w) if i else str(c).ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=False))
        )

    lines = [
        f"MetroFlow comparison  scenario={scenario}  seed={seed}",
        row(header),
        row(["-" * w for w in widths]),
    ]
    for key, label, _ in _TABLE_ROWS:
        lines.append(row([label] + [by_name[c][key] for c in cols]))

    # Headline deltas versus baseline.
    if baseline in by_name:
        base = by_name[baseline]
        lines.append("")
        lines.append(f"Change in denied boardings vs {baseline}:")
        for c in cols:
            if c == baseline:
                continue
            b = base["total_denied_boardings"]
            v = by_name[c]["total_denied_boardings"]
            pct = (v - b) / b * 100.0 if b else 0.0
            lines.append(f"  {c:<12} {v:>8}  ({pct:+.1f}%)")
    return "\n".join(lines)


def comparison_payload(summaries: list[dict]) -> dict:
    by_name = {s["controller"]: s for s in summaries}
    payload = {
        "scenario": summaries[0]["scenario"],
        "seed": summaries[0]["seed"],
        "results": by_name,
    }
    if "baseline" in by_name:
        base = by_name["baseline"]["total_denied_boardings"]
        deltas = {}
        for name, s in by_name.items():
            if name == "baseline":
                continue
            v = s["total_denied_boardings"]
            deltas[name] = {
                "denied_boardings": v,
                "delta_vs_baseline": v - base,
                "pct_vs_baseline": round((v - base) / base * 100.0, 2) if base else 0.0,
            }
        payload["denied_boardings_delta"] = deltas
    return payload

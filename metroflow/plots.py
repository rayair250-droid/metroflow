"""Matplotlib (Agg) figure generation.

All figures are written to PNG; nothing is shown interactively, so the module
works headless. Import order matters: the backend is forced before pyplot.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metroflow.line import UP  # noqa: E402

# ---------------------------------------------------------------------------- #
# Shared visual style
# ---------------------------------------------------------------------------- #
#: One muted, colour-blind-friendly palette used by every figure, so that the
#: same controller keeps the same colour across plots. Order matches the usual
#: controller ordering (baseline, reactive, predictive, optimizer).
PALETTE = ["#b0413e", "#d98c3f", "#3f8a56", "#2f5d8a"]
_ACCENT = "#7a5195"  # muted purple, for single-series plots
_GRID = "#d5d7db"
_FG = "#2a2a2a"


def apply_style() -> None:
    """Apply the shared MetroFlow figure style via matplotlib rcParams.

    Called at the top of every plotting function so all figures share a sober,
    readable look: muted fonts, a light grid, and consistent sizing. Safe to
    call repeatedly (it only mutates rcParams).
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8a8d91",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.labelcolor": _FG,
            "axes.titlecolor": _FG,
            "text.color": _FG,
            "xtick.color": _FG,
            "ytick.color": _FG,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "grid.color": _GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#c8cacd",
            "font.size": 9.5,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def _controller_colors(names: list[str]) -> list[str]:
    """Stable colour per controller name using the shared palette."""
    order = {"baseline": 0, "reactive": 1, "predictive": 2, "optimizer": 3}
    return [PALETTE[order.get(n, i) % len(PALETTE)] for i, n in enumerate(names)]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


def plot_load_heatmap(sim, path: str, n_time_bins: int = 72) -> str:
    """Time x station average train load heatmap."""
    apply_style()
    _ensure_dir(path)
    n = sim.line.n_stations
    horizon = sim.cfg.horizon
    load_sum = np.zeros((n, n_time_bins))
    load_cnt = np.zeros((n, n_time_bins))
    for rec in sim.metrics.departures:
        b = min(int(rec.t / horizon * n_time_bins), n_time_bins - 1)
        load_sum[rec.station, b] += rec.load
        load_cnt[rec.station, b] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(load_cnt > 0, load_sum / np.maximum(load_cnt, 1), np.nan)

    capacity = float(sim.cfg.train_capacity)
    # Bins with no departure are *absence of a train*, not missing data: render
    # them as a quiet neutral grey instead of distracting white holes.
    cmap = plt.get_cmap("magma").with_extremes(bad="#e4e5e8")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(False)
    im = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=capacity,
        extent=(0.0, horizon / 60.0, -0.5, n - 0.5),
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Station index")
    ax.set_yticks(range(n))
    ax.set_title(
        f"Mean train load by station and time ({sim.controller.name})  "
        "—  grey = no departure in bin"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Passengers on board (capacity = {capacity:.0f})")
    # Mark the crush-capacity end of the scale.
    cbar.ax.axhline(capacity, color="#b0413e", linewidth=1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_queue_over_time(sim, path: str, stations: list[int] | None = None) -> str:
    """Platform queue over time for a few key stations, with injection markers."""
    apply_style()
    _ensure_dir(path)
    n = sim.line.n_stations
    if stations is None:
        mid = n // 2
        stations = sorted({max(1, mid - 2), mid, min(n - 2, mid + 2)})

    series: dict[int, dict[str, list]] = {s: {"t": [], "q": []} for s in stations}
    for smp in sim.metrics.queue_samples:
        if smp.station in series and smp.direction == UP:
            series[smp.station]["t"].append(smp.t / 60.0)
            series[smp.station]["q"].append(smp.length)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(axis="both")
    line_colors = [_ACCENT, "#3f8a56", "#d98c3f"]
    for i, s in enumerate(stations):
        ax.plot(
            series[s]["t"],
            series[s]["q"],
            label=f"Station {s:02d} (up)",
            linewidth=1.5,
            color=line_colors[i % len(line_colors)],
        )
    for inj in sim.metrics.injections:
        ax.axvline(inj.t / 60.0, color="#2f5d8a", linestyle="--", linewidth=1.0, alpha=0.55)
    if sim.metrics.injections:
        ax.axvline(
            sim.metrics.injections[0].t / 60.0,
            color="#2f5d8a",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label="reserve injection",
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Platform queue length (passengers)")
    ax.set_title(f"Platform queue over time ({sim.controller.name})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_headway_comparison(sims_by_name: dict[str, object], path: str) -> str:
    """Two-panel bar chart: headway std and bunching index across controllers.

    Deliberately *not* a dual-axis chart: std (seconds) and the bunching index
    (unitless) live on incomparable scales, and twin axes invite reading one
    bar against the other's axis. Side-by-side panels keep each metric honest.
    """
    apply_style()
    _ensure_dir(path)
    names = list(sims_by_name)
    stds = []
    cvs = []
    for name in names:
        _, std, cv = sims_by_name[name].metrics.headway_stats()  # type: ignore[attr-defined]
        stds.append(std)
        cvs.append(cv)

    colors = _controller_colors(names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))

    for ax, vals, ylab, sub in (
        (ax1, stds, "Headway std (s)", "Spread of departure headways"),
        (ax2, cvs, "Bunching index (std / mean, unitless)", "Scale-free bunching measure"),
    ):
        bars = ax.bar(names, vals, color=colors)
        fmt = "{:.2f}" if max(vals) < 10 else "{:.0f}"
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_ylabel(ylab)
        ax.set_title(sub, fontsize=9.5, fontweight="normal", color="#555555")
        ax.margins(y=0.15)

    fig.suptitle(
        "Headway variability by controller (lower is better)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_bunching(labelled_results, path: str) -> str:
    """Grouped bar chart of early vs late headway CV for several run variants.

    ``labelled_results`` is a list of ``(label, BunchingResult)``; it visualises
    that headway variability grows over the run without control and is suppressed
    by holding.
    """
    apply_style()
    _ensure_dir(path)
    labels = [lbl for lbl, _ in labelled_results]
    early = [r.early_cv for _, r in labelled_results]
    late = [r.late_cv for _, r in labelled_results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, early, width=0.4, color="#3f8a56", label="early (first third)")
    ax.bar(x + 0.2, late, width=0.4, color="#b0413e", label="late (last third)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Headway coefficient of variation (unitless)")
    ax.set_title("Bunching: headway variability, early vs late in the run")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sensitivity(result, path: str) -> str:
    """Line plot of a swept parameter versus an outcome metric."""
    apply_style()
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(axis="both")
    ax.plot(
        result.values,
        result.outcomes,
        marker="o",
        markersize=5,
        linewidth=1.6,
        color=_ACCENT,
    )
    ax.set_xlabel(result.param)
    ax.set_ylabel(result.metric)
    ax.set_title(f"Sensitivity of {result.metric} to {result.param} ({result.controller})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_experiment_ci(stats_by_ctrl, metric: str, path: str) -> str:
    """Bar chart of a metric's mean with 95% CI error bars across controllers."""
    apply_style()
    _ensure_dir(path)
    order = ["baseline", "reactive", "predictive", "optimizer"]
    names = [c for c in order if c in stats_by_ctrl] + [c for c in stats_by_ctrl if c not in order]
    means = [stats_by_ctrl[c][metric].mean for c in names]
    errs = [stats_by_ctrl[c][metric].half_width for c in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        names,
        means,
        yerr=errs,
        capsize=6,
        color=_controller_colors(names),
        error_kw={"ecolor": "#3a3a3a", "elinewidth": 1.1},
    )
    ax.set_ylabel(f"{metric} (mean, error bars = 95% CI)")
    ax.set_title(f"{metric} by controller with 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_denied_comparison(summaries: list[dict], path: str) -> str:
    """Bar chart of total denied boardings across controllers."""
    apply_style()
    _ensure_dir(path)
    order = {"baseline": 0, "reactive": 1, "predictive": 2, "optimizer": 3}
    summaries = sorted(summaries, key=lambda s: order.get(s["controller"], 99))
    names = [s["controller"] for s in summaries]
    denied = [s["total_denied_boardings"] for s in summaries]

    base = next(
        (s["total_denied_boardings"] for s in summaries if s["controller"] == "baseline"),
        None,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, denied, color=_controller_colors(names))
    for b, v, name in zip(bars, denied, names, strict=True):
        label = f"{v:,}"
        # Annotate the improvement (or regression) against the baseline so the
        # headline result is readable straight off the bars.
        if base and name != "baseline":
            pct = (v - base) / base * 100.0
            label += f"\n({pct:+.1f}%)"
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            linespacing=1.3,
        )
    if base:
        ax.axhline(base, color="#b0413e", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_ylabel("Total denied boardings (passengers)")
    ax.set_title("Denied boardings by controller (same seed and scenario)")
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- #
# Multi-panel summary dashboard
# ---------------------------------------------------------------------------- #
_DASH_KPIS = [
    ("total_denied_boardings", "Denied boardings", "{:,.0f}", True),
    ("mean_wait_s", "Mean wait (s)", "{:.1f}", True),
    ("p90_wait_s", "P90 wait (s)", "{:.1f}", True),
    ("max_queue", "Max queue (pax)", "{:,.0f}", True),
    ("headway_std_s", "Headway std (s)", "{:.1f}", True),
    ("regularity_pct", "Regularity (%)", "{:.1f}", False),
    ("capacity_utilization", "Capacity util.", "{:.2f}", False),
]


def _queue_grid(sim, n_time_bins: int = 60):
    """Mean up-direction platform queue on a station x time grid."""
    n = sim.line.n_stations
    horizon = sim.cfg.horizon
    q_sum = np.zeros((n, n_time_bins))
    q_cnt = np.zeros((n, n_time_bins))
    for smp in sim.metrics.queue_samples:
        if smp.direction != UP:
            continue
        b = min(int(smp.t / horizon * n_time_bins), n_time_bins - 1)
        q_sum[smp.station, b] += smp.length
        q_cnt[smp.station, b] += 1
    with np.errstate(invalid="ignore"):
        return np.where(q_cnt > 0, q_sum / np.maximum(q_cnt, 1), np.nan)


def dashboard(sims_by_name: dict, path: str, scenario: str | None = None) -> str:
    """Five-panel summary of a ``compare`` run across controllers.

    Panels: (1) a KPI table with the best value per row highlighted, (2) total
    denied boardings as bars (with % vs baseline), (3) headway regularity (std,
    lower is better), (4) a station x time queue heatmap for the best
    (fewest-denied) controller, and (5) a full-width incident timeline (shared
    across controllers via common random numbers) that explains the spikes.
    """
    apply_style()
    _ensure_dir(path)

    order = {"baseline": 0, "reactive": 1, "predictive": 2, "optimizer": 3}
    names = sorted(sims_by_name, key=lambda c: order.get(c, 99))
    summaries = {c: sims_by_name[c].summary() for c in names}
    colors = _controller_colors(names)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.34],
        width_ratios=[1.15, 1.0],
        hspace=0.42,
        wspace=0.22,
    )
    scen = scenario or summaries[names[0]]["scenario"]
    fig.suptitle(
        f"MetroFlow control-strategy dashboard  —  scenario: {scen}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # --- Panel 1: KPI table ------------------------------------------------- #
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.axis("off")
    ax_tbl.set_title("Key performance indicators", loc="left", pad=10)
    col_labels = ["KPI"] + names
    cell_text = []
    cell_colours = []
    for key, label, fmt, lower_better in _DASH_KPIS:
        vals = [summaries[c].get(key, float("nan")) for c in names]
        row = [label] + [fmt.format(v) for v in vals]
        cell_text.append(row)
        finite = [v for v in vals if v == v]
        best = (min if lower_better else max)(finite) if finite else None
        rowc = ["white"]
        for v in vals:
            rowc.append("#dcecdc" if best is not None and v == best else "white")
        cell_colours.append(rowc)
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c8cacd")
        if r == 0:
            cell.set_facecolor("#eceef0")
            cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    ax_tbl.text(
        0.0,
        -0.04,
        "Green = best value in each row.",
        transform=ax_tbl.transAxes,
        fontsize=7.5,
        color="#666666",
    )

    # --- Panel 2: denied boardings bars ------------------------------------ #
    ax_den = fig.add_subplot(gs[0, 1])
    denied = [summaries[c]["total_denied_boardings"] for c in names]
    den_base = summaries["baseline"]["total_denied_boardings"] if "baseline" in summaries else None
    bars = ax_den.bar(names, denied, color=colors)
    for b, v, name in zip(bars, denied, names, strict=True):
        label = f"{v:,}"
        if den_base and name != "baseline":
            label += f"\n({(v - den_base) / den_base * 100.0:+.1f}%)"
        ax_den.text(
            b.get_x() + b.get_width() / 2,
            v,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=1.25,
        )
    if den_base:
        ax_den.axhline(den_base, color="#b0413e", linewidth=0.9, linestyle=":", alpha=0.7)
    ax_den.set_ylabel("Total denied boardings (passengers)")
    ax_den.set_title("Denied boardings by controller (lower is better)")
    ax_den.margins(y=0.22)

    # --- Panel 3: headway regularity --------------------------------------- #
    ax_hw = fig.add_subplot(gs[1, 0])
    stds = [summaries[c]["headway_std_s"] for c in names]
    ax_hw.bar(names, stds, color=colors)
    ax_hw.set_ylabel("Headway std (s)")
    ax_hw.set_title("Headway variability by controller (std, lower is better)")
    ax_hw.margins(y=0.12)

    # --- Panel 4: queue heatmap for the best controller -------------------- #
    ax_hm = fig.add_subplot(gs[1, 1])
    ax_hm.grid(False)
    best_ctrl = min(names, key=lambda c: summaries[c]["total_denied_boardings"])
    best_sim = sims_by_name[best_ctrl]
    grid = _queue_grid(best_sim)
    n = best_sim.line.n_stations
    horizon = best_sim.cfg.horizon
    # No sample in bin -> quiet grey, not white holes.
    hm_cmap = plt.get_cmap("magma").with_extremes(bad="#e4e5e8")
    im = ax_hm.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        origin="lower",
        cmap=hm_cmap,
        vmin=0.0,
        extent=(0.0, horizon / 60.0, -0.5, n - 0.5),
    )
    ax_hm.set_xlabel("Time (min)")
    ax_hm.set_ylabel("Station index")
    ax_hm.set_yticks(range(0, n, max(1, n // 8)))
    ax_hm.set_title(f"Platform queue, station x time  ({best_ctrl})")
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Queue length (passengers)")

    # --- Panel 5: incident timeline (full width) --------------------------- #
    # Incidents come from a dedicated RNG stream seeded identically for every
    # controller (common random numbers), so the timeline is shared context:
    # it explains *why* the queue/denial spikes above line up in time.
    ax_inc = fig.add_subplot(gs[2, :])
    _incident_timeline(ax_inc, best_sim)

    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# Muted, distinct colour per incident type (shared with any future legend).
_INCIDENT_STYLE = {
    "breakdown": ("#b0413e", "Breakdown"),
    "signal_failure": ("#2f5d8a", "Signal failure"),
    "dwell_event": ("#d98c3f", "Dwell event"),
    "surge": ("#7a5195", "Demand surge"),
}


def _incident_timeline(ax, sim) -> None:
    """Draw a compact one-row-per-type incident timeline on ``ax`` (in minutes)."""
    horizon_min = sim.cfg.horizon / 60.0
    types = list(_INCIDENT_STYLE)
    row_of = {t: i for i, t in enumerate(types)}
    counts = dict.fromkeys(types, 0)

    for inc in sim.metrics.incidents:
        kind = inc.get("type")
        if kind not in row_of:
            continue
        counts[kind] += 1
        color, _ = _INCIDENT_STYLE[kind]
        ax.scatter(
            inc["t"] / 60.0,
            row_of[kind],
            marker="|",
            s=170,
            linewidths=1.6,
            color=color,
            zorder=3,
        )

    ax.set_xlim(0, horizon_min)
    ax.set_ylim(-0.6, len(types) - 0.4)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{_INCIDENT_STYLE[t][1]}  (n={counts[t]})" for t in types], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Time (min)")
    ax.set_title(
        "Incident timeline (identical across controllers — common random numbers)",
        fontsize=9.5,
    )
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

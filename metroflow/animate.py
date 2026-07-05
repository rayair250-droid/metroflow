"""Sober schematic animation of a single MetroFlow run, saved as a GIF.

The figure is intentionally plain -- the kind of schematic you would put in a
transport-engineering report, not a marketing clip. Stations are ticks along a
horizontal axis, trains are dots moving in both directions, each station shows a
small platform-queue bar, and a reserve-train injection produces a brief marker.

Rendering uses matplotlib's ``FuncAnimation`` with the Pillow writer on the
headless ``Agg`` backend. Pillow is imported lazily so the rest of the package
still works if it is not installed; the CLI prints a clear message in that case.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from metroflow.config import SimConfig  # noqa: E402
from metroflow.controllers import make_controller  # noqa: E402
from metroflow.simulation import Simulation  # noqa: E402

# Muted, professional palette (no bright/marketing colours).
_UP_COLOR = "#2f5d8a"  # muted blue: trains running up-line
_DOWN_COLOR = "#8a5a2f"  # muted brown: trains running down-line
_INJECT_COLOR = "#3f7d54"  # muted green: reserve injection marker
_QUEUE_COLOR = "#9aa0a6"  # neutral grey: platform-queue bars
_LINE_COLOR = "#3c3c3c"  # dark grey: the track
_STATION_COLOR = "#5a5a5a"

# Muted green -> amber -> red scale for platform-queue severity. Kept desaturated
# so the figure still reads as a sober engineering schematic, not a heatmap poster.
_SEVERITY_CMAP = LinearSegmentedColormap.from_list(
    "mf_severity", ["#3f7d54", "#c9a227", "#a83232"]
)


class PillowMissingError(RuntimeError):
    """Raised when Pillow (the GIF writer backend) is unavailable."""


def _require_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised via CLI message
        raise PillowMissingError(
            "Pillow is required to write the animation GIF. Install it with "
            "`pip install pillow` (or `pip install -e .[animate]`)."
        ) from exc


def _downsample(items: list, n: int) -> list:
    """Evenly pick at most ``n`` items from ``items`` (keeping first and last)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1)
    idx = sorted({int(round(i * step)) for i in range(n)})
    return [items[i] for i in idx]


def render_animation(
    cfg: SimConfig,
    controller_name: str,
    seed: int,
    out_path: str,
    seconds: float = 8.0,
    fps: int = 10,
) -> str:
    """Run one simulation and write a short, sober schematic GIF to ``out_path``.

    ``seconds`` is the target playback length and ``fps`` the frame rate, so the
    animation shows ``seconds * fps`` frames sampled evenly across the whole run.
    The GIF is deliberately small (low resolution, few frames).
    """
    _require_pillow()

    n_frames = max(2, int(round(seconds * fps)))

    # Use a private copy of the config so we can raise the sampling rate enough
    # to feed the animation without disturbing the caller's config.
    import copy

    cfg = copy.deepcopy(cfg)
    cfg.sample_interval = max(1.0, cfg.horizon / (n_frames * 2))

    controller = make_controller(controller_name, cfg.controller)
    sim = Simulation(cfg, controller, seed)
    sim._record_frames = True
    sim.run()

    frames = _downsample(sim.metrics.frames, n_frames)
    if not frames:
        raise RuntimeError("no frames were recorded (horizon too short?)")

    line = sim.line
    length_m = max(line.length_m, 1.0)
    coords = list(line.station_coord)
    n_stations = line.n_stations

    # Peak queue across the run fixes the bar scale so heights are comparable.
    peak_q = 1.0
    for fr in sim.metrics.frames:
        if fr["queues"]:
            peak_q = max(peak_q, max(fr["queues"].values()))

    # Injection times, matched to frames by a one-frame window.
    inj_times = [inj.t for inj in sim.metrics.injections]
    inj_stations = {inj.t: inj.station for inj in sim.metrics.injections}
    frame_dt = cfg.horizon / max(1, len(frames))

    # --- static figure ----------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    q_span = 0.9  # vertical space allotted to queue bars (below track)
    track_y = 0.0
    ax.set_xlim(-0.03 * length_m, 1.03 * length_m)
    ax.set_ylim(-q_span - 0.35, 0.75)
    ax.axis("off")
    ax.set_autoscale_on(False)  # fixed frame -> much faster redraws

    # Track and station ticks.
    ax.plot([0, length_m], [track_y, track_y], color=_LINE_COLOR, lw=1.4, zorder=1)
    for x in coords:
        ax.plot([x, x], [track_y - 0.06, track_y + 0.06], color=_STATION_COLOR, lw=1.0, zorder=1)
    if n_stations <= 16:
        for i, x in enumerate(coords):
            ax.text(
                x,
                track_y + 0.12,
                line.stations[i].name,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
                color=_STATION_COLOR,
            )
    title = f"MetroFlow  —  {cfg.name} / {controller_name}"
    ax.text(0.0, 0.66, title, fontsize=9, color="#222222", transform=ax.transAxes)

    # Thin legend describing the glyphs (kept small and out of the way).
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_UP_COLOR,
            markeredgecolor=_UP_COLOR,
            markersize=6,
            label="train (up)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=_DOWN_COLOR,
            markersize=6,
            label="train (down)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=_INJECT_COLOR,
            markeredgecolor=_INJECT_COLOR,
            markersize=9,
            label="reserve injected",
        ),
        Line2D([0], [0], color=_QUEUE_COLOR, lw=4, label="platform queue"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.14),
        handletextpad=0.4,
        columnspacing=1.2,
    )

    # Queue-height scale reference: a full-height bar in the left margin tells
    # the viewer how many passengers a full-height queue bar represents, so the
    # relative bars below the track become quantitative.
    scale_x = -0.018 * length_m
    ax.vlines(scale_x, track_y, track_y - q_span, color=_QUEUE_COLOR, linewidth=5.0, zorder=2)
    ax.text(
        scale_x,
        track_y - q_span - 0.05,
        f"= {peak_q:.0f} pax\n(peak queue)",
        ha="center",
        va="top",
        fontsize=5.5,
        color="#555555",
        linespacing=1.1,
    )

    clock = ax.text(0.99, 0.66, "", ha="right", fontsize=9, color="#222222", transform=ax.transAxes)

    # Dynamic artists (rebuilt each frame).
    dyn: list = []

    def _clear_dynamic() -> None:
        while dyn:
            dyn.pop().remove()

    def update(fi: int):
        _clear_dynamic()
        fr = frames[fi]
        t = fr["t"]
        # queue bars, drawn as one thick-line collection (cheap to rebuild)
        qx, qy0, qy1 = [], [], []
        for s in range(n_stations):
            q = fr["queues"].get(s, 0)
            if q <= 0:
                continue
            qx.append(coords[s])
            qy0.append(track_y)
            qy1.append(track_y - (q / peak_q) * q_span if peak_q > 0 else track_y)
        if qx:
            lc = ax.vlines(qx, qy0, qy1, color=_QUEUE_COLOR, linewidth=5.0, zorder=2)
            dyn.append(lc)
        # trains: batch into up / down / injected scatters
        up_x, down_x, inj_x, inj_y = [], [], [], []
        for x, direction, injected in fr["trains"]:
            if injected:
                inj_x.append(x)
                inj_y.append(track_y + (0.16 if direction > 0 else -0.16))
            elif direction > 0:
                up_x.append(x)
            else:
                down_x.append(x)
        if up_x:
            dyn.append(
                ax.scatter(
                    up_x,
                    [track_y + 0.16] * len(up_x),
                    s=34,
                    marker="o",
                    facecolors=_UP_COLOR,
                    edgecolors=_UP_COLOR,
                    linewidths=1.1,
                    zorder=4,
                )
            )
        if down_x:
            dyn.append(
                ax.scatter(
                    down_x,
                    [track_y - 0.16] * len(down_x),
                    s=34,
                    marker="o",
                    facecolors="none",
                    edgecolors=_DOWN_COLOR,
                    linewidths=1.1,
                    zorder=4,
                )
            )
        if inj_x:
            dyn.append(
                ax.scatter(
                    inj_x,
                    inj_y,
                    s=70,
                    marker="*",
                    facecolors=_INJECT_COLOR,
                    edgecolors=_INJECT_COLOR,
                    zorder=4,
                )
            )
        # injection flash (brief marker at the injection station)
        for it in inj_times:
            if t - frame_dt <= it <= t + frame_dt:
                sx = coords[min(inj_stations[it], n_stations - 1)]
                dyn.append(
                    ax.scatter(
                        [sx],
                        [track_y + 0.34],
                        s=130,
                        marker="*",
                        facecolors=_INJECT_COLOR,
                        edgecolors="none",
                        zorder=5,
                    )
                )
                dyn.append(
                    ax.text(
                        sx,
                        track_y + 0.42,
                        "reserve injected",
                        ha="center",
                        fontsize=6,
                        color=_INJECT_COLOR,
                        zorder=5,
                    )
                )
        clock.set_text(f"t = {int(t // 60):02d}:{int(t % 60):02d}")
        return dyn

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / max(1, fps), blit=False)

    from matplotlib.animation import PillowWriter

    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out_path


def animate_from_config(
    cfg: SimConfig,
    controller_name: str,
    seed: int | None,
    out_path: str,
    seconds: float = 8.0,
    fps: int = 10,
) -> str:
    """Thin wrapper resolving the seed from the config when not given."""
    use_seed = cfg.seed if seed is None else seed
    return render_animation(cfg, controller_name, use_seed, out_path, seconds, fps)


# --------------------------------------------------------------------------- #
# Split-screen comparison: baseline vs predictive on the SAME seed
# --------------------------------------------------------------------------- #
def _run_recorded(cfg: SimConfig, controller_name: str, seed: int) -> Simulation:
    """Run one frame-recording simulation. Same ``seed`` => identical arrivals and
    incidents across controllers (the engine spawns independent SeedSequence
    streams for demand/incidents), so the only difference between two runs is the
    dispatch strategy -- exactly the contrast this animation is meant to show."""
    controller = make_controller(controller_name, cfg.controller)
    sim = Simulation(cfg, controller, seed)
    sim._record_frames = True
    sim.run()
    return sim


def _denied_cumulative(sim: Simulation, frame_times: list[float]) -> list[int]:
    """Cumulative denied boardings at each frame time (a live running total)."""
    deps = sorted(sim.metrics.departures, key=lambda r: r.t)
    if not deps:
        return [0 for _ in frame_times]
    import numpy as np

    dep_t = np.asarray([r.t for r in deps], dtype=float)
    cum = np.cumsum(np.asarray([r.denied for r in deps], dtype=float))
    out: list[int] = []
    for t in frame_times:
        idx = int(np.searchsorted(dep_t, t, side="right")) - 1
        out.append(int(cum[idx]) if idx >= 0 else 0)
    return out


def _draw_static_panel(
    ax,
    line,
    coords: list[float],
    length_m: float,
    q_span: float,
    title: str,
) -> None:
    """Draw the immutable parts of one schematic panel (track, ticks, labels)."""
    track_y = 0.0
    n_stations = line.n_stations
    ax.set_xlim(-0.05 * length_m, 1.03 * length_m)
    ax.set_ylim(-q_span - 0.30, 0.72)
    ax.axis("off")
    ax.set_autoscale_on(False)

    ax.plot([0, length_m], [track_y, track_y], color=_LINE_COLOR, lw=1.4, zorder=1)
    for x in coords:
        ax.plot(
            [x, x],
            [track_y - 0.06, track_y + 0.06],
            color=_STATION_COLOR,
            lw=1.0,
            zorder=1,
        )
    if n_stations <= 16:
        for i, x in enumerate(coords):
            ax.text(
                x,
                track_y + 0.12,
                line.stations[i].name,
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=45,
                color=_STATION_COLOR,
            )
    ax.text(0.0, 0.80, title, fontsize=9, color="#222222", transform=ax.transAxes)


def render_comparison_animation(
    cfg: SimConfig,
    seed: int,
    out_path: str,
    seconds: float = 7.0,
    fps: int = 8,
    baseline_controller: str = "baseline",
    predictive_controller: str = "predictive",
) -> str:
    """Render a split-screen ``baseline`` vs ``predictive`` GIF telling the core
    story: predictive reserve-train injection relieving platform saturation.

    Two simulations are run on the SAME ``seed`` and scenario, so passenger
    arrivals and incidents are byte-for-byte identical between them; only the
    dispatch strategy differs. They are drawn as two stacked schematic panels --
    baseline (no injection) on top, predictive (reserve injection) below -- with:

    * platform-queue bars coloured on a shared green->amber->red severity scale
      (keyed to train capacity), so building saturation is visually obvious and
      the two panels are directly comparable;
    * a live "denied boardings" counter on each panel, so the baseline total
      visibly races ahead of the predictive one -- the headline result, live;
    * a prominent injection flash and "reserve injected" label on the predictive
      panel (the baseline panel never injects -- that contrast is the point);
    * a shared clock and a one-line caption naming the mechanism.

    Deterministic for a fixed seed; kept short and small (few frames, dpi 80).
    """
    _require_pillow()

    n_frames = max(2, int(round(seconds * fps)))

    import copy

    cfg = copy.deepcopy(cfg)
    cfg.sample_interval = max(1.0, cfg.horizon / (n_frames * 2))

    base_sim = _run_recorded(cfg, baseline_controller, seed)
    pred_sim = _run_recorded(cfg, predictive_controller, seed)

    # The sampler ticks at the same interval over the same horizon in both runs,
    # so the recorded frame lists line up 1:1; downsample both with one index set.
    n_raw = min(len(base_sim.metrics.frames), len(pred_sim.metrics.frames))
    if n_raw == 0:
        raise RuntimeError("no frames were recorded (horizon too short?)")
    keep = _downsample(list(range(n_raw)), n_frames)
    base_frames = [base_sim.metrics.frames[i] for i in keep]
    pred_frames = [pred_sim.metrics.frames[i] for i in keep]
    frame_times = [fr["t"] for fr in base_frames]

    line = base_sim.line
    length_m = max(line.length_m, 1.0)
    coords = list(line.station_coord)
    n_stations = line.n_stations
    cap_ref = float(max(1, cfg.train_capacity))

    # Shared bar-height scale across BOTH runs so heights stay comparable.
    peak_q = 1.0
    for sim in (base_sim, pred_sim):
        for fr in sim.metrics.frames:
            if fr["queues"]:
                peak_q = max(peak_q, max(fr["queues"].values()))

    base_denied = _denied_cumulative(base_sim, frame_times)
    pred_denied = _denied_cumulative(pred_sim, frame_times)

    # Predictive-panel injections, matched to frames by a one-frame window.
    inj_times = [inj.t for inj in pred_sim.metrics.injections]
    inj_stations = {inj.t: inj.station for inj in pred_sim.metrics.injections}
    frame_dt = cfg.horizon / max(1, len(base_frames))

    q_span = 0.9
    track_y = 0.0

    # --- static figure: two stacked panels --------------------------------- #
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 5.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.12, hspace=0.42)
    _draw_static_panel(ax_top, line, coords, length_m, q_span, "Baseline (no injection)")
    _draw_static_panel(ax_bot, line, coords, length_m, q_span, "Predictive injection")

    # Severity scale reference: a full-height bar keyed to train capacity, so the
    # coloured bars below the track become quantitative and comparable.
    norm = Normalize(vmin=0.0, vmax=cap_ref)
    for ax in (ax_top, ax_bot):
        scale_x = -0.03 * length_m
        ax.vlines(
            scale_x,
            track_y,
            track_y - q_span,
            color=_SEVERITY_CMAP(0.85),
            linewidth=5.0,
            zorder=2,
        )
        ax.text(
            scale_x,
            track_y - q_span - 0.04,
            f"= {peak_q:.0f} pax\n(peak queue)",
            ha="center",
            va="top",
            fontsize=5.0,
            color="#555555",
            linespacing=1.1,
        )

    # Live denied-boardings counters (one per panel) and the shared clock.
    denied_top = ax_top.text(
        0.99, 0.80, "", ha="right", fontsize=8.5, color="#a83232", transform=ax_top.transAxes
    )
    denied_bot = ax_bot.text(
        0.99, 0.80, "", ha="right", fontsize=8.5, color="#a83232", transform=ax_bot.transAxes
    )
    clock = fig.text(0.5, 0.955, "", ha="center", fontsize=10, color="#222222")
    fig.text(
        0.5,
        0.975,
        "MetroFlow — same seed, same arrivals & incidents; only dispatch differs",
        ha="center",
        fontsize=9.5,
        color="#222222",
    )
    fig.text(
        0.5,
        0.028,
        "Mechanism: predictive reserve-train injection relieves saturation, "
        "cutting denied boardings vs a fixed baseline.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )

    # Thin shared legend across the bottom.
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=_UP_COLOR,
            markeredgecolor=_UP_COLOR, markersize=6, label="train (up)",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none",
            markeredgecolor=_DOWN_COLOR, markersize=6, label="train (down)",
        ),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor=_INJECT_COLOR,
            markeredgecolor=_INJECT_COLOR, markersize=9, label="reserve injected",
        ),
        Line2D([0], [0], color=_SEVERITY_CMAP(0.15), lw=4, label="queue: low"),
        Line2D([0], [0], color=_SEVERITY_CMAP(0.95), lw=4, label="queue: saturated"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        handletextpad=0.4,
        columnspacing=1.2,
    )

    # Per-panel dynamic artists, rebuilt each frame.
    dyn_top: list = []
    dyn_bot: list = []

    def _draw_panel(ax, dyn: list, fr: dict, show_injection: bool, t: float) -> None:
        while dyn:
            dyn.pop().remove()
        # queue bars coloured by severity (shared scale keyed to capacity)
        qx, qy0, qy1, qc = [], [], [], []
        for s in range(n_stations):
            q = fr["queues"].get(s, 0)
            if q <= 0:
                continue
            qx.append(coords[s])
            qy0.append(track_y)
            qy1.append(track_y - (q / peak_q) * q_span if peak_q > 0 else track_y)
            qc.append(_SEVERITY_CMAP(norm(min(q, cap_ref))))
        if qx:
            dyn.append(ax.vlines(qx, qy0, qy1, colors=qc, linewidth=5.0, zorder=2))
        # trains: batch up / down / injected
        up_x, down_x, inj_x, inj_y = [], [], [], []
        for x, direction, injected in fr["trains"]:
            if injected:
                inj_x.append(x)
                inj_y.append(track_y + (0.16 if direction > 0 else -0.16))
            elif direction > 0:
                up_x.append(x)
            else:
                down_x.append(x)
        if up_x:
            dyn.append(
                ax.scatter(up_x, [track_y + 0.16] * len(up_x), s=30, marker="o",
                           facecolors=_UP_COLOR, edgecolors=_UP_COLOR, linewidths=1.0, zorder=4)
            )
        if down_x:
            dyn.append(
                ax.scatter(down_x, [track_y - 0.16] * len(down_x), s=30, marker="o",
                           facecolors="none", edgecolors=_DOWN_COLOR, linewidths=1.0, zorder=4)
            )
        if inj_x:
            dyn.append(
                ax.scatter(inj_x, inj_y, s=62, marker="*", facecolors=_INJECT_COLOR,
                           edgecolors=_INJECT_COLOR, zorder=4)
            )
        # injection flash + label (predictive panel only)
        if show_injection:
            for it in inj_times:
                if t - frame_dt <= it <= t + frame_dt:
                    sx = coords[min(inj_stations[it], n_stations - 1)]
                    dyn.append(
                        ax.scatter([sx], [track_y + 0.34], s=150, marker="*",
                                   facecolors=_INJECT_COLOR, edgecolors="none", zorder=5)
                    )
                    dyn.append(
                        ax.text(sx, track_y + 0.44, "reserve injected", ha="center",
                                fontsize=6.5, color=_INJECT_COLOR, zorder=5)
                    )

    def update(fi: int):
        t = frame_times[fi]
        _draw_panel(ax_top, dyn_top, base_frames[fi], show_injection=False, t=t)
        _draw_panel(ax_bot, dyn_bot, pred_frames[fi], show_injection=True, t=t)
        denied_top.set_text(f"denied boardings: {base_denied[fi]:,}")
        denied_bot.set_text(f"denied boardings: {pred_denied[fi]:,}")
        clock.set_text(f"t = {int(t // 60):02d}:{int(t % 60):02d}")
        return dyn_top + dyn_bot

    anim = FuncAnimation(
        fig, update, frames=len(base_frames), interval=1000.0 / max(1, fps), blit=False
    )

    from matplotlib.animation import PillowWriter

    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out_path


def comparison_from_config(
    cfg: SimConfig,
    seed: int | None,
    out_path: str,
    seconds: float = 7.0,
    fps: int = 8,
) -> str:
    """Thin wrapper resolving the seed from the config when not given."""
    use_seed = cfg.seed if seed is None else seed
    return render_comparison_animation(cfg, use_seed, out_path, seconds, fps)

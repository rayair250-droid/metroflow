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

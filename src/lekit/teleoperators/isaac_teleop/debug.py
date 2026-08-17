"""Live debug dashboard for :class:`IsaacTeleop`.

Run with ``python -m lekit.teleoperators.isaac_teleop.debug`` (or directly
``python src/lekit/teleoperators/isaac_teleop/debug.py``) to see an in-place
``rich`` panel showing the controller pose, the instantaneous movement
direction, orientation, and the analog/button inputs. No scrollback spam;
panel updates in place.

The dashboard is the only place this file's helpers are used, so it
imports ``rich`` lazily inside :func:`_run_demo`.
"""

from __future__ import annotations

import time

import numpy as np

from lerobot.types import RobotAction

from .isaac_teleop import IsaacTeleop, IsaacTeleopConfig

# ------------------------------------------------------------------ rendering
# Rich-free helpers for the dashboard below. Kept here (not in
# ``isaac_teleop.py``) so the production module has no dashboard coupling.

# Horizontal-plane heading glyphs, indexed by the 45° sector of
# ``arctan2(y_left, x_forward)``: sector 0 is forward (screen up), rotating
# counter-clockwise toward the operator's left.
_HEADING_ARROWS = ("↑", "↖", "←", "↙", "↓", "↘", "→", "↗")
_HEADING_LABELS = (
    "forward",
    "forward-left",
    "left",
    "back-left",
    "back",
    "back-right",
    "right",
    "forward-right",
)


def _heading(vel: np.ndarray, deadzone: float) -> tuple[str, str]:
    """Map a base-frame velocity to an ``(arrow, label)`` pair for display.

    Args:
        vel: ``(vx_forward, vy_left, vz_up)`` in m/s.
        deadzone: Speed below which an axis reads as noise, so a resting
            controller shows "still" instead of chasing sensor jitter.
    """
    vx, vy, vz = (float(c) for c in vel)
    horizontal = float(np.hypot(vx, vy))
    vertical = "up" if vz > deadzone else "down" if vz < -deadzone else ""

    if horizontal < deadzone:
        if not vertical:
            return "·", "still"
        return ("▲", "up") if vz > 0 else ("▼", "down")

    sector = int(round(float(np.degrees(np.arctan2(vy, vx))) / 45.0)) % 8
    label = _HEADING_LABELS[sector]
    return _HEADING_ARROWS[sector], f"{label}-{vertical}" if vertical else label


def _signed_bar(value: float, span: float, half_width: int = 10) -> str:
    """Center-anchored bar: fills right of the axis for ``+``, left for ``-``.

    ``span`` is the magnitude that saturates one half of the bar.
    """
    cells = [" "] * (2 * half_width + 1)
    cells[half_width] = "│"
    n = int(round(max(-1.0, min(1.0, value / span)) * half_width))
    lo, hi = (half_width + 1, half_width + 1 + n) if n > 0 else (half_width + n, half_width)
    for i in range(lo, hi):
        cells[i] = "█"
    return "".join(cells)


def _meter(value: float, width: int = 12) -> str:
    """Left-anchored ``0..1`` meter."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * filled + "░" * (width - filled)


def _xy_plot(ee_xy: np.ndarray, span_x: float = 0.4, span_y: float = 0.2, cols: int = 31, rows: int = 11):
    """Render a 2D x-y plot as rich ``Text``.

    Integrates the consumer's view of the EE target: ``ee_xy`` is the
    accumulated position in the user convention (m). Cell size is
    ``2 * span_x / (cols - 1)`` wide and ``2 * span_y / (rows - 1)`` tall.
    The crosshair marks the origin (the engage point). The marker is
    clamped to the grid if the EE has wandered off.

    Args:
        ee_xy: ``(x_right, y_front)`` in metres. ``z`` is ignored.
        span_x: Half-width in metres. Total range is ``±span_x``.
        span_y: Half-height in metres. Total range is ``±span_y``.
    """
    from rich.text import Text as RText

    half_c = (cols - 1) // 2
    half_r = (rows - 1) // 2
    cell_x = span_x / half_c
    cell_y = span_y / half_r
    col = int(round(float(ee_xy[0]) / cell_x)) + half_c
    row = half_r - int(round(float(ee_xy[1]) / cell_y))
    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))

    text = RText()
    for r in range(rows):
        for c in range(cols):
            # Marker first so the crosshair can't paint over it.
            if r == row and c == col:
                text.append("●", style="bold red")
            elif r == half_r and c == half_c:
                text.append("┼", style="dim")
            elif r == half_r:
                text.append("─", style="dim")
            elif c == half_c:
                text.append("│", style="dim")
            else:
                text.append(" ")
        if r < rows - 1:
            text.append("\n")
    return text


# --------------------------------------------------------------------- main


def _run_demo() -> None:
    """Live terminal dashboard for eyeballing controller pose and motion.

    Renders one in-place panel (no scrollback spam) showing absolute base-frame
    position, the instantaneous movement direction derived from frame-to-frame
    deltas, orientation, and the analog/button inputs.
    """
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    target_hz = 30.0
    period = 1.0 / target_hz
    vel_alpha = 0.35  # Velocity EMA — smooths jitter without adding much lag.
    deadzone = 0.02  # m/s; below this the controller reads as "still".
    pos_span = 0.05  # Metres/frame that saturate half a bar. At 30 Hz this
    # corresponds to ~1.5 m/s — fast hand motion. Normal teleop is well
    # inside the bar; truly fast moves saturate visibly.

    # (label, name, word for +, word for -). Matches the operator-friendly
    # convention in IsaacTeleop.get_action (NOT the base frame).
    axis_rows = (
        ("X", "right", "right", "left"),
        ("Y", "fwd", "forward", "back"),
        ("Z", "up", "up", "down"),
    )

    def _section(title: str) -> Text:
        return Text(title, style="dim bold")

    def _panel(
        action: RobotAction,
        tracking: bool,
        vel: np.ndarray,
        hz: float,
        step_ms: float,
        ee_xy: np.ndarray,
    ) -> Panel:
        pos = np.asarray(action["grip_pos"], dtype=np.float32)
        roll, pitch, yaw = (float(a) for a in np.degrees(action["grip_ori"]))

        header = Table.grid(padding=(0, 3))
        header.add_row(
            Text("● tracking", style="bold green") if tracking else Text("○ no controller", style="bold red"),
            Text(f"{hz:5.1f} Hz", style="dim"),
            Text(f"step {step_ms:5.2f} ms", style="dim"),
        )

        axes = Table.grid(padding=(0, 2))
        for label, short, pos_word, neg_word in axis_rows:
            value = float(pos["XYZ".index(label)])
            axes.add_row(
                Text(f"{label} {short:<4}", style="bold"),
                Text(f"{value:+.3f}", style="white"),
                Text(_signed_bar(value, pos_span), style="cyan"),
                Text(pos_word if value >= 0 else neg_word, style="dim"),
            )

        arrow, label = _heading(vel, deadzone)
        moving = label != "still"
        accent = "bold cyan" if moving else "dim"
        move = Table.grid(padding=(0, 3))
        move.add_row(
            Text(f" {arrow} ", style=accent),
            Text(f"{label:<14}", style="bold" if moving else "dim"),
            Text(f"{float(np.linalg.norm(vel)) * 100.0:6.1f} cm/s", style=accent),
        )

        orient = Table.grid(padding=(0, 3))
        orient.add_row(
            Text(f"roll {roll:+7.1f}°", style="white"),
            Text(f"pitch {pitch:+7.1f}°", style="white"),
            Text(f"yaw {yaw:+7.1f}°", style="white"),
        )

        squeeze = float(action["squeeze"])
        trigger = float(action["trigger"])
        inputs = Table.grid(padding=(0, 2))
        inputs.add_row(
            Text("squeeze", style="bold"),
            Text(_meter(squeeze), style="green"),
            Text(f"{squeeze:.2f}", style="white"),
        )
        inputs.add_row(
            Text("trigger", style="bold"),
            Text(_meter(trigger), style="yellow"),
            Text(f"{trigger:.2f}", style="white"),
        )
        buttons = Text()
        for name, value in (("A", action["a_button"]), ("B", action["b_button"])):
            pressed = float(value) > 0.5
            buttons.append(f"{name} {'●' if pressed else '○'}   ", style="bold green" if pressed else "dim")

        # 2D x-y trace. The marker is the integrated EE position in user
        # convention; z is dropped. The grid spans ±0.4 m horizontally and
        # ±0.2 m vertically (one typical hand-reach workspace).
        plot = _xy_plot(ee_xy)
        plot_caption = Text(
            "  x → right  ·  y ↑ front  ·  ┼ = origin (engage)  ·  m",
            style="dim",
        )

        return Panel(
            Group(
                header,
                Text(),
                _section("INCREMENT  (m/frame)"),
                axes,
                Text(),
                _section("MOVE  (instantaneous)"),
                move,
                Text(),
                _section("ORIENTATION"),
                orient,
                Text(),
                _section("INPUTS"),
                inputs,
                buttons,
                Text(),
                _section("XY TRACE  (±0.4 m × ±0.2 m, z ignored)"),
                plot,
                plot_caption,
            ),
            title=f"IsaacTeleop · {config.hand_side} hand",
            subtitle="Ctrl-C to quit",
            border_style="green" if tracking else "red",
            expand=False,
            padding=(0, 2),
        )

    config = IsaacTeleopConfig()
    teleop = IsaacTeleop(config)
    console = Console()

    with teleop:
        prev_t: float | None = None
        last_loop_t: float | None = None
        vel = np.zeros(3, dtype=np.float32)
        hz = 0.0
        # Integrated EE target in user convention (m). Starts at 0; the
        # consumer-side integration is simulated here so the XY trace shows
        # where the EE would land if the deltas were summed. z is dropped.
        ee_xy = np.zeros(2, dtype=np.float32)

        with Live(console=console, refresh_per_second=15) as live:
            try:
                while True:
                    loop_t0 = time.perf_counter()
                    action = teleop.get_action()
                    step_ms = (time.perf_counter() - loop_t0) * 1000.0

                    now = time.perf_counter()
                    tracking = teleop.is_tracking
                    # Under plan-B, ``grip_pos`` is already the per-frame
                    # increment (m/frame); dividing by dt gives the
                    # instantaneous velocity in m/s. The previous-position
                    # diff (which made sense under total-from-engage) would
                    # now be a derivative of velocity i.e. acceleration, so
                    # we use the increment directly.
                    inc = np.asarray(action["grip_pos"], dtype=np.float32)
                    if not tracking or prev_t is None or now <= prev_t:
                        prev_t = now
                        vel[:] = 0.0
                    else:
                        dt = now - prev_t
                        v = inc / dt
                        vel += vel_alpha * (v - vel)
                        prev_t = now

                    # Accumulate the per-frame increment into the simulated
                    # EE position. The first engaged frame is (0,0,0) so
                    # engage itself adds nothing; subsequent frames move
                    # the marker. Released / untracked frames are also
                    # (0,0,0), so the marker freezes in place.
                    if tracking:
                        ee_xy[0] += float(inc[0])
                        ee_xy[1] += float(inc[1])

                    if last_loop_t is not None and now > last_loop_t:
                        hz += 0.2 * (1.0 / (now - last_loop_t) - hz)
                    last_loop_t = now

                    live.update(_panel(action, tracking, vel, hz, step_ms, ee_xy))

                    remaining = period - (time.perf_counter() - loop_t0)
                    if remaining > 0:
                        time.sleep(remaining)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    _run_demo()

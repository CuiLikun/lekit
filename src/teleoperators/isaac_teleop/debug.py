"""Live debug dashboard for :class:`IsaacTeleop`.

Run with ``python -m src.teleoperators.isaac_teleop.debug`` (or directly
``python src/teleoperators/isaac_teleop/debug.py``) to see an in-place
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
    pos_span = 1.0  # Metres that saturate half a position bar.

    axis_rows = (  # (label, name, meaning of +, meaning of -)
        ("X", "fwd", "forward", "back"),
        ("Y", "left", "left", "right"),
        ("Z", "up", "up", "down"),
    )

    def _section(title: str) -> Text:
        return Text(title, style="dim bold")

    def _panel(action: RobotAction, tracking: bool, vel: np.ndarray, hz: float, step_ms: float) -> Panel:
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

        return Panel(
            Group(
                header,
                Text(),
                _section("POSITION  (base frame, m)"),
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
        prev_pos: np.ndarray | None = None
        prev_t: float | None = None
        last_loop_t: float | None = None
        vel = np.zeros(3, dtype=np.float32)
        hz = 0.0

        with Live(console=console, refresh_per_second=15) as live:
            try:
                while True:
                    loop_t0 = time.perf_counter()
                    action = teleop.get_action()
                    step_ms = (time.perf_counter() - loop_t0) * 1000.0

                    now = time.perf_counter()
                    tracking = teleop.is_tracking
                    pos = np.asarray(action["grip_pos"], dtype=np.float32)
                    if not tracking:
                        # Untracked frames report zeros; differencing against
                        # them would fabricate a huge spike on re-acquire.
                        prev_pos, prev_t = None, None
                        vel[:] = 0.0
                    else:
                        if prev_pos is not None and prev_t is not None and now > prev_t:
                            vel += vel_alpha * ((pos - prev_pos) / (now - prev_t) - vel)
                        prev_pos, prev_t = pos, now

                    if last_loop_t is not None and now > last_loop_t:
                        hz += 0.2 * (1.0 / (now - last_loop_t) - hz)
                    last_loop_t = now

                    live.update(_panel(action, tracking, vel, hz, step_ms))

                    remaining = period - (time.perf_counter() - loop_t0)
                    if remaining > 0:
                        time.sleep(remaining)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    _run_demo()

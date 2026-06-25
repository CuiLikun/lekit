import contextlib
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

try:
    import termios
except ImportError:
    termios = None

import yaml
from rich import print
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from src.tools.ros_robot import RosRobot, RosRobotConfig
except ImportError:
    from src.tools.mock_robot import MockRobot as RosRobot
    from src.tools.mock_robot import MockRobotConfig as RosRobotConfig

from src.tools.proxy import Proxy
from src.utils.keyboard_utils import KeyboardListener
from src.utils.rerun_utils import RerunLogger

DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DIR))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/config.yaml")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")


def make_parts(config_path=DEFAULT_CONFIG_PATH):
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))  # noqa
    robot = RosRobot(RosRobotConfig(**config["robot"]))
    proxy = Proxy()
    tasks = config.get("tasks", [])
    batch = config.get("batch", [])
    rerun = RerunLogger(config["rerun"]) if config.get("rerun") else None
    return robot, proxy, tasks, batch, rerun


def get_field(task, key, default=None):
    """Read a field from a task, supporting both dataclass and dict (with `preset` fallback)."""
    if isinstance(task, dict):
        value = task.get(key)
        if value is None and isinstance(task.get("preset"), dict):
            value = task["preset"].get(key)
        return value if value is not None else default
    return getattr(task, key, default)


class Coordinator:
    def __init__(self):
        self.keyboard = KeyboardListener()
        self.robot, self.proxy, self.tasks, self.batch, self.rerun = make_parts()
        self.logs_stream = []
        self._live = None

        # Unified dictionary managing all state and control variables
        self.state = {
            "status": "selecting",  # Global TUI status (selecting, presetting, awaiting_confirm, executing, paused, done, aborted, error)
            "running": True,  # Main loop lifetime flag (True to run, False to exit and restore terminal)
            "curr_task": None,  # Active task dict currently selected
            "pause": False,  # Task execution paused flag
            "abort": False,  # Task execution abort flag
            "pending_confirm_prompt": None,  # Active confirmation message for the confirmation Modal
            "confirm_result": None,  # Keypress confirmation result (True for Yes/Enter, False for No/Esc)
            "show_help": False,  # Help Overlay active flag
            "elapsed": 0.0,  # Cumulative execution time of the active task in seconds
            "fps": 0,  # Real-time task execution FPS
            "steps": 0,  # Cumulative completed control steps
            "tui_render_time": 0.0,  # Last TUI layout render time in milliseconds
            "task_cursor": 0,  # Logical index of the highlighted task in self.tasks
            "task_scroll_offset": 0,  # Index of the first task visible in the Task List panel
            "batch_index": -1,  # Index into self.batch of the currently-executing task (-1 = not running a batch)
        }

        # Persistent Console used to query terminal size for scroll-window math
        self.console = Console()

        # Task execution history statistics: {task_id: {"runs": int, "aborts": int, "policy": str, "average_duration": float}}
        self.history = {}
        for t in self.tasks:
            self.history[t["id"]] = {
                "runs": 0,
                "aborts": 0,
                "policy": "",
                "average_duration": 0.0,
            }

        # Lock for safe state updates, queue for asynchronous rendering requests, and render thread
        self.state_lock = threading.Lock()
        self.render_queue = queue.Queue(maxsize=1)
        self.render_thread = threading.Thread(target=self._async_render_loop, daemon=True)

        # Save and disable terminal echo to prevent keys from disrupting the layout
        self.old_settings = None
        if termios is not None and sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                new_settings = termios.tcgetattr(sys.stdin)
                new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable ECHO
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
            except Exception:
                pass

        self.bind_keys()
        self.keyboard.start()
        self.render_thread.start()

    def _async_render_loop(self):
        """Asynchronous background rendering thread loop."""
        while self.state["running"]:
            try:
                # Block with timeout to check for rendering requests
                self.render_queue.get(timeout=0.05)
                should_render = True
            except queue.Empty:
                # If the proxy is connecting, force a refresh anyway so the spinner animates
                should_render = hasattr(self, "proxy") and getattr(self.proxy, "connection_state", None) == "connecting"
                if not should_render:
                    continue
            except Exception:
                continue

            try:
                if self._live:
                    t_start = time.perf_counter()
                    # Acquire lock before reading state for layout generation
                    with self.state_lock:
                        layout_rendered = self.build_tui_layout()
                    self._live.update(layout_rendered)
                    self._live.refresh()
                    with self.state_lock:
                        self.state["tui_render_time"] = (time.perf_counter() - t_start) * 1000.0
            except Exception:
                pass

    def refresh_tui(self):
        """Schedule an asynchronous TUI refresh request in the background queue."""
        with contextlib.suppress(queue.Full):
            self.render_queue.put_nowait(True)
        # Drop if a frame is already queued to prevent backlog latency

    def log(self, level: str, msg: str):
        """Append log to buffer and constrain stream capacity."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"{timestamp} [{level.upper()}] {msg}"
        self.logs_stream.append(log_line)
        if len(self.logs_stream) > 50:
            self.logs_stream.pop(0)
        self.refresh_tui()

    def bind_keys(self):
        """Bind keyboard keys to the coordinator methods."""
        self.keyboard.bind_key("q", self.stop_main_loop, "Quit program")
        self.keyboard.bind_key("space", self.toggle_pause, "Pause/Resume execution")
        self.keyboard.bind_key("left", self.abort_task, "Abort current task")
        self.keyboard.bind_key("right", self.finish_task, "Finish current task")
        self.keyboard.bind_key("esc", self.handle_esc, "Back to selection / Cancel")
        self.keyboard.bind_key("r", self.handle_retry, "Retry current task")
        self.keyboard.bind_key("?", self.toggle_help, "Show/Hide Key Bindings")
        for task in self.tasks:
            self.keyboard.bind_key(
                str(task["id"]),
                lambda key, t=task: self.select_task_by_key(t),
                f"Select task {task['id']}",
            )
        self.keyboard.bind_key("y", self.handle_confirm_yes, "Confirm (Yes)")
        self.keyboard.bind_key("n", self.handle_confirm_no, "Confirm (No)")
        self.keyboard.bind_key("enter", self.handle_enter, "Confirm / Replay task")
        self.keyboard.bind_key("[", self.switch_gripper, "Switch left gripper")
        self.keyboard.bind_key("]", self.switch_gripper, "Switch right gripper")
        self.keyboard.bind_key("up", self.move_cursor, "Move cursor up")
        self.keyboard.bind_key("down", self.move_cursor, "Move cursor down")
        self.keyboard.bind_key("b", self.handle_batch, "Run batch tasks (auto)")

    def handle_batch(self, _key=None):
        """Run batch tasks (auto)."""
        if self.state["status"] == "selecting":
            self.log("info", "Batch mode enabled.")
            if len(self.batch) == 0:
                self.log("warning", "Batch is empty. No tasks to execute.")
                return
            with self.state_lock:
                self.state["batch_index"] = 0
                task_id = self.batch[self.state["batch_index"]]
                self.state["curr_task"] = next((t for t in self.tasks if t["id"] == task_id), None)
                if self.state["curr_task"] is None:
                    self.log("error", f"Task {task_id} not found in batch.")
                    return
            self.refresh_tui()

    def switch_gripper(self, _key=None):
        """Switch the robot gripper state."""
        side = {"[": "left", "]": "right"}[_key]
        try:
            self.robot.switch_gripper(side)
            self.log("info", f"Switched {side} gripper state.")
        except Exception as e:
            self.log("error", f"Failed to switch {side} gripper: {e}")

    def handle_confirm_yes(self, _key=None):
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = True
            self.refresh_tui()

    def handle_confirm_no(self, _key=None):
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = False
            self.refresh_tui()

    def handle_enter(self, _key=None):
        # Handle user_confirm
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = True
        # Handle task selection
        elif self.state["status"] in ("selecting", "done", "aborted", "error"):
            # Re-align curr_task to the cursor position; this also triggers process_selected_task
            # via the main loop when status == "selecting" and no task is active.
            self.select_task_at_cursor()
            if self.state["status"] in ("done", "aborted", "error") and self.state["curr_task"] is not None:
                self.log("info", f"Replaying task {self.state['curr_task']['id']}...")
                with self.state_lock:
                    self.state["status"] = "selecting"
        self.refresh_tui()

    def handle_esc(self, _key=None):
        # Handle user_confirm
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = False
        # Handle task selection
        elif self.state["status"] in ("error", "done", "aborted", "selecting"):
            with self.state_lock:
                self.state["curr_task"] = None
                self.state["status"] = "selecting"
        self.refresh_tui()

    def toggle_help(self, _key=None):
        with self.state_lock:
            self.state["show_help"] = not self.state["show_help"]
        self.refresh_tui()

    def handle_retry(self, _key=None):
        if self.state["status"] == "error":
            self.log("info", "Retrying task...")
            with self.state_lock:
                self.state["status"] = "selecting"
            self.refresh_tui()

    def user_confirm(self, prompt: str) -> bool:
        """Non-blocking user confirmation mapped to KeyboardListener triggers."""
        with self.state_lock:
            self.state["pending_confirm_prompt"] = prompt
            self.state["confirm_result"] = None
            old_status = self.state["status"]
            self.state["status"] = "awaiting_confirm"
        self.refresh_tui()

        while self.state["confirm_result"] is None and self.state["running"]:
            time.sleep(0.02)
            self.refresh_tui()

        with self.state_lock:
            res = self.state["confirm_result"] if self.state["confirm_result"] is not None else False
            self.state["pending_confirm_prompt"] = None
            self.state["status"] = old_status
        self.refresh_tui()
        return res

    def build_tui_layout(self) -> Layout:
        """Construct the standard layout / modals for full-screen monitoring."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )

        time_str = datetime.now().strftime("%H:%M:%S")  # noqa
        header_text = f"Mobile Robot · Multi-Task Coordinator{' ' * 10}"
        header_text += f"State: {self.state['status'].upper()}{' ' * 10}"
        # header_text += f"TUI: {self.state['tui_render_time']:.0f}ms{' ' * 10}{time_str}"
        layout["header"].update(
            Align.center(
                Text(
                    header_text,
                    style="bold white on blue",
                ),
                style="bold white on blue",
            )
        )
        layout["footer"].update(
            Align.center(
                Text(
                    "Q Quit · B Batch · ↑ / ↓ Cursor · ENTER Confirm · [ ] Gripper · SPACE Pause · ← Abort · → Finish · ESC Cancel · R Retry · ? Help",
                    style="black on white",
                ),
                style="black on white",
            )
        )

        # Render Help Modal
        if self.state["show_help"]:
            help_table = Table(show_header=True, header_style="bold cyan", box=None)
            help_table.add_column("Key", style="bold yellow")
            help_table.add_column("Action", style="white")
            help_table.add_row("Q", "Quit program")
            help_table.add_row("1-9", "Select corresponding Task")
            help_table.add_row("↑ / ↓", "Move cursor in task list")
            help_table.add_row("ENTER", "Confirm / Select task")
            help_table.add_row("[ / ]", "Switch left/right gripper")
            help_table.add_row("SPACE", "Pause / Resume execution")
            help_table.add_row("← (Left)", "Abort current task")
            help_table.add_row("→ (Right)", "Finish current task")
            help_table.add_row("ESC", "Back to selector / Cancel")
            help_table.add_row("R", "Retry task (on error)")
            help_table.add_row("B", "Run batch sequence (auto)")
            help_table.add_row("?", "Show / Hide this help modal")
            modal = Panel(
                help_table,
                title="[bold green]⌨  Key Bindings[/bold green]",
                border_style="blue",
                width=50,
            )
            layout["body"].update(Align.center(modal, vertical="middle"))
            return layout

        # Main Layout: Split into Left Column (Task Card List) and Right Column
        layout["body"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))

        # Right Column: Split into StatusPanel (top) and LogsPanel (bottom)
        layout["right"].split_column(Layout(name="status", size=10), Layout(name="logs"))

        # Left Column: Task Card List (with scrolling + cursor highlight)
        visible_count = self._compute_visible_tasks()
        offset = self.state["task_scroll_offset"]
        cursor_idx = self.state["task_cursor"]
        total_tasks = len(self.tasks)
        # Clamp offset defensively (terminal may have shrunk since last render)
        offset = max(0, min(offset, max(0, total_tasks - 1)))
        self.state["task_scroll_offset"] = offset
        visible_tasks = self.tasks[offset : offset + visible_count]

        task_panels = []
        for i, t in enumerate(visible_tasks):
            absolute_idx = offset + i
            is_active = self.state["curr_task"] and self.state["curr_task"]["id"] == t["id"]
            is_cursor = absolute_idx == cursor_idx

            if is_active:
                # All entries use supplementary-plane emoji (🟢🟡🔴✅🛑🏃) so that
                # every codepoint renders as 2 cells in any modern terminal. Mixing
                # BMP+VS16 glyphs (▶⏸⚠⛔) with SMP glyphs causes the title to drift
                # left/right between renders and visually mis-align the top border.
                status_desc = {
                    "executing": "🟢 executing",
                    "paused": "🟡 paused",
                    "error": "🔴 error",
                    "done": "✅ done",
                    "aborted": "🛑 aborted",
                    "presetting": "🏃 presetting",
                }.get(self.state["status"], "👉 active")
                border_color = "yellow"
                title_style = "bold bright_yellow"
            else:
                status_desc = "· idle"
                border_color = "dim"
                title_style = "dim"

            content = Text(no_wrap=True, overflow="ellipsis")
            content.append("Prompt : ", style="bold cyan" if is_active else "dim")
            content.append(f"{t['prompt']}\n", style="bright_white" if is_active else "dim")
            content.append("Preset : ", style="bold cyan" if is_active else "dim")
            preset = t.get("preset", {})
            joint_motions = len(preset.get("joint_angles", []))
            motion_desc = f"{joint_motions} motion" if joint_motions > 0 else "none"
            content.append(
                f"Location={preset.get('location', '')}, Torso={preset.get('torso_height') or 0.5}m, Joints={motion_desc}\n",
                style="green" if is_active else "dim",
            )
            content.append("Budget : ", style="bold cyan" if is_active else "dim")
            content.append(f"{t['budget']:.0f}s", style="white" if is_active else "dim")

            # Fetch real-time runs and aborts from self.history
            stats = self.history.get(t["id"], {"runs": 0, "aborts": 0})
            runs = stats.get("runs", 0)
            aborts = stats.get("aborts", 0)
            completed = max(0, runs - aborts)
            success_rate = (completed / runs * 100.0) if runs > 0 else 0.0
            rate_str = f"{success_rate:.1f}%" if runs > 0 else "-"

            content.append("  Runs : ", style="bold cyan" if is_active else "dim")
            content.append(f"{runs}", style="white" if is_active else "dim")
            content.append("  Aborts : ", style="bold cyan" if is_active else "dim")
            content.append(f"{aborts}", style="white" if is_active else "dim")
            content.append("  SR : ", style="bold cyan" if is_active else "dim")
            content.append(f"{rate_str}", style="white" if is_active else "dim")

            # Embed live progress bar and telemetry metrics directly inside the active task card
            if is_active and self.state["status"] in (
                "executing",
                "paused",
                "done",
                "aborted",
            ):
                pct = min(self.state["elapsed"] / t.get("budget", 10.0), 1.0) if t.get("budget", 10.0) > 0 else 0.0
                filled = int(pct * 25)
                pbar = "▓" * filled + "░" * (25 - filled)
                content.append("\n")
                content.append("Status : ", style="bold yellow")
                content.append(
                    f"{pbar} {self.state['elapsed']:.1f}/{t.get('budget', 10.0):.1f}s",
                    style="yellow",
                )
                content.append(
                    f"  ⚡ {self.state['fps']}fps  steps={self.state['steps']}",
                    style="cyan",
                )

            # Cursor prefix: "👉 " in bold green when this card is under the cursor
            cursor_prefix = "[bold green]👉 [/bold green]" if is_cursor else ""
            task_panels.append(
                Panel(
                    content,
                    title=f"[{title_style}]{cursor_prefix}Task {t['id']} ({status_desc})[/{title_style}]",
                    border_style=border_color,
                    padding=(0, 1),
                )
            )

        # Append a "↓ N more" indicator when there are tasks below the visible window
        hidden_below = total_tasks - (offset + len(visible_tasks))
        hidden_above = offset
        showing_start = offset + 1 if total_tasks else 0
        showing_end = offset + len(visible_tasks)

        indicator_text = Text()
        indicator_text.append(f"showing {showing_start}-{showing_end} of {total_tasks} ", style="dim")
        if hidden_above > 0:
            indicator_text.append(f"    ↑ {hidden_above} more above ", style="dim")
        if hidden_below > 0:
            indicator_text.append(f"    ↓ {hidden_below} more below", style="dim")
        if indicator_text:
            task_panels.append(indicator_text)

        layout["left"].update(
            Panel(
                Group(*task_panels),
                title="[bold green]📋 Task List[/bold green] ",
                border_style="green",
            )
        )

        status_text = Text()

        # Batch sequence line (shown when self.batch is non-empty).
        # Highlights the currently executing task id (when in_batch is True).
        if self.batch:
            in_batch = self.state.get("batch_index", -1) >= 0
            current_idx = self.state.get("batch_index", -1) if in_batch else -1
            status_text.append("Batch: ", style="cyan")
            for i, tid in enumerate(self.batch):
                if i > 0:
                    status_text.append(" → ", style="dim")
                if i == current_idx:
                    status_text.append(str(tid), style="bold bright_yellow")
                else:
                    status_text.append(str(tid), style="dim")
            status_text.append("\n")

        # Robot Status
        robot_text = f"Robot: {self.robot.__class__.__name__ if self.robot else '(unset)'}"
        if self.robot.is_connected:
            robot_text += f" (connected ✓)"  # noqa
        status_text.append(robot_text + "\n", style="green")

        # Proxy Status
        proxy_meta = getattr(self.proxy, "policy_meta", {})
        if not isinstance(proxy_meta, dict):
            proxy_meta = {}
        policy_repo = proxy_meta.get("repo_id", "unknown")

        conn_state = getattr(self.proxy, "connection_state", "unconnected")
        connecting_addr = getattr(self.proxy, "connecting_addr", "None")
        addr = proxy_meta.get("addr", "None")
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        conn_styles = {
            "connecting": (
                f"{spinner_frames[int(time.time() * 10) % len(spinner_frames)]} Proxy: Connecting to {connecting_addr}...",
                "yellow",
            ),
            "connected": (
                f"Proxy: {addr}" + (f"  @{policy_repo}" if policy_repo != "unknown" and addr != "rule-based" else ""),
                "green",
            ),
            "failed": (f"Proxy: Connection to {connecting_addr} failed", "red"),
        }
        proxy_text, style = conn_styles.get(conn_state, (f"Proxy: {addr}", "dim white"))
        status_text.append(proxy_text + "\n", style=style)

        # Current Task Status
        curr_task_str = (
            f"{self.state['curr_task']['id']}: {self.state['curr_task']['prompt']}"
            if self.state["curr_task"]
            else "(none)"
        )
        status_text.append(f"{self.state['status'].upper()} (Task: {curr_task_str})", style="yellow")

        # Build the operation-console portion in the same Panel. Title and border reflect
        # the active operation state; defaults signal the merged identity of the panel.
        op_text = Text()
        op_color = "dim"
        status_title = "⚡ System & Operation Console"

        # Calculate fast and slow breathing square waves using the system clock
        is_breathe_fast = int(time.time() * 2.5) % 2 == 0
        is_breathe_slow = int(time.time() * 1.0) % 2 == 0

        if self.state["status"] == "awaiting_confirm" and self.state["pending_confirm_prompt"]:
            op_color = "bright_yellow" if is_breathe_fast else "yellow"
            warning_dot = "●" if is_breathe_fast else " "
            status_title = f"[blink]{warning_dot}[/blink] Confirmation Required"

            op_text.append(f"❓ {self.state['pending_confirm_prompt']}\n", style="bold white")
            op_text.append(
                "[Y] Yes / Enter",
                style="bold bright_green" if is_breathe_fast else "bold green",
            )
            op_text.append(" " * 6)
            op_text.append(
                "[N] No / Esc",
                style="bold bright_red" if is_breathe_fast else "bold red",
            )
        elif self.state["status"] == "selecting":
            op_text.append("All set. Ready to go. 👌\n", style="bold green")
            op_text.append("Select a task from list, or Q to quit.", style="cyan")
            op_color = "green"
        elif self.state["status"] == "presetting":
            op_text.append("🏃 Preset in progress...\n", style="bold yellow")
            op_text.append("Prepare robot for execution.", style="dim")
            op_color = "yellow"
        elif self.state["status"] == "executing":
            op_color = "bright_green" if is_breathe_slow else "green"
            warning_dot = "●" if is_breathe_slow else " "
            status_title = f"[blink]{warning_dot}[/blink] Policy Execution"

            op_text.append("▶️️ Policy execution in progress...\n", style="bold bright_green")
            op_text.append(
                "Real-time control loops are active. Press Space to pause, ← to abort, → to finish.",
                style="green" if is_breathe_slow else "dim green",
            )
        elif self.state["status"] == "paused":
            op_color = "bright_yellow" if is_breathe_fast else "yellow"
            warning_dot = "●" if is_breathe_fast else " "
            status_title = f"[blink]{warning_dot}[/blink] Policy Paused"

            op_text.append("⏸️ Policy execution paused.\n", style="bold bright_yellow")
            op_text.append(
                "Press Space to resume, ← to abort, or → to finish.",
                style="bright_yellow" if is_breathe_fast else "dim yellow",
            )
        elif self.state["status"] == "done":
            op_text.append("✅ Active task completed successfully!\n", style="bright_green")
            op_text.append(
                "Press Enter to repeat, Esc to clear, or select another task.",
                style="cyan",
            )
            op_color = "bright_green"
        elif self.state["status"] == "aborted":
            op_text.append("⛔️ Task execution aborted by user.\n", style="bright_red")
            op_text.append(
                "Press Enter to repeat, Esc to clear, or select another task.",
                style="cyan",
            )
            op_color = "bright_red"
        elif self.state["status"] == "error":
            op_text.append("⚠️ Task execution failed with error.\n", style="bright_red")
            op_text.append(
                "Press Enter/R to retry, or Esc to return to task selection.",
                style="white",
            )
            op_color = "bright_red"
            status_title = "🚨 System Error"

        # Empty Text gives one blank separator line between the system-status block and
        # the operation-console block so they read as distinct sections within the merged panel.
        layout["status"].update(
            Panel(
                Group(status_text, Text(""), op_text),
                title=f"[bold {op_color}]{status_title}[/bold {op_color}]",
                border_style=op_color,
            )
        )

        # Right Column bottom: LogsPanel
        logs_text = Text("\n".join(self.logs_stream[-20:]))
        layout["logs"].update(
            Panel(
                logs_text,
                title="[bold magenta]📜 Recent Logs[/bold magenta]",
                border_style="magenta",
            )
        )

        return layout

    def stop_main_loop(self, key: str = None):
        """Stop the main loop of the coordinator."""
        self.log("info", "Stopping main loop...")
        with self.state_lock:
            self.state["running"] = False
        self.stop()

    def stop(self, key: str = None):
        """Clean up resources on shutdown."""
        with self.state_lock:
            self.state["running"] = False

        # Flush terminal input buffer and restore echo settings
        if termios is not None and sys.stdin.isatty():
            try:
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
                if self.old_settings is not None:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

        # Stop and release all active components
        for component, name in [
            (self.keyboard, "Keyboard"),
            (self.robot, "Robot"),
            (self.proxy, "Proxy"),
        ]:
            try:
                component.stop()
            except Exception:
                self.log("error", f"Error stopping {name} component: {sys.exc_info()[0]}")

    def run(self):
        """Run the main coordinator screen loop."""
        with Live(self.build_tui_layout(), screen=False, auto_refresh=False) as live:
            self._live = live
            self.log("info", "Multi-Task Coordinator TUI started.")
            while self.state["running"]:
                self._live.update(self.build_tui_layout())
                self._live.refresh()
                time.sleep(0.05)

                if self.state["curr_task"] and self.state["status"] == "selecting":
                    try:
                        self.process_selected_task()
                    except Exception as e:
                        self.log("error", f"Task execution failed: {e}")
                        self.state["status"] = "error"

    def process_selected_task(self):
        """Execution pipeline helper for selected tasks.

        After execute_single_task returns, mirror handle_enter's replay pattern:
        if we're partway through a batch, advance to the next batch task and
        reset status to "selecting" so the main loop re-triggers this method.
        Otherwise leave the final status (done/aborted/error) for the user to
        decide via keys.
        """
        self.state["status"] = "presetting"
        self.log("info", f"Starting Task {self.state['curr_task']['id']} process")
        try:
            self.set_proxy(self.state["curr_task"])
            self.set_robot(self.state["curr_task"])
        except Exception as e:
            self.log("error", f"Presetting failed: {e}")
            self.state["status"] = "error"
            self.refresh_tui()
            return

        self.state["status"] = "executing"
        self.refresh_tui()

        self.execute_single_task(self.state["curr_task"])

        # Batch auto-advance: if batch_index is set, move on to the next task
        # (or finish the batch). Same replay trick as handle_enter — set status
        # back to "selecting" so the main loop re-enters this method.
        if self.state.get("batch_index", -1) >= 0:
            # If the task did NOT finish cleanly, ask the user whether to retry
            # the same task before moving on.
            if self.state["status"] != "done" and self.user_confirm(
                f"Batch task {self.state['curr_task']['id']} ended with '{self.state['status']}'. Retry?"
            ):
                self.log("info", f"Batch: retrying task {self.state['curr_task']['id']}.")
                with self.state_lock:
                    self.state["status"] = "selecting"
                return
            next_idx = self.state["batch_index"] + 1
            if next_idx < len(self.batch):
                next_task = next((t for t in self.tasks if t["id"] == self.batch[next_idx]), None)
                if next_task is None:
                    self.log(
                        "error",
                        f"Batch task id {self.batch[next_idx]} not found; stopping.",
                    )
                    with self.state_lock:
                        self.state["batch_index"] = -1
                        self.state["curr_task"] = None
                        self.state["status"] = "selecting"
                    return
                self.log("info", f"Batch: advancing to task {self.batch[next_idx]}.")
                with self.state_lock:
                    self.state["batch_index"] = next_idx
                    self.state["curr_task"] = next_task
                    self.state["status"] = "selecting"
            else:
                self.log("info", f"Batch completed: {len(self.batch)} tasks.")
                with self.state_lock:
                    self.state["batch_index"] = -1
                    self.state["curr_task"] = None
                    self.state["status"] = "selecting"

    def select_task_by_key(self, task: dict):
        """Select a task from the list via key binds."""
        if self.state.get("batch_index", -1) >= 0:
            # Lock task selection while a batch is running.
            self.log("warn", "Cannot select task manually while a batch is running.")
            return
        if self.state["status"] in ("selecting", "error", "done", "aborted"):
            self.state["curr_task"] = task
            self.state["status"] = "selecting"
            # Sync the cursor highlight to the selected row so the TUI cursor
            # and the active card stay on the same line. Without this, pressing
            # "5" moves the active state to row 4 while the highlight stays on
            # row 0, and the scroll window never auto-adjusts to reveal the
            # chosen row when it lies outside the visible window.
            if task in self.tasks:
                with self.state_lock:
                    self.state["task_cursor"] = self.tasks.index(task)
                    self._adjust_scroll()
            # self.log("info", f"Selected task {task['id']} via keypress.")

    # Statuses in which arrow-key cursor navigation is allowed. While a task is
    # executing/paused/presetting or the user is being asked to confirm, the
    # selection should be frozen to avoid accidental changes.
    _NAV_ALLOWED = ("selecting", "error", "done", "aborted")
    # Idle task card: top border (with title) + 3 content lines + bottom border = 5 lines.
    # Active cards add one progress line; the slice below tolerates the resulting +1 overflow.
    _TASK_CARD_HEIGHT = 5

    def _compute_visible_tasks(self) -> int:
        """Estimate how many task cards fit in the left panel."""
        try:
            total_height = self.console.size.height
        except Exception:
            total_height = 24
        # Header (1) + footer (1) + outer panel top/bottom borders (2) = 4 lines
        # of chrome that are not available for task card content.
        available = total_height - 4
        return max(1, available // self._TASK_CARD_HEIGHT)

    def _adjust_scroll(self) -> None:
        """Shift task_scroll_offset so task_cursor stays in the visible window."""
        visible = self._compute_visible_tasks()
        total = len(self.tasks)
        cursor = self.state["task_cursor"]
        offset = self.state["task_scroll_offset"]
        if cursor < offset:
            offset = cursor
        elif cursor >= offset + visible:
            offset = cursor - visible + 1
        max_offset = max(0, total - visible)
        self.state["task_scroll_offset"] = max(0, min(offset, max_offset))

    def move_cursor(self, _key: str = None) -> None:
        """Move the task-list cursor up or down (selected by _key) with wrap-around."""
        if not self.tasks or self.state["status"] not in self._NAV_ALLOWED:
            return
        step = -1 if _key == "up" else 1 if _key == "down" else 0
        if step == 0:
            return
        with self.state_lock:
            self.state["task_cursor"] = (self.state["task_cursor"] + step) % len(self.tasks)
            self._adjust_scroll()
        self.refresh_tui()

    def select_task_at_cursor(self) -> None:
        """Reuse the existing 1-9 selection path with the task under the cursor."""
        if not self.tasks:
            return
        cursor = self.state["task_cursor"] % len(self.tasks)
        self.select_task_by_key(self.tasks[cursor])

    def set_proxy(self, task):
        """Set the proxy policy based on the selected task."""
        policy_addr = get_field(task, "policy")
        # Special case for rule-based policy
        if policy_addr == "rule-based":
            self.log("info", "Using rule-based policy.")
            self.proxy.stop()  # Stop any existing proxy
            self.proxy.policy_meta = {"addr": "rule-based"}
            self.proxy.connection_state = "connected"
            return
        # General case: switch to a remote policy server
        self.proxy.switch_policy(policy_addr)
        if self.proxy.policy_meta["addr"] != policy_addr:
            self.log(
                "error",
                f"Mismatch in proxy policy address! Expected {policy_addr}, got {self.proxy.policy_meta['addr']}",
            )
            raise ValueError("Proxy policy address mismatch")
        self.log("info", f"Switch policy to: {policy_addr}")

    def set_robot(self, task):
        """Preset the robot based on the task's preset configuration."""
        location = get_field(task, "location")
        torso_height = get_field(task, "torso_height")
        joint_angles = get_field(task, "joint_angles")
        is_rule_based = get_field(task, "policy") == "rule-based"
        # if not self.user_confirm(
        #     f"Preset robot? Location={location}, Torso Height={torso_height}, Joint Angles={'[skipped]' if joint_angles is None else f'{len(joint_angles)} waypoints'}"
        # ):
        #     self.log("info", "Robot preset cancelled.")
        #     raise ValueError("Preset cancelled")
        if location is not None:
            self.log("info", f"Moving chasis to: {location}")
            self.robot.move_to_location(location)
        if torso_height is not None:
            self.log("info", f"Set torso height: {torso_height}m")
            self.robot.set_torso_height(torso_height)
        if joint_angles is not None and not is_rule_based:
            self.log("info", f"Set joint angles: {len(joint_angles)} waypoints")
            # Rule-based policies execute joint_angles through robot.move_arms_by_action
            # during the executing phase; doing it here would race with that motion.
            for value in joint_angles:  # each step is a joint angle vector
                self.robot.set_joint_angles(value)
            # Alternative: set joint angles by robot.move_arms_by_action
            # self.robot.move_arms_by_action(joint_angles, timeout_sec=30.0)
        elif is_rule_based:
            self.log("info", "Rule-based policy detected, skipping joint angles preset.")
        # self.log("info", "Robot preset completed.")

    def execute_single_task(self, task, fps: int = 30):
        """Execute the main logic of the task using the proxy."""
        budget = get_field(task, "budget", 10.0)
        prompt = get_field(task, "prompt", "")
        # Confirm task execution, skip in batch mode
        if not self.state.get("batch_index", -1) >= 0 and not self.user_confirm(
            f"Start task {task['id']}? prompt='{prompt}' budget={budget}s"
        ):
            self.log("info", "Task execution cancelled.")
            self.state["status"] = "selecting"
            self.state["curr_task"] = None
            return

        # Increment total execution runs counter
        self.history[task["id"]]["runs"] += 1

        # Record the policy repo_id in history
        policy_addr = self.proxy.policy_meta.get("addr", "N/A")  # fallback to policy address if unavailable
        self.history[task["id"]]["policy"] = self.proxy.policy_meta.get("repo_id", policy_addr)

        # Initialize state variables for the execution loop
        dt = 1.0 / fps
        self.state["steps"], self.state["elapsed"] = 0, 0.0
        self.state["pause"] = False
        self.state["abort"] = False
        self.state["finish"] = False
        self.state["fps"] = fps

        if self.rerun is not None:
            self.rerun.switch_record()
            self.log("info", f"Rerun recording started for task {task['id']}.")

        # Main loop
        while self.state["elapsed"] < budget and self.state["running"]:
            if self.state["abort"] or self.state["finish"]:
                break
            if self.state["pause"]:
                self.state["status"] = "paused"
                time.sleep(0.05)
                self.refresh_tui()
                continue

            self.state["status"] = "executing"
            cycle_start = time.perf_counter()
            t0 = time.perf_counter()

            # Special case: rule-based policies execute joint_angles through robot.move_arms_by_action
            if self.proxy.policy_meta.get("addr") == "rule-based":
                self.log("info", "Executing rule-based policy via robot.move_arms_by_action.")
                self.robot.move_arms_by_action(get_field(task, "joint_angles", []), timeout_sec=budget)
                self.state["elapsed"] += time.perf_counter() - cycle_start
                break  # Rule-based policy completed, exit loop
            else:
                obs = self.robot.capture_observation()
                obs["task"] = prompt
                act = self.proxy.require_action(obs)

            if act is None:
                self.log("warn", "No action received from proxy!")
            else:
                self.robot.send_action(act)

            if self.rerun is not None and obs is not None and act is not None:
                label = "policy" if self.robot.policy_control else "teleop"
                label = "policy"  # override to always log as policy action for now
                self.rerun.log({**obs, label: act})

            self.refresh_tui()
            time.sleep(max(0, dt - (time.perf_counter() - t0)))
            self.state["elapsed"] += time.perf_counter() - cycle_start
            self.state["fps"] = round(1.0 / (time.perf_counter() - t0))
            self.state["steps"] += 1

        # Check if the task was aborted within budget
        if self.state["abort"] and self.state["elapsed"] < budget:
            self.history[task["id"]]["aborts"] += 1

        # Update average duration for the task
        if self.history[task["id"]]["average_duration"] == 0.0:
            self.history[task["id"]]["average_duration"] = self.state["elapsed"]
        else:
            self.history[task["id"]]["average_duration"] = (
                self.history[task["id"]]["average_duration"] * (self.history[task["id"]]["runs"] - 1)
                + self.state["elapsed"]
            ) / self.history[task["id"]]["runs"]

        self.state["status"] = "aborted" if self.state["abort"] else "done"
        self.log("info", f"Task {task['id']} completed with status: {self.state['status']}")

    def toggle_pause(self, _key: str = None):
        """Toggle pause/resume state."""
        if self.state["status"] in ("executing", "paused"):
            self.state["pause"] = not self.state["pause"]
            self.state["status"] = "paused" if self.state["pause"] else "executing"

    def abort_task(self, _key: str = None):
        """Signal the running task to exit its loop."""
        if self.state["status"] in ("executing", "paused"):
            self.state["abort"] = True

    def finish_task(self, _key: str = None):
        """Signal the running task to finish and complete normally."""
        if self.state["status"] in ("executing", "paused"):
            self.state["finish"] = True

    def save_and_print_statistics(self):
        """Log summary statistics to a file in structured JSON format and print a beautiful summary Table to the console on exit."""

        console = Console()
        table = Table(
            title="[bold bright_green]📊 Task Execution Telemetry Summary[/bold bright_green]",
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column("ID", justify="center", style="bold yellow")
        table.add_column("Task Prompt", style="white")
        table.add_column("Runs", justify="center", style="green")
        table.add_column("Aborts", justify="center", style="red")
        table.add_column("Success Rate", justify="right", style="bright_cyan")
        table.add_column("Avg Time", justify="right", style="bright_magenta")
        table.add_column("Policy", style="white")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_runs = 0
        total_aborts = 0
        tasks_stats = []

        for t in self.tasks:
            t_id = t["id"]
            stats = self.history.get(t_id, {"runs": 0, "aborts": 0, "policy": "", "average_duration": 0.0})
            runs = stats["runs"]
            aborts = stats["aborts"]
            completed = max(0, runs - aborts)
            success_rate = (completed / runs * 100.0) if runs > 0 else 0.0
            average_duration = stats["average_duration"] if stats["average_duration"] > 0.0 else None
            rate_str = f"{success_rate:.1f}%" if runs > 0 else "-"
            time_str = f"{average_duration:.2f}s" if average_duration is not None else "-"

            total_runs += runs
            total_aborts += aborts

            table.add_row(
                str(t_id),
                t["prompt"],
                str(runs),
                str(aborts),
                rate_str,
                time_str,
                stats["policy"],
            )

            tasks_stats.append(
                {
                    "task_id": t_id,
                    "prompt": t["prompt"],
                    "policy": stats["policy"],
                    "runs": runs,
                    "aborts": aborts,
                    "success_rate": round(success_rate, 2) if runs > 0 else None,
                    "average_duration": round(average_duration, 2) if average_duration is not None else None,
                }
            )

        json_report = {
            "timestamp": timestamp,
            "summary": {
                "total_runs": total_runs,
                "total_aborts": total_aborts,
            },
            "tasks": tasks_stats,
        }

        if total_runs == 0:
            # No tasks were actually run; skip saving the report.
            console.print("[yellow]No tasks executed this session; skipping telemetry report.[/yellow]")
            return

        os.makedirs("outputs", exist_ok=True)
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(DEFAULT_OUTPUT_PATH, f"run_task_{timestamp_file}.json")
        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(json_report, f, indent=4, ensure_ascii=False)
            console.print(table)
            console.print(
                f"\n[green]✔ Telemetry JSON report written to: [bold underline]{log_file_path}[/bold underline][/green]"
            )
        except Exception as e:
            console.print(f"[red]Failed to write statistics to log file: {e}[/red]")


def main():
    coordinator = Coordinator()
    try:
        coordinator.run()
    except KeyboardInterrupt:
        print("\n[red]Shutting down...[/red]")
        coordinator.stop()
    finally:
        coordinator.save_and_print_statistics()


if __name__ == "__main__":
    main()

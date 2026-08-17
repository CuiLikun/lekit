"""Multi-Task Coordinator TUI built with Textual.

Refactor of :mod:`dashboard`. Two main changes vs. the rich-based original:

1. The TUI surface (``rich.layout.Layout``/``rich.live.Live``) is replaced by a
   Textual ``App`` with widget tree and CSS-driven layout. Static panels are
   refreshed on a periodic ``set_interval`` timer to keep spinners and
   breathing animations animated.

2. The custom ``pynput``-based ``KeyboardListener`` is removed entirely; key
   input is handled by Textual. Static bindings live on the ``App.BINDINGS``
   class attribute, modal-scoped keys live on each ``ModalScreen``, and the
   dynamic "task id -> select task" keys are routed through the App's
   ``on_key`` event handler.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

try:
    from lekit.tools.ros_robot import RosRobot, RosRobotConfig
except ImportError:
    from lekit.tools.mock_robot import MockRobot as RosRobot
    from lekit.tools.mock_robot import MockRobotConfig as RosRobotConfig

from lekit.tools.proxy import Proxy
from lekit.utils.rerun_utils import RerunLogger

DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DIR))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/config.yaml")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Pure helpers (carried over from the rich version)
# ---------------------------------------------------------------------------


def make_parts(config_path: str = DEFAULT_CONFIG_PATH):
    """Build the robot/proxy/tasks/rerun components from a YAML config file."""
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))  # noqa: SIM115
    robot = RosRobot(RosRobotConfig(**config["robot"]))
    proxy = Proxy()
    tasks = config.get("tasks", [])
    batch = config.get("batch", [])
    rerun = RerunLogger(config["rerun"]) if config.get("rerun") else None
    return robot, proxy, tasks, batch, rerun


def get_field(task: Any, key: str, default: Any = None) -> Any:
    """Read a field from a task, supporting both dataclass and dict (with `preset` fallback)."""
    if isinstance(task, dict):
        value = task.get(key)
        if value is None and isinstance(task.get("preset"), dict):
            value = task["preset"].get(key)
        return value if value is not None else default
    return getattr(task, key, default)


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class HelpScreen(ModalScreen[None]):
    """Modal screen that lists every key binding."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Container {
        width: 60;
        max-width: 96;
        height: auto;
        border: round blue;
        background: $panel;
        padding: 1 2;
    }

    HelpScreen .help-title {
        color: green;
        text-style: bold;
        content-align: center middle;
        width: 100%;
        margin-bottom: 1;
    }

    HelpScreen .help-row {
        height: 1;
    }

    HelpScreen .help-key {
        width: 16;
        color: yellow;
        text-style: bold;
    }

    HelpScreen .help-action {
        width: 1fr;
        color: white;
    }
    """

    # Escape or "?" closes the help modal regardless of which key the user presses.
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Close"),
        Binding("question_mark", "app.pop_screen", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("⌨  Key Bindings", classes="help-title")
            for key, action in [
                ("Q", "Quit program"),
                ("1-9", "Select corresponding Task"),
                ("↑ / ↓", "Move cursor in task list"),
                ("ENTER", "Confirm / Select task"),
                ("[ / ]", "Switch left/right gripper"),
                ("SPACE", "Pause / Resume execution"),
                ("← (Left)", "Abort current task"),
                ("→ (Right)", "Finish current task"),
                ("ESC", "Back to selector / Cancel"),
                ("R", "Retry task (on error)"),
                ("B", "Run batch sequence (auto)"),
                ("?", "Show / Hide this help modal"),
            ]:
                with Horizontal(classes="help-row"):
                    yield Static(f"{key:<14}", classes="help-key")
                    yield Static(action, classes="help-action")


class ConfirmScreen(ModalScreen[bool]):
    """Modal Yes/No confirmation dialog. Dismisses with True/False."""

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }

    ConfirmScreen > Container {
        width: 70;
        max-width: 96;
        height: auto;
        border: round $warning;
        background: $panel;
        padding: 1 2;
    }

    ConfirmScreen #confirm-prompt {
        text-style: bold;
        color: white;
        margin-bottom: 1;
    }

    ConfirmScreen .confirm-buttons {
        height: 1;
    }

    ConfirmScreen #confirm-yes {
        width: 1fr;
        content-align: center middle;
        color: green;
        text-style: bold;
    }

    ConfirmScreen #confirm-no {
        width: 1fr;
        content-align: center middle;
        color: red;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("y", "yes", "Yes"),
        Binding("n", "no", "No"),
        Binding("escape", "no", "Cancel"),
        Binding("enter", "yes", "OK"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt_text = prompt

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(f"❓ {self.prompt_text}", id="confirm-prompt")
            with Horizontal(classes="confirm-buttons"):
                yield Static("[Y/Enter] Yes", id="confirm-yes")
                yield Static("[N/Esc] No", id="confirm-no")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Main coordinator application
# ---------------------------------------------------------------------------


class CoordinatorApp(App):
    """Multi-Task Coordinator TUI."""

    BINDINGS = [
        Binding("q", "stop_main_loop", "Quit"),
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("left", "abort_task", "Abort"),
        Binding("right", "finish_task", "Finish"),
        Binding("escape", "handle_esc", "Back/Cancel"),
        Binding("r", "handle_retry", "Retry"),
        Binding("question_mark", "toggle_help", "Help"),
        Binding("y", "handle_confirm_yes", "Yes"),
        Binding("n", "handle_confirm_no", "No"),
        Binding("enter", "handle_enter", "Confirm"),
        Binding("left_square_bracket", "switch_gripper_left", "Gripper L"),
        Binding("right_square_bracket", "switch_gripper_right", "Gripper R"),
        Binding("up", "move_cursor_up", "Cursor ↑"),
        Binding("down", "move_cursor_down", "Cursor ↓"),
        Binding("b", "handle_batch", "Run batch"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #app-header {
        height: 1;
        background: blue;
        color: white;
        text-style: bold;
        content-align: center middle;
    }

    #app-footer {
        height: 1;
        background: white;
        color: black;
        content-align: center middle;
    }

    #app-body {
        height: 1fr;
        layout: horizontal;
    }

    #left-column {
        width: 1fr;
        border: round green;
        padding: 0 1;
    }

    #right-column {
        width: 1fr;
        layout: vertical;
    }

    #status-panel {
        height: 10;
        border: round yellow;
        padding: 0 1;
    }

    #map-panel {
        height: 25;
        border: round cyan;
        padding: 0 1;
    }

    #logs-panel {
        height: 1fr;
        border: round magenta;
        padding: 0 1;
    }
    """

    # Idle task card height (top border + title + 3 content lines + bottom border).
    # Active cards add one progress line; the slice below tolerates the +1 overflow.
    _TASK_CARD_HEIGHT = 5
    # Statuses in which arrow-key cursor navigation is allowed.
    _NAV_ALLOWED = ("selecting", "error", "done", "aborted")

    def __init__(self) -> None:
        super().__init__()
        self.robot, self.proxy, self.tasks, self.batch, self.rerun = make_parts()
        self.locations: dict[str, Any] = {
            x["preset"]["location"]["name"]: x["preset"]["location"]["position"] for x in self.tasks
        }
        self.logs_stream: list[str] = []

        # Unified dictionary managing all state and control variables. The lock below
        # guards writes from any thread; periodic renders read without a lock for
        # throughput (a stale render is harmless, atomic transitions matter).
        self.state: dict[str, Any] = {
            "status": "selecting",  # selecting, presetting, awaiting_confirm, executing, paused, done, aborted, error
            "running": True,  # Main loop lifetime flag (False to exit)
            "curr_task": None,  # Active task dict
            "pause": False,
            "abort": False,
            "retry": False,
            "finish": False,
            "pending_confirm_prompt": None,
            "confirm_result": None,
            "show_help": False,
            "elapsed": 0.0,
            "fps": 0,
            "steps": 0,
            "task_cursor": 0,
            "task_scroll_offset": 0,
            "batch_index": -1,
        }

        # Permanent trajectory of robot positions (deduped on 5cm threshold).
        self.trajectory: list[tuple[float, float]] = []

        # Task execution history statistics: {task_id: {"runs": int, "aborts": int, "policy": str, "average_duration": float}}
        self.history: dict[Any, dict[str, Any]] = {}
        for t in self.tasks:
            self.history[t["id"]] = {"runs": 0, "aborts": 0, "policy": "", "average_duration": 0.0}

        # State lock for cross-thread state mutations.
        self.state_lock = threading.Lock()

        # Confirmation plumbing for blocking calls coming from worker threads.
        self._confirm_event: threading.Event | None = None
        # Reference to the currently running task worker (so we can detect completion).
        self._task_worker = None

    # ------------------------------------------------------------------ #
    # Layout / compose
    # ------------------------------------------------------------------ #

    def compose(self) -> ComposeResult:
        """Build the persistent TUI widget tree.

        The four right-column panels (status / map / logs) share one body
        container; the left column hosts the scrollable task list. Header
        and footer are single-line bars styled via CSS.
        """
        yield Static(self._build_header_text(), id="app-header")

        with Container(id="app-body"):
            with Vertical(id="left-column"):
                yield Static(id="task-list", markup=True)
            with Vertical(id="right-column"):
                yield Static(id="status-content", markup=True)
                yield Static(id="map-content", markup=True)
                yield Static(id="logs-content", markup=True)

        yield Static(self._build_footer_text(), id="app-footer")

    def on_mount(self) -> None:
        """Set up the TUI when the app starts."""
        self.log_msg("info", "Multi-Task Coordinator TUI started.")

        # Two intervals driving the TUI:
        # * 100 ms panel refresh -> smooth spinners / breathing animations
        # * 50 ms worker check    -> quickly starts the next task after selection
        self.set_interval(0.1, self._refresh_panels)
        self.set_interval(0.05, self._check_and_run_task)

    # ------------------------------------------------------------------ #
    # Periodic refresh
    # ------------------------------------------------------------------ #

    def _refresh_panels(self) -> None:
        """Update every dynamic panel widget's content."""
        try:
            self.query_one("#app-header", Static).update(self._build_header_text())
            self.query_one("#app-footer", Static).update(self._build_footer_text())
            self.query_one("#task-list", Static).update(self._build_task_list_content())
            self.query_one("#status-content", Static).update(self._build_status_content())
            self.query_one("#map-content", Static).update(self._build_map_content())
            self.query_one("#logs-content", Static).update(self._build_logs_content())
        except Exception:
            pass

    def _check_and_run_task(self) -> None:
        """Start the task worker when a task is queued but not yet running."""
        if not self.state["running"]:
            return
        if self.state["curr_task"] and self.state["status"] == "selecting":
            if self._task_worker is None or not self._task_worker.is_running:
                self._task_worker = self.run_worker(
                    self.process_selected_task, thread=True, exclusive=True
                )

    # ------------------------------------------------------------------ #
    # Header / footer text builders
    # ------------------------------------------------------------------ #

    def _build_header_text(self) -> str:
        return (
            f"Mobile Robot · Multi-Task Coordinator{' ' * 10}"
            f"State: {self.state['status'].upper()}"
        )

    @staticmethod
    def _build_footer_text() -> str:
        return (
            "Q Quit · B Batch · ↑ / ↓ Cursor · ENTER Confirm · [ ] Gripper · "
            "SPACE Pause · ← Abort · → Finish · ESC Cancel · R Retry · ? Help"
        )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_msg(self, level: str, msg: str) -> None:
        """Append log to buffer and cap it at 50 entries (FIFO eviction).

        Named ``log_msg`` (not ``log``) to avoid shadowing Textual's
        :attr:`textual.app.App.log` property, which returns the framework's
        :class:`textual.Logger` instance.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"{timestamp} [{level.upper()}] {msg}"
        self.logs_stream.append(log_line)
        if len(self.logs_stream) > 50:
            self.logs_stream.pop(0)

    # ------------------------------------------------------------------ #
    # Key handler
    # ------------------------------------------------------------------ #

    def on_key(self, event) -> None:  # type: (Any) -> None
        """Route dynamic task-ID keys to the matching task.

        Most bindings live on ``BINDINGS``; only the *task id -> select* path
        needs dynamic dispatch because the set of tasks comes from the YAML
        config and isn't known at class-definition time.
        """
        # Modal screens handle their own bindings; don't steal keys.
        if isinstance(self.screen, ModalScreen):
            return
        if self.state["status"] not in ("selecting", "error", "done", "aborted"):
            return
        key_str = event.key
        if not key_str:
            return
        for task in self.tasks:
            if str(task["id"]) == key_str:
                self.select_task_by_key(task)
                event.prevent_default()
                event.stop()
                return

    # ------------------------------------------------------------------ #
    # Action methods (bound to keys via BINDINGS)
    # ------------------------------------------------------------------ #

    def action_stop_main_loop(self) -> None:
        self.log_msg("info", "Stopping main loop...")
        with self.state_lock:
            self.state["running"] = False
        self.exit()

    def action_toggle_pause(self) -> None:
        self.toggle_pause()

    def action_abort_task(self) -> None:
        self.abort_task()

    def action_finish_task(self) -> None:
        self.finish_task()

    def action_handle_esc(self) -> None:
        self.handle_esc()

    def action_handle_retry(self) -> None:
        self.handle_retry()

    def action_toggle_help(self) -> None:
        self.toggle_help()

    def action_handle_confirm_yes(self) -> None:
        self.handle_confirm_yes()

    def action_handle_confirm_no(self) -> None:
        self.handle_confirm_no()

    def action_handle_enter(self) -> None:
        self.handle_enter()

    def action_switch_gripper_left(self) -> None:
        self._switch_gripper_side("left")

    def action_switch_gripper_right(self) -> None:
        self._switch_gripper_side("right")

    def action_move_cursor_up(self) -> None:
        self.move_cursor("up")

    def action_move_cursor_down(self) -> None:
        self.move_cursor("down")

    def action_handle_batch(self) -> None:
        self.handle_batch()

    # ------------------------------------------------------------------ #
    # Handlers (kept as plain methods so other code can call them)
    # ------------------------------------------------------------------ #

    def handle_batch(self, _key: str | None = None) -> None:
        """Run batch tasks (auto)."""
        if self.state["status"] == "selecting":
            self.log_msg("info", "Batch mode enabled.")
            if len(self.batch) == 0:
                self.log_msg("warning", "Batch is empty. No tasks to execute.")
                return
            with self.state_lock:
                self.state["batch_index"] = 0
                task_id = self.batch[self.state["batch_index"]]
                self.state["curr_task"] = next((t for t in self.tasks if t["id"] == task_id), None)
                if self.state["curr_task"] is None:
                    self.log_msg("error", f"Task {task_id} not found in batch.")
                    return

    def _switch_gripper_side(self, side: str) -> None:
        """Switch the robot gripper state on the named side."""
        try:
            self.robot.switch_gripper(side)
            self.log_msg("info", f"Switched {side} gripper state.")
        except Exception as e:
            self.log_msg("error", f"Failed to switch {side} gripper: {e}")

    def handle_confirm_yes(self, _key: str | None = None) -> None:
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = True
            # If a confirm modal is currently on top, dismiss it with True.
            if isinstance(self.screen, ConfirmScreen):
                self.screen.dismiss(True)

    def handle_confirm_no(self, _key: str | None = None) -> None:
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = False
            if isinstance(self.screen, ConfirmScreen):
                self.screen.dismiss(False)

    def handle_enter(self, _key: str | None = None) -> None:
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = True
            if isinstance(self.screen, ConfirmScreen):
                self.screen.dismiss(True)
                return
        elif self.state["status"] in ("selecting", "done", "aborted", "error"):
            self.select_task_at_cursor()
            if self.state["status"] in ("done", "aborted", "error") and self.state["curr_task"] is not None:
                self.log_msg("info", f"Replaying task {self.state['curr_task']['id']}...")
                with self.state_lock:
                    self.state["status"] = "selecting"

    def handle_esc(self, _key: str | None = None) -> None:
        if self.state["status"] == "awaiting_confirm":
            with self.state_lock:
                self.state["confirm_result"] = False
            if isinstance(self.screen, ConfirmScreen):
                self.screen.dismiss(False)
                return
        elif self.state["status"] in ("error", "done", "aborted", "selecting"):
            with self.state_lock:
                self.state["curr_task"] = None
                self.state["status"] = "selecting"

    def toggle_help(self, _key: str | None = None) -> None:
        with self.state_lock:
            self.state["show_help"] = not self.state["show_help"]
        if self.state["show_help"]:
            self.push_screen(HelpScreen())
        else:
            if isinstance(self.screen, HelpScreen):
                self.pop_screen()

    def handle_retry(self, _key: str | None = None) -> None:
        if self.state["status"] == "error":
            self.log_msg("info", "Retrying task...")
            with self.state_lock:
                self.state["status"] = "selecting"

    def select_task_by_key(self, task: dict) -> None:
        """Select a task from the list via key binds."""
        if self.state.get("batch_index", -1) >= 0:
            self.log_msg("warn", "Cannot select task manually while a batch is running.")
            return
        if self.state["status"] in ("selecting", "error", "done", "aborted"):
            with self.state_lock:
                self.state["curr_task"] = task
                self.state["status"] = "selecting"
                # Sync the cursor highlight to the selected row so the TUI cursor
                # and the active card stay on the same line.
                if task in self.tasks:
                    self.state["task_cursor"] = self.tasks.index(task)
                    self._adjust_scroll()

    def select_task_at_cursor(self) -> None:
        """Reuse the existing 1-9 selection path with the task under the cursor."""
        if not self.tasks:
            return
        cursor = self.state["task_cursor"] % len(self.tasks)
        self.select_task_by_key(self.tasks[cursor])

    def move_cursor(self, _key: str | None = None) -> None:
        """Move the task-list cursor up or down (selected by _key) with wrap-around."""
        if not self.tasks or self.state["status"] not in self._NAV_ALLOWED:
            return
        step = -1 if _key == "up" else 1 if _key == "down" else 0
        if step == 0:
            return
        with self.state_lock:
            self.state["task_cursor"] = (self.state["task_cursor"] + step) % len(self.tasks)
            self._adjust_scroll()

    def toggle_pause(self, _key: str | None = None) -> None:
        """Toggle pause/resume state."""
        if self.state["status"] in ("executing", "paused"):
            with self.state_lock:
                self.state["pause"] = not self.state["pause"]
                self.state["status"] = "paused" if self.state["pause"] else "executing"

    def abort_task(self, _key: str | None = None) -> None:
        """Signal the running task to exit its loop."""
        if self.state["status"] in ("executing", "paused"):
            with self.state_lock:
                self.state["abort"] = True

    def finish_task(self, _key: str | None = None) -> None:
        """Signal the running task to finish and complete normally."""
        if self.state["status"] in ("executing", "paused"):
            with self.state_lock:
                self.state["finish"] = True

    # ------------------------------------------------------------------ #
    # User confirmation (blocking — called from worker thread)
    # ------------------------------------------------------------------ #

    def user_confirm(self, prompt: str) -> bool:
        """Block until the user confirms Yes/No.

        Safe to call from a worker thread. Schedules the modal push on the
        event loop via ``call_from_thread``, then blocks on a
        :class:`threading.Event` until the modal dismisses.
        """
        with self.state_lock:
            self.state["pending_confirm_prompt"] = prompt
            self.state["confirm_result"] = None
            self.state["status"] = "awaiting_confirm"

        confirm_event = threading.Event()
        self._confirm_event = confirm_event
        # Ask the event-loop thread to push the modal. Fire-and-forget from this side.
        self.call_from_thread(self._push_confirm_modal, prompt)

        # Block worker thread until dismissal.
        confirm_event.wait()
        self._confirm_event = None

        with self.state_lock:
            res = self.state["confirm_result"] if self.state["confirm_result"] is not None else False
            self.state["pending_confirm_prompt"] = None
        return res

    def _push_confirm_modal(self, prompt: str) -> None:
        """Push confirmation modal on the event-loop thread."""
        self.push_screen(ConfirmScreen(prompt), self._on_confirm_dismissed)

    def _on_confirm_dismissed(self, result: bool | None) -> None:
        """Confirm-screen dismissal callback (runs on the event-loop thread)."""
        with self.state_lock:
            self.state["confirm_result"] = result
        # Unblock the worker thread that called user_confirm().
        if self._confirm_event is not None:
            self._confirm_event.set()
            self._confirm_event = None

    # ------------------------------------------------------------------ #
    # Task execution pipeline (runs in worker thread)
    # ------------------------------------------------------------------ #

    def process_selected_task(self) -> None:
        """Execution pipeline helper for selected tasks.

        Runs in a worker thread (set up by :meth:`_check_and_run_task`).
        Mirrors the original ``process_selected_task`` with one difference:
        any cleanup happens here rather than from the main loop.
        """
        if not self.state["running"]:
            return
        with self.state_lock:
            self.state["status"] = "presetting"
        task_id = self.state["curr_task"]["id"] if self.state.get("curr_task") else None
        self.log_msg("info", f"Starting Task {task_id} process")
        try:
            self.set_proxy(self.state["curr_task"])
            self.set_robot(self.state["curr_task"])
        except Exception as e:
            self.log_msg("error", f"Presetting failed: {e}")
            with self.state_lock:
                self.state["status"] = "error"
            return

        with self.state_lock:
            self.state["status"] = "executing"

        self.execute_single_task(self.state["curr_task"])

        # Batch auto-advance: if batch_index is set, move on to the next task.
        with self.state_lock:
            current_batch_index = self.state.get("batch_index", -1)

        if current_batch_index >= 0:
            if self.state["status"] != "done" and self.user_confirm(
                f"Batch task {self.state['curr_task']['id']} ended with '{self.state['status']}'. Retry?"
            ):
                self.log_msg("info", f"Batch: retrying task {self.state['curr_task']['id']}.")
                with self.state_lock:
                    self.state["status"] = "selecting"
                    self.state["retry"] = True
                return
            with self.state_lock:
                next_idx = self.state["batch_index"] + 1
            if next_idx < len(self.batch):
                next_task = next((t for t in self.tasks if t["id"] == self.batch[next_idx]), None)
                if next_task is None:
                    self.log_msg("error", f"Batch task id {self.batch[next_idx]} not found; stopping.")
                    with self.state_lock:
                        self.state["batch_index"] = -1
                        self.state["curr_task"] = None
                        self.state["status"] = "selecting"
                    return
                self.log_msg("info", f"Batch: advancing to task {self.batch[next_idx]}.")
                with self.state_lock:
                    self.state["batch_index"] = next_idx
                    self.state["curr_task"] = next_task
                    self.state["status"] = "selecting"
            else:
                self.log_msg("info", f"Batch completed: {len(self.batch)} tasks.")
                with self.state_lock:
                    self.state["batch_index"] = -1
                    self.state["curr_task"] = None
                    self.state["status"] = "selecting"

    # ------------------------------------------------------------------ #
    # Status / Map / Logs content builders
    # ------------------------------------------------------------------ #

    def _build_task_list_content(self) -> str:
        """Render the task-list panel content."""
        visible_count = self._compute_visible_tasks()
        offset = self.state["task_scroll_offset"]
        cursor_idx = self.state["task_cursor"]
        total_tasks = len(self.tasks)
        # Clamp offset defensively (terminal may have shrunk since last render).
        offset = max(0, min(offset, max(0, total_tasks - 1)))
        self.state["task_scroll_offset"] = offset
        visible_tasks = self.tasks[offset : offset + visible_count]

        lines: list[str] = ["[bold green]📋 Task List[/bold green]"]
        for i, t in enumerate(visible_tasks):
            absolute_idx = offset + i
            is_active = self.state["curr_task"] and self.state["curr_task"]["id"] == t["id"]
            is_cursor = absolute_idx == cursor_idx

            if is_active:
                # All entries use supplementary-plane emoji (🟢🟡🔴✅🛑🏃) so
                # that every codepoint renders as 2 cells in any modern terminal.
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

            preset = t.get("preset", {})
            joint_motions = len(preset.get("joint_angles", []))
            motion_desc = f"{joint_motions} motion" if joint_motions > 0 else "none"
            location = preset.get("location", {}).get("name", "")

            stats = self.history.get(t["id"], {"runs": 0, "aborts": 0})
            runs = stats.get("runs", 0)
            aborts = stats.get("aborts", 0)
            completed = max(0, runs - aborts)
            success_rate = (completed / runs * 100.0) if runs > 0 else 0.0
            rate_str = f"{success_rate:.1f}%" if runs > 0 else "-"

            label_style = "bold cyan" if is_active else "dim"
            value_style = "bright_white" if is_active else "dim"

            cursor_prefix = "[bold green]👉 [/bold green]" if is_cursor else ""
            lines.append(
                f"[{title_style}]╭─ {cursor_prefix}Task {t['id']} ({status_desc})[/{title_style}]"
            )
            lines.append(
                f"[{border_color}]│[/] [{label_style}]Prompt :[/] [{value_style}]{t['prompt']}[/{value_style}]"
            )
            lines.append(
                f"[{border_color}]│[/] [{label_style}]Preset :[/] "
                f"[{value_style}]Location={location}, "
                f"Torso={preset.get('torso_height') or 0.5}m, "
                f"Joints={motion_desc}[/{value_style}]"
            )
            lines.append(
                f"[{border_color}]│[/] [{label_style}]Budget :[/] [{value_style}]{t['budget']:.0f}s[/]  "
                f"[{label_style}]Runs :[/] [{value_style}]{runs}[/]  "
                f"[{label_style}]Aborts :[/] [{value_style}]{aborts}[/]  "
                f"[{label_style}]SR :[/] [{value_style}]{rate_str}[/{value_style}]"
            )

            # Embed live progress bar and telemetry metrics directly inside the active task card.
            if is_active and self.state["status"] in ("executing", "paused", "done", "aborted"):
                pct = min(self.state["elapsed"] / t.get("budget", 10.0), 1.0) if t.get("budget", 10.0) > 0 else 0.0
                filled = int(pct * 25)
                pbar = "▓" * filled + "░" * (25 - filled)
                lines.append(
                    f"[{border_color}]│[/] [bold yellow]Status :[/] {pbar} "
                    f"[yellow]{self.state['elapsed']:.1f}/{t.get('budget', 10.0):.1f}s[/]  "
                    f"[cyan]⚡ {self.state['fps']}fps  steps={self.state['steps']}[/cyan]"
                )

            lines.append(f"[{border_color}]╰──────────────────────────────────────────────╯[/{border_color}]")

        hidden_below = total_tasks - (offset + len(visible_tasks))
        hidden_above = offset
        showing_start = offset + 1 if total_tasks else 0
        showing_end = offset + len(visible_tasks)
        indicator = f"[dim]showing {showing_start}-{showing_end} of {total_tasks}[/dim]"
        if hidden_above > 0:
            indicator += f"    [dim]↑ {hidden_above} more above [/dim]"
        if hidden_below > 0:
            indicator += f"    [dim]↓ {hidden_below} more below[/dim]"
        lines.append(indicator)

        return "\n".join(lines)

    def _build_status_content(self) -> str:
        """Render the status panel content (top of right column)."""
        status_text = ""

        # Batch sequence line (shown when self.batch is non-empty).
        if self.batch:
            in_batch = self.state.get("batch_index", -1) >= 0
            current_idx = self.state.get("batch_index", -1) if in_batch else -1
            batch_str_parts: list[str] = []
            for i, tid in enumerate(self.batch):
                if i > 0:
                    batch_str_parts.append("[dim] → [/dim]")
                if i == current_idx:
                    batch_str_parts.append(f"[bold bright_yellow]{tid}[/bold bright_yellow]")
                else:
                    batch_str_parts.append(f"[dim]{tid}[/dim]")
            status_text += "[cyan]Batch:[/cyan] " + "".join(batch_str_parts) + "\n"

        # Robot Status
        robot_text = f"Robot: {self.robot.__class__.__name__ if self.robot else '(unset)'}"
        if self.robot.is_connected:
            robot_text += " (connected ✓)"
        status_text += f"[green]{robot_text}[/green]\n"

        # Proxy Status
        proxy_meta = getattr(self.proxy, "policy_meta", {})
        if not isinstance(proxy_meta, dict):
            proxy_meta = {}
        policy_repo = proxy_meta.get("repo_id", "unknown")
        conn_state = getattr(self.proxy, "connection_state", "unconnected")
        connecting_addr = getattr(self.proxy, "connecting_addr", "None")
        addr = proxy_meta.get("addr", "None")
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        if conn_state == "connecting":
            proxy_text = (
                f"{spinner_frames[int(time.time() * 10) % len(spinner_frames)]} "
                f"Proxy: Connecting to {connecting_addr}..."
            )
            style = "[yellow]"
        elif conn_state == "connected":
            extra = (
                f"  @{policy_repo}" if policy_repo != "unknown" and addr != "rule-based" else ""
            )
            proxy_text = f"Proxy: {addr}{extra}"
            style = "[green]"
        elif conn_state == "failed":
            proxy_text = f"Proxy: Connection to {connecting_addr} failed"
            style = "[red]"
        else:
            proxy_text = f"Proxy: {addr}"
            style = "[dim white]"
        status_text += f"{style}{proxy_text}[/]\n"

        # Current Task Status
        curr_task_str = (
            f"{self.state['curr_task']['id']}: {self.state['curr_task']['prompt']}"
            if self.state["curr_task"]
            else "(none)"
        )
        status_text += f"[yellow]{self.state['status'].upper()} (Task: {curr_task_str})[/yellow]\n"

        # Operation Console
        op_text = ""
        op_color = "dim"
        status_title = "⚡ System & Operation Console"
        is_breathe_fast = int(time.time() * 2.5) % 2 == 0
        is_breathe_slow = int(time.time() * 1.0) % 2 == 0

        if self.state["status"] == "awaiting_confirm" and self.state["pending_confirm_prompt"]:
            op_color = "bright_yellow" if is_breathe_fast else "yellow"
            warning_dot = "●" if is_breathe_fast else " "
            status_title = f"[blink]{warning_dot}[/blink] Confirmation Required"
            op_text = f"[bold white]❓ {self.state['pending_confirm_prompt']}[/bold white]\n"
            yes_style = "bold bright_green" if is_breathe_fast else "bold green"
            no_style = "bold bright_red" if is_breathe_fast else "bold red"
            op_text += (
                f"[{yes_style}][Y] Yes / Enter[/]      "
                f"[{no_style}][N] No / Esc[/]"
            )
        elif self.state["status"] == "selecting":
            op_text = (
                "[bold green]All set. Ready to go. 👌[/bold green]\n"
                "[cyan]Select a task from list, or Q to quit.[/cyan]"
            )
            op_color = "green"
        elif self.state["status"] == "presetting":
            op_text = (
                "[bold yellow]🏃 Preset in progress...[/bold yellow]\n"
                "[dim]Prepare robot for execution.[/dim]"
            )
            op_color = "yellow"
        elif self.state["status"] == "executing":
            op_color = "bright_green" if is_breathe_slow else "green"
            warning_dot = "●" if is_breathe_slow else " "
            status_title = f"[blink]{warning_dot}[/blink] Policy Execution"
            op_text = (
                "[bold bright_green]▶️️ Policy execution in progress...[/bold bright_green]\n"
                f"[{'green' if is_breathe_slow else 'dim green'}]"
                "Real-time control loops are active. Press Space to pause, "
                "← to abort, → to finish.[/]"
            )
        elif self.state["status"] == "paused":
            op_color = "bright_yellow" if is_breathe_fast else "yellow"
            warning_dot = "●" if is_breathe_fast else " "
            status_title = f"[blink]{warning_dot}[/blink] Policy Paused"
            op_text = (
                "[bold bright_yellow]⏸️ Policy execution paused.[/bold bright_yellow]\n"
                f"[{'bright_yellow' if is_breathe_fast else 'dim yellow'}]"
                "Press Space to resume, ← to abort, or → to finish.[/]"
            )
        elif self.state["status"] == "done":
            op_text = (
                "[bright_green]✅ Active task completed successfully![/bright_green]\n"
                "[cyan]Press Enter to repeat, Esc to clear, or select another task.[/cyan]"
            )
            op_color = "bright_green"
        elif self.state["status"] == "aborted":
            op_text = (
                "[bright_red]⛔️ Task execution aborted by user.[/bright_red]\n"
                "[cyan]Press Enter to repeat, Esc to clear, or select another task.[/cyan]"
            )
            op_color = "bright_red"
        elif self.state["status"] == "error":
            op_text = (
                "[bright_red]⚠️ Task execution failed with error.[/bright_red]\n"
                "[white]Press Enter/R to retry, or Esc to return to task selection.[/white]"
            )
            op_color = "bright_red"
            status_title = "🚨 System Error"

        return (
            f"[bold {op_color}]{status_title}[/bold {op_color}]\n"
            f"{status_text}\n{op_text}"
        )

    def _build_map_content(self) -> str:
        """Render the map panel content (middle of right column)."""
        info = self.robot.get_chassis_info() or {}
        pos = info.get("position") or [0, 0]
        if hasattr(pos, "x"):
            x, y = pos.x, pos.y
        elif isinstance(pos, dict):
            x, y = pos.get("x", 0.0), pos.get("y", 0.0)
        else:
            x, y = (list(pos) + [0, 0])[:2]

        # Permanent trajectory; skip duplicate consecutive points within 5cm
        # so a stationary robot doesn't bloat the list with noise.
        last = self.trajectory[-1] if self.trajectory else None
        if last is None or abs(x - last[0]) > 0.05 or abs(y - last[1]) > 0.05:
            self.trajectory.append((x, y))

        # Range derived from A/B so A sits at the top row and B at the bottom
        # row of the grid.
        def _xy(p: Any) -> tuple[float, float] | None:
            if not p:
                return None
            try:
                return p[0], p[1]
            except (TypeError, IndexError, KeyError):
                try:
                    return p.x, p.y
                except AttributeError:
                    return None

        a_xy = _xy(self.locations.get("A"))
        b_xy = _xy(self.locations.get("B"))
        if a_xy and b_xy and a_xy[1] > b_xy[1]:
            y_max, y_min = a_xy[1] + 0.1, b_xy[1] - 0.1
        else:
            y_min, y_max = -2.2, 3.1  # fallback

        xys = [p for p in (_xy(v) for v in self.locations.values()) if p]
        if xys:
            xs = [p[0] for p in xys]
            x_min, x_max = min(xs) - 0.3, max(xs) + 0.3
        else:
            x_min, x_max = -1.2, 2.9

        x_range, y_range = x_max - x_min, y_max - y_min
        try:
            w, _ = self.size
        except Exception:
            w = 120
        panel_w = max(5, w // 2 - 2)
        panel_h = max(5, 25 - 3)  # map layout size = 25
        cell = max(x_range / panel_w, y_range / panel_h)
        # +1 so the grid covers both endpoints of the range (round() of
        # x_range / cell is the number of *intervals*, not cells).
        grid_w = int(round(x_range / cell)) + 1
        grid_h = int(round(y_range / cell)) + 1

        grid = [[" "] * grid_w for _ in range(grid_h)]

        def to_cell(wx: float, wy: float) -> tuple[int, int]:
            # Round half up (avoids banker's rounding shifting a point at an
            # exact midpoint by one cell in either direction).
            return int((wx - x_min) / cell + 0.5), int((y_max - wy) / cell + 0.5)

        # Named locations (overwrite trajectory).
        for name, loc in self.locations.items():
            if not name or not loc:
                continue
            try:
                lx, ly = loc[0], loc[1]
            except Exception:
                try:
                    lx, ly = loc.x, loc.y
                except Exception:
                    continue
            i, j = to_cell(lx, ly)
            if 0 <= i < grid_w and 0 <= j < grid_h:
                grid[j][i] = name[0].upper()

        # Robot at its projected position (overwrites anything at that cell).
        ri, rj = to_cell(x, y)
        if 0 <= ri < grid_w and 0 <= rj < grid_h:
            grid[rj][ri] = "🤖"

        title = "[bold cyan]📍 Robot Location[/bold cyan]"
        body_lines: list[str] = []
        body_lines.append(f"[cyan]Position: ({x:.2f}, {y:.2f}) [{info.get('status', 'unknown')}]")
        for row in grid:
            body_lines.append(f"[dim]{''.join(row)}[/dim]")
        return title + "\n" + "\n".join(body_lines)

    def _build_logs_content(self) -> str:
        """Render the logs panel content (bottom of right column)."""
        return (
            "[bold magenta]📜 Recent Logs[/bold magenta]\n"
            + "\n".join(self.logs_stream[-20:])
        )

    # ------------------------------------------------------------------ #
    # Layout math
    # ------------------------------------------------------------------ #

    def _compute_visible_tasks(self) -> int:
        """Estimate how many task cards fit in the left panel."""
        try:
            total_height = self.size.height
        except Exception:
            total_height = 24
        # Header (1) + footer (1) + outer panel top/bottom borders (2) = 4 lines
        # of chrome that are not available for task card content.
        available = total_height - 4
        return max(1, available // self._TASK_CARD_HEIGHT)

    def _adjust_scroll(self) -> None:
        """Shift ``task_scroll_offset`` so ``task_cursor`` stays in the visible window."""
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

    # ------------------------------------------------------------------ #
    # Task execution (worker thread)
    # ------------------------------------------------------------------ #

    def set_proxy(self, task: Any) -> None:
        """Set the proxy policy based on the selected task."""
        policy_addr = get_field(task, "policy")
        # Special case for rule-based policy
        if policy_addr == "rule-based":
            self.log_msg("info", "Using rule-based policy.")
            self.proxy.stop()  # Stop any existing proxy
            self.proxy.policy_meta = {"addr": "rule-based"}
            self.proxy.connection_state = "connected"
            return
        # General case: switch to a remote policy server
        self.proxy.switch_policy(policy_addr)
        if self.proxy.policy_meta["addr"] != policy_addr:
            self.log_msg(
                "error",
                f"Mismatch in proxy policy address! Expected {policy_addr}, got {self.proxy.policy_meta['addr']}",
            )
            raise ValueError("Proxy policy address mismatch")
        self.log_msg("info", f"Switch policy to: {policy_addr}")

    def set_robot(self, task: Any) -> None:
        """Preset the robot based on the task's preset configuration."""
        location = get_field(task, "location").get("name", None)
        torso_height = get_field(task, "torso_height")
        joint_angles = get_field(task, "joint_angles")
        is_rule_based = get_field(task, "policy") == "rule-based"
        thread_arm = None  # ensure closure references a stable name
        if joint_angles is not None and not is_rule_based:
            self.log_msg("info", f"Set joint angles: {len(joint_angles)} waypoints")
            thread_arm = threading.Thread(
                target=self.robot.move_arms_by_action, args=(joint_angles, 2.0, 30.0)
            )
            thread_arm.start()
        if location is not None:
            self.log_msg("info", f"Moving chasis to: {location}")
            thread_loc = threading.Thread(target=self.robot.move_to_location, args=(location,))
            thread_loc.start()
        else:
            thread_loc = None
        if torso_height is not None:
            self.log_msg("info", f"Set torso height: {torso_height}m")
            self.robot.set_torso_height(torso_height)
        if not is_rule_based and thread_arm is not None:
            thread_arm.join()
        if thread_loc is not None:
            thread_loc.join()
        self.log_msg("info", "Robot preset completed.")

    def execute_single_task(self, task: dict, fps: int = 30) -> None:
        """Execute the main logic of the task using the proxy."""
        budget = get_field(task, "budget", 10.0)
        prompt = get_field(task, "prompt", "")
        # Confirm task execution, skip in batch mode
        if (
            not self.state.get("batch_index", -1) >= 0
            and not self.user_confirm(f"Start task {task['id']}? prompt='{prompt}' budget={budget}s")
        ):
            self.log_msg("info", "Task execution cancelled.")
            with self.state_lock:
                self.state["status"] = "selecting"
                self.state["curr_task"] = None
            return

        # Confirm retry if the task is being retried
        if self.state.get("retry", False):
            if not self.user_confirm(f"Retry task {task['id']}? prompt='{prompt}' budget={budget}s"):
                self.log_msg("info", "Task retry cancelled.")
                with self.state_lock:
                    self.state["status"] = "selecting"
                    self.state["curr_task"] = None
                return
            else:
                with self.state_lock:
                    self.state["retry"] = False  # Reset retry flag

        # Increment total execution runs counter
        self.history[task["id"]]["runs"] += 1

        # Record the policy repo_id in history
        policy_addr = self.proxy.policy_meta.get("addr", "N/A")
        self.history[task["id"]]["policy"] = self.proxy.policy_meta.get("repo_id", policy_addr)

        # Initialize state variables for the execution loop
        dt = 1.0 / fps
        with self.state_lock:
            self.state["steps"], self.state["elapsed"] = 0, 0.0
            self.state["pause"] = False
            self.state["abort"] = False
            self.state["finish"] = False
            self.state["fps"] = fps

        if self.rerun is not None:
            self.rerun.switch_record()
            self.log_msg("info", f"Rerun recording started for task {task['id']}.")

        # Main loop
        while self.state["elapsed"] < budget and self.state["running"]:
            if self.state["abort"] or self.state["finish"]:
                break
            if self.state["pause"]:
                with self.state_lock:
                    self.state["status"] = "paused"
                time.sleep(0.05)
                continue

            with self.state_lock:
                self.state["status"] = "executing"
            cycle_start = time.perf_counter()
            t0 = time.perf_counter()

            act = None
            # Special case: rule-based policies execute joint_angles through robot.move_arms_by_action
            if self.proxy.policy_meta.get("addr") == "rule-based":
                self.log_msg("info", "Executing rule-based policy via robot.move_arms_by_action.")
                self.robot.move_arms_by_action(get_field(task, "joint_angles", []), timeout_sec=budget)
                with self.state_lock:
                    self.state["elapsed"] += time.perf_counter() - cycle_start
                break  # Rule-based policy completed, exit loop
            else:
                obs = self.robot.capture_observation()
                obs["task"] = prompt
                act = self.proxy.require_action(obs)

            if act is None:
                self.log_msg("warn", "No action received from proxy!")
            else:
                self.robot.send_action(act)

            if self.rerun is not None and obs is not None and act is not None:
                label = "policy" if self.robot.policy_control else "teleop"
                label = "policy"  # override to always log as policy action for now
                self.rerun.log({**obs, label: act})

            time.sleep(max(0, dt - (time.perf_counter() - t0)))
            with self.state_lock:
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

        with self.state_lock:
            self.state["status"] = "aborted" if self.state["abort"] else "done"
        self.log_msg("info", f"Task {task['id']} completed with status: {self.state['status']}")

    # ------------------------------------------------------------------ #
    # Cleanup & statistics
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Stop components and break the app out of its run loop."""
        with self.state_lock:
            self.state["running"] = False
        for component, name in [
            (self.robot, "Robot"),
            (self.proxy, "Proxy"),
            (self.rerun, "Rerun"),
        ]:
            if component is None:
                continue
            try:
                component.stop()
            except Exception:
                self.log_msg("error", f"Error stopping {name} component: {type(component).__name__}")

    def save_and_print_statistics(self) -> None:
        """Log summary statistics to a file in structured JSON format and print a beautiful summary Table on exit."""
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
        tasks_stats: list[dict[str, Any]] = []

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

            table.add_row(str(t_id), t["prompt"], str(runs), str(aborts), rate_str, time_str, stats["policy"])

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
                f"\n[green]✔ Telemetry JSON report written to: "
                f"[bold underline]{log_file_path}[/bold underline][/green]"
            )
        except Exception as e:
            console.print(f"[red]Failed to write statistics to log file: {e}[/red]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = CoordinatorApp()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        app.save_and_print_statistics()


if __name__ == "__main__":
    main()

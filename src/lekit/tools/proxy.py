"""ZMQ REQ/REP client that streams policy actions from a remote server asynchronously."""

import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp

import torch
import torch.nn.functional as F  # noqa
import zmq
from rich import print
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


# ---------- Temporal Ensembling Aggregators ----------
def _fold(alpha: float) -> Callable[[list[torch.Tensor]], torch.Tensor]:
    """Sequential fold: result = α·prev + (1-α)·next across the history list."""

    def fn(history: list[torch.Tensor]) -> torch.Tensor:
        out = history[0]
        for action in history[1:]:
            out = alpha * out + (1 - alpha) * action
        return out

    return fn


def _adaptive_ensemble(history: list[torch.Tensor], alpha: float = 3.0) -> torch.Tensor:
    """Cosine-similarity-weighted average; newer actions weigh more when similar."""
    if len(history) == 1:
        return history[0]
    ref = history[-1].reshape(-1)
    sims = [F.cosine_similarity(c.reshape(-1), ref, dim=0, eps=1e-7).item() for c in history]
    weights = torch.tensor([exp(alpha * s) for s in sims])
    weights /= weights.sum()
    return sum(w * a for w, a in zip(weights.tolist(), history, strict=True))


_AGGREGATOR_FACTORIES: dict[str, Callable[..., Callable]] = {
    "latest_only": lambda alpha: lambda hist: hist[-1],
    "weighted_average": lambda alpha: _fold(0.3),
    "average": lambda alpha: _fold(0.5),
    "conservative": lambda alpha: _fold(0.7),
    "adaptive_ensemble": lambda alpha: lambda hist: _adaptive_ensemble(hist, alpha),
}


def _make_aggregator(name: str, alpha: float) -> Callable:
    try:
        return _AGGREGATOR_FACTORIES[name](alpha)
    except KeyError as e:
        raise ValueError(f"Unknown aggregate function {name!r}. Available: {list(_AGGREGATOR_FACTORIES)}") from e


# ---------- ProxyConfig ----------
@dataclass
class ProxyConfig:
    """Configuration for the async inference proxy."""

    chunk_size_threshold: float = field(
        default=1.0, metadata={"help": "Buffer fill ratio before allowing new observations"}
    )
    aggregate_fn_name: str = field(
        default="adaptive_ensemble", metadata={"help": f"Aggregator. Options: {list(_AGGREGATOR_FACTORIES)}"}
    )
    adaptive_ensemble_alpha: float = field(default=3.0, metadata={"help": "Sharpness for adaptive_ensemble weighting"})
    handshake_timeout: float = field(default=3.0, metadata={"help": "ZMQ handshake and request timeout in seconds"})

    def __post_init__(self):
        if not 0 <= self.chunk_size_threshold <= 1.0:
            raise ValueError(f"chunk_size_threshold must be in [0, 1.0], got {self.chunk_size_threshold}")
        if self.adaptive_ensemble_alpha < 0:
            raise ValueError(f"adaptive_ensemble_alpha must be non-negative, got {self.adaptive_ensemble_alpha}")
        if self.handshake_timeout <= 0:
            raise ValueError(f"handshake_timeout must be positive, got {self.handshake_timeout}")
        _make_aggregator(self.aggregate_fn_name, self.adaptive_ensemble_alpha)


# ---------- Proxy ----------
class Proxy:
    def __init__(self, config: ProxyConfig | None = None):
        self.config = config or ProxyConfig()
        self._init_state()

    def _init_state(self) -> None:
        self.policy_meta: dict = {}
        self.connection_state = "unconnected"
        self.connecting_addr = None

        # Core runtime state
        self.timestep = 0
        self.chunk_size = 1
        self.stop_event = threading.Event()
        self._obs_event = threading.Event()
        self._last_observation: tuple[dict, int] | None = None

        # History buffer storage (timestep -> list of raw incoming actions)
        self._history: dict[int, list[torch.Tensor]] = {}
        # Action buffer storage (timestep -> aggregated action)
        self._action_buffer: dict[int, torch.Tensor] = {}

        self._request_in_flight = False
        self.action_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._aggregator = _make_aggregator(self.config.aggregate_fn_name, self.config.adaptive_ensemble_alpha)

    def _connect(self, addr: str) -> None:
        """Establish ZMQ socket and perform handshake verification."""
        self.context = zmq.Context()
        self._socket = self.context.socket(zmq.REQ)
        self._socket.connect(addr)
        self._addr = addr

        # Get Policy Server metadata, raise TimeoutError if timeout or fail
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        self._socket.send(pickle.dumps({"__request_policy_meta__": True}))

        deadline_ms = int(self.config.handshake_timeout * 1000)
        if dict(poller.poll(timeout=deadline_ms)):
            msg = pickle.loads(self._socket.recv())
            self.policy_meta = msg.get("policy_meta", {})
            self.policy_meta["addr"] = addr
        else:
            raise TimeoutError(
                f"ZMQ handshake connection to {addr!r} timed out after {self.config.handshake_timeout}s."
            )
        self.start()

    def start(self) -> None:
        if self.action_thread is not None and self.action_thread.is_alive():
            return
        self.stop_event.clear()
        self.action_thread = threading.Thread(target=self.request_policy, daemon=True)
        self.action_thread.start()

    def stop(self) -> None:
        try:
            self.stop_event.set()
            if self.action_thread is not None and self.action_thread.is_alive():
                self.action_thread.join()
            if hasattr(self, "_socket"):
                self._socket.close(linger=0)
                self.context.term()
        except Exception as e:
            print(f"[bright_red]Error stopping proxy: {e}[/bright_red]")

    def switch_policy(self, addr: str) -> None:
        """Disconnect, reset state, and connect to a different policy server."""
        self.stop()
        self._init_state()
        self.connection_state = "connecting"
        self.connecting_addr = addr
        try:
            self._connect(addr)
            self.connection_state = "connected"
        except Exception as e:
            self.connection_state = "failed"
            raise e

    # ---------- Observation & Buffer Operations ----------
    def update_observation(self, observation: dict, timestep: int) -> None:
        """Provide the latest observation (no-op if a request is in flight or pending)."""
        with self._lock:
            busy = self._request_in_flight
            obs_pending = self._obs_event.is_set()
            buffer_full = len(self._action_buffer) / self.chunk_size > self.config.chunk_size_threshold

            if busy or obs_pending or buffer_full:
                return

            self._last_observation = (observation, timestep)
        self._obs_event.set()

    def _aggregate_incoming(self, incoming: dict[int, torch.Tensor]) -> None:
        with self._lock:
            current = self.timestep
            for ts, action in incoming.items():
                if ts >= current:
                    self._history.setdefault(ts, []).append(action)
            self._history = {ts: h for ts, h in self._history.items() if ts >= current}
            self._action_buffer = {ts: self._aggregator(self._history[ts]) for ts in self._history}

    def _pop_action(self, expected_timestep: int) -> torch.Tensor | None:
        with self._lock:
            action = self._action_buffer.pop(expected_timestep, None)
            self._history.pop(expected_timestep, None)
        return action

    # ---------- Internal Daemon Request Thread ----------
    def request_policy(self) -> None:
        """Request policy server inference results and aggregate them."""
        while not self.stop_event.is_set():
            if not self._obs_event.wait(timeout=0.1):
                continue
            self._obs_event.clear()

            with self._lock:
                obs_data = self._last_observation
                if obs_data is not None:
                    observation, timestep = obs_data
                else:
                    observation, timestep = None, None

            if observation is None:
                continue

            payload = {"timestep": timestep, "timestamp": time.time(), "observation": observation}
            try:
                with self._lock:
                    self._request_in_flight = True
                self._socket.send(pickle.dumps(payload))
                msg = self._socket.recv()
            except zmq.error.Again:
                continue
            finally:
                with self._lock:
                    self._request_in_flight = False

            actions = pickle.loads(msg)
            with self._lock:
                self.chunk_size = max(self.chunk_size, len(actions))
            self._aggregate_incoming(actions)

    # ---------- Main Thread API ----------
    def require_action(self, observation: dict | None, timeout_s: float = 3.0) -> torch.Tensor | None:
        if not hasattr(self, "_socket"):
            raise RuntimeError(
                "Proxy is not connected. Call proxy.switch_policy(addr) to connect to a policy server first."
            )

        with self._lock:
            current_ts = self.timestep

        if observation is not None:
            self.update_observation(observation, current_ts)

        deadline = time.perf_counter() + timeout_s
        warned = False
        while True:
            with self._lock:
                expected = self.timestep
            action = self._pop_action(expected)
            if action is not None:
                with self._lock:
                    self.timestep = expected + 1
                return action
            if not warned and time.perf_counter() >= deadline:
                print(
                    f"[bright_red]Timeout requiring action after {timeout_s:.0f}s[/bright_red]",
                    end="\r",
                    flush=True,
                )
                warned = True
            time.sleep(0.001)


# ---------- Demo CLI ----------
class _DemoMockRobot:
    name = "mock_robot"

    def __init__(self, obs_features: dict[str, dict]):
        self._obs = {
            k: (
                torch.randint(0, 256, f["shape"], dtype=torch.uint8)
                if "image" in k
                else torch.randn(f["shape"], dtype=torch.float32)
            )
            for k, f in obs_features.items()
        }

    def get_observation(self) -> dict:
        return self._obs

    def send_action(self, _action) -> None:
        pass


def main():
    proxy = Proxy()
    try:
        dt = 1.0 / 30.0
        for addr in ("tcp://172.20.66.32:9001", "tcp://172.20.66.32:9002"):
            proxy.switch_policy(addr)
            robot = _DemoMockRobot(obs_features=proxy.policy_meta.get("input_features", {}))
            with Progress(
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                BarColumn(bar_width=30),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[bold green]FPS: {task.fields[fps]:>6.2f}[/bold green]"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(addr, total=100, fps=0.0)
                for _ in range(100):
                    t0 = time.perf_counter()
                    obs = robot.get_observation()
                    obs["task"] = "do nothing"
                    action = proxy.require_action(obs)
                    if action is not None:
                        robot.send_action(action)
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                    fps = 1.0 / (time.perf_counter() - t0)
                    progress.update(task, advance=1, fps=fps)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        proxy.stop()


if __name__ == "__main__":
    main()

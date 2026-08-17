import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from math import exp
from queue import Queue

import draccus
import torch
import torch.nn.functional as functional
import zmq
from rich import print


class MockRobot:
    name = "mock_robot"

    def __init__(self, obs_features: dict[str, dict]):
        self.obs_features = obs_features
        self._random_observation = {
            key: torch.randint(0, 256, feature["shape"], dtype=torch.uint8)
            if "image" in key
            else torch.randn(feature["shape"], dtype=torch.float32)
            for key, feature in self.obs_features.items()
        }

    def get_observation(self) -> dict:
        # Return cached zero-filled observation for minimal overhead.
        return self._random_observation

    def send_action(self, action_dict):
        pass


# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
    "adaptive_ensemble": None,
}


@dataclass
class PolicyClientConfig:
    # Network configuration
    host: str = field(default="localhost", metadata={"help": "Server host"})
    port: int = field(default=8001, metadata={"help": "Server port"})

    # The threshold for the chunk size before sending a new observation to the server
    chunk_size_threshold: float = field(default=1.0, metadata={"help": "Threshold for chunk size"})

    # Aggregate function configuration
    aggregate_fn_name: str = field(
        default="adaptive_ensemble",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )
    adaptive_ensemble_alpha: float = field(
        default=3.0,
        metadata={"help": "Alpha used by adaptive_ensemble weighting across repeated timestep predictions."},
    )

    def __post_init__(self):
        if not self.host:
            raise ValueError("host cannot be empty")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in range [1, 65535], got {self.port}")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1.0:
            raise ValueError(f"chunk_size_threshold must be in [0, 1.0], got {self.chunk_size_threshold}")

        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            available = list(AGGREGATE_FUNCTIONS.keys())
            raise ValueError(f"Unknown aggregate function '{self.aggregate_fn_name}'. Available: {available}")
        self.aggregate_fn = AGGREGATE_FUNCTIONS[self.aggregate_fn_name]

        if self.adaptive_ensemble_alpha < 0:
            raise ValueError(
                f"adaptive_ensemble_alpha must be non-negative, got {self.adaptive_ensemble_alpha}"
            )


class PolicyClient:
    def __init__(self, config: PolicyClientConfig):
        self.config = config

        self.context = zmq.Context()
        self._socket = self.context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self.config.host}:{self.config.port}")

        # Get policy metadata from the server.
        self.policy_meta = self.request_policy_meta()
        self.policy_name = self.policy_meta.get("name", "N/A")
        self.policy_config = {
            key: value for key, value in self.policy_meta.items() if key not in {"name", "ckpt"}
        }
        self.policy_repo_id = self.policy_config.get("repo_id", "N/A")
        self.input_features = self.policy_config.get("input_features", {})
        self.output_features = self.policy_config.get("output_features", {})
        self.policy_name_with_repo_id = f"{self.policy_name}<{self.policy_repo_id}>"
        print(f"[bright_yellow]Using policy: {self.policy_name_with_repo_id}[/bright_yellow]")
        print("[bright_yellow]Input features:[/bright_yellow]")
        for key, value in self.input_features.items():
            print(f"  - {key}: {value}")
        print("[bright_yellow]Output features:[/bright_yellow]")
        for key, value in self.output_features.items():
            print(f"  - {key}: {value}")
        print(f"Connected to server: {self.config.host}:{self.config.port}")

        # Initialize client side variables
        self.timestep = 0  # Track the number of timesteps of action
        self.timestep_lock = threading.Lock()  # Protect timestep variable
        self._last_observation = None
        self._last_observation_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._request_in_flight = False
        self._request_in_flight_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self._action_history_by_timestep: dict[int, list[torch.Tensor]] = {}
        self.action_queue_size = []
        self.action_thread = None
        self.chunk_size = 1

        self.start()

    def start(self):
        """"""
        if self.action_thread is not None and self.action_thread.is_alive():
            print("[bright_yellow]PolicyClient already running[/bright_yellow]")
            return

        self.stop_event.clear()
        self.action_thread = threading.Thread(target=self.request_actions, daemon=True)
        self.action_thread.start()
        # Block until we have at least one observation and an action available.
        print("[bright_yellow]PolicyClient started and action thread launched[/bright_yellow]")

    def stop(self):
        self.stop_event.set()
        if self.action_thread is not None and self.action_thread.is_alive():
            self.action_thread.join()
        self._socket.close(linger=0)
        self.context.term()
        print("[bright_yellow]PolicyClient stopped[/bright_yellow]")

    def reset(self):
        with self.action_queue_lock:
            self.action_queue = Queue()
            self._action_history_by_timestep = {}
        self.action_queue_size = []
        with self.timestep_lock:
            self.timestep = 0  # Track the number of timesteps of action
        with self._last_observation_lock:
            self._last_observation = None
        with self._request_in_flight_lock:
            self._request_in_flight = False
        self._obs_event.clear()

    def _request(self, payload: dict, response_key: str, default=None):
        self._socket.send(pickle.dumps(payload))
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        start_time, end_time = time.time(), None
        while True:
            events = dict(poller.poll(timeout=1000))
            if self._socket in events:
                message = pickle.loads(self._socket.recv())
                if not message.get(response_key):
                    raise RuntimeError(f"Failed to get '{response_key}' from the policy server.")
                break
            else:
                elapsed = int(time.time() - start_time)
                print(
                    f"[bright_yellow]waiting for ZMQ server ({elapsed}s)...[/bright_yellow]",
                    end="\r",
                    flush=True,
                )
        if end_time is not None:
            print(f"[bright_yellow]Connected to server after {elapsed}s.[/bright_yellow]")
        return message.get(response_key, default)

    def request_policy_meta(self):
        return self._request({"__request_policy_meta__": True}, "policy_meta")

    def update_observation(self, observation: dict) -> None:
        """Update the current observation from external code and enqueue it for the action request thread."""
        # Do not enqueue a new observation while one request is already pending.
        with self._request_in_flight_lock:
            request_in_flight = self._request_in_flight

        if not request_in_flight and not self._obs_event.is_set():
            with self._last_observation_lock:
                self._last_observation = observation
            self._obs_event.set()

    @property
    def last_observation(self) -> dict | None:
        with self._last_observation_lock:
            return self._last_observation

    def _adaptive_ensemble_action(self, action_history: list[torch.Tensor]) -> torch.Tensor:
        if len(action_history) == 1:
            return action_history[0]

        reference = action_history[-1].reshape(-1)
        similarities = []
        for candidate in action_history:
            similarities.append(
                functional.cosine_similarity(candidate.reshape(-1), reference, dim=0, eps=1e-7).item()
            )

        weight_values = [exp(self.config.adaptive_ensemble_alpha * similarity) for similarity in similarities]
        weight_sum = sum(weight_values)
        normalized_weights = [weight / weight_sum for weight in weight_values]

        aggregated_action = torch.zeros_like(action_history[-1])
        for weight, action in zip(normalized_weights, action_history, strict=True):
            aggregated_action = aggregated_action + action * weight

        return aggregated_action

    def _aggregate_action_history(
        self, action_history: list[torch.Tensor], aggregate_fn: Callable | None = None
    ) -> torch.Tensor:
        if self.config.aggregate_fn_name == "adaptive_ensemble":
            return self._adaptive_ensemble_action(action_history)

        if aggregate_fn is None:
            return action_history[-1]

        aggregated_action = action_history[0]
        for action in action_history[1:]:
            aggregated_action = aggregate_fn(aggregated_action, action)
        return aggregated_action

    def _rebuild_action_queue_locked(self, aggregate_fn: Callable | None = None) -> None:
        rebuilt_queue = Queue()
        for ts in sorted(self._action_history_by_timestep):
            rebuilt_queue.put(
                {
                    "timestep": ts,
                    "action": self._aggregate_action_history(
                        self._action_history_by_timestep[ts], aggregate_fn
                    ),
                }
            )
        self.action_queue = rebuilt_queue

    def aggregate_actions_in_queue(self, incoming_actions: dict, aggregate_fn: Callable = None):
        """Aggregate actions in the queue (SAFE & CORRECT VERSION)"""
        with self.timestep_lock:
            current_timestep = self.timestep

        with self.action_queue_lock:
            self._action_history_by_timestep = {
                ts: history
                for ts, history in self._action_history_by_timestep.items()
                if ts >= current_timestep
            }

            for ts, new_act in incoming_actions.items():
                if ts < current_timestep:
                    continue

                self._action_history_by_timestep.setdefault(ts, []).append(new_act)

            self._rebuild_action_queue_locked(aggregate_fn)

    def request_actions(self):
        while not self.stop_event.is_set():
            # Wait for the latest observation provided via update_observation().
            if not self._obs_event.wait(timeout=0.1):
                continue

            self._obs_event.clear()
            with self._last_observation_lock:
                observation = self._last_observation
            if observation is None:
                continue

            payload = {}
            # Include current timestep in the payload for better synchronization and debugging
            with self.timestep_lock:
                payload["timestep"] = self.timestep

            # Add timestamp to payload for debugging and latency measurement
            payload["timestamp"] = time.time()
            payload["observation"] = observation
            try:
                with self._request_in_flight_lock:
                    self._request_in_flight = True

                self._socket.send(pickle.dumps(payload))
                message = self._socket.recv()
            except zmq.error.Again:
                continue
            finally:
                with self._request_in_flight_lock:
                    self._request_in_flight = False

            actions = pickle.loads(message)
            self.chunk_size = max(self.chunk_size, len(actions))

            # Aggregate actions with the same timestep in the queue
            self.aggregate_actions_in_queue(actions, aggregate_fn=self.config.aggregate_fn)

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _pop_action_for_timestep(self, expected_timestep: int) -> dict | None:
        """Atomically pop the action for expected_timestep if present.

        Drops stale actions (older timesteps) and preserves queue ordering when
        only future actions are currently available.
        """
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None

            future_head = None
            matched_action = None

            while not self.action_queue.empty():
                action = self.action_queue.get_nowait()
                ts = action["timestep"]

                if ts < expected_timestep:
                    self._action_history_by_timestep.pop(ts, None)
                    continue

                if ts == expected_timestep:
                    matched_action = action
                    self._action_history_by_timestep.pop(ts, None)
                    break

                # ts > expected_timestep: keep this and all remaining future actions.
                future_head = action
                break

            if future_head is not None:
                remaining = []
                while not self.action_queue.empty():
                    remaining.append(self.action_queue.get_nowait())
                self.action_queue.put(future_head)
                for item in remaining:
                    self.action_queue.put(item)

            if matched_action is not None:
                self.action_queue_size.append(self.action_queue.qsize())
                return matched_action

            return None

    def ready_to_send_observation(self, observation: dict) -> bool:
        """Check if the observation is ready to be sent."""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.chunk_size <= self.config.chunk_size_threshold

    def require_action(self, observation: dict, timeout_s: float = 3.0) -> torch.Tensor | None:
        if observation is not None and self.ready_to_send_observation(observation):
            self.update_observation(observation)

        deadline = time.perf_counter() + timeout_s
        wait_start = time.perf_counter()
        action = None
        while action is None:
            with self.timestep_lock:
                expected_timestep = self.timestep

            action = self._pop_action_for_timestep(expected_timestep)
            if action is not None:
                break

            if time.perf_counter() >= deadline:
                elapsed = time.perf_counter() - wait_start
                print(
                    f"[bright_red]Timeout requiring action after {elapsed:.0f} seconds.[/bright_red]",
                    end="\r",
                    flush=True,
                )
                return None
            time.sleep(0.001)

        # Increment timestep after getting action
        with self.timestep_lock:
            self.timestep = max(self.timestep, action["timestep"] + 1)

        return action["action"]


@draccus.wrap()
def async_client_zmq(config: PolicyClientConfig):
    print(asdict(config))

    policy = PolicyClient(config)
    robot = MockRobot(obs_features=policy.input_features)

    try:
        dt = 1.0 / 30.0
        while True:
            step_start = time.perf_counter()
            observation = robot.get_observation()
            observation["task"] = "do nothing"
            actions = policy.require_action(observation)
            if actions is not None:
                robot.send_action(actions)
            time.sleep(max(0.0, dt - (time.perf_counter() - step_start)))
            actual_fps = 1.0 / (time.perf_counter() - step_start)
            print(f"Actual FPS: {actual_fps:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        import matplotlib.pyplot as plt

        plt.plot(policy.action_queue_size)
        plt.title("Action Queue Size Over Time")
        plt.savefig("action_queue_size.png")

        policy.stop()


if __name__ == "__main__":
    async_client_zmq()

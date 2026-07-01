# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python scripts/tools/policy_server.py \
    --host=0.0.0.0 \
    --port=8001 \
    --policy_type="act" \
    --pretrained_name_or_path=<path/to/checkpoint>
```
"""

import pickle  # nosec
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
import zmq
from rich import print
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def get_local_ip():
    """Get the local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except Exception:
        return "127.0.0.1"


def hwc_2_chw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from HWC to CHW format."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor in HWC format, got {tensor.ndim}D tensor")
    if tensor.dtype == torch.uint8:
        tensor = (tensor.clamp(0, 255) / 255.0).to(torch.float32)
    h, w, c = tensor.shape
    if c in (1, 3, 4) and h > c and w > c:
        return tensor.permute(2, 0, 1).contiguous()
    return tensor


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8000, metadata={"help": "Port number to bind the server to"})

    # Policy configuration
    policy_type: str = field(default="act", metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(default="", metadata={"help": "Pretrained model name or path"})
    policy_device: str = field(default="cuda", metadata={"help": "Device for policy inference"})
    rename_map: dict[str, str] = field(default_factory=dict)

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(default=50, metadata={"help": "Number of actions per chunk"})

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if (Path(self.pretrained_name_or_path) / "pretrained_model").exists():
            self.pretrained_name_or_path = str(Path(self.pretrained_name_or_path) / "pretrained_model")
            if not (Path(self.pretrained_name_or_path) / "config.json").exists():
                print(f"[bright_red]Invalid checkpoint path: {self.pretrained_name_or_path}.[/bright_red]")
                raise ValueError(f"Config file does not exist: {self.pretrained_name_or_path}")
        else:
            print(f"[bright_red]Invalid checkpoint path: {self.pretrained_name_or_path}.[/bright_red]")
            raise ValueError(f"Invalid checkpoint path: {self.pretrained_name_or_path}.[/bright_red]")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")


class PolicyServer:
    def __init__(self, config: PolicyServerConfig):
        # Load policy
        policy_class = get_policy_class(config.policy_type)
        self.policy_name = policy_class.__name__
        self.ckpt = config.pretrained_name_or_path
        print(f'[bright_yellow]Loading model: "{config.pretrained_name_or_path}" ...[/bright_yellow]')
        t0 = time.perf_counter()
        self.policy = policy_class.from_pretrained(config.pretrained_name_or_path)
        print(self.policy.config)
        print(
            f"[bright_green]Taken {time.perf_counter() - t0:.2f}s to put policy on {config.policy_device}.[/bright_green]"
        )
        self.chunk_size = config.actions_per_chunk
        self.action_dim = 0
        # Make preprocessor and postprocessor
        rename_map = config.rename_map or {
            str(k).replace("_camera", ""): str(k)
            for k in self.policy.config.input_features
            if "camera" in str(k)
        }  # In compatibility with older checkpoints used to have "camera" in the observation keys
        print(f"Using rename_map: {rename_map}")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=config.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": {"device": config.policy_device},
                "rename_observations_processor": {"rename_map": rename_map},
            },
            postprocessor_overrides={"device_processor": {"device": config.policy_device}},
        )
        print("preprocessor steps:")
        for idx, step in enumerate(self.preprocessor.steps):
            device = getattr(step, "device", None)
            print(f"  [{idx}] {step.__class__.__name__} {'device=' + str(device) if device else ''}")
        print("postprocessor steps:")
        for idx, step in enumerate(self.postprocessor.steps):
            device = getattr(step, "device", None)
            print(f"  [{idx}] {step.__class__.__name__} {'device=' + str(device) if device else ''}")

        self._warmup_policy()

        self.stop_event = threading.Event()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{config.host}:{config.port}")
        self.bind_port = config.port
        self.console = Console()

    def _warmup_policy(self, steps=3):
        """Warm up the policy by running dummy inferences."""
        start_time = time.monotonic()
        for _ in range(steps):
            obs = {k: torch.rand(v.shape) for k, v in self.policy.config.input_features.items()}
            obs["task"] = "do nothing"
            output = self.predict_action_chunk(obs, i0=0)
        elapsed = time.perf_counter() - start_time
        print(f"[bright_green]Taken {elapsed:.2f}s to warm up the policy.[/bright_green]")
        print(f"[bright_green]Output shape: {output[next(iter(output))].shape}[/bright_green]")

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    @property
    def policy_config(self):
        return asdict(self.policy.config)

    def predict_action_chunk(self, observation: dict[str, torch.Tensor], i0: int) -> dict[int, torch.Tensor]:
        """Predict action chunk for the given observation.
        Args:
           observation: A dictionary of observation features.
           i0: The timestep to start the action chunk from.
        Returns:
              A dictionary mapping timestep to action tensor.
        """
        # Check observation dict values for torch.Tensor type
        for k in list(observation.keys()):  # use list to avoid iteration error
            v = observation[k]
            if k == "task":
                continue
            if not isinstance(v, torch.Tensor):
                observation[k] = torch.from_numpy(v)
            if "images" in k:
                observation[k] = hwc_2_chw(observation[k])
            v = observation[k]

        observation = self.preprocessor(observation)
        action_tensor = self.policy.predict_action_chunk(observation)
        action_tensor = action_tensor[:, : self.chunk_size, :]
        _, self.chunk_size, self.action_dim = action_tensor.shape  # (B, chunk_size, action_dim)

        # Process each action in the chunk
        processed_actions = []
        for i in range(self.chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        action_tensor = action_tensor.detach().cpu()

        # Convert to dict with timestep keys
        actions = {i0 + i: action for i, action in enumerate(action_tensor)}

        return actions

    def run(self) -> None:
        ip = get_local_ip()
        print(f"[bright_green]Policy server listen on: tcp://{ip}:{self.bind_port}[/bright_green]")
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        last_step = 0
        try:
            with Live(console=self.console, refresh_per_second=4) as live:
                while self.stop_event.is_set() is False:
                    delay_time = time.perf_counter()
                    events = dict(poller.poll(timeout=200))
                    delay_time = (time.perf_counter() - delay_time) * 1000  # convert to ms

                    if self.socket not in events:
                        continue

                    unpac_time = time.perf_counter()
                    try:
                        payload = self.socket.recv()
                        message = pickle.loads(payload)
                    except Exception as e:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"[bold red]{timestamp}: Failed to receive or unpickle message: {e}[/bold red]")
                        self.socket.send(b"")
                        continue
                    unpac_time = (time.perf_counter() - unpac_time) * 1000  # convert to ms

                    if not isinstance(message, dict):
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"[bold red]{timestamp}: Invalid message: {type(message).__name__}[/bold red]")
                        self.socket.send(b"")
                        continue

                    if message.get("__request_policy_meta__", False):
                        client_endpoint = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"[bold cyan]{timestamp}: Connected from {client_endpoint}[/bold cyan]")
                        meta = {"name": self.policy_name, "ckpt": self.ckpt, **self.policy_config}
                        self.socket.send(pickle.dumps({"policy_meta": meta}))
                        continue

                    observation = message.get("observation", {})
                    obs_info = {
                        k: (v.dtype, tuple(v.shape))
                        for k, v in observation.items()
                        if isinstance(v, (torch.Tensor, np.ndarray))
                    }

                    timestamp = message.get("timestamp", time.time())
                    timestep = message.get("timestep", 0)

                    infer_time = time.perf_counter()
                    actions = self.predict_action_chunk(observation, i0=timestep)
                    infer_time = (time.perf_counter() - infer_time) * 1000  # convert to ms
                    self.socket.send(pickle.dumps(actions))

                    # Update live panel
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

                    table = Table(show_header=False, box=None, padding=(0, 1))
                    table.add_column(style="bold cyan", no_wrap=True)
                    table.add_column(style="white", no_wrap=True)
                    table.add_column(style="bold cyan", no_wrap=True)
                    table.add_column(style="white", no_wrap=True)

                    skip = timestep - last_step
                    table.add_row("time steps", f"{timestep} (+{skip})", "delay_time", f"{delay_time:.2f} ms")
                    last_step = timestep
                    table.add_row("chunk_size", str(self.chunk_size), "unpac_time", f"{unpac_time:.2f} ms")
                    table.add_row("action_dim", str(self.action_dim), "infer_time", f"{infer_time:.2f} ms")
                    table.add_row("", "", "", "")
                    if "task" in observation:
                        table.add_row("task", observation["task"], "", "")
                    for k, v in obs_info.items():
                        table.add_row(k, str(v[0]), str(v[1]))
                    panel = Panel(
                        table,
                        title=f"{self.policy_name} <{self.policy_config.get('repo_id', '')}>",
                        subtitle=f"{timestamp}",
                    )
                    live.update(panel)
        except KeyboardInterrupt:
            print("\n[bright_yellow]Received Ctrl+C, shutting down policy server...[/bright_yellow]")
        finally:
            self.stop_event.set()
            poller.unregister(self.socket)
            self.socket.close(linger=0)
            self.context.term()
            print("Policy Server terminated")


@draccus.wrap()
def serve_policy(config: PolicyServerConfig):
    print(asdict(config))
    server = PolicyServer(config)
    server.run()


if __name__ == "__main__":
    serve_policy()

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
python scripts/value_server.py \
    --host=0.0.0.0 \
    --port=8000 \
    --pretrained_name_or_path=outputs/value_train/record_0429/checkpoints/010000

Protocol:
- Request policy name: {"__request_policy_name__": True}
- Request policy config: {"__request_policy_config__": True}
- Predict value: {"observation": {...}}

Response for prediction:
{"value": Tensor[B]}
"""

import pickle  # nosec
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import draccus
import numpy as np
import torch
import zmq
from rich import print

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.values.pistar06.modeling_pistar06 import Pistar06Policy


def get_local_ip():
    """Get the local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except Exception:
        return "127.0.0.1"


@dataclass
class ValueServerConfig:
    port: int = field(default=8002, metadata={"help": "Port number to bind the server to"})
    pretrained_name_or_path: str = field(default="", metadata={"help": "Path to pretrained value checkpoint"})


class ValueServer:
    def __init__(self, config: ValueServerConfig):
        self.config = config
        self.stop_event = threading.Event()

        print(f'[bright_yellow]Loading value model: "{config.pretrained_name_or_path}" ...[/bright_yellow]')
        self.policy = Pistar06Policy.from_pretrained(config.pretrained_name_or_path)
        self.policy.eval()

        self.preprocessor, _ = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=config.pretrained_name_or_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://0.0.0.0:{config.port}")

    def _to_torch_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy arrays to torch tensors and add batch dimension (dim 0)."""
        converted: dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                t = torch.from_numpy(value)
                converted[key] = t.unsqueeze(0)  # [*shape] -> [1, *shape]
            elif isinstance(value, torch.Tensor):
                converted[key] = value.unsqueeze(0)
            else:
                converted[key] = value
        return converted

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        # Adds batch dim; preprocessor expects [B, ...] inputs.
        torch_obs = self._to_torch_observation(observation)
        preprocessed_data = self.preprocessor(torch_obs)
        value = self.policy.predict_value(preprocessed_data)
        # value shape: [B] — squeeze batch dim for scalar return.
        return value.detach().cpu().squeeze(0)

    def run(self) -> None:
        ip = get_local_ip()
        print(f"[bright_green]Value server listen on: tcp://{ip}:{self.config.port}[/bright_green]")
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        try:
            while not self.stop_event.is_set():
                events = dict(poller.poll(timeout=200))
                if self.socket not in events:
                    continue

                payload = self.socket.recv()
                message = pickle.loads(payload)

                if not isinstance(message, dict):
                    self.socket.send(b"")
                    continue

                observation = message.get("observation")
                if observation is None:
                    self.socket.send(pickle.dumps({"error": "missing 'observation'"}))
                    continue

                try:
                    t0 = time.time()
                    value = self.infer(observation)
                    t1 = time.time()
                    print(f"[bright_yellow]Inference time: {(t1 - t0) * 1000:.2f} ms[/bright_yellow]")
                    self.socket.send(pickle.dumps({"value": value}))
                except Exception as exc:
                    self.socket.send(pickle.dumps({"error": str(exc)}))
        except KeyboardInterrupt:
            print("\n[bright_yellow]Received Ctrl+C, shutting down value server...[/bright_yellow]")
        finally:
            self.stop_event.set()
            poller.unregister(self.socket)
            self.socket.close(linger=0)
            self.context.term()
            print("Value Server terminated")


@draccus.wrap()
def serve_value(config: ValueServerConfig) -> None:
    print(asdict(config))
    server = ValueServer(config)
    server.run()


if __name__ == "__main__":
    serve_value()

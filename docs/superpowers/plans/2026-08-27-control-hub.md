# Lekit Control Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Hub-managed distributed control system that automatically discovers standard LeRobot Robot and Controller Nodes, grants exclusive expiring Control Handles, keeps all control state observable, and carries 60 Hz actions directly from Controller to Robot.

**Architecture:** Pure domain and scheduling code owns Handle semantics, while a single Runtime Interface isolates management, discovery, and latest-frame action transport. `MemoryRuntime` provides deterministic tests and `ZmqRuntime` provides UDP multicast discovery, ROUTER/DEALER management, and direct PUB/SUB actions; Isaac Teleop and Piper remain device-specific adapters around generic `ControllerNode` and `RobotNode`.

**Tech Stack:** Python 3.12, LeRobot 0.6.0, ZeroMQ/pyzmq 27.1+, FastAPI 0.115+, Uvicorn 0.30+, SQLite through `sqlite3`, pytest, Ruff, compact JSON management messages, length-prefixed JSON plus opaque bytes for action envelopes.

**Spec:** `docs/superpowers/specs/2026-08-27-control-hub-design.md`

## Global Constraints

- Python remains `>=3.12`; LeRobot remains exactly `0.6.0`.
- ZeroMQ is the first Runtime Adapter; Dora is outside this implementation.
- User-facing process names are exactly `lekit hub`, `lekit teleop`, and `lekit robot`.
- Hub is never in the 60 Hz action path; Controller actions travel directly to Robot.
- Controller action rate defaults to 60 Hz; Robot stale threshold to 100 ms; status to 10 Hz; heartbeat to 2 Hz; discovery to 1 Hz; renewal to 1 s; Handle TTL to 3 s.
- A Control Handle binds one Controller session to one Robot session, is exclusive, expiring, revocable, fenced, and is never reused after a terminal transition.
- Robot local HOLD and SAFETY remain authoritative when Hub or Controller is unavailable.
- Action validation, stale rejection, fencing, HOLD, and SAFETY checks cannot be disabled.
- Management/status may use compact JSON; one action is one atomic ZeroMQ message with a versioned outer envelope and opaque payload bytes.
- Action PUB is non-blocking with bounded queues; action SUB uses `RCVHWM=1`, `CONFLATE=1`, one-part messages, and `LINGER=0`.
- SQLite stores identity, epochs, Handle transitions, audit events, and low-rate snapshots; it never stores high-rate action payloads.
- Existing Isaac frame schema, reconnect/re-arm behavior, Piper processor mapping, workspace limits, gripper behavior, and fixed-width TUI remain regression-covered.
- Automated tests must never open Piper CAN, connect CloudXR, or move hardware.
- Initial deployment assumes a trusted IPv4 local network; TLS, RBAC, public-Internet exposure, blended control, and automatic scheduling are excluded.
- This plan does not authorize git commits; create commits only after a separate user request.

---

## File Structure

### New control package

- `src/lekit/control/__init__.py` — stable public imports for `Hub`, `ControllerNode`, `RobotNode`, models, and configs.
- `src/lekit/control/model.py` — enums, immutable wire/domain dataclasses, Node identity persistence, timing defaults, and validation.
- `src/lekit/control/codec.py` — strict management JSON codec and atomic binary action-envelope codec.
- `src/lekit/control/handles.py` — pure Handle transition reducer, idempotency rules, terminal-state checks, and desired/observed correlation.
- `src/lekit/control/store.py` — transactional SQLite epochs, Nodes, fencing counters, Handles, transitions, audit events, and diagnostic snapshots.
- `src/lekit/control/runtime.py` — Runtime, HubChannel, NodeChannel, ActionPublisher, and LatestActionReceiver Protocols plus received-message types.
- `src/lekit/control/memory_runtime.py` — deterministic in-process Runtime Adapter with latest-only action slots.
- `src/lekit/control/zmq_runtime.py` — ZeroMQ management and direct action channels; no domain decisions.
- `src/lekit/control/discovery.py` — versioned IPv4 UDP multicast Hub beacon and configured-seed fallback.
- `src/lekit/control/hub.py` — Hub registry, compatibility, scheduling, renewal, liveness, routed commands, mismatch response, and live read model.
- `src/lekit/control/controller.py` — generic Controller registration and `take_over` / `hand_over` lifecycle with Handle-wrapped publication.
- `src/lekit/control/robot.py` — generic standard-LeRobot Robot owner, authorization checks, watchdog, processor invocation, HOLD/SAFETY, and bounded shutdown.
- `src/lekit/control/web.py` — FastAPI HTTP/WebSocket presentation over the public Hub Interface.
- `src/lekit/control/hub.html` — dependency-free Hub inventory, assignment, control, metrics, alerts, and history client.
- `src/lekit/control/cli.py` — `lekit hub|teleop|robot` argument parsing and process wiring.

### Device adapters and existing files

- `src/lekit/teleoperators/isaac_teleop/teleop_node.py` — optionally compose a `ControllerNode`; preserve standalone legacy raw publishing when no control client is supplied.
- `src/lekit/teleoperators/isaac_teleop/__init__.py` — export the Hub-managed Isaac construction helper.
- `src/lekit/robots/piper/robot_node.py` — Piper `RobotNode` factory, Isaac payload decoder/processor wiring, and active TCP hold callback.
- `src/lekit/robots/piper/__init__.py` — export Piper Node configuration and factory.
- `pyproject.toml` — add the `lekit` console entry point and package the Hub HTML asset.
- `README.md` — document the three-process local/LAN startup and explicitly separate automated checks from approved hardware gates.

### Tests

- `tests/control/test_model_codec.py` — strict models, identity, management codec, and binary action envelope.
- `tests/control/test_handles_store.py` — transition table, idempotency, fencing, transactions, restart invalidation, and audit persistence.
- `tests/control/test_memory_runtime.py` — discovery, management routing, latest-only action delivery, disconnection, and bounded behavior.
- `tests/control/test_hub.py` — registration, compatibility, exclusive assignment, renewal, liveness, desired/observed correlation, and management actions.
- `tests/control/test_controller_node.py` — Handle target checks, take-over, stream sessions, publication, hand-over, reconnect, and expiry.
- `tests/control/test_robot_node.py` — standard Robot lifecycle, frame authorization, processor path, watchdog, HOLD, SAFETY, and shutdown.
- `tests/control/test_zmq_runtime.py` — real loopback discovery, seed fallback, ROUTER/DEALER, PUB/SUB conflation, reconnect, and zero-linger shutdown.
- `tests/control/test_web.py` — Hub API, WebSocket snapshots, operator actions, and static UI.
- `tests/control/test_end_to_end.py` — complete MemoryRuntime and loopback ZMQ flows without hardware.
- Existing Isaac, Piper, and script tests — unchanged behavior regression suite.

## Public Interface Contract

The tasks below must use these names and signatures consistently:

```python
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

class PayloadProcessor(Protocol):
    accepted_payload_schemas: frozenset[str]
    def __call__(self, payload: bytes, observation: RobotObservation) -> RobotAction:
        raise NotImplementedError
    def reset(self) -> None:
        raise NotImplementedError
    def status(self) -> Mapping[str, Any]:
        raise NotImplementedError

class Hub:
    def assign(
        self, robot: str, controller: str, *, control_mode: str = "teleop",
        actor: str | None = None,
    ) -> ControlHandle:
        raise NotImplementedError
    def renew(self, handle: ControlHandle | str, *, actor: str | None = None) -> ControlHandle:
        raise NotImplementedError
    def request_take_over(self, handle: ControlHandle | str, *, actor: str | None = None) -> None:
        raise NotImplementedError
    def request_hand_over(self, handle: ControlHandle | str, *, actor: str | None = None) -> None:
        raise NotImplementedError
    def revoke(
        self, handle: ControlHandle | str, *, reason: str, actor: str | None = None,
    ) -> None:
        raise NotImplementedError
    def force_hold(self, robot: str, *, reason: str, actor: str | None = None) -> None:
        raise NotImplementedError
    def list_nodes(self) -> tuple[NodeSnapshot, ...]:
        raise NotImplementedError
    def get_snapshot(self, handle_id: str | None = None) -> HubSnapshot | ControlSnapshot:
        raise NotImplementedError
    def watch(self, after_version: int = -1, timeout_s: float | None = None) -> HubSnapshot:
        raise NotImplementedError
    def list_history(self, *, limit: int = 200) -> tuple[Mapping[str, Any], ...]:
        raise NotImplementedError
    def run(self, *, stop_event: Any) -> None:
        raise NotImplementedError

class ControllerNode:
    def start(self) -> None:
        raise NotImplementedError
    def take_over(self, handle: ControlHandle) -> None:
        raise NotImplementedError
    def publish(self, payload: bytes, *, captured_monotonic_ns: int, captured_utc_ns: int) -> bool:
        raise NotImplementedError
    def hand_over(self, handle: ControlHandle) -> None:
        raise NotImplementedError
    def stop(self) -> None:
        raise NotImplementedError

class RobotNode:
    def start(self) -> None:
        raise NotImplementedError
    def run_cycle(self) -> RobotObservation:
        raise NotImplementedError
    def run(self, *, stop_event: Any, max_cycles: int | None = None) -> None:
        raise NotImplementedError
    def enter_safety(self, reason: str) -> None:
        raise NotImplementedError
    def clear_safety(self) -> None:
        raise NotImplementedError
    def stop(self) -> None:
        raise NotImplementedError
```

`Hub.request_take_over` and `Hub.request_hand_over` are operator conveniences: Hub routes a command to the Controller, and that Controller invokes its own public `take_over(handle)` or `hand_over(handle)`. Hub still owns Handle scheduling and sees every transition.

### Task 1: Versioned Domain Models and Codecs

**Files:**
- Create: `src/lekit/control/model.py`
- Create: `src/lekit/control/codec.py`
- Create: `src/lekit/control/__init__.py`
- Test: `tests/control/test_model_codec.py`

**Interfaces:**
- Consumes: Python standard library only plus LeRobot feature metadata represented as JSON-compatible mappings.
- Produces: `NodeRole`, `RuntimeState`, `RobotControlState`, `ControllerControlState`, `HandleState`, `NodeDescriptor`, `NodeReport`, `ControlHandle`, `ActionEnvelope`, `ControlSnapshot`, `NodeSnapshot`, `HubSnapshot`, `ManagementMessage`, `TimingConfig`, `load_or_create_node_id`, `normalize_feature_metadata`, `encode_management`, `decode_management`, `encode_action_envelope`, and `decode_action_envelope`.

- [ ] **Step 1: Write failing validation and round-trip tests**

```python
def test_action_envelope_round_trip_keeps_payload_opaque():
    envelope = ActionEnvelope(
        handle_id="handle-1", hub_epoch="epoch-1", fencing_token=7,
        controller_id="quest", controller_session_id="controller-session",
        stream_session_id="stream-1", sequence=3,
        captured_monotonic_ns=11, captured_utc_ns=12,
        payload_schema="lekit.isaac_teleop.action.v1", payload=b"\x00opaque\xff",
    )
    assert decode_action_envelope(encode_action_envelope(envelope)) == envelope

def test_action_codec_rejects_truncated_header():
    with pytest.raises(ValueError, match="header length"):
        decode_action_envelope(b"\x00\x00\x00\x10{}")

def test_node_id_is_stable_across_loads(tmp_path):
    path = tmp_path / "node-id"
    assert load_or_create_node_id(path) == load_or_create_node_id(path)
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the package does not exist**

Run: `uv run pytest tests/control/test_model_codec.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'lekit.control'`.

- [ ] **Step 3: Implement strict immutable models and timing validation**

```python
PROTOCOL_VERSION = 1

class HandleState(StrEnum):
    ASSIGNED = "assigned"
    TAKING_OVER = "taking_over"
    ACTIVE = "active"
    HANDING_OVER = "handing_over"
    RELEASED = "released"
    REVOKING = "revoking"
    REVOKED = "revoked"
    EXPIRED = "expired"
    FAULT = "fault"

TERMINAL_HANDLE_STATES = frozenset({
    HandleState.RELEASED, HandleState.REVOKED,
    HandleState.EXPIRED, HandleState.FAULT,
})

class NodeRole(StrEnum):
    CONTROLLER = "controller"
    ROBOT = "robot"

class RuntimeState(StrEnum):
    STARTING = "starting"
    ONLINE = "online"
    DEGRADED = "degraded"
    FAULT = "fault"
    STOPPED = "stopped"

class RobotControlState(StrEnum):
    HOLD = "hold"
    CONTROLLING = "controlling"
    SAFETY = "safety"

class ControllerControlState(StrEnum):
    IDLE = "idle"
    TAKING_OVER = "taking_over"
    STREAMING = "streaming"
    HANDING_OVER = "handing_over"
    FAULT = "fault"

@dataclass(frozen=True, slots=True)
class NodeDescriptor:
    protocol_version: int
    schema_version: int
    node_id: str
    session_id: str
    role: NodeRole
    display_name: str
    administratively_enabled: bool
    capabilities: tuple[str, ...]
    action_schemas: tuple[str, ...]
    control_modes: tuple[str, ...]
    action_endpoint: str | None
    observation_features: Mapping[str, Any]
    action_features: Mapping[str, Any]
    software_version: str
    diagnostics: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class ControlHandle:
    handle_id: str
    hub_epoch: str
    robot_id: str
    robot_session_id: str
    controller_id: str
    controller_session_id: str
    controller_action_endpoint: str
    action_schema: str
    control_mode: str
    fencing_token: int
    issued_at_ns: int
    expires_at_ns: int

    def __post_init__(self) -> None:
        for name in (
            "handle_id", "hub_epoch", "robot_id", "robot_session_id",
            "controller_id", "controller_session_id", "controller_action_endpoint",
            "action_schema", "control_mode",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.fencing_token < 1 or self.issued_at_ns < 0 or self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("Handle timing and fencing values are invalid")

@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    handle_id: str
    hub_epoch: str
    fencing_token: int
    controller_id: str
    controller_session_id: str
    stream_session_id: str
    sequence: int
    captured_monotonic_ns: int
    captured_utc_ns: int
    payload_schema: str
    payload: bytes

@dataclass(frozen=True, slots=True)
class NodeReport:
    node_id: str
    session_id: str
    runtime_state: RuntimeState
    robot_control_state: RobotControlState | None
    controller_control_state: ControllerControlState | None
    handle_id: str | None
    fencing_token: int | None
    action_rate_hz: float
    frame_age_ms: float | None
    last_sequence: int | None
    tracking: bool | None
    engaged: bool | None
    processor_state: str | None
    active_hold: bool | None
    error: str | None
    reported_at_ns: int

@dataclass(frozen=True, slots=True)
class ManagementMessage:
    protocol_version: int
    kind: str
    correlation_id: str
    sender_id: str
    sender_session_id: str
    sequence: int
    sent_at_ns: int
    body: Mapping[str, Any]
```

Add strict `__post_init__` checks to every network/domain dataclass: non-empty IDs, non-negative sequence/timestamps, finite positive timing values, JSON-compatible descriptor metadata, and role-specific endpoint/schema requirements. Define `ControlSnapshot` with exactly the fields in the design spec, plus `mismatch_codes: tuple[str, ...]`; define `NodeSnapshot(descriptor, online, last_seen_ns, report)` and `HubSnapshot(version, hub_epoch, generated_at_ns, nodes, controls, alerts)`. `TimingConfig` defaults are `action_rate_hz=60.0`, `action_stale_s=0.1`, `status_rate_hz=10.0`, `heartbeat_rate_hz=2.0`, `discovery_rate_hz=1.0`, `renewal_interval_s=1.0`, and `handle_ttl_s=3.0`; expose exact integer properties `action_stale_ns`, `heartbeat_interval_ns`, `renewal_interval_ns`, and `handle_ttl_ns` by rounding seconds times `1_000_000_000`.

`normalize_feature_metadata` converts scalar LeRobot feature types to stable names (`float`, `int`, `bool`), shape tuples to integer JSON arrays, recursively normalizes existing feature mappings, and rejects tensors, images, or arbitrary objects rather than serializing their contents. Only feature metadata is registered; observation values are never sent to Hub. `administratively_enabled=False` keeps a Node observable but makes it ineligible for assignment or take-over.

- [ ] **Step 4: Implement management JSON and one-message action encoding**

```python
_HEADER_LENGTH = struct.Struct("!I")

def encode_action_envelope(envelope: ActionEnvelope) -> bytes:
    header = json.dumps(
        {
            "schema_name": "lekit.control.action",
            "schema_version": 1,
            "handle_id": envelope.handle_id,
            "hub_epoch": envelope.hub_epoch,
            "fencing_token": envelope.fencing_token,
            "controller_id": envelope.controller_id,
            "controller_session_id": envelope.controller_session_id,
            "stream_session_id": envelope.stream_session_id,
            "sequence": envelope.sequence,
            "captured_monotonic_ns": envelope.captured_monotonic_ns,
            "captured_utc_ns": envelope.captured_utc_ns,
            "payload_schema": envelope.payload_schema,
        },
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return _HEADER_LENGTH.pack(len(header)) + header + envelope.payload
```

`decode_action_envelope` must require exactly the header fields above, cap the header at 64 KiB, reject malformed UTF-8/JSON, booleans in integer fields, empty payloads, and invalid models. Management messages use exact top-level fields `protocol_version`, `kind`, `correlation_id`, `sender_id`, `sender_session_id`, `sequence`, `sent_at_ns`, and `body`.

- [ ] **Step 5: Verify the task and inspect formatting**

Run: `uv run pytest tests/control/test_model_codec.py -q`

Expected: all tests pass.

Run: `uv run ruff check src/lekit/control/model.py src/lekit/control/codec.py tests/control/test_model_codec.py && git diff --check`

Expected: both commands exit 0. Do not commit without a separate user request.

### Task 2: Handle Reducer and Transactional SQLite Store

**Files:**
- Create: `src/lekit/control/handles.py`
- Create: `src/lekit/control/store.py`
- Test: `tests/control/test_handles_store.py`

**Interfaces:**
- Consumes: Task 1 `ControlHandle`, `HandleState`, `NodeDescriptor`, `NodeReport`, and snapshots.
- Produces: `HandleRecord`, `HandleTransition`, `transition_handle(record, target, *, transition_sequence, correlation_id, at_ns, reason=None)`, `correlate_control(record, robot_report, controller_report, *, now_ns)`, and `HubStore` methods shown below.

- [ ] **Step 1: Write the complete transition-table and store transaction tests**

```python
@pytest.mark.parametrize(("source", "target"), [
    (HandleState.ASSIGNED, HandleState.TAKING_OVER),
    (HandleState.TAKING_OVER, HandleState.ACTIVE),
    (HandleState.ACTIVE, HandleState.HANDING_OVER),
    (HandleState.HANDING_OVER, HandleState.RELEASED),
    (HandleState.ACTIVE, HandleState.REVOKING),
    (HandleState.REVOKING, HandleState.REVOKED),
])
def test_valid_transitions(source, target, handle_record):
    current = replace(handle_record, state=source, transition_sequence=4)
    updated = transition_handle(
        current, target, transition_sequence=5,
        correlation_id="correlation-5", at_ns=500,
    )
    assert updated.state is target

def test_terminal_handle_cannot_reactivate(handle_record):
    current = replace(handle_record, state=HandleState.REVOKED, transition_sequence=9)
    with pytest.raises(InvalidHandleTransition):
        transition_handle(current, HandleState.ACTIVE, transition_sequence=10,
                          correlation_id="late", at_ns=600)

def test_store_assigns_monotonic_fencing_and_exclusive_handle(tmp_path, descriptors):
    store = HubStore(tmp_path / "hub.sqlite3")
    first = store.create_assignment(*descriptors, now_ns=10, ttl_ns=3_000_000_000)
    store.transition(first.handle_id, HandleState.REVOKED, 1, "revoke", 20, "test")
    second = store.create_assignment(*descriptors, now_ns=30, ttl_ns=3_000_000_000)
    assert second.fencing_token == first.fencing_token + 1
    assert second.handle_id != first.handle_id
```

- [ ] **Step 2: Run the focused tests and confirm missing symbols fail**

Run: `uv run pytest tests/control/test_handles_store.py -q`

Expected: collection fails on missing `lekit.control.handles` or `lekit.control.store`.

- [ ] **Step 3: Implement the pure transition reducer and mismatch correlation**

```python
_ALLOWED = {
    HandleState.ASSIGNED: {HandleState.TAKING_OVER, HandleState.REVOKING,
                           HandleState.EXPIRED, HandleState.FAULT},
    HandleState.TAKING_OVER: {HandleState.ACTIVE, HandleState.REVOKING,
                              HandleState.EXPIRED, HandleState.FAULT},
    HandleState.ACTIVE: {HandleState.HANDING_OVER, HandleState.REVOKING,
                         HandleState.EXPIRED, HandleState.FAULT},
    HandleState.HANDING_OVER: {HandleState.RELEASED, HandleState.REVOKING,
                               HandleState.EXPIRED, HandleState.FAULT},
    HandleState.REVOKING: {HandleState.REVOKED, HandleState.EXPIRED, HandleState.FAULT},
}

def transition_handle(record, target, *, transition_sequence, correlation_id, at_ns, reason=None):
    if transition_sequence < record.transition_sequence:
        raise StaleTransition("transition sequence regressed")
    if transition_sequence == record.transition_sequence:
        if target is record.state and correlation_id == record.correlation_id:
            return record
        raise StaleTransition("sequence already used by another transition")
    if target not in _ALLOWED.get(record.state, set()):
        raise InvalidHandleTransition(f"{record.state} -> {target} is forbidden")
    return replace(record, state=target, transition_sequence=transition_sequence,
                   correlation_id=correlation_id, updated_at_ns=at_ns, reason=reason)
```

`correlate_control` must classify at least `desired_active_robot_hold`, `controller_streaming_robot_not_accepting`, `controller_offline_robot_controlling`, `orphan_observed_control`, `terminal_handle_accepted`, and `robot_safety_with_active_desire`; severe mismatches set `healthy=False` and a stable machine-readable `error`.

- [ ] **Step 4: Implement SQLite schema and single-transaction assignment**

`HubStore` exposes these exact methods:

```python
class HubStore:
    def begin_epoch(self, *, started_at_ns: int) -> str:
        raise NotImplementedError
    def upsert_node(self, descriptor: NodeDescriptor, *, seen_at_ns: int) -> None:
        raise NotImplementedError
    def create_assignment(
        self, robot: NodeDescriptor, controller: NodeDescriptor,
        *, now_ns: int, ttl_ns: int, action_schema: str | None = None,
        control_mode: str = "teleop",
    ) -> ControlHandle:
        raise NotImplementedError
    def get_handle(self, handle_id: str) -> HandleRecord:
        raise NotImplementedError
    def renew_handle(self, handle_id: str, *, expires_at_ns: int, at_ns: int) -> ControlHandle:
        raise NotImplementedError
    def transition(
        self, handle_id: str, target: HandleState, transition_sequence: int,
        correlation_id: str, at_ns: int, reason: str | None = None,
    ) -> HandleRecord:
        raise NotImplementedError
    def invalidate_previous_epochs(self, current_epoch: str, *, at_ns: int) -> tuple[str, ...]:
        raise NotImplementedError
    def append_audit(self, *, event: str, at_ns: int, actor: str | None,
                     correlation_id: str, details: Mapping[str, Any]) -> None:
        raise NotImplementedError
    def save_snapshot(self, snapshot: HubSnapshot) -> None:
        raise NotImplementedError
    def list_history(self, *, limit: int = 200) -> tuple[Mapping[str, Any], ...]:
        raise NotImplementedError
```

Use `BEGIN IMMEDIATE` for `create_assignment`, enforce partial unique indexes for non-terminal Robot and Controller Handles, increment `robot_fencing.robot_id` atomically, enable foreign keys and WAL, and commit Handle plus initial transition plus audit event together. Store JSON with sorted keys and compact separators.

- [ ] **Step 5: Verify restart invalidation and transaction rollback**

Run: `uv run pytest tests/control/test_handles_store.py -q`

Expected: all tests pass, including a forced insert failure that leaves no Handle and does not advance the fencing counter.

Run: `uv run ruff check src/lekit/control/handles.py src/lekit/control/store.py tests/control/test_handles_store.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 3: Runtime Interface and Deterministic MemoryRuntime

**Files:**
- Create: `src/lekit/control/runtime.py`
- Create: `src/lekit/control/memory_runtime.py`
- Test: `tests/control/test_memory_runtime.py`

**Interfaces:**
- Consumes: Task 1 codecs and `ManagementMessage` / `ActionEnvelope`.
- Produces: `ReceivedManagement`, `HubChannel`, `NodeChannel`, `ActionPublisher`, `LatestActionReceiver`, `Runtime`, and `MemoryRuntime`.

- [ ] **Step 1: Write tests for routed management and latest-only action behavior**

```python
def test_management_routes_between_hub_and_named_node():
    runtime = MemoryRuntime()
    hub = runtime.open_hub("memory://hub", hub_epoch="epoch-1")
    node = runtime.open_node("robot-1", "session-1", hub_seed="memory://hub")
    node.send(message("register", sender="robot-1"))
    received = hub.receive(timeout_s=0.0)
    assert received.peer_id == "robot-1"
    hub.send("robot-1", message("registered", sender="hub"))
    assert node.receive(timeout_s=0.0).message.kind == "registered"

def test_action_receiver_keeps_only_latest_atomic_envelope():
    runtime = MemoryRuntime()
    publisher = runtime.open_action_publisher("memory://quest/actions")
    receiver = runtime.open_action_receiver("memory://quest/actions")
    publisher.send(envelope(sequence=1))
    publisher.send(envelope(sequence=2))
    assert receiver.receive_latest(timeout_s=0.0).envelope.sequence == 2
    assert receiver.receive_latest(timeout_s=0.0) is None
```

- [ ] **Step 2: Run tests and observe missing Runtime abstractions**

Run: `uv run pytest tests/control/test_memory_runtime.py -q`

Expected: import errors for `lekit.control.runtime`.

- [ ] **Step 3: Define transport-only Protocols**

```python
@dataclass(frozen=True, slots=True)
class ReceivedManagement:
    peer_id: str
    message: ManagementMessage
    peer_host: str | None
    received_monotonic_ns: int

@dataclass(frozen=True, slots=True)
class ReceivedAction:
    envelope: ActionEnvelope
    received_monotonic_ns: int

class HubChannel(Protocol):
    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        raise NotImplementedError
    def send(self, peer_id: str, message: ManagementMessage) -> bool:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError

class NodeChannel(Protocol):
    def receive(self, *, timeout_s: float = 0.0) -> ReceivedManagement | None:
        raise NotImplementedError
    def send(self, message: ManagementMessage) -> bool:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError

class ActionPublisher(Protocol):
    def send(self, envelope: ActionEnvelope) -> bool:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError

class LatestActionReceiver(Protocol):
    def receive_latest(self, *, timeout_s: float = 0.0) -> ReceivedAction | None:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError

class Runtime(Protocol):
    def open_hub(
        self, endpoint: str, *, hub_epoch: str, advertise_endpoint: str | None = None,
    ) -> HubChannel:
        raise NotImplementedError
    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None) -> NodeChannel:
        raise NotImplementedError
    def open_action_publisher(self, endpoint: str) -> ActionPublisher:
        raise NotImplementedError
    def open_action_receiver(self, endpoint: str) -> LatestActionReceiver:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 4: Implement thread-safe MemoryRuntime**

Use a `threading.Condition` per management inbox and one overwrite slot per action endpoint. `send` never waits for a consumer; management inboxes have a configurable maximum of 256 and coalesce `status`/`heartbeat` by `(peer_id, kind)`, while action sends overwrite the previous unread envelope. Closing a channel wakes blocked receivers and causes subsequent sends to return `False`. Closed receivers return `None`; no close path raises merely because another thread is polling.

```python
def send(self, envelope: ActionEnvelope) -> bool:
    with self._slot.condition:
        if self._closed or self._slot.closed:
            return False
        self._slot.value = ReceivedAction(envelope, self._runtime.monotonic_ns())
        self._slot.condition.notify_all()
        return True
```

- [ ] **Step 5: Verify deterministic timeout, close, and queue bounds**

Run: `uv run pytest tests/control/test_memory_runtime.py -q`

Expected: all tests pass without sleeps longer than 20 ms.

Run: `uv run ruff check src/lekit/control/runtime.py src/lekit/control/memory_runtime.py tests/control/test_memory_runtime.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 4: Hub Registry, Scheduling, and Live Correlation

**Files:**
- Create: `src/lekit/control/hub.py`
- Test: `tests/control/test_hub.py`

**Interfaces:**
- Consumes: `HubStore`, `HubChannel`, Task 1 models, Task 2 reducer/correlation, and `TimingConfig`.
- Produces: `HubConfig`, public `Hub`, `Hub.run_once(timeout_s=0.0)`, `Hub.tick()` for deterministic service driving, and `Hub.run(stop_event=...)` for the process loop.

- [ ] **Step 1: Write failing scheduling and liveness tests**

```python
def test_assign_checks_schema_mode_sessions_and_exclusivity(hub, registered_pair):
    handle = hub.assign("piper-01", "quest3-main")
    assert handle.action_schema == "lekit.isaac_teleop.action.v1"
    with pytest.raises(ControlConflict):
        hub.assign("piper-01", "quest3-main")

def test_active_requires_both_ready_and_streaming_reports(hub, registered_pair):
    handle = hub.assign("piper-01", "quest3-main")
    hub.receive_report(robot_ready(handle))
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.TAKING_OVER
    hub.receive_report(controller_streaming(handle))
    assert hub.get_snapshot(handle.handle_id).handle_state is HandleState.ACTIVE

def test_three_missed_heartbeats_marks_node_offline_and_revokes(hub, clock, active_handle):
    clock.advance(1.51)
    hub.tick()
    assert not node(hub, "quest3-main").online
    assert hub.get_snapshot(active_handle.handle_id).handle_state is HandleState.REVOKING

def test_robot_hold_is_sufficient_to_finish_release_when_controller_ack_is_lost(
    hub, active_handle,
):
    hub.request_hand_over(active_handle)
    hub.receive_report(robot_holding(active_handle))
    assert hub.get_snapshot(active_handle.handle_id).handle_state is HandleState.RELEASED
```

- [ ] **Step 2: Run the Hub tests and confirm missing implementation**

Run: `uv run pytest tests/control/test_hub.py -q`

Expected: collection fails on missing `Hub`.

- [ ] **Step 3: Implement registration, compatibility, and assignment**

```python
@dataclass(kw_only=True)
class HubConfig:
    management_endpoint: str = "tcp://0.0.0.0:5560"
    advertise_endpoint: str | None = None
    database_path: Path = Path(".lekit/control-hub.sqlite3")
    timing: TimingConfig = field(default_factory=TimingConfig)
    auto_revoke_mismatches: bool = True

class Hub:
    def assign(
        self, robot: str, controller: str, *, control_mode: str = "teleop",
        actor: str | None = None,
    ) -> ControlHandle:
        with self._lock:
            robot_node = self._require_schedulable_robot(robot)
            controller_node = self._require_online_controller(controller)
            schema = self._select_schema(robot_node, controller_node, control_mode)
            handle = self._store.create_assignment(
                robot_node.descriptor, controller_node.descriptor,
                now_ns=self._utc_ns(), ttl_ns=self._timing.handle_ttl_ns,
                action_schema=schema, control_mode=control_mode,
            )
            self._desired[handle.handle_id] = HandleState.ASSIGNED
            self._send_grant(handle)
            self._publish_snapshot_locked()
            return handle
```

Registration rejects protocol mismatch, stale session messages, wildcard advertised consumer endpoints, and descriptors inconsistent with role. Resolve a Controller bind endpoint such as `tcp://0.0.0.0:5557` to the management peer host before storing the current descriptor. Assignment also rejects a Robot in SAFETY, FAULT, DEGRADED-with-control-error, or administratively disabled state. Hub liveness uses `ReceivedManagement.received_monotonic_ns`; it never trusts a Node-supplied wall or monotonic timestamp to decide online/offline state.

- [ ] **Step 4: Implement routed lifecycle, renewal, force-HOLD, liveness, and snapshots**

`run_once` decodes one received management message and dispatches `register`, `heartbeat`, `status`, `take_over_requested`, `robot_ready`, `controller_streaming`, `hand_over_requested`, `robot_holding`, `controller_released`, `fault`, and command acknowledgements. Hub acknowledges each valid Node heartbeat with `heartbeat_ack` carrying its current epoch and the request correlation ID. `tick` performs one-second renewals, expiry, three-heartbeat offline detection, retry of undelivered grants, mismatch correlation, and optional revoke/force-HOLD response. It overwrites the last diagnostic SQLite snapshot at no more than 1 Hz and keeps 10 Hz action metrics only in memory. `run` alternates bounded `run_once` waits with `tick`, so Web and signal-handling threads remain responsive.

```python
def watch(self, after_version: int = -1, timeout_s: float | None = None) -> HubSnapshot:
    with self._snapshot_changed:
        self._snapshot_changed.wait_for(
            lambda: self._snapshot.version > after_version,
            timeout=timeout_s,
        )
        return self._snapshot
```

`Hub.start()` opens the store/runtime, creates a new epoch, invalidates old non-terminal Handles, and starts in-memory state empty. It then calls `runtime.open_hub(management_endpoint, hub_epoch=epoch, advertise_endpoint=...)`, so only the Runtime Adapter owns beacon sockets; a wildcard management bind requires a non-wildcard advertise endpoint before discovery is enabled. `Hub.stop()` closes channels but never restores persisted liveness. `Hub.request_take_over` and `request_hand_over` send commands to the Controller; `revoke` transitions through `REVOKING`; `force_hold` is valid with or without a Handle and is always audited.

- [ ] **Step 5: Verify all domain behavior and no action payload reaches Hub**

Run: `uv run pytest tests/control/test_hub.py -q`

Expected: all tests pass, including a spy Runtime assertion that Hub only sees management messages and never an `ActionEnvelope`.

Run: `uv run ruff check src/lekit/control/hub.py tests/control/test_hub.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 5: Generic ControllerNode Lifecycle

**Files:**
- Create: `src/lekit/control/controller.py`
- Test: `tests/control/test_controller_node.py`

**Interfaces:**
- Consumes: `Runtime`, `NodeDescriptor`, `ControlHandle`, `ActionEnvelope`, `TimingConfig`, and management messages.
- Produces: `ControllerNodeConfig`, `ControllerNode`, `ControllerNode.run_management_once`, and the public lifecycle methods in the contract.

- [ ] **Step 1: Write failing take-over, publication, and hand-over tests**

```python
def test_take_over_requires_a_granted_current_handle(controller, handle):
    with pytest.raises(HandleNotGranted):
        controller.take_over(handle)
    controller.receive_grant(handle)
    controller.take_over(handle)
    assert controller.control_state is ControllerControlState.TAKING_OVER

def test_robot_ready_starts_new_stream_and_wraps_payload(controller, handle, action_receiver):
    controller.receive_grant(handle)
    controller.take_over(handle)
    controller.receive_robot_ready(handle)
    assert controller.publish(b"frame", captured_monotonic_ns=10, captured_utc_ns=20)
    received = action_receiver.receive_latest()
    assert received.envelope.handle_id == handle.handle_id
    assert received.envelope.payload == b"frame"
    assert received.envelope.sequence == 0

def test_hand_over_stops_publication_before_management_request(controller, handle, events):
    activate(controller, handle)
    controller.hand_over(handle)
    assert controller.publish(b"late", captured_monotonic_ns=30, captured_utc_ns=40) is False
    assert events[-1].kind == "hand_over_requested"
```

- [ ] **Step 2: Run the Controller tests and observe missing implementation**

Run: `uv run pytest tests/control/test_controller_node.py -q`

Expected: collection fails on missing `ControllerNode`.

- [ ] **Step 3: Implement identity, registration, and explicit lifecycle**

```python
@dataclass(kw_only=True)
class ControllerNodeConfig:
    node_id_path: Path
    display_name: str
    action_endpoint: str = "tcp://0.0.0.0:5557"
    action_schemas: tuple[str, ...]
    control_modes: tuple[str, ...] = ("teleop",)
    hub_seed: str | None = None
    timing: TimingConfig = field(default_factory=TimingConfig)

def take_over(self, handle: ControlHandle) -> None:
    with self._lock:
        self._validate_grant_target(handle)
        if self._handle_is_terminal(handle.handle_id):
            raise HandleExpired(handle.handle_id)
        if self._current_handle_id not in (None, handle.handle_id):
            raise ControlConflict("Controller already has another Handle")
        self._streaming_enabled = False
        self._send("take_over_requested", handle)
        self._control_state = ControllerControlState.TAKING_OVER
```

`start` creates a new process session, opens management and action channels, and registers. A grant alone never enables publication. `robot_ready` creates a fresh UUID `stream_session_id`, resets sequence to zero, enables publication, reports `controller_streaming`, and changes state to `STREAMING`. Expiry, epoch change, session mismatch, revoke, management loss, or `stop` disables publication synchronously.

`start` also launches one bounded management thread that calls `run_management_once` independently of XR sampling, emits heartbeats at 2 Hz, and emits coalesced status at 10 Hz. A grant or renewal computes its local deadline as `monotonic_ns() + min(timing.handle_ttl_ns, max(0, handle.expires_at_ns - utc_ns()))`, so clock skew can expire authority early but can never extend it beyond the configured TTL; action publication never compares clocks from another process. Three missed `heartbeat_ack` intervals or a failed management send constitute management loss. The thread then disables streaming, closes the old channel, generates a new process `session_id`, rediscovers, and registers again; the old Handle can no longer target that session and is never resumed.

- [ ] **Step 4: Implement non-blocking Handle-wrapped publication and idempotent release**

```python
def publish(self, payload: bytes, *, captured_monotonic_ns: int, captured_utc_ns: int) -> bool:
    with self._lock:
        if not self._streaming_enabled or self._handle is None or self._stream_session_id is None:
            return False
        if self._monotonic_ns() >= self._local_expiry_monotonic_ns:
            self._disable_streaming("handle_expired")
            return False
        envelope = ActionEnvelope(
            handle_id=self._handle.handle_id, hub_epoch=self._handle.hub_epoch,
            fencing_token=self._handle.fencing_token,
            controller_id=self.node_id, controller_session_id=self.session_id,
            stream_session_id=self._stream_session_id, sequence=self._sequence,
            captured_monotonic_ns=captured_monotonic_ns, captured_utc_ns=captured_utc_ns,
            payload_schema=self._handle.action_schema, payload=payload,
        )
        self._sequence += 1
    return self._action_publisher.send(envelope)
```

`hand_over` disables streaming before sending any management message. Repeated calls for the same terminal Handle succeed. Management loss never extends cached expiry and never starts a new stream automatically after reconnect.

- [ ] **Step 5: Verify Controller failure and session-change behavior**

Run: `uv run pytest tests/control/test_controller_node.py -q`

Expected: all tests pass, including expiry while Hub is absent, stale command rejection, new process session registration, and no automatic resume.

Run: `uv run ruff check src/lekit/control/controller.py tests/control/test_controller_node.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 6: Generic RobotNode Authorization and Safety Loop

**Files:**
- Create: `src/lekit/control/robot.py`
- Test: `tests/control/test_robot_node.py`

**Interfaces:**
- Consumes: standard LeRobot `Robot`, `PayloadProcessor`, `Runtime`, `ControlHandle`, and management models.
- Produces: `RobotNodeConfig`, `RobotNode`, `PassiveHold`, `HoldResult`, and deterministic rejection counters/status.

- [ ] **Step 1: Write a fake standard Robot and failing safety tests**

```python
class FakeRobot(Robot):
    name = "fake"
    config_class = FakeRobotConfig
    def connect(self, calibrate=True): self.connected = True
    def get_observation(self): return {"x": self.x}
    def send_action(self, action): self.sent.append(dict(action)); return action
    def disconnect(self): self.connected = False
    @property
    def observation_features(self): return {"x": float}
    @property
    def action_features(self): return {"x": float}
    @property
    def is_connected(self): return self.connected
    @property
    def is_calibrated(self): return True
    def calibrate(self): return None
    def configure(self): return None

def test_wrong_fencing_and_duplicate_sequence_never_reach_robot(robot_node, active_handle):
    robot_node.inject(envelope(active_handle, fencing_token=active_handle.fencing_token - 1))
    robot_node.run_cycle()
    robot_node.inject(envelope(active_handle, sequence=4))
    robot_node.run_cycle()
    robot_node.inject(envelope(active_handle, sequence=4))
    robot_node.run_cycle()
    assert len(robot_node.robot.sent) == 1
    assert robot_node.rejections["wrong_fencing"] == 1
    assert robot_node.rejections["sequence_regressed"] == 1

def test_stale_action_enters_hold_and_resets_processor(robot_node, clock, active_stream):
    robot_node.run_cycle()
    clock.advance(0.101)
    robot_node.run_cycle()
    assert robot_node.control_state is RobotControlState.HOLD
    assert robot_node.processor.reset_count == 1
```

- [ ] **Step 2: Run the Robot tests and confirm the class is absent**

Run: `uv run pytest tests/control/test_robot_node.py -q`

Expected: collection fails on missing `RobotNode`.

- [ ] **Step 3: Implement standard Robot ownership and grant/take-over flow**

```python
@dataclass(kw_only=True)
class RobotNodeConfig:
    node_id_path: Path
    display_name: str
    accepted_payload_schemas: tuple[str, ...]
    control_modes: tuple[str, ...] = ("teleop",)
    control_enabled: bool = True
    control_rate_hz: float = 60.0
    action_stale_s: float = 0.1
    hub_seed: str | None = None
    timing: TimingConfig = field(default_factory=TimingConfig)

class RobotNode:
    def __init__(
        self, robot: Robot, processor: PayloadProcessor, config: RobotNodeConfig,
        *, runtime: Runtime, hold: Callable[[Robot, RobotObservation | None, str], HoldResult] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.robot = robot
        self.processor = processor
        self.config = config
        self.runtime = runtime
        self._hold = hold or PassiveHold()
        self._monotonic_ns = monotonic_ns
        self._utc_ns = utc_ns
```

`start` connects exactly once, registers actual `observation_features` / `action_features` and `administratively_enabled=config.control_enabled`, begins in HOLD, and starts one bounded management thread that emits 2 Hz heartbeats and 10 Hz coalesced status. A Hub grant is cached independently. A routed take-over must match that grant and current epoch/session/fencing/schema, then open one latest receiver, reset processor, remain HOLD, and report `robot_ready`. A disabled Node rejects grants and take-over locally even if Hub is misconfigured. No packet can become active before this step. Grant and renewal use the same TTL-clamped local monotonic deadline formula as ControllerNode, so remote clock skew cannot extend authority. Three missed `heartbeat_ack` intervals or a failed management send immediately enters HOLD; reconnection closes the old channel, creates a new process `session_id`, rediscovers, and registers without reusing the old grant or reopening its action receiver.

- [ ] **Step 4: Implement the strict receive gate, processor path, and watchdog**

Process gates in this exact order: local SAFETY, current grant, local Handle expiry, receiver present, frame available, local receive age, Hub epoch, Handle ID, fencing, Controller ID/session, stream session, strictly increasing sequence, payload schema, processor validation, complete finite Robot action compatible with `robot.action_features`. Record local monotonic receive time from `ReceivedAction`, not Controller clocks.

```python
def run_cycle(self) -> RobotObservation:
    observation = self.robot.get_observation()
    received = None if self._receiver is None else self._receiver.receive_latest(timeout_s=0.0)
    if received is None:
        self._watchdog(observation)
        return observation
    reason = self._validate_frame(received)
    if reason is not None:
        self._reject_and_hold(reason, observation)
        return observation
    action = self.processor(received.envelope.payload, observation)
    if not action:
        self._enter_hold("processor_not_armed", observation)
        return observation
    self._validate_robot_action(action)
    self.robot.send_action(action)
    self._accept(received)
    return observation
```

First valid stream frame establishes `stream_session_id`; any subsequent new stream ID forces HOLD and requires another routed take-over. Tracking loss or an empty processor output means HOLD and processor reset. Status reports include action rate, frame age, sequence, rejections, passive/active hold, Robot connection, processor state, and error, coalesced at 10 Hz.

- [ ] **Step 5: Implement hand-over, revoke, force-HOLD, SAFETY, and bounded shutdown**

`_enter_hold` synchronously stops frame acceptance, resets processor once per transition, invokes the optional hold callback, reports HOLD, and closes the receiver for terminal/revoke/hand-over reasons. `enter_safety` preempts and rejects grants until an explicit local `clear_safety`; Controller or Hub messages cannot clear it. `stop` requests HOLD, disconnects Robot in `finally`, closes Runtime channels with zero authority restoration, and preserves the primary exception if cleanup also fails.

- [ ] **Step 6: Verify all rejection paths and standard LeRobot lifecycle**

Run: `uv run pytest tests/control/test_robot_node.py -q`

Expected: all tests pass, and the fake Robot call order is `connect`, repeated `get_observation`, authorized `send_action` only, then `disconnect`.

Run: `uv run ruff check src/lekit/control/robot.py tests/control/test_robot_node.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 7: ZeroMQ Direct Action Adapter

**Files:**
- Create: `src/lekit/control/zmq_runtime.py`
- Test: `tests/control/test_zmq_runtime.py`

**Interfaces:**
- Consumes: Task 1 action codec and Task 3 action Protocols.
- Produces initially: `ZmqActionPublisher`, `ZmqLatestActionReceiver`, and shared `ZmqContextOwner`; Task 8 extends the same file with management and Runtime factory methods.

- [ ] **Step 1: Write real loopback tests for non-blocking one-part latest delivery**

```python
def test_slow_receiver_observes_latest_complete_frame(zmq_context, free_tcp_endpoint):
    publisher = ZmqActionPublisher(free_tcp_endpoint, context=zmq_context)
    receiver = ZmqLatestActionReceiver(free_tcp_endpoint, context=zmq_context)
    wait_for_subscription_settle()
    for sequence in range(100):
        publisher.send(envelope(sequence=sequence, payload=f"frame-{sequence}".encode()))
    received = eventually_receive(receiver)
    assert received.envelope.sequence == 99
    assert received.envelope.payload == b"frame-99"

def test_close_is_zero_linger_and_does_not_block(zmq_context, free_tcp_endpoint):
    publisher = ZmqActionPublisher(free_tcp_endpoint, context=zmq_context)
    started = time.monotonic()
    publisher.close()
    assert time.monotonic() - started < 0.1
```

- [ ] **Step 2: Run the action Adapter tests and observe missing classes**

Run: `uv run pytest tests/control/test_zmq_runtime.py -q -k 'action or linger'`

Expected: import errors for the ZeroMQ action classes.

- [ ] **Step 3: Implement required ZeroMQ socket semantics**

```python
class ZmqActionPublisher:
    def __init__(self, endpoint: str, *, context: zmq.Context):
        self._socket = context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 10)
        self._socket.bind(endpoint)

    def send(self, envelope: ActionEnvelope) -> bool:
        try:
            self._socket.send(encode_action_envelope(envelope), flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

class ZmqLatestActionReceiver:
    def __init__(self, endpoint: str, *, context: zmq.Context,
                 monotonic_ns: Callable[[], int] = time.monotonic_ns):
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
```

`receive_latest` records `monotonic_ns()` immediately after `recv(NOBLOCK)` and before decoding. It returns `None` for no packet, raises a typed `MalformedAction` for bad bytes, and never uses multipart send/receive.

- [ ] **Step 4: Verify malformed data, bounded queues, and context ownership**

Run: `uv run pytest tests/control/test_zmq_runtime.py -q -k 'action or linger'`

Expected: all selected tests pass, including repeated open/close under a shared context.

Run: `uv run ruff check src/lekit/control/zmq_runtime.py tests/control/test_zmq_runtime.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 8: UDP Discovery and ZeroMQ Management Adapter

**Files:**
- Create: `src/lekit/control/discovery.py`
- Modify: `src/lekit/control/zmq_runtime.py`
- Test: `tests/control/test_zmq_runtime.py`

**Interfaces:**
- Consumes: Task 3 complete Runtime Protocol and management codec.
- Produces: `DiscoveryBeacon`, `HubBeaconPublisher`, `HubBeaconListener`, `ZmqHubChannel`, `ZmqNodeChannel`, and complete `ZmqRuntime`.

- [ ] **Step 1: Write loopback discovery, seed fallback, routing, and reconnect tests**

```python
def test_listener_accepts_only_current_protocol_beacon(udp_group, free_udp_port):
    publisher = HubBeaconPublisher(udp_group, free_udp_port, rate_hz=100)
    listener = HubBeaconListener(udp_group, free_udp_port)
    publisher.publish(DiscoveryBeacon(1, "epoch-1", "tcp://192.0.2.10:5560"))
    assert listener.receive(timeout_s=0.2).hub_epoch == "epoch-1"

def test_hub_seed_connects_when_multicast_yields_nothing(zmq_runtime, hub_endpoint):
    hub = zmq_runtime.open_hub(
        hub_endpoint, hub_epoch="epoch-1", advertise_endpoint=hub_endpoint,
    )
    node = zmq_runtime.open_node("robot-1", "session-1", hub_seed=hub_endpoint)
    node.send(message("register", sender="robot-1"))
    assert eventually(hub.receive).peer_id == "robot-1"

def test_restarted_node_same_id_uses_new_dealer_session(zmq_runtime, hub_endpoint):
    hub = zmq_runtime.open_hub(
        hub_endpoint, hub_epoch="epoch-1", advertise_endpoint=hub_endpoint,
    )
    first = zmq_runtime.open_node("robot-1", "session-1", hub_seed=hub_endpoint)
    first.send(message("register", sender="robot-1", session="session-1"))
    assert eventually(hub.receive).message.sender_session_id == "session-1"
    first.close()

    second = zmq_runtime.open_node("robot-1", "session-2", hub_seed=hub_endpoint)
    second.send(message("register", sender="robot-1", session="session-2"))
    assert eventually(hub.receive).message.sender_session_id == "session-2"
    assert hub.send("robot-1", message("registered", sender="hub", session="hub-session"))
    assert eventually(second.receive).message.kind == "registered"
    assert first.receive(timeout_s=0.0) is None
```

- [ ] **Step 2: Run discovery/management tests and confirm missing support**

Run: `uv run pytest tests/control/test_zmq_runtime.py -q -k 'discovery or management or seed or reconnect'`

Expected: selected tests fail because discovery and management classes are absent.

- [ ] **Step 3: Implement a strict one-second multicast beacon with seed fallback**

```python
@dataclass(frozen=True, slots=True)
class DiscoveryBeacon:
    protocol_version: int
    hub_epoch: str
    management_endpoint: str

DEFAULT_MULTICAST_GROUP = "239.255.42.99"
DEFAULT_DISCOVERY_PORT = 45990
```

Encode only `protocol_version`, `hub_epoch`, and `management_endpoint`; cap datagrams at 1 KiB, disable loopback only when explicitly configured for production, validate IPv4 multicast group and reachable non-wildcard endpoint, and use `SO_REUSEADDR`. `ZmqNodeChannel` races beacon discovery with `hub_seed`, selects the first valid endpoint, and reconnects with the same stable `node_id` but current `session_id`.

- [ ] **Step 4: Implement ROUTER/DEALER management with explicit peer identity**

DEALER identity is UTF-8 `node_id/session_id`. ROUTER accepts exactly two frames `[identity, payload]`; malformed multipart input is dropped and counted. Both sockets set `LINGER=0`; management send is `NOBLOCK`; ROUTER `SNDHWM` and `RCVHWM` default to 256; DEALER uses heartbeat options as a transport hint but application liveness still comes from Node reports.

```python
class ZmqRuntime:
    def open_hub(self, endpoint: str, *, hub_epoch: str,
                 advertise_endpoint: str | None = None) -> HubChannel:
        channel = ZmqHubChannel(endpoint, context=self._context,
                                monotonic_ns=self._monotonic_ns)
        channel.start_beacon(hub_epoch, advertise_endpoint or endpoint, self._discovery)
        return channel
    def open_node(self, node_id: str, session_id: str, *, hub_seed: str | None) -> NodeChannel:
        endpoint = self._discovery.resolve(hub_seed=hub_seed)
        return ZmqNodeChannel(endpoint, node_id, session_id,
                              context=self._context, monotonic_ns=self._monotonic_ns)
    def open_action_publisher(self, endpoint: str) -> ActionPublisher:
        return ZmqActionPublisher(endpoint, context=self._context)
    def open_action_receiver(self, endpoint: str) -> LatestActionReceiver:
        return ZmqLatestActionReceiver(endpoint, context=self._context,
                                       monotonic_ns=self._monotonic_ns)
```

- [ ] **Step 5: Verify reconnect, bounded shutdown, and no domain imports**

Run: `uv run pytest tests/control/test_zmq_runtime.py -q`

Expected: all real loopback tests pass; multicast-specific environments may use loopback injection rather than broad network assumptions.

Run: `! rg -n "from lekit.control.(hub|controller|robot|handles)" src/lekit/control/zmq_runtime.py src/lekit/control/discovery.py`

Expected: no matches, proving the Adapter contains no domain decisions.

Run: `uv run ruff check src/lekit/control/discovery.py src/lekit/control/zmq_runtime.py tests/control/test_zmq_runtime.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 9: Hub Service Loop, FastAPI API, WebSocket, and UI

**Files:**
- Create: `src/lekit/control/web.py`
- Create: `src/lekit/control/hub.html`
- Test: `tests/control/test_web.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: public synchronous `Hub` Interface and serializable snapshots/history.
- Produces: `create_hub_app(hub: Hub) -> FastAPI`, REST endpoints, `/ws`, and package data for `hub.html`.

- [ ] **Step 1: Write failing API and WebSocket tests using a fake Hub**

```python
def test_assignment_uses_public_hub_interface(client, fake_hub):
    response = client.post("/api/assign", json={
        "robot": "piper-01", "controller": "quest3-main", "control_mode": "teleop"
    }, headers={"X-Operator": "operator-1"})
    assert response.status_code == 201
    assert fake_hub.calls[-1] == ("assign", "piper-01", "quest3-main", "teleop", "operator-1")

def test_force_hold_requires_nonempty_reason(client, fake_hub):
    response = client.post("/api/robots/piper-01/force-hold", json={"reason": ""})
    assert response.status_code == 422

def test_websocket_streams_only_new_snapshot_versions(client, fake_hub):
    with client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["version"] == 1
        fake_hub.publish(version=2)
        assert socket.receive_json()["version"] == 2
```

- [ ] **Step 2: Run Web tests and confirm the app is absent**

Run: `uv run pytest tests/control/test_web.py -q`

Expected: collection fails on missing `create_hub_app`.

- [ ] **Step 3: Implement thin HTTP handlers over Hub**

Routes are exactly:

```text
GET  /                         packaged hub.html
GET  /api/snapshot             Hub.get_snapshot()
GET  /api/nodes                Hub.list_nodes()
GET  /api/history?limit=200    Hub history read
POST /api/assign
POST /api/handles/{id}/take-over
POST /api/handles/{id}/hand-over
POST /api/handles/{id}/renew
POST /api/handles/{id}/revoke
POST /api/robots/{id}/force-hold
WS   /ws                       Hub.watch() snapshots
```

Use Pydantic request models with non-empty, length-limited IDs/reasons. Map domain `NotFound` to 404, `ControlConflict` to 409, and compatibility/safety rejection to 422. Pass an optional `X-Operator` request header, otherwise the request client address, into Hub audit calls. Handlers never import ZeroMQ or `sqlite3`. Run blocking `Hub.watch(after_version, timeout_s=1.0)` through `anyio.to_thread.run_sync`, check WebSocket disconnect between bounded waits, and close cleanly on browser disconnect.

- [ ] **Step 4: Build a stable, dependency-free live UI**

`hub.html` uses fixed-layout tables (`table-layout: fixed`) with fixed widths for Node, role, runtime, control, session, assignment, Handle, TTL, rate, frame age, sequence, error, and actions. Render inventory, desired/observed mismatch badges, SAFETY red alerts, TTL countdown from server timestamps, history, and explicit confirmation dialogs for revoke and force-HOLD. Buttons call only the REST routes above; live state comes only from `/ws` with reconnect backoff.

Add this entry to the existing `[tool.setuptools.package-data]` table:

```toml
"lekit.control" = ["hub.html"]
```

while preserving all existing package-data entries.

- [ ] **Step 5: Verify Web behavior and asset packaging**

Run: `uv run pytest tests/control/test_web.py -q`

Expected: all tests pass, including HTML column labels and route wiring.

Run: `uv build && unzip -l dist/lekit-*.whl | rg 'lekit/control/hub.html'`

Expected: wheel listing contains `lekit/control/hub.html`.

Run: `uv run ruff check src/lekit/control/web.py tests/control/test_web.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 10: Adapt Isaac Teleop into a Managed Controller

**Files:**
- Modify: `src/lekit/teleoperators/isaac_teleop/teleop_node.py`
- Modify: `src/lekit/teleoperators/isaac_teleop/__init__.py`
- Test: `tests/teleoperators/test_isaac_teleop_node.py`
- Test: `tests/teleoperators/test_isaac_teleop_protocol.py`

**Interfaces:**
- Consumes: existing `TeleopFrame`, `encode_action_frame`, `TeleopNodeState`, and Task 5 `ControllerNode`.
- Produces: `IsaacControllerNodeConfig`, `make_isaac_controller_node`, and optional `control_node` composition in existing `TeleopNode`.

- [ ] **Step 1: Add failing managed-mode tests without weakening standalone tests**

```python
def test_managed_teleop_publishes_encoded_frame_only_when_handle_active(fake_control_node):
    node = TeleopNode(config, control_node=fake_control_node,
                      controller_factory=fake_xr_factory, monitor_factory=fake_monitor)
    node.run(max_frames=1)
    payload, metadata = fake_control_node.published[0]
    decoded = decode_action_frame(payload)
    assert decoded.action.keys() == neutral_action().keys()
    assert metadata["captured_monotonic_ns"] == decoded.captured_monotonic_ns

def test_managed_mode_starts_and_stops_control_node_even_when_xr_connect_fails(fake_control_node):
    with pytest.raises(ExpectedXRFailure):
        TeleopNode(config, control_node=fake_control_node,
                   controller_factory=failing_xr_factory).run(max_frames=1)
    assert fake_control_node.events == ["start", "stop"]
```

- [ ] **Step 2: Run current and new Isaac tests**

Run: `uv run pytest tests/teleoperators/test_isaac_teleop_node.py tests/teleoperators/test_isaac_teleop_protocol.py -q`

Expected: only the new managed-mode tests fail because `control_node` is not accepted.

- [ ] **Step 3: Compose rather than subclass the generic ControllerNode**

```python
def __init__(
    self,
    config: TeleopNodeConfig,
    *,
    control_node: ControllerNode | None = None,
    controller_factory: Callable[[IsaacTeleopConfig], Any] = IsaacXRController,
    publisher_factory: Callable[[str], Any] = ZmqTeleopPublisher,
    monitor_factory: Callable[[TeleopNodeState, str, int], Any] = MonitorServer,
    monotonic: Callable[[], float] = time.monotonic,
    utc_ns: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    self.config = config
    self._control_node = control_node
    self._controller_factory = controller_factory
    self._publisher_factory = publisher_factory
    self._monitor_factory = monitor_factory
    self._monotonic = monotonic
    self._utc_ns = utc_ns
    self._sleep = sleep

def _publish_frame(self, frame: TeleopFrame) -> bool:
    if self._control_node is not None:
        return self._control_node.publish(
            encode_action_frame(frame),
            captured_monotonic_ns=frame.captured_monotonic_ns,
            captured_utc_ns=frame.captured_utc_ns,
        )
    assert self._publisher is not None
    return self._publisher.publish_action(frame)
```

Managed mode calls `control_node.start()` before XR acquisition and `control_node.stop()` in the outer `finally`. The existing monitor remains device-local and unchanged; Hub registration, Handle, and distributed control state are shown in the Hub UI. The XR sampling loop remains 60 Hz and never calls Hub or waits for management.

- [ ] **Step 4: Add the Isaac Controller factory and preserve raw standalone behavior**

`IsaacControllerNodeConfig` holds stable ID path, display name, action endpoint, Hub seed, and existing `TeleopNodeConfig`. `make_isaac_controller_node(config, runtime)` declares exactly `ACTION_SCHEMA` plus version as `lekit.isaac_teleop.action.v1` and `control_modes=("teleop",)`. Calling `TeleopNode(config)` without `control_node` continues using `ZmqTeleopPublisher` and all old tests remain unchanged.

- [ ] **Step 5: Verify all Isaac transport, subscriber, monitor, and reconnect regressions**

Run: `uv run pytest tests/teleoperators/test_isaac_teleop_protocol.py tests/teleoperators/test_isaac_teleop_transport.py tests/teleoperators/test_isaac_teleop_node.py tests/teleoperators/test_isaac_teleop_node_monitor.py tests/teleoperators/test_isaac_teleop_subscriber.py -q`

Expected: all tests pass; dual-controller actions remain atomic and Quest session changes still require release-to-rearm.

Run: `uv run ruff check src/lekit/teleoperators/isaac_teleop tests/teleoperators && git diff --check`

Expected: exit 0; existing user changes in `teleop_node_monitor.html` remain intact; no commit is created.

### Task 11: Piper RobotNode Adapter and Active HOLD

**Files:**
- Create: `src/lekit/robots/piper/robot_node.py`
- Modify: `src/lekit/robots/piper/__init__.py`
- Test: `tests/robots/test_piper_robot_node.py`
- Test: `tests/robots/test_piper_teleop_processor.py`

**Interfaces:**
- Consumes: existing `PiperRobot`, `PiperRobotConfig`, `make_piper_isaac_processor`, Isaac `decode_action_frame`, and generic `RobotNode` / `PayloadProcessor`.
- Produces: `PiperNodeConfig`, `PiperIsaacPayloadProcessor`, `piper_active_hold`, and `make_piper_robot_node(config, runtime)`.

- [ ] **Step 1: Write failing pure adapter tests with fake Piper objects**

```python
def test_piper_payload_processor_reuses_existing_retargeting_pipeline(fake_pipeline, observation):
    adapter = PiperIsaacPayloadProcessor(fake_pipeline)
    frame = TeleopFrame("xr-session", 4, 10, 20, neutral_action())
    result = adapter(encode_action_frame(frame), observation)
    assert fake_pipeline.calls == [(frame.action, observation)]
    assert result == fake_pipeline.result

def test_piper_active_hold_sends_current_complete_tcp_only(fake_piper):
    observation = {key: value for key, value in zip(PiperRobot._EEF_KEYS, range(6), strict=True)}
    result = piper_active_hold(fake_piper, observation, "watchdog")
    assert fake_piper.sent == [observation]
    assert result.active is True

def test_incomplete_tcp_falls_back_to_passive_hold(fake_piper):
    result = piper_active_hold(fake_piper, {"ee.x": 0.1}, "feedback_incomplete")
    assert fake_piper.sent == []
    assert result.active is False
```

- [ ] **Step 2: Run adapter and existing processor tests**

Run: `uv run pytest tests/robots/test_piper_robot_node.py tests/robots/test_piper_teleop_processor.py -q`

Expected: new adapter tests fail on missing module; existing processor tests pass.

- [ ] **Step 3: Implement a thin payload processor over validated existing code**

```python
class PiperIsaacPayloadProcessor:
    accepted_payload_schemas = frozenset({"lekit.isaac_teleop.action.v1"})

    def __init__(self, pipeline: RobotProcessorPipeline):
        self._pipeline = pipeline
        self._hand = pipeline.steps[0].config.hand
        self._tracking = False
        self._engaged = False

    def __call__(self, payload: bytes, observation: RobotObservation) -> RobotAction:
        frame = decode_action_frame(payload)
        self._tracking = bool(frame.action[f"{self._hand}.is_tracking"])
        self._engaged = bool(frame.action[f"{self._hand}.is_engaged"])
        return self._pipeline((frame.action, observation))

    def reset(self) -> None:
        self._pipeline.reset()

    def status(self) -> Mapping[str, Any]:
        step = self._pipeline.steps[0]
        return {
            "processor_state": getattr(getattr(step, "state", None), "value", "unknown"),
            "tracking": self._tracking,
            "engaged": self._engaged,
            "error": getattr(step, "fault_reason", None),
        }
```

Do not duplicate coordinate transforms, anchor behavior, tracking, gripper, workspace, or re-arm logic. The generic envelope sequence protects transport; the nested Isaac sequence/session validation remains in `decode_action_frame` and processor input tests.

- [ ] **Step 4: Implement safe construction and Piper active hold**

`PiperNodeConfig` contains the existing `PiperRobotConfig`, existing `PiperTeleopProcessorConfig`, generic `RobotNodeConfig`, and `enable_motion: bool = False`. Factory behavior:

1. Set `robot.auto_enable=False` when motion is not explicitly enabled.
2. Build `PiperRobot` and the existing pipeline.
3. Intersect `include_gripper` exactly as the current teleop script does.
4. Construct generic `RobotNode` with the adapter and `piper_active_hold`.
5. In dry-run mode set `RobotNodeConfig.control_enabled=False`, connect/read/report, reject take-over, and never invoke `send_action` or active hold.

`piper_active_hold` extracts all six `PiperRobot._EEF_KEYS`, requires finite real values, and sends that measured TCP action once. If feedback is incomplete or the SDK call fails, return `HoldResult(active=False, detail=...)` and preserve/report the original control fault.

- [ ] **Step 5: Verify no hardware construction occurs in imports or tests**

Run: `uv run pytest tests/robots/test_piper_robot_node.py tests/robots/test_piper_robot.py tests/robots/test_piper_teleop_processor.py -q`

Expected: all tests pass using fakes; no CAN interface is opened.

Run: `uv run ruff check src/lekit/robots/piper tests/robots/test_piper_robot_node.py && git diff --check`

Expected: exit 0; no commit is created.

### Task 12: Three-Process CLI and End-to-End Integration

**Files:**
- Create: `src/lekit/control/cli.py`
- Modify: `src/lekit/control/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/control/test_end_to_end.py`
- Create: `tests/control/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Hub`, `ZmqRuntime`, Web app/server, Isaac factory, Piper factory.
- Produces: `lekit` console command with `hub`, `teleop`, and `robot` subcommands plus a no-hardware integration harness.

- [ ] **Step 1: Write failing CLI parsing and MemoryRuntime full-flow tests**

```python
def test_cli_exposes_exact_process_names():
    parser = build_parser()
    assert parser.parse_args(["hub"]).command == "hub"
    assert parser.parse_args(["teleop"]).command == "teleop"
    assert parser.parse_args(["robot", "--kind", "piper"]).command == "robot"

def test_complete_flow_assign_take_over_action_hand_over(memory_system):
    memory_system.start()
    handle = memory_system.hub.assign("fake-robot", "fake-controller")
    memory_system.hub.request_take_over(handle)
    memory_system.pump_until(lambda: memory_system.hub.get_snapshot(handle.handle_id).healthy)
    memory_system.controller.publish(b"action", captured_monotonic_ns=10, captured_utc_ns=20)
    memory_system.robot.run_cycle()
    assert memory_system.fake_robot.sent == [{"x": 1.0}]
    memory_system.hub.request_hand_over(handle)
    memory_system.pump_until(lambda: memory_system.hub.get_snapshot(handle.handle_id).handle_state
                             is HandleState.RELEASED)
    assert memory_system.robot.control_state is RobotControlState.HOLD
```

- [ ] **Step 2: Run CLI and end-to-end tests and observe missing wiring**

Run: `uv run pytest tests/control/test_cli.py tests/control/test_end_to_end.py -q`

Expected: CLI imports or full-flow assertions fail before wiring exists.

- [ ] **Step 3: Implement exact console entry point and safe defaults**

Add:

```toml
[project.scripts]
lekit = "lekit.control.cli:main"
```

`lekit hub` options: `--management-endpoint tcp://0.0.0.0:5560`, required `--advertise-host` when the bind host is wildcard, `--database .lekit/control-hub.sqlite3`, `--web-host 127.0.0.1`, `--web-port 8080`, `--multicast-group 239.255.42.99`, and `--discovery-port 45990`. The CLI derives `advertise_endpoint` by preserving the management port and replacing only the wildcard host.

`lekit teleop` options preserve existing CloudXR/head-yaw/monitor arguments and add `--node-id-file .lekit/nodes/quest3-main`, `--display-name quest3-main`, `--action-endpoint tcp://0.0.0.0:5557`, and `--hub-seed`.

`lekit robot --kind piper` accepts the current Piper/processor options through config files or LeRobot parser-compatible dotted arguments, plus `--node-id-file .lekit/nodes/piper-01`, `--display-name piper-01`, `--hub-seed`, and explicit `--enable-motion`. Without `--enable-motion`, it performs registration and read-only Robot observation only.

Every command installs SIGINT/SIGTERM handlers, performs bounded joins, enters local HOLD before Robot disconnect, and prints Hub/Web/action endpoints without printing secrets.

- [ ] **Step 4: Add no-hardware MemoryRuntime and real loopback ZMQ system tests**

Cover complete registration, grant-delivery acknowledgement/retry, assign, routed take-over, direct action, ACTIVE correlation, hand-over with and without Controller acknowledgement, reclaim, revoke during stream, force-HOLD, SAFETY preemption, missed renewal expiry, Controller death, Robot death, Hub epoch restart, duplicate/out-of-order management messages, wrong Handle/fencing/session, and recovery requiring a new Handle plus new stream session. The loopback test spies on Hub management sockets to prove action bytes travel only between Controller publisher and Robot receiver.

- [ ] **Step 5: Document operational startup and hardware approval boundary**

README commands:

```bash
uv run lekit hub --advertise-host 192.168.5.24
uv run lekit teleop --hub-seed tcp://192.168.5.24:5560
uv run lekit robot --kind piper --hub-seed tcp://192.168.5.24:5560
```

Document that the third command is read-only by default. Show `--enable-motion` separately under a warning that requires cleared workspace, reachable emergency stop, current Piper calibration, successful read-only status, explicit assignment, and an explicit take-over. State that Hub restart, Handle expiry, revoke, hand-over, tracking loss, or action staleness requires release-to-rearm and a fresh take-over; none auto-resume motion.

- [ ] **Step 6: Verify CLI, integration, and package installation**

Run: `uv run pytest tests/control/test_cli.py tests/control/test_end_to_end.py -q`

Expected: all tests pass without CloudXR or CAN.

Run: `uv sync && uv run lekit --help && uv run lekit hub --help && uv run lekit teleop --help && uv run lekit robot --help`

Expected: all three exact subcommands appear and help exits 0.

Run: `uv run ruff check src/lekit/control tests/control && git diff --check`

Expected: exit 0; no commit is created.

### Task 13: Full Regression, Performance Invariants, and Hardware Gate Runbook

**Files:**
- Modify only if a regression exposes a required fix: files already listed in Tasks 1–12.
- Test: all repository tests relevant to control, Isaac, Piper, and existing teleop script.
- Document: `README.md` hardware gate section from Task 12.

**Interfaces:**
- Consumes: complete implementation.
- Produces: verified first deliverable and a staged, operator-approved hardware checklist; this task itself performs no physical motion.

- [ ] **Step 1: Run the complete automated control and regression suite**

Run:

```bash
uv run pytest \
  tests/control \
  tests/teleoperators/test_isaac_teleop_protocol.py \
  tests/teleoperators/test_isaac_teleop_transport.py \
  tests/teleoperators/test_isaac_teleop_node.py \
  tests/teleoperators/test_isaac_teleop_node_monitor.py \
  tests/teleoperators/test_isaac_teleop_subscriber.py \
  tests/robots/test_piper_robot.py \
  tests/robots/test_piper_teleop_processor.py \
  tests/robots/test_piper_robot_node.py \
  tests/scripts/test_piper_teleop.py -q
```

Expected: all selected tests pass; no hardware device is opened.

- [ ] **Step 2: Run static and packaging checks**

Run:

```bash
uv run ruff check src/lekit/control src/lekit/teleoperators/isaac_teleop src/lekit/robots/piper tests/control
uv run ruff format --check src/lekit/control src/lekit/teleoperators/isaac_teleop src/lekit/robots/piper tests/control
uv build
git diff --check
```

Expected: every command exits 0 and the wheel contains the CLI plus Hub HTML.

- [ ] **Step 3: Measure latest-only and watchdog invariants under synthetic load**

Run: `uv run pytest tests/control/test_zmq_runtime.py tests/control/test_end_to_end.py -q -k 'slow or stale or watchdog or no_hub_relay or shutdown' --count=20`

If `pytest-repeat` is not installed, run the same command from a shell loop twenty times without adding a project dependency. Expected each run: slow consumers see the newest sequence, no action enters Hub, stale input reaches HOLD by the first Robot cycle after 100 ms, and shutdown remains bounded.

- [ ] **Step 4: Confirm dirty-worktree preservation and implementation scope**

Run: `git status --short && git diff --stat && git diff --check`

Expected: pre-existing edits to `src/lekit/scripts/teleop.py`, `src/lekit/teleoperators/isaac_teleop/teleop_node_monitor.html`, and `tests/scripts/test_piper_teleop.py` remain present and are not overwritten. New changes are limited to the files enumerated in this plan plus fixes directly required by a failing listed regression.

- [ ] **Step 5: Stop before any hardware motion and present staged gates for separate approval**

Present these gates one at a time, with no gate implying approval for the next:

1. Hub plus Piper Node read-only registration and observation.
2. Quest Controller registration and telemetry while no Handle exists.
3. Assignment and take-over while Piper remains physically held/HOLD.
4. Bounded translation.
5. Bounded rotation.
6. Bounded gripper.
7. Hand-over and Hub revoke.
8. Controller loss, Hub loss/TTL expiry, and recovery/re-arm.

Do not execute any motion gate until the user separately confirms cleared workspace, reachable emergency stop, and the exact bounded test. Do not create a git commit unless separately requested.

## Completion Criteria

- `Hub.assign`, renew, routed take-over/hand-over, revoke, force-HOLD, registry, snapshots, and watch work through both MemoryRuntime and ZeroMQ.
- A Controller cannot publish authorized frames before Robot readiness, after hand-over/revoke/expiry, or across a process/stream session change.
- A Robot sends actions only after every authorization/freshness/schema/processor check and enters HOLD on every specified failure.
- Hub restart invalidates all old non-terminal Handles and no process restart resumes motion.
- Hub Web UI shows Nodes, sessions, desired and observed state, Handle/TTL, metrics, mismatches, errors, controls, and history in stable-width tables.
- `lekit hub`, `lekit teleop`, and `lekit robot` install and run with safe defaults.
- Existing Isaac/Piper behavior and user changes remain intact.
- Automated verification passes without accessing physical hardware.

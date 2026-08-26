# Independent Isaac Teleop Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long-lived Isaac XR publisher, a standard LeRobot subscriber, and a read-only browser monitoring dashboard without connecting to a robot.

**Architecture:** `IsaacXRController` remains the only CloudXR/OpenXR owner. A versioned protocol and ZeroMQ transport carry atomic dual-controller frames to independent subscribers; a thread-safe state store feeds a FastAPI dashboard. Downstream code sees a normal LeRobot `Teleoperator` and implements local stale-frame safety.

**Tech Stack:** Python 3.12, NumPy, ZeroMQ/pyzmq, FastAPI, Uvicorn, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-08-26-independent-teleop-node-design.md`

## Global Constraints

- Do not connect to or command a robot.
- Preserve all existing user changes in the dirty worktree.
- Keep all runtime modules under `src/lekit/teleoperators/isaac_teleop/`.
- Keep the 28 action fields atomic in one published frame.
- Use subscriber-local monotonic time for stale detection.
- Default network binds remain on `127.0.0.1`.
- Follow strict red-green-refactor for production behavior.

---

### Task 1: Versioned action protocol

**Files:**
- Create: `src/lekit/teleoperators/isaac_teleop/protocol.py`
- Modify: `src/lekit/teleoperators/isaac_teleop/xr_controller.py`
- Modify: `src/lekit/teleoperators/isaac_teleop/quest3_visualizer.py`
- Test: `tests/teleoperators/test_isaac_teleop_protocol.py`

**Interfaces:**
- Produces: `action_features() -> dict`, `neutral_action() -> RobotAction`, `TeleopFrame`, `encode_action_frame(frame) -> bytes`, and `decode_action_frame(payload) -> TeleopFrame`.
- Consumes: the existing 28-key `IsaacXRController.get_action()` schema.

- [ ] Write protocol tests for literal schema keys, array/scalar types, round trip, and malformed data.
- [ ] Run the protocol tests and verify they fail because the module is missing.
- [ ] Implement only the protocol behavior required by the tests.
- [ ] Replace duplicate action-feature and neutral-action declarations with the shared helpers.
- [ ] Run protocol, controller, and visualizer tests until green.

### Task 2: Bounded ZeroMQ transport

**Files:**
- Create: `src/lekit/teleoperators/isaac_teleop/transport.py`
- Test: `tests/teleoperators/test_isaac_teleop_transport.py`

**Interfaces:**
- Consumes: encoded `TeleopFrame` bytes from Task 1.
- Produces: `ZmqTeleopPublisher.publish_action`, `publish_status`, and `ZmqTeleopReceiver.receive_latest`.

- [x] Write real localhost PUB/SUB tests for late subscription, newest-frame conflation, topic filtering, and close behavior.
- [ ] Run the transport tests and verify expected missing-interface failures.
- [x] Implement bounded, zero-linger PUB/SUB sockets and latest-frame conflation.
- [ ] Run the transport tests until green.

### Task 3: Standard LeRobot subscriber

**Files:**
- Create: `src/lekit/teleoperators/isaac_teleop/subscriber.py`
- Modify: `src/lekit/teleoperators/isaac_teleop/__init__.py`
- Test: `tests/teleoperators/test_isaac_teleop_subscriber.py`

**Interfaces:**
- Consumes: `ZmqTeleopReceiver.receive_latest()` and `TeleopFrame`.
- Produces: `IsaacTeleopNodeConfig` and `IsaacTeleopNodeSubscriber(Teleoperator)`.

- [ ] Write tests for config validation, first-frame connection, timeout, non-blocking latest action, stale neutralization, sequence rejection, and new-session release-to-rearm.
- [ ] Run subscriber tests and verify expected failures.
- [ ] Implement the minimal LeRobot adapter and safety state machine.
- [ ] Run subscriber and protocol/transport tests until green.

### Task 4: Monitoring state and Web page

**Files:**
- Create: `src/lekit/teleoperators/isaac_teleop/node_state.py`
- Create: `src/lekit/teleoperators/isaac_teleop/teleop_node_monitor.html`
- Modify: `pyproject.toml`
- Test: `tests/teleoperators/test_isaac_teleop_node_monitor.py`

**Interfaces:**
- Produces: `TeleopNodeState`, `TeleopNodeSnapshot`, and `create_monitor_app(state)`.
- Consumes: normalized actions from Task 1.

- [ ] Write real FastAPI route tests for `/`, `/api/status`, and `/healthz`, including all 28 action fields.
- [ ] Run monitor tests and verify expected failures.
- [ ] Implement the thread-safe snapshot store, read-only endpoints, and polling dashboard asset.
- [ ] Run monitor tests until green.

### Task 5: Long-lived node runner and CLI

**Files:**
- Create: `src/lekit/teleoperators/isaac_teleop/teleop_node.py`
- Test: `tests/teleoperators/test_isaac_teleop_node.py`

**Interfaces:**
- Consumes: `IsaacXRController`, `ZmqTeleopPublisher`, and `TeleopNodeState`.
- Produces: `TeleopNodeConfig`, `TeleopNode.run`, and `python -m lekit.teleoperators.isaac_teleop.teleop_node`.

- [ ] Write finite fake-source tests for waiting, streaming, atomic publication, rate metrics, reconnect state, and clean stop.
- [ ] Run node tests and verify expected failures.
- [ ] Implement the node lifecycle, monitor-server thread, reconnect loop, signal handling, and CLI.
- [ ] Run node tests until green.

### Task 6: Usage documentation and full verification

**Files:**
- Modify: `src/lekit/teleoperators/isaac_teleop/README.md`

**Interfaces:**
- Documents: node startup, SSH port forwarding, subscriber usage, endpoint exposure, and safety behavior.

- [ ] Add exact server, SSH tunnel, browser, and subscriber commands.
- [ ] Run every Isaac teleop test.
- [ ] Run Ruff check and format verification over changed source and tests.
- [ ] Review the final diff against the design spec and verify no robot-control code was introduced.

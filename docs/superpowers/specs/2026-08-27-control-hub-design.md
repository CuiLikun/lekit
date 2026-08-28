# Lekit Control Hub Design

## Goal

Provide a small distributed control system in which independently running Robots and Controllers automatically discover and register with a central Hub. The Hub distributes short-lived exclusive Control Handles, observes desired and actual control state, and can revoke or force-hold control at any time. Controllers send real-time actions directly to Robots so the Hub never becomes part of the per-frame control path.

The first deployment connects the existing Isaac Quest 3 Teleop Controller to one Piper Robot, while keeping the model general enough for later Policy Controllers and other standard LeRobot Robots.

## Scope

- `Hub` replaces the earlier user-facing term "control-plane".
- `Controller` covers Teleop, Policy, and future action-producing Nodes.
- `Robot` is a Node that owns one standard LeRobot `Robot` instance.
- Nodes automatically discover and register with the Hub on a trusted IPv4 local network.
- The Hub schedules exclusive Robot control by distributing and reclaiming Control Handles.
- Controllers expose `take_over(handle)` and `hand_over(handle)`.
- The Hub observes both its desired control state and the actual state reported by both endpoints.
- ZeroMQ is the first communication implementation.
- Real-time Controller-to-Robot actions travel directly and use bounded latest-frame semantics.
- SQLite persists Node identity metadata, Handle transitions, and audit events.
- The existing Piper retargeting processor remains responsible for Isaac XR-to-Piper action mapping, tracking, clutching, anchoring, workspace limiting, and gripper behavior.

## Non-goals

- Dora integration in the first implementation.
- Public-Internet deployment, TLS, user RBAC, or hostile-network security.
- Automatic scheduling policies beyond explicit Robot-to-Controller assignment.
- Simultaneous blended control from multiple Controllers.
- Restoring motion automatically after any process restart, safety transition, or expired Handle.
- Relaying, recording, or persisting every real-time action through the Hub.
- A universal active-stop implementation for hardware not represented by the standard LeRobot `Robot` Interface.

## Domain model

The canonical vocabulary is recorded in the repository root `CONTEXT.md`.

### Node

An independently running participant registered with the Hub. Every Node has a stable `node_id`, a role, a display name, capabilities, status, a process session ID, and one Hub management connection.

### Controller

A Node that produces control frames. Teleop and Policy are Controller types. A Controller may publish device-relative input, absolute Robot actions, or another declared action schema. It may control a Robot only while holding that Robot's valid Control Handle.

### Robot

A Node that owns a physical or simulated standard LeRobot `Robot`. It is the final local authority over accepted actions and HOLD/SAFETY behavior. It accepts frames from at most one active Control Handle.

### Hub

The central authority for Node registry, compatibility checking, Control Handle scheduling, desired-state management, live-state correlation, operator management, and audit history. It does not relay action frames.

### Control Handle

A one-assignment, exclusive, expiring, revocable authorization binding exactly one Controller to exactly one Robot. A released, revoked, expired, or faulted Handle is never reused. Reclaiming a Handle terminates it and makes the Robot eligible for a newly minted Handle.

## Architecture

```text
                     +--------------------------------+
                     |              Hub               |
                     | registry, handles, live state, |
                     | management, Web UI, SQLite     |
                     +---------------+----------------+
                          management | management
                         and status  | and status
                    +----------------+----------------+
                    v                                 v
             +-------------+                   +-------------+
             | Controller  |                   |    Robot    |
             | Teleop/     |                   | RobotNode + |
             | Policy      |                   | LeRobot     |
             +------+------+                   +-------------+
                    |                                 ^
                    | direct latest-only actions     |
                    +=================================+
```

Three communication paths remain distinct:

1. **Discovery:** the Hub announces its management endpoint and Hub epoch through a low-rate UDP multicast beacon. Nodes on the same local network listen for the beacon. A configured `hub_seed` provides a deterministic fallback for multicast-disabled or cross-subnet environments.
2. **Management:** the Hub owns a ZeroMQ ROUTER socket; each Node connects through a DEALER socket using its stable identity. Registration, routed RPC, commands, acknowledgements, Handle transitions, heartbeats, metrics, and errors use this path. Controller-to-Robot management requests are routed through the Hub so every transition is observable.
3. **Actions:** a Controller publishes directly to the assigned Robot. The receiver is non-blocking, retains only the newest complete frame, and never waits for the Hub.

The Hub and Nodes use one internal Runtime seam. `ZmqRuntime` is the production Adapter and `MemoryRuntime` is the deterministic test Adapter. A future `DoraRuntime` may satisfy the same Interface without changing Hub scheduling, Handle semantics, Robot action validation, or Controller public methods.

The Runtime Interface owns discovery, registration, management RPC/events, status publication, and direct action streams. Robot and Controller implementations must not import or construct ZeroMQ sockets outside the ZeroMQ Adapter.

## Naming and public Interface

User-facing process names are concise:

```text
lekit hub
lekit teleop
lekit robot
```

The public Python model is explicit:

```python
hub = Hub(...)
robot = RobotNode(robot=piper, processor=piper_processor)
teleop = TeleopNode(teleoperator=isaac_teleop)
```

Network configuration, discovery, registration, heartbeat, reconnection, status publication, and socket lifecycle are internal. A Robot still requires a LeRobot Robot or Robot configuration; a truly argument-free `RobotNode()` cannot know which hardware to construct.

The central control Interface is:

```python
handle = hub.assign(robot="piper-01", controller="quest3-main")
hub.renew(handle)
hub.revoke(handle, reason="operator_request")
hub.force_hold(robot="piper-01", reason="safety_check")
hub.list_nodes()
hub.get_snapshot()
hub.watch()
```

Every Controller exposes:

```python
controller.take_over(handle)
controller.hand_over(handle)
```

These methods have the same semantics for Teleop and Policy Controllers. They are local methods backed by Hub-routed management RPC and direct action-stream operations when the participants run in separate processes.

## Automatic discovery and registration

The Hub starts a new random `hub_epoch` on every process start and broadcasts a versioned discovery beacon at one-second intervals. The beacon contains only the protocol version, Hub epoch, and reachable management endpoint.

A Node starts as follows:

1. Load or create its stable `node_id`; create a fresh process `session_id`.
2. Listen for a compatible Hub beacon. If `hub_seed` is configured, attempt it in parallel.
3. Connect its DEALER management socket to the Hub ROUTER endpoint.
4. Send a complete registration descriptor.
5. Receive the current Hub epoch and registration acknowledgement.
6. Publish heartbeats and status until disconnected.
7. On management loss, retain local safe behavior, rediscover, reconnect, and register a fresh process session without restoring motion.

A registration descriptor includes:

- protocol and schema versions;
- stable Node ID, process session ID, role, and display name;
- Controller or Robot capabilities;
- supported action schemas and control modes;
- direct action endpoint when the Node is a Controller;
- Robot observation/action feature metadata when the Node is a Robot;
- software version and diagnostic metadata.

The Hub uses the management peer address to resolve wildcard action endpoints into reachable addresses. Nodes may bind to `0.0.0.0`, but they must never advertise that wildcard as a consumer endpoint.

The live registry is derived from current sessions and heartbeats. A Node is marked offline after three consecutive heartbeat intervals without a valid report. SQLite retains identity metadata and history, but a persisted Node is not considered online until it registers in the current Hub epoch.

## Compatibility and scheduling

Before assignment, the Hub verifies:

- Robot and Controller are online in the current Hub epoch;
- neither is committed to an incompatible active Handle;
- the Controller output schema is accepted by the Robot or its configured processor;
- requested control mode is supported;
- Robot is not in SAFETY, FAULT, or administratively disabled state.

The first implementation allows at most one non-terminal Handle per Robot and one non-terminal Handle per Controller. This can be relaxed later without changing Handle identity or fencing semantics.

`Hub.assign` uses one Hub store transaction to:

1. check compatibility and exclusivity;
2. increment the Robot's monotonic `fencing_token`;
3. create a random one-time `handle_id` and expiry;
4. persist the Handle in `ASSIGNED` state;
5. commit the desired assignment.

After commit, the Hub grants the Handle to the Robot, distributes it to the Controller, records acknowledgements, and exposes delivery progress. Network delivery is not part of the SQLite transaction; failed delivery leaves the Handle observable in `ASSIGNED` until retry, revoke, or expiry.

The Handle includes:

```text
handle_id
hub_epoch
robot_id
robot_session_id
controller_id
controller_session_id
controller_action_endpoint
action_schema
control_mode
fencing_token
issued_at_ns
expires_at_ns
```

The Hub refreshes a non-terminal Handle every second while the assignment remains desired and both Node sessions are current. The default TTL is three seconds. Renewal changes expiry but never identity or fencing token.

## Handle lifecycle

The normal lifecycle is:

```text
ASSIGNED -> TAKING_OVER -> ACTIVE -> HANDING_OVER -> RELEASED
```

Exceptional terminal paths are:

```text
ASSIGNED / TAKING_OVER / ACTIVE / HANDING_OVER
    -> REVOKING -> REVOKED
    -> EXPIRED
    -> FAULT
```

Terminal Handles cannot transition back to an active state. Robot availability is derived from the absence of a non-terminal Handle and from current Robot health; `AVAILABLE` is therefore a Robot scheduling state, not a reusable Handle state.

Every transition is idempotent and identified by `handle_id`, `fencing_token`, transition sequence, and correlation ID. Duplicate or out-of-order transitions cannot reactivate an older Handle.

## Take-over flow

After the Hub has granted and distributed an `ASSIGNED` Handle, the Controller invokes:

```python
controller.take_over(handle)
```

The flow is:

1. The Controller checks that the Handle targets its current Node and process session.
2. It sends a take-over request to the Hub. The Hub records `TAKING_OVER` and forwards the request to the Robot through the Robot's existing management connection.
3. The Robot verifies its independently received Hub grant, Hub epoch, Robot and Controller identity/session, action schema, expiry, and current fencing token.
4. The Robot disconnects any terminal or stale action input and connects its latest-only receiver to the Controller action endpoint.
5. The Robot resets the processor and remains in HOLD until it receives a fresh valid Handle-tagged frame and all robot-specific re-arm conditions pass.
6. The Robot acknowledges `ready` to the Hub; the Hub forwards readiness to the Controller; the Controller then acknowledges `streaming`.
7. The Hub marks the Handle `ACTIVE` only after correlating both acknowledgements.

The Controller may keep its hardware or inference source running while unassigned. It may publish an internal local stream, but a Robot accepts only frames wrapped in the currently active Control Handle envelope.

## Action envelope and direct data path

One complete action sample is transported atomically in a versioned envelope:

```text
schema_name
schema_version
handle_id
hub_epoch
fencing_token
controller_id
controller_session_id
stream_session_id
sequence
captured_monotonic_ns
captured_utc_ns
payload_schema
payload
```

The payload is opaque to the Runtime. For Isaac Teleop, it is the existing complete dual-controller `TeleopFrame` payload. For a future Policy, it may be a standard LeRobot `RobotAction`. A Robot processor declares which payload schemas it accepts and maps accepted payloads to the wrapped Robot's `action_features`.

`stream_session_id` is newly generated for every successful take-over attempt. Reusing a Handle after an action-stream interruption still requires a new stream session and Robot re-arm; receiving a new packet alone cannot resume the previous stream.

The first ZeroMQ data Adapter preserves these invariants:

- one complete envelope per message;
- non-blocking Controller publication;
- zero linger on shutdown;
- bounded send queues;
- Robot `RCVHWM=1` and `CONFLATE=1`;
- no multipart messages on a conflated input;
- local monotonic receive time recorded before decode/processing;
- duplicate, regressed, wrong-session, wrong-Handle, and wrong-fencing frames rejected;
- only the newest valid frame considered in each Robot control cycle.

The Hub receives summarized metrics, not action payloads.

## Hand-over flow

The Controller invokes:

```python
controller.hand_over(handle)
```

The flow is:

1. The Controller stops publishing frames for the Handle and reports `HANDING_OVER`.
2. It sends a hand-over request to the Hub. The Hub records `HANDING_OVER` and forwards it to the Robot.
3. The Robot stops accepting the Handle, enters HOLD, closes the action receiver, resets the processor, and acknowledges release.
4. Controller and Robot independently report release to the Hub.
5. The Hub marks the Handle `RELEASED` only after the Robot confirms HOLD. Controller acknowledgement may be absent if the Controller failed.
6. The Hub reclaims the terminal Handle and exposes the Robot as eligible for a new assignment.

`hand_over` is idempotent. Calling it again for a terminal Handle succeeds without restoring any connection.

## Local Robot control behavior

RobotNode wraps the standard LeRobot Robot lifecycle:

- `connect()` on Node startup;
- `get_observation()` on the configured control cycle;
- processor mapping from the accepted Controller payload to `RobotAction`;
- `send_action()` only for a fresh, authorized action;
- `disconnect()` during bounded shutdown.

The Robot maintains independent runtime and control states. Runtime states include starting, online, degraded, fault, and stopped. Control states include HOLD, CONTROLLING, and SAFETY. Runtime connectivity must not imply motion authority.

The Robot enters HOLD when any of these occurs:

- no action has arrived within the default 100 ms stale threshold;
- Handle expires or is revoked;
- Handle, Hub epoch, fencing token, Controller identity, or session mismatches;
- action sequence duplicates or regresses;
- payload schema or action validation fails;
- tracking or processor-specific re-arm conditions fail;
- Controller or direct action input disconnects;
- Robot communication reports a recoverable control fault;
- Hub sends force-hold.

HOLD immediately stops forwarding new actions and resets the processor's active mapping state. RobotNode accepts an optional hardware-specific hold callback. Without one, the generic minimum is to stop calling `send_action()` and report that only passive HOLD is available. Piper supplies an active hold behavior appropriate to its SDK and retains its existing processor safety checks.

SAFETY preempts every Handle and cannot be cleared by Controller input or automatic reconnection.

## Hub observability model

The Hub never equates assignment with successful control. It keeps separate desired and observed state.

The desired state comes from Handle scheduling. The observed state is correlated from Robot and Controller reports. A live `ControlSnapshot` includes:

```text
handle_id
robot_id
controller_id
desired_state
handle_state
robot_control_state
controller_control_state
action_rate_hz
frame_age_ms
last_sequence
issued_at_ns
expires_at_ns
last_updated_ns
healthy
error
```

Transitions and faults are reported immediately. Action rate, frame age, sequence, tracking, engagement, processor state, and Robot connection metrics are coalesced at ten hertz. Node heartbeat defaults to two hertz. High-rate metrics remain in memory and are streamed to browser clients over WebSocket.

The Hub explicitly detects and surfaces mismatches, including:

- desired ACTIVE while Robot remains HOLD;
- Controller streaming while Robot is not accepting its Handle;
- Controller offline while Robot reports CONTROLLING;
- no desired Handle while either endpoint reports active control;
- Robot accepting a terminal or stale Handle;
- Robot SAFETY while a Handle remains desired ACTIVE.

The Hub may respond by alerting, revoking the Handle, or issuing force-hold. Robot local HOLD/SAFETY remains authoritative if the Hub is unreachable.

## Hub management and Web UI

The first Web UI provides:

- automatically discovered Robot and Controller inventory;
- online/offline, current session, capabilities, and errors;
- compatible Robot-to-Controller assignment;
- Handle phase and remaining TTL;
- expected versus observed control state;
- Robot connection, tracking, engagement, action rate, frame age, and sequence;
- assignment, renewal, revocation, and force-hold actions;
- mismatch and safety alerts;
- Handle transition and fault history.

FastAPI owns HTTP and WebSocket presentation. UI handlers call the same Hub Interface used by Python callers and do not manipulate ZeroMQ sockets or SQLite rows directly.

## Persistence and restart semantics

SQLite persists:

- stable Node identity metadata and capabilities;
- Hub epoch history;
- Handle records and every state transition;
- operator/request identity when available;
- timestamps, reasons, correlation IDs, acknowledgements, and faults;
- the last known low-rate Node and control snapshot for diagnostics.

High-frequency action metrics and payloads are not persisted.

On Hub restart:

1. create a new Hub epoch;
2. mark every previously non-terminal Handle terminal and invalid;
3. broadcast the new epoch;
4. require every Node to re-register;
5. require a newly minted Handle and a fresh take-over before motion can resume.

Persisted Node records never imply current liveness. Persisted ACTIVE state is diagnostic history, not restored authority.

## Fault handling

### Controller failure

The Robot action watchdog enters HOLD within the stale threshold. The Hub marks the Controller offline after missed heartbeats and revokes or expires its Handle. Recovery never reuses the old Handle.

### Robot failure

The Hub marks the Handle FAULT and the Robot unavailable. A restarted Robot registers a new process session and starts in HOLD.

### Hub failure or partition

Controller and Robot may continue only until the locally cached Handle expires. Default renewal is one second and default TTL is three seconds. No Node extends a Handle without a current Hub renewal. Local action stale and safety checks continue independently.

### Action-network failure

Robot frame age exceeds the stale threshold and causes HOLD. Recovery requires a non-terminal Handle, a current fencing token, a fresh stream session, and processor re-arm; packets arriving after a gap cannot automatically resume motion.

### Lost hand-over acknowledgement

The Controller stops publication before requesting hand-over. The Hub retains `HANDING_OVER` or transitions to `REVOKING`; it does not reassign until Robot HOLD is observed or the old Handle expires and the Robot confirms no active Handle.

### Duplicate and out-of-order management messages

Every command and transition is idempotent under Handle ID, fencing token, transition sequence, and correlation ID. An older desired state cannot overwrite a newer terminal state.

## Performance properties

- The 60 Hz action path does not call Hub, FastAPI, SQLite, discovery, or the management socket.
- Network operations never block XR sampling or Robot control cycles.
- Slow status consumers receive coalesced recent state rather than an unbounded history.
- Action queues are latest-only and cannot accumulate latency.
- JSON remains acceptable for low-rate management/status messages.
- The existing compact action encoding may be retained initially, but the outer Handle envelope is versioned behind the Runtime seam so a binary codec can be introduced without changing control semantics.
- Full observations, camera images, and tensors are not sent to the Hub status path.

## Initial configuration defaults

```text
Controller action rate        60 Hz
Robot action stale threshold  100 ms
status metrics rate           10 Hz
Node heartbeat rate            2 Hz
Hub discovery beacon           1 Hz
Handle renewal interval        1 s
Handle TTL                     3 s
```

Timing values are configurable, but Handle validation, stale-frame rejection, fencing, HOLD, and SAFETY checks cannot be disabled.

## Testing

### Domain and state tests

- Node registration and current Hub epoch validation.
- compatibility checks and exclusive assignment.
- one-time Handle identity and monotonically increasing fencing tokens.
- every valid Handle transition and rejection of invalid transitions.
- duplicate and out-of-order command idempotency.
- desired/observed state correlation and mismatch classification.
- Hub restart invalidation of all old Handles.

### MemoryRuntime integration tests

- automatic registration and re-registration.
- complete assign, take-over, action, hand-over, and reclaim flow.
- Controller, Robot, and Hub failure independently.
- missed renewal, expiry, revoke, force-hold, and SAFETY preemption.
- Controller or Robot process session changes.
- passive versus configured hardware hold reporting.

### Real ZeroMQ integration tests

- discovery on the local network plus configured-seed fallback.
- ROUTER/DEALER reconnection and identity handling.
- direct Controller-to-Robot actions without Hub relay.
- bounded, conflated latest-frame delivery under a slow Robot consumer.
- wrong Handle, old fencing token, duplicate sequence, stale stream, and malformed payload rejection.
- Hub revocation during active streaming.
- network loss and recovery without automatic motion resumption.
- clean shutdown with zero linger and no blocked control loop.

### Existing behavior regression tests

- Isaac dual-controller frames remain atomic and schema-compatible.
- Quest reconnect and session changes still require release-to-rearm.
- Piper translation, rotation, gripper, workspace limits, and fixed-width TUI behavior remain unchanged outside the new Node lifecycle.
- Standard LeRobot Robot connect, observation, action, and disconnect behavior remains intact.

### Hardware gates

Automated tests never open Piper CAN or move hardware. Hardware validation remains separately approved and staged: read-only registration, Controller telemetry, take-over while held, bounded translation, rotation, gripper, hand-over, Handle revoke, Controller loss, Hub loss, and recovery/re-arm.

## First implementation deliverable

The first usable slice includes:

- Hub process with UDP discovery, ZeroMQ management, SQLite audit, FastAPI/WebSocket UI, and in-memory live read model;
- generic RobotNode wrapping a standard LeRobot Robot;
- generic Controller lifecycle supporting `take_over` and `hand_over`;
- adaptation of the existing Isaac Teleop process into a Controller;
- Piper RobotNode configuration using the existing retargeting processor;
- direct latest-only ZeroMQ action path;
- MemoryRuntime and ZeroMQ tests for registration, Handle lifecycle, observability, revocation, expiry, and failure behavior.

Policy Controller implementation, Dora, public-network security, and multi-Controller blending remain later work.

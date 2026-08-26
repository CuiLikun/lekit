# Piper Isaac Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standard absolute TCP control to `PiperRobot` and implement a LeRobot `RobotProcessorPipeline` plus `teleop.py` loop that safely retargets Isaac XR cumulative relative poses to a physical Piper arm.

**Architecture:** `IsaacXRController` remains an independent source of cumulative engage-relative hand poses. A stateful `PiperIsaacRetargetingStep` consumes `(isaac_action, piper_observation)` and emits canonical absolute `ee.*`/gripper actions. `PiperRobot` owns final TCP feedback, workspace/lead limiting, TCP-to-flange conversion, and `move_p()` submission; `teleop.py` owns lifecycle, pacing, dry-run, and Rich telemetry.

**Tech Stack:** Python 3.12, LeRobot 0.6.0 processors, NumPy, SciPy Rotation, pyAgxArm, Draccus, Rich, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-piper-isaac-teleop-design.md`

## Global Constraints

- Keep `IsaacXRController` robot-independent and unchanged.
- Public robot units are metres and radians; `ee.*` always means configured TCP in the Piper base frame.
- Never call `move_js()`; Cartesian commands use controller-planned `move_p()` only.
- Do not put raw `left.*`/`right.*` fields in `PiperRobot.action_features`.
- Joint and complete Cartesian command families are mutually exclusive; gripper may accompany either.
- All automated tests use fakes and must not access CAN or move hardware.
- `enable_motion` defaults to `False`; hardware validation is a separate, explicitly approved phase.
- Do not create commits unless the user explicitly requests them.

---

## File map

- Modify `src/lekit/robots/piper/piper_robot.py`: standard TCP features, feedback, configuration,
  Cartesian safety limiting, TCP-to-flange conversion, and `move_p()` dispatch.
- Create `src/lekit/robots/piper/teleop_processor.py`: LeRobot processor config, state machine,
  coordinate retargeting, trigger mapping, telemetry, and pipeline factory.
- Modify `src/lekit/robots/piper/__init__.py`: export the new processor API.
- Implement `src/lekit/scripts/teleop.py`: Piper/Isaac configuration, Rich status, dry-run and real loop.
- Modify `tests/robots/test_piper_robot.py`: fake TCP SDK behavior and Cartesian robot contract tests.
- Create `tests/robots/test_piper_teleop_processor.py`: pure processor/state-machine tests.
- Create `tests/scripts/test_piper_teleop.py`: loop ordering, dry-run, hold, and cleanup tests.
- Modify `src/lekit/robots/piper/README.md`: TCP offset, dry-run, startup command, control mapping, and
  physical safety procedure.

---

### Task 1: Piper TCP configuration, features, and feedback

**Files:**
- Modify: `src/lekit/robots/piper/piper_robot.py`
- Test: `tests/robots/test_piper_robot.py`

**Interfaces:**
- Produces: `PiperRobot._EEF_KEYS`, `PiperRobot._read_tcp_pose()`, TCP fields in action/observation
  features, and validated `PiperRobotConfig` Cartesian settings.
- Consumes: pinned pyAgxArm `set_tcp_offset()` and `get_tcp_pose()` APIs.

- [ ] **Step 1: Extend the fake SDK and write failing feature/configuration tests**

Add these fields to the existing fake arm initializer and add the two methods:

```python
self.tcp_pose = [0.35, -0.10, 0.30, 0.10, -0.20, 0.30]
self.tcp_offset = [0.0] * 6

def set_tcp_offset(self, pose: list[float]) -> None:
    self.tcp_offset = list(pose)
    self.events.append(("set_tcp_offset", list(pose)))

def get_tcp_pose(self):
    return SimpleNamespace(msg=list(self.tcp_pose), hz=100.0, timestamp=1.0)
```

Add tests asserting all six `ee.*` fields appear in both schemas, the configured offset is passed during
`connect()`, TCP feedback is returned in SI units, and malformed/non-finite TCP feedback raises
`PiperFeedbackError`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run --extra piper pytest tests/robots/test_piper_robot.py -k 'tcp or features' -v
```

Expected: failures because the configuration fields, features, and `_read_tcp_pose()` do not exist.

- [ ] **Step 3: Implement the minimal TCP schema and feedback path**

Add these public configuration fields and validation:

```python
tcp_offset: tuple[float, float, float, float, float, float] = (0.0,) * 6
eef_workspace_min_m: tuple[float, float, float] = (-0.65, -0.65, 0.02)
eef_workspace_max_m: tuple[float, float, float] = (0.65, 0.65, 0.75)
max_eef_target_lead_m: float | None = 0.005
max_eef_target_lead_rad: float | None = math.radians(2.0)
```

Add:

```python
_EEF_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")

def _read_tcp_pose(self) -> dict[str, float]:
    feedback = self.arm.get_tcp_pose()
    values = getattr(feedback, "msg", None)
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise PiperFeedbackError("Piper TCP pose feedback is unavailable or malformed.")
    pose = [float(value) for value in values]
    if any(not math.isfinite(value) for value in pose):
        raise PiperFeedbackError("Piper TCP pose feedback contains non-finite values.")
    return dict(zip(self._EEF_KEYS, pose, strict=True))
```

Call `arm.set_tcp_offset(list(config.tcp_offset))` from `configure()`, add TCP features, and include TCP
feedback in `get_observation()`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: selected tests pass with no hardware access.

- [ ] **Step 5: Run all existing Piper robot tests**

```bash
uv run --extra piper pytest tests/robots/test_piper_robot.py -v
```

Update existing expected schemas/complete returned actions only where the approved standard TCP contract
requires it; do not weaken joint/gripper assertions.

---

### Task 2: Absolute Cartesian `send_action()` with final robot-side safety

**Files:**
- Modify: `src/lekit/robots/piper/piper_robot.py`
- Test: `tests/robots/test_piper_robot.py`

**Interfaces:**
- Consumes: complete base-frame TCP actions under `PiperRobot._EEF_KEYS`.
- Produces: `_action_representation()`, `_send_eef_action()`, and complete applied robot action; invokes
  `arm.get_tcp2flange_pose()` then `arm.move_p()`.

- [ ] **Step 1: Write failing Cartesian dispatch and validation tests**

Extend `FakeArm`:

```python
def get_tcp2flange_pose(self, tcp_pose: list[float]) -> list[float]:
    self.events.append(("get_tcp2flange_pose", list(tcp_pose)))
    return [tcp_pose[0], tcp_pose[1], tcp_pose[2] - 0.10, *tcp_pose[3:]]

def move_p(self, flange_pose: list[float]) -> None:
    self.events.append(("move_p", list(flange_pose)))
```

Tests must cover these named behaviors with direct assertions on `FakeArm.events`:

- `test_complete_tcp_action_is_bounded_converted_and_sent`
- `test_joint_and_tcp_fields_cannot_be_mixed`
- `test_partial_tcp_action_is_rejected`
- `test_invalid_gripper_rejects_before_move_p`
- `test_tcp_translation_is_clamped_to_workspace`
- `test_tcp_target_lead_is_limited_from_measured_pose`
- `test_tcp_rotation_uses_shortest_so3_limit_across_euler_wrap`
- `test_tcp_action_rejects_unhealthy_arm_before_any_motion`

For example, the complete-action test sends all six `ee.*` values, asserts that
`get_tcp2flange_pose` precedes `move_p`, and compares the returned `ee.*` values to the TCP argument
passed into `get_tcp2flange_pose`. The mixed/partial/invalid tests assert that neither `move_p` nor the
gripper appears in the fake event log.

- [ ] **Step 2: Run the Cartesian tests and verify RED**

```bash
uv run --extra piper pytest tests/robots/test_piper_robot.py -k 'tcp_action or tcp_translation or tcp_target or tcp_rotation or mixed or partial_tcp' -v
```

Expected: failures because `send_action()` rejects `ee.*` and never calls `move_p()`.

- [ ] **Step 3: Implement representation selection and complete-pose validation**

Use this contract:

```python
def _action_representation(
    self, action: RobotAction, *, name: str = "action"
) -> Literal["joints", "eef"] | None:
    fields = set(action)
    has_joints = not fields.isdisjoint(self._JOINT_KEYS)
    has_eef = not fields.isdisjoint(self._EEF_KEYS)
    if has_joints and has_eef:
        raise ValueError(f"{name} cannot mix joint and TCP fields")
    if has_eef and not set(self._EEF_KEYS).issubset(fields):
        missing = sorted(set(self._EEF_KEYS) - fields)
        raise ValueError(f"{name} has an incomplete TCP pose; missing {missing}")
    return "joints" if has_joints else "eef" if has_eef else None
```

Validate every supplied value before any arm or gripper command.

Require `bool(self.arm.is_ok())` before Cartesian dispatch. If the SDK health check raises or returns
false, raise `PiperFeedbackError` and do not call `move_p()` or the gripper.

- [ ] **Step 4: Implement measured-pose translation/orientation limiting and SDK dispatch**

Use SciPy `Rotation` and this behavior:

```python
current_rotation = Rotation.from_euler("xyz", current_pose[3:])
target_rotation = Rotation.from_euler("xyz", requested_pose[3:])
delta = target_rotation * current_rotation.inv()
angle = delta.magnitude()
if max_angle is not None and angle > max_angle:
    delta = Rotation.from_rotvec(delta.as_rotvec() * (max_angle / angle))
safe_rotation = delta * current_rotation
safe_pose[3:] = safe_rotation.as_euler("xyz")

flange_target = self.arm.get_tcp2flange_pose(safe_pose)
self.arm.move_p(list(flange_target))
```

Clamp XYZ to configured workspace first, then clamp the translation vector from measured TCP to
`max_eef_target_lead_m`. Validate current joints against model limits before a Cartesian command.

- [ ] **Step 5: Verify focused and complete Piper tests GREEN**

```bash
uv run --extra piper pytest tests/robots/test_piper_robot.py -v
```

Expected: all Piper robot tests pass and fake event logs prove no `move_j()`/`move_p()` partial dispatch.

---

### Task 3: LeRobot Isaac-to-Piper processor

**Files:**
- Create: `src/lekit/robots/piper/teleop_processor.py`
- Modify: `src/lekit/robots/piper/__init__.py`
- Create: `tests/robots/test_piper_teleop_processor.py`

**Interfaces:**
- Produces: `PiperTeleopProcessorConfig`, `PiperTeleopState`,
  `PiperIsaacRetargetingStep`, and `make_piper_isaac_processor()`.
- Consumes: `(RobotAction, RobotObservation)` through
  `robot_action_observation_to_transition`; emits `RobotAction` through
  `transition_to_robot_action`.

- [ ] **Step 1: Write failing processor API, feature, and state tests**

Build selected-hand sample helpers:

```python
def xr_frame(*, engaged: bool, tracked: bool = True, translation=(0, 0, 0), rotation=(0, 0, 0, 1), trigger=0.0):
    return {
        "right.translation": np.asarray(translation, dtype=np.float32),
        "right.rotation": np.asarray(rotation, dtype=np.float32),
        "right.trigger": trigger,
        "right.is_tracking": tracked,
        "right.is_engaged": engaged,
    }
```

Write independent tests for:

- initial held squeeze stays `UNARMED` and emits `{}`;
- observing release arms `IDLE`;
- neutral engage latches measured `ee.*` with no jump;
- cumulative operator forward maps to Piper `+X`, right to `-Y`, up to `+Z`;
- quaternion delta is left-composed onto measured TCP orientation;
- release emits exactly one measured hold action, then `{}`;
- re-engage uses the new measured observation;
- tracking loss emits hold and requires release before rearm;
- malformed vectors/quaternions enter `FAULT` and emit one measured TCP hold action;
- trigger 0/1 maps to configured max/min gripper widths while engaged;
- `reset()` clears all state;
- `transform_features()` removes XR device fields and declares canonical `ee.*`/gripper fields.

- [ ] **Step 2: Run processor tests and verify RED**

```bash
uv run --extra piper pytest tests/robots/test_piper_teleop_processor.py -v
```

Expected: import failure because `teleop_processor.py` does not exist.

- [ ] **Step 3: Implement validated configuration and state enum**

Create:

```python
class PiperTeleopState(str, Enum):
    UNARMED = "unarmed"
    IDLE = "idle"
    ENGAGED = "engaged"
    FAULT = "fault"

@dataclass
class PiperTeleopProcessorConfig:
    hand: Literal["left", "right"] = "right"
    translation_scale: float = 1.0
    rotation_scale: float = 0.0
    operator_to_base_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    max_translation_from_anchor_m: float = 0.15
    max_rotation_from_anchor_rad: float = math.radians(30.0)
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.07
    neutral_translation_tolerance_m: float = 1e-4
    neutral_rotation_tolerance_rad: float = math.radians(0.5)
```

Validate finiteness, positive limits, increasing gripper range, and a proper orthonormal rotation matrix
with determinant `+1`.

- [ ] **Step 4: Implement the stateful registered processor step**

Use `RobotActionProcessorStep`, inspect `self.transition[TransitionKey.OBSERVATION]`, and copy input
actions rather than mutating caller data. The core target calculation is:

```python
delta_p_base = operator_to_base @ (translation_scale * delta_p_operator)
delta_p_base = clamp_norm(delta_p_base, max_translation_from_anchor_m)
target_p = anchor_p + delta_p_base

delta_r_operator = Rotation.from_quat(rotation_xyzw)
delta_rotvec = delta_r_operator.as_rotvec() * rotation_scale
delta_rotvec = clamp_norm(delta_rotvec, max_rotation_from_anchor_rad)
delta_r_operator = Rotation.from_rotvec(delta_rotvec)
delta_r_base = Rotation.from_matrix(operator_to_base @ delta_r_operator.as_matrix() @ operator_to_base.T)
target_r = delta_r_base * anchor_r
```

On release/tracking loss, return the six measured `ee.*` fields once. Invalid selected-hand data enters
`FAULT`, records `fault_reason`, and returns the same measured hold action instead of propagating a raw
numeric exception through an active control loop. While idle return `{}`. Preserve diagnostic properties
`state`, `last_target`, and `fault_reason`; implement `reset()`.

- [ ] **Step 5: Implement feature transformation and pipeline factory**

```python
def make_piper_isaac_processor(config: PiperTeleopProcessorConfig) -> RobotProcessorPipeline:
    return RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[PiperIsaacRetargetingStep(config=config)],
        name="piper_isaac_retargeting",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
```

Register the step under a Piper-specific stable name and export all four public symbols from
`src/lekit/robots/piper/__init__.py`.

- [ ] **Step 6: Run processor tests and verify GREEN**

```bash
uv run --extra piper pytest tests/robots/test_piper_teleop_processor.py -v
```

---

### Task 4: `teleop.py` control loop and Rich status

**Files:**
- Implement: `src/lekit/scripts/teleop.py`
- Create: `tests/scripts/test_piper_teleop.py`

**Interfaces:**
- Consumes: `PiperRobot`, `IsaacXRController`, and the processor factory from Task 3.
- Produces: `PiperIsaacTeleopConfig`, `run_teleop_loop()`, `teleoperate()`, and module CLI.

- [ ] **Step 1: Write failing loop behavior tests with fakes**

Define a finite loop using fake robot/teleop objects and implement these tests:

- `test_loop_orders_observation_before_teleop_and_processor` records the call sequence and asserts
  `get_observation`, `get_action`, processor, then `send_action`.
- `test_dry_run_never_calls_send_action` runs two frames with a non-empty processor result and asserts
  the fake robot action list stays empty.
- `test_motion_mode_sends_only_processor_output` asserts raw `right.*` fields never reach the robot.
- `test_release_hold_output_is_sent_once` feeds engaged/released/idle frames and asserts one absolute
  `ee.*` hold action on the release edge.
- `test_exception_attempts_measured_tcp_hold_and_disconnects` raises from the fake teleoperator and
  asserts the latest complete measured TCP pose is sent before both disconnect methods run.
- `test_loop_uses_configured_frame_limit_without_sleeping_after_last_frame` uses `max_frames=2` and
  asserts two reads and one pacing sleep.

Inject `sleep_fn`, `clock`, and `max_frames` so tests remain deterministic and do not import the XR
runtime or hardware SDK.

- [ ] **Step 2: Run script tests and verify RED**

```bash
uv run --extra piper pytest tests/scripts/test_piper_teleop.py -v
```

Expected: import/API failures because `teleop.py` is empty.

- [ ] **Step 3: Implement configuration and testable loop**

```python
@dataclass
class PiperIsaacTeleopConfig:
    robot: PiperRobotConfig = field(default_factory=PiperRobotConfig)
    teleop: IsaacTeleopConfig = field(default_factory=IsaacTeleopConfig)
    processor: PiperTeleopProcessorConfig = field(default_factory=PiperTeleopProcessorConfig)
    fps: int = 30
    enable_motion: bool = False
    max_frames: int | None = None
```

Implement:

```python
def run_teleop_loop(
    robot: PiperRobot,
    teleop: IsaacXRController,
    processor: RobotProcessorPipeline,
    *,
    fps: int,
    enable_motion: bool,
    max_frames: int | None = None,
    sleep_fn: Callable[[float], None] = precise_sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    period_s = 1.0 / fps
    frame_index = 0
    last_observation = None
    try:
        while max_frames is None or frame_index < max_frames:
            started_at = clock()
            observation = robot.get_observation()
            last_observation = observation
            isaac_action = teleop.get_action()
            piper_action = processor((isaac_action, observation))
            if enable_motion and piper_action:
                robot.send_action(piper_action)
            frame_index += 1
            if max_frames is None or frame_index < max_frames:
                sleep_fn(max(period_s - (clock() - started_at), 0.0))
    finally:
        if enable_motion and last_observation is not None:
            hold = {key: float(last_observation[key]) for key in PiperRobot._EEF_KEYS}
            robot.send_action(hold)
```

The loop reads observation first, then teleop action, then processor output. Empty output means no arm
command. Dry-run still processes and displays targets but never calls `send_action()`.

- [ ] **Step 4: Implement lifecycle and best-effort hold**

Connect the teleoperator before enabling/connecting the robot. When `enable_motion=False`, force
`robot.config.auto_enable=False`. `run_teleop_loop()` attempts one complete measured `ee.*` hold action
from its latest observation before returning or propagating an exception. Then disconnect teleop and
robot independently so one cleanup failure cannot skip the other. Use nested cleanup:

```python
try:
    teleop.disconnect()
finally:
    robot.disconnect()
```

- [ ] **Step 5: Add Rich live status and Draccus entry point**

Render a compact table with control state, selected hand, tracking, engage, loop Hz, measured TCP,
target TCP, dry-run/motion mode, and latest fault. Use `lerobot.configs.parser.wrap()` for the module CLI:

```python
@parser.wrap()
def teleoperate(cfg: PiperIsaacTeleopConfig) -> None:
    robot = PiperRobot(cfg.robot)
    teleop = IsaacXRController(cfg.teleop)
    processor = make_piper_isaac_processor(cfg.processor)
    if not cfg.enable_motion:
        robot.config.auto_enable = False
    try:
        teleop.connect()
        robot.connect()
        run_teleop_loop(
            robot,
            teleop,
            processor,
            fps=cfg.fps,
            enable_motion=cfg.enable_motion,
            max_frames=cfg.max_frames,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

if __name__ == "__main__":
    teleoperate()
```

- [ ] **Step 6: Run script tests and verify GREEN**

```bash
uv run --extra piper pytest tests/scripts/test_piper_teleop.py -v
```

---

### Task 5: Documentation and non-hardware verification

**Files:**
- Modify: `src/lekit/robots/piper/README.md`
- Verify: all files changed in Tasks 1-4

**Interfaces:**
- Documents the exact CLI and physical checkout sequence.
- Does not connect to the robot.

- [ ] **Step 1: Document dry-run and motion commands**

Include commands equivalent to:

```bash
# Observe and retarget only; robot motion remains disabled.
uv run --extra piper --extra teleop python -m lekit.scripts.teleop \
  --robot.channel=can0 \
  --robot.firmware_version=v188

# First translation-only physical checkout. Supply the measured TCP offset.
uv run --extra piper --extra teleop python -m lekit.scripts.teleop \
  --robot.channel=can0 \
  --robot.firmware_version=v188 \
  --robot.tcp_offset='[0,0,0,0,0,0]' \
  --robot.speed_percent=10 \
  --processor.rotation_scale=0 \
  --enable_motion=true
```

Explain the default right-hand controls, axis preset, trigger/gripper mapping, engage/re-engage behavior,
TCP offset measurement, configuration overrides, and physical emergency-stop requirement.

- [ ] **Step 2: Run format and lint checks**

```bash
uv run ruff format --check \
  src/lekit/robots/piper src/lekit/scripts/teleop.py \
  tests/robots/test_piper_robot.py tests/robots/test_piper_teleop_processor.py \
  tests/scripts/test_piper_teleop.py

uv run ruff check \
  src/lekit/robots/piper src/lekit/scripts/teleop.py \
  tests/robots/test_piper_robot.py tests/robots/test_piper_teleop_processor.py \
  tests/scripts/test_piper_teleop.py
```

- [ ] **Step 3: Run the complete relevant non-hardware suite**

```bash
uv run --extra piper pytest \
  tests/robots/test_piper_robot.py \
  tests/robots/test_piper_demo.py \
  tests/robots/test_piper_teleop_processor.py \
  tests/scripts/test_piper_teleop.py \
  tests/teleoperators/test_isaac_teleop_relative_pose.py \
  tests/teleoperators/test_isaac_teleop_session.py -v
```

Expected: zero failures, zero CAN access, and zero physical commands.

- [ ] **Step 4: Review the diff against the design spec**

Check every constraint and state transition in
`docs/superpowers/specs/2026-08-26-piper-isaac-teleop-design.md`. Search for accidental `move_js`, raw
Isaac fields in `PiperRobot`, partial TCP completion, and a motion-enabled default. Report any gap rather
than proceeding to hardware.

---

### Task 6: Separately approved physical checkout

**Files:**
- No source changes unless a hardware observation produces a failing regression test first.
- Evidence: terminal/Rich output and measured before/after poses.

**Interfaces:**
- Consumes the verified CLI from Task 5.
- Produces evidence about v188 `move_p()` overwrite, hold, axes, and 6DoF behavior.

- [ ] **Step 1: Request approval for read-only dry-run connection**

State the CAN channel, firmware selection, auto-enable state, and that no `send_action()` call will be
made. Run only after approval.

- [ ] **Step 2: Verify live TCP and XR telemetry with motion disabled**

Confirm stable TCP feedback, correct selected-hand tracking, neutral engage frames, and target direction
display. Stop on stale/malformed feedback or unexpected axis mapping.

- [ ] **Step 3: Request approval for a translation-only 10 mm checkout**

List the exact axis, direction, maximum controller displacement, `speed_percent=10`,
`rotation_scale=0`, `max_eef_target_lead_m=0.005`, and emergency-stop arrangement.

- [ ] **Step 4: Verify engage, each translation axis, release hold, tracking loss, and re-engage**

Exercise one axis per approval. Compare commanded and measured TCP changes. Specifically verify whether
a release hold supersedes an in-flight `move_p()` target on S-V1.8-8; do not describe release as a
deadman stop without this evidence.

- [ ] **Step 5: Request approval for 2-degree single-axis rotation checkout**

Set nonzero rotation scale only after translation tests pass. Test roll, pitch, and yaw independently,
then a combined small 6DoF motion.

- [ ] **Step 6: Request approval for trigger/gripper checkout**

Keep the arm stationary, verify current gripper calibration, then test one bounded open/close cycle.

- [ ] **Step 7: Record findings and add regression tests for any discrepancy**

If firmware behavior differs from the contract, first add a failing fake/integration test reproducing the
observed behavior, then return to the RED-GREEN cycle before another physical run.

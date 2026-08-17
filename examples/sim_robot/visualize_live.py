"""Live visualisation of a scripted-policy rollout through SimRobot.

This is the SimRobot counterpart to ``lerobot.record --display_data=true``:
it feeds a smooth per-step sine trajectory through :class:`SimRobot` and
streams every state/action/image update to three optional backends:

* a MuJoCo 3D viewer (``mujoco.viewer.launch_passive``) so you can *see* the
  simulated arm move in real time;
* one OpenCV window per configured camera so you can watch what the
  simulator's cameras are seeing;
* the standard LeRobot visualisation backends (rerun / foxglove), exposed
  the same way as in ``lerobot.record --display_mode=rerun``.

Scene sources
-------------

The driver can load one of three scenes, in priority order:

* ``--scene <path>`` loads an explicit MJCF/XML. Joint/actuator/camera names
  are introspected from the model so the example works with any standard
  6-DOF / 7-DOF arm (e.g. ``assets/SO101/scene.xml``).
* ``--scene-name so101`` loads ``assets/SO101/scene.xml`` (resolved relative
  to the lekit repo root), uses the canonical SO-100/101 motor layout and
  enables ``enable_position_servos=True`` so the URDF-imported ``<motor>``
  actuators track position targets rather than acting as open-loop force
  drivers.
* ``--scene-name demo`` (default) loads the bundled ``scene.xml`` next to
  this script (the small SO-101-like 3-link arm with two cameras).

Usage::

    # Desktop (all three viewers):
    MUJOCO_GL=glfw python examples/sim_robot/visualize_live.py \\
        --display_mujoco_viewer --display_camera_windows \\
        --display_lerobot_backend rerun --steps 120

    # Specific scene (e.g. the bundled SO-101 asset):
    python examples/sim_robot/visualize_live.py \\
        --scene /home/sorel/workspace/lekit/assets/SO101/scene.xml \\
        --display_mujoco_viewer --display_camera_windows --steps 9999
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

# Make ``from lekit.robots.sim_robot import …`` resolve when launched with PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
_TMP_HOME = Path(tempfile.gettempdir()) / "sim_robot_hf"
_TMP_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_TMP_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(_TMP_HOME / "datasets"))


from lekit.robots.sim_robot import (  # noqa: E402
    SimCameraConfig,
    SimMotorSpec,
    SimRobot,
    SimRobotConfig,
    SimRobotVisualizer,
)


# ── Scene introspection ────────────────────────────────────────────────────


def _resolve_xml(xml_path: Path, _seen: set | None = None) -> str:
    """Read an XML file with ``<include file="...">`` directives inlined.

    MuJoCo scenes are commonly a tiny wrapper that defers to a "main" XML
    via ``<include>``; we need actuator-type counts to include the included
    files, so this helper concatenates them depth-first into a single string.
    """
    if _seen is None:
        _seen = set()
    real = xml_path.resolve()
    if real in _seen:
        return ""
    _seen.add(real)
    text = xml_path.read_text()
    import re

    chunks = [text]
    for match in re.finditer(r'<include\s+file="([^"]+)"\s*/>', text):
        chunks.append(_resolve_xml(xml_path.parent / match.group(1), _seen))
    return "\n".join(chunks)


def _introspect_scene(xml_path: Path) -> dict:
    """Load the MJCF and return auto-derived motor/actuator/camera specs.

    The result is suitable for unpacking into
    ``SimRobotConfig(motors=..., cameras=..., enable_position_servos=True)``.

    Joint ↔ actuator pairing: a joint ``x`` is paired with the actuator named
    ``x`` if one exists (the SO-100/101 layout) or with the i-th actuator in
    declaration order otherwise.
    """
    import mujoco

    # MuJoCo resolves ``<include file="...">`` paths relative to the model
    # file's directory. Chdir there so include resolution works, then restore
    # the original cwd afterwards.
    original_cwd = os.getcwd()
    m = mujoco.MjModel.from_xml_path(str(xml_path.parent / xml_path.name))
    os.chdir(original_cwd)

    joints: list[tuple[str, int, list[float]]] = []
    hinge_id = int(mujoco.mjtJoint.mjJNT_HINGE)
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name or name in ("root", "world"):
            continue
        # Only 1-DoF hinge joints for now -- SimRobot addresses each motor as
        # ``data.qpos[jnt_qposadr[jid]]`` and that layout is only meaningful
        # for hinges.
        if int(m.jnt_type[i]) != hinge_id:
            continue
        rng = m.jnt_range[i].tolist() if bool(m.jnt_limited[i]) else [-3.14, 3.14]
        joints.append((name, int(m.jnt_qposadr[i]), rng))

    actuators_by_name: dict[str, int] = {}
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuators_by_name[name] = i
    actuators_by_index: list[int] = list(range(m.nu))

    # Pair: if a joint's name matches an actuator, use that pairing. Otherwise
    # fall back to positional pairing.
    motors: dict[str, SimMotorSpec] = {}
    joint_idx = 0
    actuator_idx = 0
    for name, _qadr, _rng in joints:
        if name in actuators_by_name:
            motors[name] = SimMotorSpec(joint=name, actuator=name)
            joint_idx += 1
            actuator_idx += 1
        elif actuator_idx < len(actuators_by_index):
            # Skip name-matching already, fall back to next actuator by index.
            mot_actuator = next(
                (n for n, idx in actuators_by_name.items() if idx == actuators_by_index[actuator_idx]),
                None,
            )
            motors[name] = SimMotorSpec(joint=name, actuator=mot_actuator)
            joint_idx += 1
            actuator_idx += 1
        else:
            # Joint without an actuator -- kinematic control via qpos.
            motors[name] = SimMotorSpec(joint=name)

    cameras: dict[str, SimCameraConfig] = {}
    for i in range(m.ncam):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            cameras.setdefault(name, SimCameraConfig(name=name, width=128, height=96, fps=30))

    # Count how many of the declared actuators are generic ``<motor>`` (the
    # URDF-imported ``<motor>`` style) vs. anything else (``<position>``,
    # ``<velocity>``, ...). If we have at least one ``<motor>``, let
    # ``SimRobot.connect`` decide on a per-motor basis in ``"auto"`` mode:
    # it rewrites the gainprm/biasprm on bare ``<motor>``s and leaves
    # PD-already-encoded actuators alone.
    resolved = _resolve_xml(xml_path)
    plain_motor_count = resolved.count("<motor ") + resolved.count("<motor>")
    position_servo_count = resolved.count("<position ") + resolved.count("<position>")
    has_plain_motors = plain_motor_count > 0 and position_servo_count == 0

    return {
        "motors": motors,
        "cameras": cameras,
        "default_joint_positions": {n: 0.0 for n in motors},
        # ``"auto"`` lets ``SimRobot.connect`` apply the gainprm rewrite to
        # URDF-imported ``<motor>`` actuators only; pass ``"off"`` if you
        # know your model already encodes PD control on every actuator.
        "enable_position_servos": "auto" if has_plain_motors else "off",
    }


# ── Scene presets ──────────────────────────────────────────────────────────


def _resolve_scene(args: argparse.Namespace) -> Path:
    if args.scene:
        return Path(args.scene).resolve()
    if args.scene_name == "so101":
        repo_root = Path(__file__).resolve().parents[2]
        return (repo_root / "assets" / "SO101" / "scene.xml").resolve()
    return (Path(__file__).parent / "scene.xml").resolve()


def build_config(args: argparse.Namespace) -> SimRobotConfig:
    scene_path = _resolve_scene(args)
    if not scene_path.is_file():
        raise SystemExit(f"Scene XML not found: {scene_path}")

    scene_info = _introspect_scene(scene_path)
    # Camera windows in the SO-101 asset (and many other real-world scenes)
    # have no `<camera>` elements; if the introspection didn't find any, skip
    # the cv2 window channel so the user doesn't see a silent no-op.
    has_cameras = bool(scene_info["cameras"])
    if not has_cameras and args.display_camera_windows:
        print(
            "[viz] scene has no <camera> elements; disabling --display_camera_windows.",
            file=sys.stderr,
            flush=True,
        )
        args.display_camera_windows = False

    return SimRobotConfig(
        id="visualize_live_example",
        xml_path=str(scene_path),
        motors=scene_info["motors"],
        cameras=scene_info["cameras"],
        default_joint_positions=scene_info["default_joint_positions"],
        max_relative_target=0.4,
        n_substeps=4,
        enable_position_servos=scene_info["enable_position_servos"],  # auto/off/all
        position_kp=40.0,
        position_kv=12.0,
        # Visualization flags:
        display_mujoco_viewer=args.display_mujoco_viewer,
        display_camera_windows=args.display_camera_windows,
        display_lerobot_backend=args.display_lerobot_backend,
        display_lerobot_ip=args.display_lerobot_ip,
        display_lerobot_port=args.display_lerobot_port,
        display_lerobot_session_name=args.display_lerobot_session_name,
        display_lerobot_compress_images=args.display_lerobot_compress_images,
        calibration_dir=args.dataset_root / "calibration",
    )


# ── Scripted policy ────────────────────────────────────────────────────────


def scripted_action(t: float, motor_names: list[str]) -> dict[str, float]:
    """Smooth sine targets scaled by motor index for obvious 3D motion."""
    return {name: 0.25 * math.sin(0.5 * t + 0.4 * i) for i, name in enumerate(motor_names)}


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=None, help="Path to an MJCF/XML scene. Overrides --scene-name.")
    parser.add_argument(
        "--scene-name",
        choices=("demo", "so101"),
        default="demo",
        help="Bundled scene to load when --scene is unset. Default 'demo' = scene.xml next to this script.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/sim_robot_dataset"))
    parser.add_argument(
        "--steps", type=int, default=200, help="Loop length. Ignored if a UI is open and the user closes it first."
    )

    # ── Display flags (mirror ``lerobot.record --display_data=...``) ───────
    parser.add_argument(
        "--display_mujoco_viewer",
        action="store_true",
        help="Open the MuJoCo 3D passive viewer (needs a display, glfw).",
    )
    parser.add_argument(
        "--display_camera_windows", action="store_true", help="Open one cv2 window per configured camera."
    )
    parser.add_argument(
        "--display_lerobot_backend",
        default="off",
        choices=("off", "rerun", "foxglove"),
        help="LeRobot visualisation backend (rerun / foxglove).",
    )
    parser.add_argument(
        "--display_lerobot_ip",
        default=None,
        help="For rerun: IP of the remote Rerun server. For foxglove: bind interface.",
    )
    parser.add_argument(
        "--display_lerobot_port",
        type=int,
        default=None,
        help="For rerun: port of the remote Rerun server. For foxglove: WebSocket port.",
    )
    parser.add_argument("--display_lerobot_session_name", default="sim_robot_visualize")
    parser.add_argument("--display_lerobot_compress_images", action="store_true")
    args = parser.parse_args()

    args.dataset_root.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    print(
        f"[viz] motors={list(cfg.motors)} cameras={list(cfg.cameras)} "
        f"enable_position_servos={cfg.enable_position_servos}",
        file=sys.stderr,
        flush=True,
    )
    robot = SimRobot(cfg)
    robot.connect()
    print(
        f"[viz] connected. mujoco_gui={cfg.display_mujoco_viewer} "
        f"cv2_cams={cfg.display_camera_windows} backend={cfg.display_lerobot_backend}",
        file=sys.stderr,
        flush=True,
    )
    try:
        motor_names = list(cfg.motors)
        with SimRobotVisualizer(robot) as viz:
            for step in range(args.steps):
                if not viz.is_running:
                    print("[viz] viewer closed -- stopping loop early.", file=sys.stderr, flush=True)
                    break
                action = scripted_action(step / 30.0, motor_names)
                sent = robot.send_action(action)
                obs = robot.get_observation()
                viz.render(observation=obs, action=sent)
    finally:
        robot.disconnect()
    print("[viz] done.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

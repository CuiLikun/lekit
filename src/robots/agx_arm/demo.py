# doc： https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/piper/piper_api.md#piper-%E6%9C%BA%E6%A2%B0%E8%87%82-api-%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3
import time
from platform import system

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

# Nero firmware: <= 1.10 → NeroFW.DEFAULT; 1.11 → NeroFW.V111; >= 1.12 → NeroFW.V112.
platform_system = system()
if platform_system == "Windows":
    interface = "agx_cando"
    channel = "0"
elif platform_system == "Linux":
    interface = "socketcan"
    channel = "can0"
elif platform_system == "Darwin":
    interface = "slcan"
    channel = "/dev/ttyACM0"
else:
    raise RuntimeError(
        "pyAgxArm currently documents Linux `socketcan`, Windows `agx_cando`, and macOS `slcan`."
    )

cfg = create_agx_arm_config(
    robot=ArmModel.PIPER_X,
    firmeware_version=PiperFW.V188,
    interface=interface,
    channel=channel,
)
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)


def wait_until_enabled(max_wait_s: float = 5.0) -> bool:
    """Try enabling all joints until success or timeout."""
    start_t = time.monotonic()
    while time.monotonic() - start_t < max_wait_s:
        if robot.enable():
            return True
        time.sleep(0.05)
    return False


def wait_gripper_status(max_wait_s: float = 2.0):
    """Wait for one fresh gripper status message."""
    start_t = time.monotonic()
    while time.monotonic() - start_t < max_wait_s:
        gs = end_effector.get_gripper_status()
        if gs is not None:
            return gs
        time.sleep(0.05)
    return None


def move_gripper_and_print(value: float, force: float) -> None:
    before = wait_gripper_status()
    end_effector.move_gripper_m(value=value, force=force)
    time.sleep(0.8)
    after = wait_gripper_status()
    ctrl = end_effector.get_gripper_ctrl_states()
    print(f"gripper_cmd value={value}, force={force}")
    print(f"before={before.msg if before else None}")
    print(f"after={after.msg if after else None}")
    print(f"ctrl_feedback={ctrl.msg if ctrl else None}")


def ensure_gripper_ready() -> bool:
    # Follower/standby mode avoids teaching-mode interference for tool commands.
    robot.set_follower_mode()
    time.sleep(0.2)

    gs = wait_gripper_status()
    print(f"gripper_status_before_prepare={gs.msg if gs else None}")
    if gs is None:
        return False

    # If homing is false, gripper commands may be accepted but have no physical effect.
    if not getattr(gs.msg.foc_status, "homing_status", False):
        print("gripper homing is false, trying reset+calibrate")
        _ = end_effector.reset_gripper()  # Return value is not reliable in current SDK.
        end_effector.disable_gripper()
        time.sleep(0.3)
        input("Please move gripper to ZERO position manually, then press Enter...")
        calib_ok = end_effector.calibrate_gripper(timeout=3.0)
        print(f"calibrate_gripper={calib_ok}")
        time.sleep(0.8)

    gs2 = wait_gripper_status(max_wait_s=3.0)
    print(f"gripper_status_after_prepare={gs2.msg if gs2 else None}")
    if gs2 is None:
        return False
    return bool(getattr(gs2.msg.foc_status, "homing_status", False))


robot.connect()
print(f"comm_error={robot.has_comm_error()}, comm_error_detail={robot.get_comm_error()}")

# A common reason for "command has no effect" is not enabled or in joint-error state.
clear_ok = robot.clear_joint_error()
print(f"clear_joint_error={clear_ok}")
enable_ok = wait_until_enabled()
print(f"enable_ok={enable_ok}, all_joint_enabled={robot.get_joint_enable_status(255)}")

# move eef
print(f"{robot.get_firmware()=}")
print(f"{robot.get_arm_status().msg=}")
print(f"{robot.get_fps()=}")
print(f"{robot.joint_nums=}")

print(f"{end_effector.is_ok()=}")
first_gs = wait_gripper_status()
print(f"first_gripper_status={first_gs.msg if first_gs else None}")
gripper_ready = ensure_gripper_ready()
print(f"gripper_ready={gripper_ready}")
if not gripper_ready:
    raise RuntimeError(
        "Gripper homing_status is still False after calibration. "
        "Control commands are likely ignored by gripper firmware/hardware."
    )

move_gripper_and_print(value=0.05, force=10.0)
# end_effector.move_gripper_m(value=0.0, force=1.0)

# a = robot.clear_joint_error()
# print(a)

# print(robot.enable())
# 2. 获取关节角度
# while True:
#     ja = robot.get_joint_angles()
#     if ja is not None:
#         print(ja.msg)
#         print(ja.hz, ja.timestamp)
#     time.sleep(0.005)

# 3. 获取法兰盘位姿
# while True:
#     fp = robot.get_flange_pose()
#     if fp is not None:
#         print(fp.msg)
#         print(fp.hz, fp.timestamp)
#     time.sleep(0.005)

# 4. 设置运行速度, 适用于 move_j / move_p / move_l / move_c。
# robot.set_speed_percent(100)

# 5. 设置 TCP 偏移，
# robot.set_tcp_offset([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])

# 6. 获取 TCP 位姿
# while True:
#     tcp = robot.get_tcp_pose()
#     if tcp is not None:
#         print(tcp.msg)
#         print(tcp.hz, tcp.timestamp)
#     time.sleep(0.02)

# 7. 读取 IK 关节角，仅在调用 move_p() 之后可用。
# while True:
#     ika = robot.get_ik_joint_angles()
#     if ika is not None:
#         print(ika.msg)
#         print(ika.hz, ika.timestamp)
#     time.sleep(0.005)

## 8. 关节运动
# robot.set_speed_percent(5)  # 以 100% 速度运行

# # tensor([-0.4161,  0.6676, -0.9647,  1.2113,  0.0496, -0.0581,  0.0508,  0.6713,
# #          0.7686, -1.1062,  1.1863,  0.0268,  0.2221,  0.0472],
# robot.move_j(
#     [
#         -0.03508111796508603,
#         0.443523069516799,
#         -0.40029126394489944,
#         0.508798383541387,
#         0.1866455102082736,
#         0.14529866022852791,
#     ]
# )

# # robot.move_j([0.6713,0.7686, -1.1062,  1.1863,  0.0268,  0.2221])


## 9. 点到点运动
# robot.move_p( [0.36010385439565823, 0.12566657121382174, 0.15999378166219616, -3.132167875629024, -8.726646259971648e-05, -1.5659268581818324])


## 10. 直线运动
# robot.move_l([0.1, 0.0, 0.3, 0.0, 1.570796326794896619, 0.0])

# command_joints = [0.023474678439323732, -0.011658799403322121, -0.04443608275577564, 0.11321950857687216, 0.04621631859280985, 0.03354522822333101]
# # ## 11. MIT 控制
# for i in range(1, robot.joint_nums + 1):
#     robot.move_mit(
#         joint_index=i,
#         p_des=command_joints[i-1],
#         v_des=0.0,
#         kp=12.0,
#         kd=0.8,
#         t_ff=0.0,
#     )

# time.sleep(5)

# 12.获取夹爪的状态
gs = wait_gripper_status()
if gs is not None:
    print("value=", gs.msg.value, "mode=", gs.msg.mode, "force(N)=", gs.msg.force)
    print("hz=", gs.hz, "timestamp=", gs.timestamp)
else:
    print("gripper status timeout")

# 13. 控制夹爪
# 张开到 5cm，力 1N
move_gripper_and_print(value=0.05, force=10.0)

# 闭合（行程 0）
move_gripper_and_print(value=0.0, force=10.0)
gs = wait_gripper_status()
if gs is not None:
    print("value=", gs.msg.value, "mode=", gs.msg.mode, "force(N)=", gs.msg.force)
    print("hz=", gs.hz, "timestamp=", gs.timestamp)
else:
    print("gripper status timeout")

# # 14. 设置遥操时夹爪的柔软程度
# end_effector.set_gripper_teaching_pendant_param(
#     teaching_range_per=100,
#     max_range_config=0.07,
#     # teaching_friction=7,
#     teaching_friction=1,
# )

# # 15. 设置运行速度，适用于 move_j / move_p / move_l / move_c
# robot.set_speed_percent(100) # 以 100% 速度运行


## 16. 设置末端负载, 可选值：'empty'（空载）/ 'half'（半载）/ 'full'（满载），默认：'empty'
# robot.set_payload(robot.OPTIONS.PAYLOAD.EMPTY)

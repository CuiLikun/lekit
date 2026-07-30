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

robot.connect()

print("robotic arm fps =", robot.get_fps(), "Hz")
# 1. 获取 firmware 的版本
fw = robot.get_firmware()
if fw is not None:
    print(fw)

# a = robot.clear_joint_error()
# print(a)

# print(robot.enable())
# 2. 获取关节角度
while True:
    ja = robot.get_joint_angles()
    if ja is not None:
        print(ja.msg)
        print(ja.hz, ja.timestamp)
    time.sleep(0.005)

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

# tensor([-0.4161,  0.6676, -0.9647,  1.2113,  0.0496, -0.0581,  0.0508,  0.6713,
#          0.7686, -1.1062,  1.1863,  0.0268,  0.2221,  0.0472],
# robot.move_j([-0.03508111796508603, 0.443523069516799, -0.40029126394489944, 0.508798383541387, 0.1866455102082736, 0.14529866022852791])

# robot.move_j([0.6713,0.7686, -1.1062,  1.1863,  0.0268,  0.2221])


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
# while True:
#     gs = end_effector.get_gripper_status()
#     if gs is not None:
#         print("value=", gs.msg.value, "mode=", gs.msg.mode, "force(N)=", gs.msg.force)
#         print("hz=", gs.hz, "timestamp=", gs.timestamp)
#         break
#     time.sleep(0.05)

# 13. 控制夹爪
# 张开到 5cm，力 1N
# end_effector.move_gripper_m(value=0.05, force=1.0)
# time.sleep(1.0)

# 闭合（行程 0）
# end_effector.move_gripper_m(value=0.0, force=1.0)

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

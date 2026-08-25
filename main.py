import time
from platform import system

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

# Nero 固件：≤ 1.10 选 NeroFW.DEFAULT；1.11 选 NeroFW.V111；1.12 选 NeroFW.V112；≥ 1.20 选 NeroFW.V120。
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
    raise RuntimeError("pyAgxArm 当前公开说明包含 Linux `socketcan`、Windows `agx_cando` 与 macOS `slcan`。")

cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,
    interface=interface,
    channel=channel,
)
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

while True:
    ja = robot.get_joint_angles()
    if ja is not None:
        print(ja.msg)
        print(ja.hz, ja.timestamp)
    time.sleep(0.005)

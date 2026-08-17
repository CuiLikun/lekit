import random
import time

import jkrc

IO_CABINET = 0  # 控制柜面板IO
IO_TOOL = 1  # 工具IO
IO_EXTEND = 2  # 扩展IO
robot = jkrc.RC("192.168.1.31")  # jaka 机器人的 IP 地址
robot.login()

for i in range(10):
    t0 = time.time()
    value = random.randint(0, 1000)
    ret = robot.set_analog_output(
        iotype=IO_EXTEND, index=3, value=value
    )  # 设置夹爪宽度随机为 0 - 1000，0 代表完全闭合夹爪，1000 代表完全打开夹爪
    print(f"第 {i + 1} 次:", ret)
    print(f"设置夹爪宽度耗时: {(time.time() - t0) * 1000:.3f} 毫秒")
    time.sleep(1)
robot.logout()  # 登出

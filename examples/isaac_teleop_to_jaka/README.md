# Isaac Teleop to JAKA

这个示例使用 XR 手柄遥操 JAKA 机械臂，并记录 LeRobot 数据集。主机不执行逆运动学，
而是把绝对 TCP 位姿目标持续发送给 JAKA SDK 的 Cartesian Servo Move (`servo_p`)。

## 坐标系

本示例使用以下约定：

- OpenXR anchor：`+X` 向右，`+Y` 向上，`-Z` 向前。
- JAKA 基坐标：`+X` 向右，`+Y` 向前，`+Z` 向上。
- JAKA SDK 使用毫米和弧度；LeRobot 接口使用米和弧度。
- 遥操目标位于机器人基坐标，因此必须使用 `user_frame_id=0`。

期望的直觉映射是：

| 操作者动作 | JAKA TCP |
| --- | --- |
| 向身体右侧移动 | `+X` |
| 向身体前方移动 | `+Y` |
| 向上移动 | `+Z` |

## 方向错乱的根因

手柄位置由 OpenXR anchor 表示，而不是由操作者身体坐标表示。OpenXR anchor 在一次会话中
基本固定；操作者转身后，身体的“向前”和“向右”会在 anchor 坐标中旋转。如果始终使用一个
固定的 `base_T_anchor`，映射只会在某个特定操作者朝向下符合直觉，转身后就会出现换轴、反向
或两个水平轴同时移动。

之前曾从一次采集拟合出固定 `operator_yaw_deg=-10.0364`。它把当时的水平串轴从约
`92%/100%` 降到约 `2.1%`，但操作者改变朝向后仍然失效。这说明固定 yaw 只能校准一个
朝向，不能解决“方向相对操作者”的问题。

## 头显相对映射

默认配置 `use_head_yaw=true` 会在每次 squeeze clutch 接合时读取头显朝向，并锁存其水平
yaw。OpenXR 中头显局部前方是 `[0, 0, -1]`。设头显四元数对应的旋转矩阵为
`anchor_R_head`，首先计算其水平前向：

```text
forward_anchor = normalize_horizontal(anchor_R_head @ [0, 0, -1])
heading = atan2(forward_anchor.x, -forward_anchor.z) + operator_yaw_offset
```

然后构造 OpenXR anchor 到 JAKA base 的旋转：

```text
base_R_anchor(heading) =
    [ cos(heading), 0,  sin(heading) ]   # operator right   -> JAKA +X
    [ sin(heading), 0, -cos(heading) ]   # operator forward -> JAKA +Y
    [            0, 1,             0 ]   # OpenXR up        -> JAKA +Z
```

因此，无论操作者面向哪个水平方向，只要在按下 squeeze 时面向自己认为的“前方”，身体
右移和前移都会分别映射到 JAKA `+X` 和 `+Y`。

### 为什么只在 clutch 接合时锁存

不能在每一帧都让映射跟随头显，否则手保持不动时仅仅转头就会改变坐标变换，造成 TCP
漂移。当前控制时序是：

```text
squeeze 松开
    -> TCP 保持上一目标
    -> 允许操作者转身或重新站位

squeeze 按下边沿
    -> 读取头显 yaw
    -> 锁存 base_R_anchor
    -> 同时锁存手柄原点和实测 TCP home

squeeze 持续按住
    -> 固定使用已锁存的 base_R_anchor
    -> 转头不会改变映射

squeeze 再次松开
    -> 结束本次映射；下次按下重新读取操作者朝向
```

如果 clutch 接合时头显跟踪无效，控制不会退回不确定的固定方向，而是保持 TCP 不动并等待
有效头显数据。

## Clutch 位移原理

手柄位姿先转换到 JAKA base。接合瞬间记录转换后的手柄位置 `p_grip_origin` 和实测 TCP
位置 `p_tcp_home`，后续目标为：

```text
p_tcp_target = p_tcp_home + (p_grip_current - p_grip_origin)
```

所以接合第一帧的位移恒为零，不会因为操作者位置或 OpenXR 世界原点较远而跳变。释放后
`HoldLatch` 保持最后一个命令目标，不会用滞后的反馈把机械臂拉回去。

手柄方向四元数使用同一锁存旋转变换到 JAKA base，再由 clutch 计算相对旋转。使用
`--teleop.lock_pose=true` 时，接合瞬间的实测 TCP roll/pitch/yaw 会被锁定，只跟随平移。

## 静态兼容模式

固定工位或没有头显姿态时可以关闭头显相对映射：

```bash
--teleop.use_head_yaw=false \
--teleop.operator_yaw_deg=-10.0364
```

静态模式只对标定时的操作者朝向成立。也可以用 `--teleop.base_T_anchor` 完整覆盖 4x4
变换。默认头显相对模式下，`operator_yaw_deg` 表示“头显视线前方”到“操作者期望前方”的
固定微调，而不是 OpenXR anchor 到机器人的绝对工位角。

## 安装与运行

从仓库根目录安装：

```bash
uv pip install -e .
uv pip install "isaacteleop[cloudxr,retargeters-lite]~=1.3.131" "scipy>=1.14"
python -m isaacteleop.cloudxr --accept-eula
```

运行示例：

```bash
python -m examples.isaac_teleop_to_jaka.record \
    --robot.type=jaka_robot \
    --robot.ip=192.168.1.31 \
    --robot.id=jaka_arm \
    --robot.user_frame_id=0 \
    --teleop.type=xr_controller \
    --teleop.use_head_yaw=true \
    --teleop.operator_yaw_deg=0 \
    --teleop.lock_pose=true \
    --dataset.repo_id="sorel/pick-cube" \
    --dataset.single_task="Pick up the object" \
    --dataset.fps=30 \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=false
```

`record.sh` 包含同样的头显相对配置。

## 响应速度

XR 默认线速度、加速度和 jerk 分别为 `0.15 m/s`、`0.8 m/s^2`、`8 m/s^3`。JAKA
驱动以约 8 ms 周期插值并发送 Servo P。此前低速验证使用的 `0.02 m/s` 会使 15 cm 动作
需要约 7.5 秒，它只适合方向确认，不代表正常遥操延迟。

不要先关闭 `cartesian_nlf` 来解决迟滞。该滤波器用于抑制固定目标附近的 Servo P 抖动；
应先通过以下参数调整速度轮廓：

```text
teleop.servo_linear_velocity_m_s
teleop.servo_linear_acceleration_m_s2
teleop.servo_linear_jerk_m_s3
```

## 控制 Trace

添加以下参数可记录同步诊断数据：

```bash
--control_trace_csv=artifacts/jaka_control_trace.csv
```

CSV 包含原始/转换后手柄位置、头显四元数、头显跟踪状态、clutch 锁存 yaw、
requested/applied/actual TCP、内部 Servo target/commanded position、循环时间、发送频率和
queue depth。方向问题应先比较原始手柄、锁存 yaw 与 requested TCP；速度问题应比较
requested、applied、commanded 和 actual，避免把输入限幅误判为机器人延迟。

## 安全

运行前确认 tool frame、user frame、负载、碰撞等级、工作空间和急停。首次验证新的坐标映射
时使用小位移和低速度。任何异常方向都应立即松开 squeeze 并停止程序。

实现依据 JAKA Python SDK 1.7.2：
https://www.jaka.com/docs/guide/1.7.2/SDK/Python.html

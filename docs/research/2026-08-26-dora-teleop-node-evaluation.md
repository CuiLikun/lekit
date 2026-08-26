# Dora 作为独立 `teleop-node` 分布式设计参考的评估

## 结论

Dora 很适合作为 `teleop-node` 的**架构参考**，尤其值得借鉴以下设计：硬件由单一独立节点持有、数据与状态分通道发布、输入使用有界队列、消息带会话和序列信息、订阅端显式处理超时与上游重启、节点由守护进程监管，并且本机与跨机部署使用同一份数据流描述。[1][2][3]

现阶段不建议把 Dora 直接设为 Lekit `isaac-teleop` 的唯一或必选运行时。当前需求的核心是“一个长期运行的 Quest 3 输入服务，任意控制链路可按需订阅或退出”，而 Dora 的订阅关系通常属于一个已启动的 dataflow；Coordinator、每机 Daemon、YAML 图和 Arrow 数据协议也明显重于仓库已经具备的 ZeroMQ。Dora 的运行期节点增删正在快速完善，但发布版、`main` 和部分官方文档之间仍有可观察到的语义差异。[4][5][6]

推荐分两步走：

1. 第一阶段保留一个很薄的 transport 接口，用 ZeroMQ 实现独立常驻的 `teleop-node` 和标准 LeRobot subscriber；协议和故障语义按本文借鉴 Dora 的方式设计。
2. 当项目确认需要跨机编排、统一监控、节点自动恢复、记录回放或更多机器人节点时，再增加实验性的 `DoraTransport`，通过故障注入验证后决定是否升级为主要后端。

本文在 2026-08-26 以 Dora 官方仓库 `main` 的提交 [`6715a798`](https://github.com/dora-rs/dora/commit/6715a798adc0551215c423b253165cfdb28c6de9) 为源码基线，同时区分已发布的稳定版 `v0.5.0` 和预发布版 `v1.0.0-rc.4`。[5]

## 当前 Lekit 控制链路的约束

当前 `IsaacXRController.get_action()` 每次只推进一次 XR step，再从同一结果中生成左右手输入，因此一次返回的 28 个字段是同一采样步的完整动作快照。每只手包含 Grip/Aim 相对位姿、模拟量、摇杆、按键和三个状态位；总计为 44 个 `float32` 标量和 6 个布尔标量。相对位姿和 squeeze clutch 属于硬件输入边界，应继续留在唯一持有 CloudXR/OpenXR 会话的进程中。[18]

`src/lekit/scripts/teleop.py` 当前以 30 Hz 调用 `teleop.get_action()`，取得一个最新动作后立即送入 Piper processor；控制循环没有历史回放需求。Piper processor 在 tracking 丢失、clutch 释放或输入非法时进入 idle/fault/hold 路径。因此分布式改造必须保持以下语义：[19][20]

- 一个 XR 会话只允许一个所有者；subscriber 不得重新创建 CloudXR/OpenXR 会话。
- 一次消息必须原子地包含左右手全部字段，不能把 28 个字段拆成独立 topic 后在订阅端重新拼接。
- 允许丢弃旧帧，禁止积累控制延迟或重放旧动作。
- 没有首帧、消息过期、上游重启或传输断开时，必须立即产生失效/中性动作，至少令双手 `is_tracking=False`、`is_aim_tracking=False`、`is_engaged=False`。
- 节点重启后 clutch 原点会重置；subscriber 必须识别新会话，机器人侧仍需重新释放并接合，而不能沿用旧锚点。

## Dora 架构与生命周期

Dora 把应用描述为节点和端口组成的有向图。分布式模式的控制拓扑是：

```text
CLI -> Coordinator -> 每台机器一个 Daemon -> 独立 Node / Runtime Operator
```

Coordinator 负责 dataflow 生命周期、节点放置和跨 daemon 协调；Daemon 负责本机节点进程的启动、停止、通信、健康检查和重启；Node 是独立 OS 进程；Operator 则共享 Runtime 进程。硬件驱动型 `isaac-teleop` 应使用独立 Node，不应作为共享进程内的 Operator。[1]

普通 Node 的典型生命周期如下：[1][7]

1. Daemon 根据 dataflow 描述启动脚本或二进制，并通过环境变量注入节点身份和通信配置。
2. Node 向 Daemon 注册并订阅事件流。
3. Node 在事件循环中读取输入、发送输出，并响应 `STOP`、输入关闭和上游重启等控制事件。
4. 进程退出后，Daemon 根据 `restart_policy`、重启次数、时间窗口和退避配置决定是否重启。

本机开发可使用 `dora run dataflow.yml`。后台或跨机运行使用 `dora up`/`dora start`，或者将每机 daemon 安装为 systemd 服务。后者更接近“后台常驻并可运维”的目标。[2][3]

这里存在一个关键适配问题：Dora 管理的是 **dataflow 的生命周期**，并非一个地址稳定、任何外部客户端随时 SUB 的通用消息服务。若把 `isaac-teleop-node` 置于专用的长期 dataflow 中，控制器和可视化器通常也要成为该 dataflow 的节点，或者通过运行期拓扑命令加入。相比之下，ZeroMQ PUB/SUB 天然允许未知数量的订阅者独立出现和消失。

## Python API、自定义节点与动态节点

官方 Python 包名为 `dora-rs`，导入名为 `dora`。普通 Python Node 使用 `Node()` 从 Daemon 注入的环境读取身份；`Node(node_id="...")` 用于手工启动并连接到已声明为 `path: dynamic` 的节点。主要接收接口包括阻塞的 `next(timeout=...)`、非阻塞的 `try_recv()`、一次取走当前缓冲的 `drain()`，发送接口为 `send_output()`；载荷可以是 bytes 或 PyArrow Array。[7][8]

```python
from dora import Node

node = Node()
node.send_output("action", action_array, metadata)

for event in node:
    if event["type"] == "INPUT":
        consume(event["value"], event["metadata"])
    elif event["type"] == "STOP":
        break
```

Dora 中有两类容易混淆的“动态”：

- `path: dynamic` 表示节点进程由用户手工启动，而不是由 Daemon 启动。官方 Rust API 文档明确列出限制：它不能用于 `dora run`，节点 ID 必须在所有运行中的 dataflow 中唯一，跨机时还要由用户在正确机器上启动。[9]
- `dora node add/remove/connect/disconnect/replace` 表示修改运行中 dataflow 的拓扑。[4]

二者都不能完全等价于“任意进程知道 endpoint 后即可订阅”。特别是跨机的晚加入订阅者曾触发过初始化卡死，相关修复和多 daemon 端到端测试直到 2026-08-03/04 才进入 `main`，晚于 `v1.0.0-rc.4`。[10][11][12] 因此，如果用 Dora 实现按需订阅，应固定到包含这些修复的版本，并专门测试控制链路在 teleop 已运行之后加入、退出、重启和跨机重连的场景。

## YAML 数据流映射

Dora 的端口和连线在 YAML 中显式声明。按当前目标，一个合理的 Dora 图形状是：[2]

```yaml
health_check_interval: 1.0

nodes:
  - id: isaac-teleop
    path: teleop_node.py
    outputs:
      - action
      - status
    restart_policy: on-failure
    max_restarts: 5
    restart_delay: 1.0

  - id: robot-control
    path: robot_control.py
    inputs:
      teleop_action:
        source: isaac-teleop/action
        queue_size: 1
        queue_policy: drop_oldest
```

建议发布两个输出：

- `action`：一个不可分割的双手动作帧，包含现有 28 个字段。
- `status`：连接阶段、Quest 是否已连接、最近错误、发布频率、丢帧计数和会话 ID；控制链路不应通过“有没有 action”间接猜测硬件状态。

不要把每个手柄字段做成单独 output。Dora 对不同输入 ID 使用独立队列并以公平策略调度，不同输入之间不保证保持原始全局时间顺序；拆分字段会引入跨 topic 拼帧和一致性问题。[13]

## 跨进程和跨机器传输

Dora 在上层保持同一套端口模型，在底层根据节点位置选择传输：[1][3]

- 同机节点由 Daemon 建立本机通信，并可使用 Zenoh/shared-memory 快路径。
- 跨机节点使用 Zenoh；当前 `main` 会尝试建立 producer node 到 consumer node 的直接 Zenoh 链路，使 Daemon 退出数据路径。
- 直连未建立时保留 Daemon 转发路径，避免启动阶段静默丢消息。
- 远端 dynamic consumer、producer 没有可拨地址、连接建立超时，以及远端输入配置了 `input_timeout` 时，会固定使用 Daemon 路径。[3][14]

双手动作帧只有约 182 字节的原始数值数据，远低于 Dora 默认 4 KiB 零拷贝阈值，因此共享内存零拷贝不是这个场景的主要收益；30 Hz 下带宽也很小。真正有价值的是跨机拓扑、监督、时间戳、监控和记录回放，而不是吞吐量。[1][14]

Dora 的协调面可以启用 bearer-token 身份验证，Zenoh 数据面可通过 JSON5 overlay 增加 TLS 和认证。官方部署文档把默认跨机节点网络定位在 tailnet 或 ACL 后方；不应直接把默认端口暴露到不可信网络。[3]

## 消息格式与时间戳

Dora 的数据载荷使用 Apache Arrow，消息元数据包含 U-HLC 混合逻辑时钟时间戳；Python 订阅端看到的是带 UTC 时区的 `datetime`。用户元数据支持 bool、整数、浮点、字符串、列表和时间戳，并已有 `session_id`、`seq`、`flush` 等约定键。[1][7]

对于 Lekit，建议一个 `action` 输出承载一行固定 schema 的 Arrow Struct，或者承载版本化二进制结构；无论选择哪一种，都应额外定义：

| 字段 | 用途 |
| --- | --- |
| `schema_name` / `schema_version` | 拒绝不兼容的字段布局 |
| `session_id` | teleop-node 每次启动或重新建立 XR 会话时变化 |
| `sequence` | 单调递增，诊断丢帧、重复和乱序 |
| `source_monotonic_ns` | 同一 producer 会话内测量采样间隔，不用于跨主机直接算年龄 |
| `source_utc_ns` 或 Dora timestamp | 日志、链路观测；跨机使用需要 NTP/PTP 或等价时钟同步 |

机器人安全不能只依赖 producer 时间戳。跨机时，subscriber 应以自己的单调时钟记录“最后一次收到合格帧”的时刻，并据此判断链路是否 stale；这样不依赖两台机器的墙钟同步。

## 队列、背压与最新帧语义

Dora 默认并不是最新值寄存器。`main` 中每个输入默认缓存 10 条，默认 `drop_oldest`；`backpressure` 会扩大缓冲，在硬上限处仍会丢弃并记录 ERROR。源码把 `queue_size: 0` 的 `drop_oldest` 容量钳制到 1，并明确称为 latest-only；为了配置可读性，本文仍建议显式写 `queue_size: 1`。[2][13][15]

`queue_size: 1` 加 `drop_oldest` 只保证 **Node API 尚未消费的该输入缓冲** 最多保留一个最新事件，并不构成端到端“绝不执行旧帧”的保证：

- 已经被 subscriber 取出的旧事件不会被撤销。
- 网络和传输层仍可能存在在途数据。
- subscriber 如果处理变慢，仍可能执行已经过时的帧。
- Dora 的 `flush: true` 可以清理某个输入中更旧的非关联事件，但它主要是流式中断语义，不能代替消息年龄检查。[7][13]

因此 Dora subscriber 仍应在每个 30 Hz 控制周期先 `drain()`，只采用本批次最后一个 `action`，然后验证 `session_id`、`sequence` 和本地接收年龄。若没有新帧或年龄超过安全阈值，必须交给 Piper processor 一个明确失效的中性动作，不能沿用 Dora/Rust `InputTracker` 缓存的最后值继续控制机械臂。[7][16]

## 超时与安全边界

Dora 的 `input_timeout` 很有用，但不能直接承担机械臂的快速失联保护：

1. 当前源码只有在输入至少收到一条消息后才启动 deadline；从未收到首帧的输入不会超时。这是为合法的按需 service 输入设计的，但意味着“teleop-node 已运行、Quest 从未连接”不能靠它检测。[17]
2. timeout 的扫描频率由 dataflow 级 `health_check_interval` 控制，默认 5 秒。把机械臂安全完全交给该巡检会过慢。[2][16]
3. 远端输入一旦配置 `input_timeout`，Dora 会让该边保留在 Daemon 转发路径，因为直连会绕过用于刷新 deadline 的 consumer Daemon。[3][14]

推荐把 Dora timeout 用作较慢的运维级 circuit breaker；控制级 watchdog 仍放在 `teleop subscriber` 内，以本地单调时钟每周期检查。首次等待和流中断都要有独立阈值。

## 故障、重启与监控

Dora `main` 提供每节点 `never`、`on-failure`、`always` 重启策略、重启次数、指数退避、重启窗口和连接后的健康检查；下游能收到 `NodeRestarted`，输入超时和恢复会产生 `InputClosed`/`InputRecovered`。[2][16]

这些能力与 Quest/CloudXR 的脆弱外部会话很契合，但要保留应用级状态机：

- Node 进程活着不表示 Quest 在 tracking；`status` 必须明确区分 `waiting_for_headset`、`streaming`、`degraded` 和 `fault`。
- Node 重启后生成新的 `session_id` 并重置 clutch。
- subscriber 收到 `NodeRestarted`、session 改变、`InputClosed` 或 stale 时立即失效动作。
- restart 不得使机器人自动重新 engage；必须重新 tracking、释放 squeeze，再由现有 Piper 状态机接合。

官方工具包括结构化日志、`dora top` 的 CPU/RSS/队列深度/网络流量/重启次数、`topic echo/hz/info`，以及 `.drec` record/replay。topic 检查需要启用 debug inspection，`dora top` 和普通日志不需要。记录 Quest 动作并离线回放对 subscriber 与 processor 的故障测试很有价值，但任何 replay 必须与实机运动使能隔离。[6][16]

`main` 已在 2026-06 修复“Coordinator 断开会杀死所有 Node”的问题，并增加 daemon reconnect 的端到端测试；但是官方 fault-tolerance 指南在当前基线仍保留旧的 known-limitation 文本。这是版本高速演进和文档滞后的直接例子，选型时应以固定提交的源码与测试为准，而不是混用 latest 文档和旧 wheel。[21][22]

## 部署依赖、许可证和项目状态

官方安装方式是 `cargo install dora-cli` 和 `pip install dora-rs`；Python Node 依赖 PyArrow。托管多机模式还依赖 SSH，安装 systemd service 和滚动升级需要远端权限。当前 `main` 的 Rust MSRV 是 1.95，Python 文档写的是 3.11+；Lekit 自身要求 Python 3.12，因此语言版本没有冲突。[1][23][24]

版本状态必须特别注意：

- PyPI 当前稳定包仍是 `dora-rs 0.5.0`，发布于 2026-03-25；它已有 `queue_size` 和 Python `drain()`/`try_recv()`，但没有 `main` 文档中的 `queue_policy` 和 `input_timeout` 完整配置面。[24][25]
- GitHub 最新正式标记为 stable 的版本也是 `v0.5.0`；`v1.0.0-rc.4` 发布于 2026-07-22，仍为 pre-release。[5]
- 调研基线 `main` 在最近 30 天有 299 个提交，最近一次提交为 2026-08-25，说明项目非常活跃，也说明未固定版本会承受较高变化风险。[26]

官方仓库和 Cargo workspace 声明 Apache-2.0。[27][28] PyPI `0.5.0` 页面当前却显示 MIT，这属于发布元数据与仓库许可证不一致；若正式引入，需要在依赖清单或法务审查中记录并向上游确认。[24]

## 适配度评估

| 目标 | Dora 适配度 | 说明 |
| --- | --- | --- |
| 唯一硬件所有者、独立进程 | 高 | 普通 Node 与 Daemon supervision 完全匹配 |
| 一个 producer、多消费者 | 高 | dataflow 内 fan-out 原生支持 |
| 任意外部控制进程随时订阅/退出 | 中 | 需要成为 dataflow 节点或运行期修改拓扑，不如裸 PUB/SUB 自然 |
| 30 Hz 最新帧控制 | 中高 | `queue_size: 1`、`drop_oldest`、`drain()` 可实现，但 stale 检查必须应用自己做 |
| 同机与跨机统一 | 高 | 同一端口模型，Zenoh 跨机，具备放置和集群管理 |
| 机械臂安全失联保护 | 中 | 有 timeout/restart 事件，但首次无帧、巡检粒度和过期动作仍需 subscriber watchdog |
| 后台运行与自动恢复 | 高 | Daemon、systemd、restart policy、日志和指标完整 |
| 轻量接入现有 Lekit | 低到中 | 新增 CLI/Daemon/Coordinator/PyArrow/YAML 运维面，明显重于现有 pyzmq |
| 当前版本稳定性 | 中 | 1.0 仍为 RC，`main` 与已发布版本差异较大 |

## 建议从 Dora 吸收的 `teleop-node` 设计

即使第一阶段继续使用 ZeroMQ，公共模块也应采用下列 Dora 风格的边界：

```text
IsaacXRController（硬件适配器，只在 producer 进程内）
        |
        v
TeleopNode（采样、验证、session/sequence、action/status 发布）
        |
        +---- action ----> LeRobotTeleopSubscriber ----> Piper processor
        |
        +---- status ----> visualizer / monitor
```

建议公共抽象只暴露稳定的领域语义：

- `TeleopFrame`：版本、会话、序列、采样时间和完整的双手 `RobotAction`。
- `TeleopStatus`：硬件连接、XR streaming、最近错误、频率和统计。
- `TeleopPublisher`：发布 action/status，不知道具体机器人。
- `TeleopSubscriber`：实现标准 LeRobot `Teleoperator`；只持有最新帧缓存和安全 watchdog，不知道 CloudXR。
- `TeleopTransport`：隐藏 ZeroMQ 或 Dora 的启动、编码、订阅和关闭细节。

第一版 ZeroMQ 与未来 Dora 后端必须共享同一组协议测试：schema 拒绝、动作原子性、慢消费者丢旧帧、首次无帧、stale、publisher 重启、session 变化、sequence 乱序、双手 tracking 失效、subscriber 晚加入、多个 subscriber 和干净退出。这样 Dora 是可替换的运行时，而不会侵入 `IsaacXRController` 或 Piper 控制逻辑。

## 若要直接采用 Dora，建议先做的验证

在把 Dora 放入实机控制链路前，建议固定一个包含 2026-08 多 daemon late-subscriber 修复的 commit，完成一个不连接机器人运动的 spike：

1. 普通 Python `isaac-teleop-node` 发布一个原子 `action` Struct 和独立 `status`。
2. 本机与两机各运行一次，控制 subscriber 设置 `queue_size: 1`，循环只取 drain 后最后一帧。
3. 测量 p50/p95/p99 接收年龄和最长停顿，而不只看平均 Hz。
4. 依次杀死 Quest client、teleop Node、Daemon、Coordinator 和网络接口，验证 subscriber 在预定阈值内输出失效动作。
5. teleop 已运行后再加入/移除/restart subscriber，验证不挂起且不会执行加入前的历史动作。
6. 验证 `dora record/replay` 的 action schema，同时确保 replay 模式永远不能意外使能实机运动。

只有当这些测试通过，并且团队愿意维护 Dora 集群运行面时，才值得用 Dora 替换第一阶段的 ZeroMQ transport。

## Sources

1. Dora, [Architecture](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md), accessed 2026-08-26.
2. Dora, [Dataflow YAML specification](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/dataflow-yaml.md), accessed 2026-08-26.
3. Dora, [Distributed deployment](https://github.com/dora-rs/dora/blob/main/docs/distributed-deployment.md), accessed 2026-08-26.
4. Dora, [Dynamic topology](https://github.com/dora-rs/dora/blob/main/guide/src/operations/dynamic-topology.md), accessed 2026-08-26.
5. Dora, [GitHub releases](https://github.com/dora-rs/dora/releases), accessed 2026-08-26.
6. Dora, [Debugging and observability](https://github.com/dora-rs/dora/blob/main/guide/src/operations/debugging.md), accessed 2026-08-26.
7. Dora, [Python API reference](https://github.com/dora-rs/dora/blob/main/guide/src/languages/python.md), accessed 2026-08-26.
8. Dora, [Python Node implementation](https://github.com/dora-rs/dora/blob/main/apis/python/node/src/lib.rs), accessed 2026-08-26.
9. Dora, [Rust Node API dynamic-node contract](https://github.com/dora-rs/dora/blob/main/apis/rust/node/src/lib.rs), accessed 2026-08-26.
10. Dora, [Late subscriber fix `596f27b2`](https://github.com/dora-rs/dora/commit/596f27b2), accessed 2026-08-26.
11. Dora, [Multi-daemon late-subscriber E2E `7f9b0b78`](https://github.com/dora-rs/dora/commit/7f9b0b78), accessed 2026-08-26.
12. Dora, [Nightly multi-daemon late-subscriber coverage `313ca4cc`](https://github.com/dora-rs/dora/commit/313ca4cc), accessed 2026-08-26.
13. Dora, [Input event scheduler](https://github.com/dora-rs/dora/blob/main/apis/rust/node/src/event_stream/scheduler.rs), accessed 2026-08-26.
14. Dora, [Daemon output-routing policy](https://github.com/dora-rs/dora/blob/main/binaries/daemon/src/output_routing.rs), accessed 2026-08-26.
15. Dora, [Queue policy and capacity source](https://github.com/dora-rs/dora/blob/main/libraries/message/src/config.rs), accessed 2026-08-26.
16. Dora, [Fault tolerance](https://github.com/dora-rs/dora/blob/main/guide/src/operations/fault-tolerance.md), accessed 2026-08-26.
17. Dora, [InputDeadline source](https://github.com/dora-rs/dora/blob/main/binaries/daemon/src/running_dataflow.rs), accessed 2026-08-26.
18. Lekit, [`IsaacXRController`](../../src/lekit/teleoperators/isaac_teleop/xr_controller.py), accessed 2026-08-26.
19. Lekit, [Piper teleoperation loop](../../src/lekit/scripts/teleop.py), accessed 2026-08-26.
20. Lekit, [Piper Isaac retargeting safety state machine](../../src/lekit/robots/piper/teleop_processor.py), accessed 2026-08-26.
21. Dora, [Preserve nodes across coordinator reconnect `d3edfdae`](https://github.com/dora-rs/dora/commit/d3edfdaec7a96fc5bb601d8ab8c53225e500174d), accessed 2026-08-26.
22. Dora, [Daemon reconnect E2E](https://github.com/dora-rs/dora/blob/main/tests/daemon-reconnect-e2e.rs), accessed 2026-08-26.
23. Dora, [Installation guide](https://github.com/dora-rs/dora/blob/main/guide/src/getting-started/installation.md), accessed 2026-08-26.
24. PyPI, [`dora-rs`](https://pypi.org/project/dora-rs/), accessed 2026-08-26.
25. Dora, [`v0.5.0` input config](https://github.com/dora-rs/dora/blob/v0.5.0/libraries/message/src/config.rs), accessed 2026-08-26.
26. Dora, [`main` commit history](https://github.com/dora-rs/dora/commits/main/), accessed 2026-08-26.
27. Dora, [Apache-2.0 LICENSE](https://github.com/dora-rs/dora/blob/main/LICENSE), accessed 2026-08-26.
28. Dora, [Cargo workspace metadata](https://github.com/dora-rs/dora/blob/main/Cargo.toml), accessed 2026-08-26.

# Dora-rs 作为 teleop-node、robot-node 与 control-plane 通信中间件的可行性

调研日期：2026-08-27。本文只采用 Dora 官方文档、`dora-rs/dora` 官方仓库（含 release、源码、issue）和 PyPI 官方项目页作为 Dora 的事实来源；仓库现状来自本工作区代码。这里的“Control Plane”特指 Lekit 拟建设的 FastAPI + SQLite **业务控制面**，不要和 Dora 的 Coordinator 混为一谈。

## 结论

**推荐 C：Dora 作为可选的边缘运行时/实时数据面，Lekit Control Plane 独立，且由 robot-node 本地安全内核最终裁决。** 先保持现有 ZeroMQ transport 作为基线；只有在多机器人、跨机传感器/推理图、节点监督、可观测性或录制回放已成为实际需求时，才以固定 Dora 版本引入一条受限的 Dora dataflow。

不推荐 A（Dora 完全替代并兼任业务编排）。Dora 能编排**进程和数据流**，却没有本项目需要的持久、可审计、按机器人资源范围进行 compare-and-swap 的控制权租约模型，也不能定义或证明机械臂的 SAFETY/HOLD 转移。B（仅实时数据面、独立 Control Plane）可行，但没有充分利用 Dora 的节点运行时价值；C 在不把安全与业务语义外包给 middleware 的前提下，保留 B 的边界并按需采用 Dora 的部署、监督、指标与 record/replay。

对当前单一 Quest 3 → Piper 的 30 Hz、小于 4 KiB 的原子动作帧而言，Dora 的零拷贝吞吐优势不是决定因素；现有 ZeroMQ 接收端设置 `RCVHWM=1` 和 `CONFLATE=1`，并用 250 ms 本地单调时钟 watchdog 输出 neutral action，正是合适的最新帧基线。Dora 的收益主要是跨机图、统一节点生命周期和运维，不是替代这条安全语义。

## 事实核验与版本边界

| 项目 | 截至 2026-08-27 的结论 | 选型含义 |
| --- | --- | --- |
| 已发布稳定版 | GitHub 最新非 prerelease 为 [`v0.5.0`](https://github.com/dora-rs/dora/releases/tag/v0.5.0)，发布于 2026-03-25；PyPI 当前稳定包也是 [`dora-rs 0.5.0`](https://pypi.org/project/dora-rs/)。 | 生产 spike 应从这一可复现 release 开始，不能把 `main` 当稳定 API。 |
| 预发布 | 最新候选为 [`v1.0.0-rc.4`](https://github.com/dora-rs/dora/releases/tag/v1.0.0-rc.4)，GitHub 标记为 prerelease，发布于 2026-07-22。 | 仅可用于隔离实验；升级必须重新跑验收。 |
| Python 3.12 | PyPI 元数据要求 Python `>=3.8`，并发布 `cp37-abi3` wheels；因此 CPython 3.12 在 ABI/声明范围内。官方 `main` README 则把 Python node 的起点写为 3.11+。见 [PyPI](https://pypi.org/project/dora-rs/) 和 [README](https://github.com/dora-rs/dora/blob/main/README.md)。 | Lekit 的 `requires-python >=3.12` 不构成版本阻塞；仍要在目标 Linux/架构安装 wheel 并验证 PyArrow 组合。 |
| 许可证 | 仓库 [`LICENSE`](https://github.com/dora-rs/dora/blob/main/LICENSE) 是 Apache-2.0；PyPI 0.5.0 元数据显示 MIT。 | 这是发布元数据与源码不一致的供应链/法务风险；引入前锁定具体 artifact 的 `LICENSE`/SBOM，并向上游确认，不可自行假定。 |

下文凡写“`main`”均表示官方主干的当前能力，**不等于** 0.5.0 已发布保证。尤其动态拓扑、Zenoh 数据面、队列策略和 fault tolerance 近月变化快，必须以固定 tag/commit 和本项目测试结果为准。

## Dora 能提供什么，不能提供什么

官方架构把应用建模为有向 dataflow：独立 OS 进程的 **Node** 通过输入/输出端口通信；较轻的 **Operator** 在共享 Runtime 内执行。每台机器一个 **Daemon** 负责拉起/监视节点和本机通信；多机 **Coordinator** 通过 WebSocket 协调 daemon 与 dataflow 生命周期。详见官方 [architecture](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md) 和 [CLI reference](https://dora-rs.ai/dora/operations/cli)。

适合 Lekit 的映射如下：

```text
Quest/OpenXR ─> teleop-node (Node，唯一硬件所有者)
                         │ action/status（实时数据面）
                         v
                 robot-node (Node，本地安全裁决、唯一机器人所有者)
                         │ 状态/遥测（可汇聚、可丢旧）
                         v
            FastAPI + SQLite Control Plane（业务命令、租约、审计）
                         │ 已授权的 intent / revoke / HOLD
                         └──────────────────────────────> robot-node
```

teleop 与物理机器人驱动必须是 Node，而不是 Operator：它们需要进程隔离、独占硬件句柄以及可被 daemon 重启。Operator 仅适合无硬件所有权、可容忍同进程失败域的轻量转换/过滤；官方架构也明确区分了 standalone node 和 in-process runtime operator。[architecture](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md)

### 同机、跨机与动态性

官方主干说明的链路是：Daemon↔Node 用本地 TCP/共享内存控制通道；Daemon↔Daemon 用 Zenoh；对大于 4096 bytes 的数据，Node↔Node 可走 Zenoh shared memory 的直接路径。本机共享内存不等于跨机零拷贝：远端仍要网络复制。见 [架构的通信协议和阈值](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md) 与 [Zenoh SHM 设计/迁移讨论](https://github.com/dora-rs/dora/blob/main/docs/plan-zenoh-shared-memory.md)。

当前 teleop 原子帧约为数百字节量级，低于 4 KiB 阈值，预期走小消息路径；对它追求 SHM 零拷贝没有明显回报。相机、点云、张量等大 payload 才是 Dora/Zenoh SHM 值得 benchmark 的对象。Python Node 的 payload 是 bytes 或 Arrow/PyArrow 数据；官方协议把 Arrow 类型信息/metadata 携带在消息中。[Python API](https://github.com/dora-rs/dora/blob/main/guide/src/languages/python.md)；[message/Arrow 架构](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md)

动态节点有两层含义，不能把它误读成“任意外部客户端随时 SUB 即可”：

- `path: dynamic` 是节点由用户手工启动、接入已声明 dataflow 的契约；官方 Rust Node API 给出 `dora run` 不支持、全局 node ID 唯一、跨机需自行在正确机器启动等限制。[Node API](https://github.com/dora-rs/dora/blob/main/apis/rust/node/src/lib.rs)
- `dora node add/remove/connect/disconnect/replace` 是修改运行中 dataflow 的拓扑。[dynamic topology](https://github.com/dora-rs/dora/blob/main/guide/src/operations/dynamic-topology.md)

因此 Dora 能支持受控的晚加入和拓扑变更，却比 ZeroMQ PUB/SUB 的无协调订阅者更重。更重要的是，近期官方 issue 仍记录 Zenoh 节点 listener bind race 可导致 silent partition；它被标为高影响、尚未关闭，虽为代码路径分析但必须纳入 fault injection。[#2762](https://github.com/dora-rs/dora/issues/2762)

### 队列、背压和“最新帧”

Dataflow YAML 的 input 具有 `queue_size`（默认 10）和 `queue_policy`；官方文档定义 `drop_oldest` 在满时丢弃最旧项，`backpressure` 可增长到 `10 × queue_size` 后仍会丢弃并记 ERROR。[YAML specification](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/dataflow-yaml.md)

对 `teleop/action -> robot-node`，应显式使用 `queue_size: 1` + `drop_oldest`，绝不用 `backpressure`。接收 Python Node 每个控制周期应 `drain()` 并只处理最后一个有效 frame；`drain()`、`try_recv()` 和事件 API 见官方 [Python API](https://github.com/dora-rs/dora/blob/main/guide/src/languages/python.md)。但这只限制 **尚未被应用取走的输入队列**，并非端到端 latest-only：在途包、已经取到的旧帧和 Python 调度停顿不能被撤销。

故 robot-node 必须继续执行下列深层策略：每帧含 `session_id`、严格单调 `sequence`、schema version、source timestamp；以**自身单调时钟**记录合格帧抵达时刻；每周期 drain 后校验 session/seq/本地接收年龄；首帧缺失、乱序、会话变化、到期或 transport 断开立即进入 HOLD 并向底层输出安全命令。Dora 的 metadata 已有 `session_id`、`seq` 与 `flush` 约定，但 `flush` 是队列清理提示，不能替代上述实时安全检查。[architecture metadata](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md)

Arrow 的权衡也应据实评估：它给跨语言结构化 schema、向量/图像的大块 buffer 传递和统一记录提供好处；但对 30 Hz 的小型控制帧，Python↔PyO3↔PyArrow 创建/类型信息/metadata 编码和对象分配很可能主导成本，且未见官方为该具体小帧场景给出的延迟保证。故不可从“4 KiB 以上零拷贝”推导“Python 小控制帧更快”；验收必须直接测量。控制动作可先使用版本化固定 bytes，或只在实测优于/等价时用单个 Arrow Struct，绝不可拆为多个 field topic 后再拼帧。

### 监督、健康、可观测性和录制

Daemon 提供节点 spawn、进程监控以及 `restart_policy` 等 dataflow 级机制；节点事件包含 `Stop`、`InputClosed`、`InputRecovered` 与 `NodeRestarted`。Coordinator 与 daemon 也有 heartbeat、status/metrics/log 事件。官方 [fault tolerance](https://github.com/dora-rs/dora/blob/main/guide/src/operations/fault-tolerance.md) 和 [architecture](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md) 给出这些接口。

官方 CLI 还提供日志、`dora inspect top`/topic 检查、record/replay；架构中包含 record/replay node 和 `.drec` 格式。[debugging/observability](https://github.com/dora-rs/dora/blob/main/guide/src/operations/debugging.md)，[architecture](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md)。这些是 C 方案有价值的运维增益，不过：

- 存活/heartbeat 不能证明 Quest tracking、驱动链路或制动器安全；robot-node 还需应用健康信号与本地 watchdog。
- `input_timeout` 和 daemon health scan 应当是运维级降级/告警，不能是毫秒至百毫秒级最终 safety interlock。官方 fault-tolerance 配置本身应按目标 release 再核验。[fault tolerance](https://github.com/dora-rs/dora/blob/main/guide/src/operations/fault-tolerance.md)
- record/replay 是诊断工具；replay dataflow 必须在配置和硬件许可两层都禁止 `motion-enable`，不得因回放误驱动实机。
- 2026-07 的官方 issue 显示 Zenoh runtime 卡住时 node shutdown 仍可能无限挂起，虽有部分 timeout 修复。这要求 robot-node 的外层 supervisor、强制 HOLD 和 bounded kill/restart 设计。[#2583](https://github.com/dora-rs/dora/issues/2583)

### 安全、部署与权限

官方架构说明 Coordinator 的控制 WebSocket 使用 bearer token；其认证 token 由随机字节生成并存于工作目录，WebSocket routes 采用常量时间比较。Zenoh 网络层的 TLS/认证是通过 Zenoh 配置设置，而不是 Dora 业务权限系统。见 [architecture authentication](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md) 和 [distributed deployment](https://github.com/dora-rs/dora/blob/main/docs/distributed-deployment.md)。

这意味着 Dora 能帮助保护其 coordinator/transport，但不能替代本项目的用户、机器人、租约、审计与授权模型。部署上需要 CLI、每机 daemon；多机还需 coordinator、Zenoh 可达性/路由与防火墙/证书配置。将 coordinator 或 Zenoh 默认端口直接暴露给不可信网段不是安全设计；应使用受管网络、mTLS/Zenoh auth、最小网络 ACL、密钥轮换和独立的 Control Plane 身份验证。

## 三种设计比较

| 设计 | 责任划分 | 优势 | 致命/主要代价 | 判定 |
| --- | --- | --- | --- | --- |
| A. Dora 完全替代 ZeroMQ，承担数据面 + 节点编排 | Coordinator/Daemon 编排，Dora port 同时传实时动作和业务命令 | 一个框架覆盖多机 dataflow、节点监督、metrics、record/replay。 | 把业务状态耦合到短生命周期 dataflow；仍需另造租约、审计、Safety 状态机，且错误地诱导团队将其放在 middleware；对当前小流量运维过重，RC/main 演进风险高。 | **不推荐。** |
| B. Dora 仅实时数据面，Control Plane 独立 | FastAPI+SQLite 负责业务；Dora 仅 `teleop->robot`/媒体主题 | 边界清晰；可逐步替换 ZMQ；保留 C-P 的持久性与安全所有权。 | 没充分利用 daemon 的节点运行时；依然要定义两条通道如何授权、撤销、降级。 | 可行的过渡。 |
| C. Dora 边缘运行时 + 数据面；业务 Control Plane 独立 | Dora：部署 graph、进程监督、非权威实时事件/遥测/回放。Control Plane：身份、租约、策略、审计、期望态。robot-node：本地最终裁决、HOLD。 | 最符合失效域：中心服务或网络失败仍能在边缘 HOLD；可先在非关键 topic 引入 Dora，再扩大；Dora 价值与业务模型互不侵蚀。 | 两套系统的关联需设计：稳定 robot ID、correlation ID、版本、幂等命令、可观测性汇聚；部署复杂度仍存在。 | **推荐。** |

推荐的职责合同：Control Plane 发出持久、已授权且带 `lease_id`/`fencing_token`/expiry 的 `ControlIntent`，并持久化审计；robot-node 本地验证 token 未过期、主体/机器人/模式匹配、fencing token 新于已接受值、teleop 和机器人健康，才把输入映射为 actuator command。Dora action/status 是低延迟、可丢旧的 transport；它不拥有“谁能动机器人”的真相。中心 C-P 可请求 `HOLD`，但 robot-node 的局部 watchdog 可随时越权将其强制为 HOLD。

## 为什么 Dora 不能直接承担接管、独占、审计与 SAFETY/HOLD

不能直接承担。官方能力是 dataflow 的 node 生命周期、端口路由、restart、input timeout、topic 检查/record；其 Coordinator state 是 dataflow 编排状态（默认内存、可选 redb），不是带业务资源、身份主体、事务语义和不变量的机器人控制权账本。[architecture coordinator/state](https://github.com/dora-rs/dora/blob/main/guide/src/concepts/architecture.md)

业务级接管租约至少需要：按 `robot_id` 原子取得/续期/撤销；在并发控制面实例下维持单写者或一致的 CAS；失效时钟与 fencing token；把操作员身份、理由、审批、版本和结果写入不可篡改/可保留审计记录；以及能以可检索的方式证明“当时谁有权控制”。这些既非 YAML queue/restart 配置，也不能由“一个 node 正在运行”推导出来。

SAFETY/HOLD 更不能委托给远程 coordinator 或普通消息中间件：其转移取决于本地实时事实（frame age、sequence、tracking、CAN/驱动状态、限位、急停、watchdog），并且在 Coordinator、Daemon、Zenoh、网络或 Python 任一方死锁时仍必须 fail closed。它应是 robot-node 内的深模块，暴露小而可测试的接口，例如 `grant(intent)`、`revoke(reason)`、`accept_frame(frame)`、`tick(now)`、`emergency_stop()`；内部拥有状态转换、fencing、去抖、故障原因和每次 actuator 输出的审计关联。Dora 事件只能作为输入，不应成为 safety truth。

## 风险清单与缓解

| 风险 | 等级 | 缓解/准入要求 |
| --- | --- | --- |
| 0.5.0 与 RC/`main` 的特性和文档不一致 | 高 | 固定 release/commit、wheel hash 和 Zenoh config；任何升级按完整验收重新认证。 |
| Zenoh 跨机 late join/listener race 或网络分区导致静默断流 | 高 | 以本地 frame-age watchdog HOLD；双机 late join/端口竞争/分区测试；跟踪官方 [#2762](https://github.com/dora-rs/dora/issues/2762)。 |
| Node/Zenoh teardown 挂起，Dora restart 不及时 | 高 | 外层 systemd/container liveness deadline + kill；robot-node 不依赖优雅 shutdown 才进入 HOLD；测试官方 [#2583](https://github.com/dora-rs/dora/issues/2583) 相关故障。 |
| 误把 queue_size=1 当作端到端实时保证 | 高 | 每周期 drain、只接受最新合格帧、local monotonic age deadline、禁止 backpressure 于动作边。 |
| Python Arrow 对小帧的额外延迟/GC 尾延迟 | 中高 | 与现有 ZMQ 固定 CPU/负载对照测 p99.9 和最大值；小帧优先固定 bytes，除非 Arrow 实测达标。 |
| Coordinator/Zenoh 凭据或端口暴露 | 高 | 私网/ACL、TLS/认证、密钥轮换、最小权限、独立 C-P RBAC；不得将 token 视为业务授权。 |
| record/replay 被接到实机动作 | 极高 | replay build/runtime 强制 dry-run；机器人侧额外拒绝 `replay=true` 或无新鲜 lease 的 motion enable。 |
| 许可证元数据不一致 | 中 | 引入前完成依赖许可证审查并保留证据。 |

## 实机前 benchmark 与 fault-injection 验收门槛

在不连接运动使能、或连接硬件但物理制动/仿真替身有效的条件下，按同一台与两台目标机分别对 **ZMQ 基线、Dora 0.5.0、候选 RC** 运行至少 30 分钟；CPU 固定核、记录负载、NTP/PTP 状态、版本、Zenoh 配置和 payload 格式。每项以 robot-node 的单调时钟测量，不能只报 producer Hz 或平均延迟。

| 项目 | 必须达到的门槛 |
| --- | --- |
| 最新帧与延迟 | 30 Hz 动作流在额定和 2× 消费压力下无旧帧执行；`queue_size:1/drop_oldest + drain` 可观测丢旧。记录 p50/p95/p99/p99.9、最大接收年龄和连续控制停顿；候选 Dora 的 p99.9 与最大值不得劣于已批准 ZMQ 基线，或获得明确的安全预算批准。 |
| 安全反应 | 首帧超时、连续 frame-age 超阈、seq 倒退/重复、schema 不匹配、session 变化、tracking=false、lease 到期/撤销，均在 robot-node 定义的 watchdog deadline 内进入 HOLD；每次都产生带原因/correlation ID 的审计记录；恢复绝不自动 motion-enable。 |
| 进程故障 | 分别 SIGKILL teleop-node、robot-node、daemon、coordinator；使 Python event loop 卡住；验证机器人在 deadline 内 HOLD，重启后 fencing token 和 re-arm 流程阻止旧控制者恢复动作。 |
| 网络故障 | 断开/恢复两机网络、丢包/乱序/延迟/带宽限制、Zenoh router/peer 重启、late subscriber、subscriber 重启、端口占用 race；无静默继续执行陈旧动作，恢复后仍须新鲜 frame + 有效 lease + 人工/显式 re-arm。 |
| 压力与资源 | 加相机/推理大 payload、慢 consumer 和磁盘 record；CPU/RSS/FD/SHM/队列深度有上限且告警；动作通道不因其他 topic 的 backpressure 堆积。 |
| 安全与恢复 | 无 token/错误 token/过期证书/未授权 subject/重放的 intent 均被拒绝；Coordinator 不可达和 SQLite 暂时不可用时 robot-node fail closed；record/replay 无法使能真实 actuator。 |

只有上述测试在固定版本上全部通过，且能够由自动化测试复现，才允许将 Dora 从“可选数据面”扩大到实机控制路径。即使通过，也保留 ZeroMQ 后端至少到 Dora 跨机部署的稳定 release 和团队运维能力均经过一段现场验证为止。

## 本仓库依据

- [`pyproject.toml`](../../pyproject.toml)：项目要求 Python 3.12+，已有 FastAPI、pyzmq。
- [`transport.py`](../../src/lekit/teleoperators/isaac_teleop/transport.py)：PUB/SUB，subscriber `RCVHWM=1` + `CONFLATE=1`，完整动作帧单消息发布。
- [`subscriber.py`](../../src/lekit/teleoperators/isaac_teleop/subscriber.py)：首帧等待、单调时钟 stale 检查、session/sequence 与 re-arm/neutral-action 行为。

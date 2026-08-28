# 机器人 Agent 与 Skill 进化设计

## 文档状态

本文记录一套拟议架构。在 Lekit 现有的 Hub 管理式机器人控制系统之上，增加能够理解任务的 Robot Agent、自动发现的 Skill，以及一套受控的 Skill 进化流程。本文属于设计记录，不是实施计划。

本设计扩展现有的 [Control Hub 设计](2026-08-27-control-hub-design.md)。Hub 继续掌握运动控制权，Controller 继续产生实时动作，Robot 继续保有本地 HOLD 与 SAFETY 权限。

对应的英文文档见 [Robot Agent and Evolving Skills Design](2026-08-27-robot-agent-skill-evolution-design.md)。

## 目标

增加一个具备以下能力的 Agent。

- 理解人类给出的高层目标
- 构建并修订结构化 Plan
- 自动发现兼容的 Skill
- 通过统一的任务级 Interface 调用 Code-as-Policy、VLA/WAM 和 Human Teleop
- 观察执行过程，恢复执行、重新规划或升级为人工控制
- 从成功与失败的执行记录中学习可复用流程
- 提出并评估新的 Skill 版本，同时阻止未经验证的行为控制硬件

## 非目标

- 把 LLM 放进 Robot 实时动作循环
- 允许 Agent 签发、伪造或绕过 Control Handle
- 允许生成代码直接访问 Robot SDK、CAN 接口或动作套接字
- 自动修改 Robot 安全限值、Handle 校验、watchdog 或急停行为
- 在 Stable Skill 运行期间原地修改它
- 把新生成的 Policy 直接提升到不受限制的实机执行环境

## 设计原则

1. **认知与控制分离。** Agent 在任务时间尺度上推理，Controller 与 Robot 运行实时循环。
2. **Skill 可以由 Controller 提供执行能力，两者保留各自的领域含义。** Skill 表示任务级能力，Controller 表示产生动作的执行资源。
3. **Hub 继续掌握运动控制权。** Agent 和 Skill 的决策不能替代 Handle 调度、fencing、到期回收或 Robot 本地安全机制。
4. **发现过程采用渐进加载。** Agent 开始时只看到精简的 Skill 元数据，选出候选项以后再加载完整指令或 schema。
5. **进化过程产生不可变候选版本。** 每项新行为都创建带有谱系和证据的新版本。
6. **晋级依赖证据。** Stable 状态之前依次经过静态检查、历史重放、仿真、Shadow 和受限实机 Canary。
7. **安全机制不参与自动进化。** 学习过程可以改进流程与 Policy，但不能削弱确定性的安全约束。
8. **每项决策都应可观测。** 一个 Mission 应能一路追踪到 Plan、Skill Run、Controller、Handle、动作流会话和最终结果。

## 领域语言

以下术语扩展现有的机器人控制语言。

### Agent

负责解释目标、构建 Plan、选择 Skill、监督执行和重新规划的 Node。Agent 不发布 Robot 动作，也不持有 Control Handle。

### Mission

用户给出的端到端目标，例如“把红色方块放入左侧托盘”。一个 Mission 拥有一个或多个 Plan，并且最终进入终态。

### Plan

一组经过版本管理的结构化 Step。Plan 记录依赖关系、前置条件、预期效果、失败策略和完成条件。重新规划会创建新的 Plan revision，已经发生的执行历史保持不变。

### Step

Plan 中一次已经规划的 Skill 调用。

### Skill

Agent 可以发现和调用的任务能力。Skill 带有结构化输入、输出、运行要求、风险元数据和执行语义。

### Agent Skill

教 Agent 在什么情况下、用什么方式完成一类工作的程序性指令。通常由 `SKILL.md`、可选参考资料和脚本表示。

### Control Skill

需要独占 Robot 运动权限的 Skill。它由持有有效 Control Handle 的 Controller 执行。

### Composite Skill

负责协调其他 Skill 的能力，例如 `locate-object -> approach-object -> grasp-object -> place-object`。如果它的实现本身不发布动作，就不需要为自身申请 Handle。

### Skill Provider

能够执行一个或多个 Skill 的 Node 或本地 Adapter。Controller 通常充当 Control Skill 的 Provider。感知或规划等无运动能力也可以由非 Controller Provider 提供。

### Skill Run

Mission Step 对某个 Skill 发起的一次有状态执行。Skill Run 与 Control Handle 分开建模，因此在尚未获得控制权或已经释放控制权时仍然可以存在。

### Skill Candidate

由人、Agent、轨迹提取流程或训练流水线提出的不可变非 Stable Skill 版本。

### Promotion

依据证据把 Skill 版本提升到更高信任阶段的过程。Promotion 只改变生命周期状态，不修改该版本的内容。

## 两条控制循环

系统包含两条时间尺度和权限不同的循环。

```text
认知循环，通常以秒计
Mission -> Plan -> Skill -> Skill Event -> 重新规划或完成

实时循环，通常为 10 到 100 Hz
Controller -> Action Envelope -> Robot -> Observation
```

认知循环可以暂停、重试、查询模型或等待人工。实时循环必须保持有界、只保留最新动作并接受独立监督。Agent 消失时，实时循环仍须进入安全状态。

## 架构

```text
用户或外部 Agent
        |
        v
+--------------------------------+
| Agent Node                     |
| 目标理解、规划、Skill 选择与监督  |
+----------------+---------------+
                 | 任务级调用与事件
                 v
+--------------------------------+
| Agent Gateway + Skill Runtime  |
| schema、策略、审批、Skill Run、取消|
+----------+------------+--------+
           |            |
           |            +-----------------------+
           v                                    v
+-------------------+               +----------------------+
| Skill Registry    |               | Evolution Engine     |
| 包、在线 Provider、|               | Episode、候选版本、   |
| 信任、版本与证据    |               | 评测与晋级             |
+---------+---------+               +----------------------+
          |
          | 分配意图与执行关联
          v
+-------------------+     Control Handle     +-------------------+
| Hub               |----------------------->| Controller        |
| 调度、审计、撤销、  |                        | Code/VLA/Teleop   |
| force HOLD        |                         +---------+---------+
+-------------------+                                   |
                                                        | 直接动作流
                                                        v
                                              +-------------------+
                                              | Robot             |
                                              | fencing、限值、    |
                                              | HOLD 与 SAFETY    |
                                              +-------------------+
```

### Agent Node

Agent Node 负责 Mission 推理和 Plan revision。它消费精简的 Robot 状态、Skill 元数据、Skill Event 和人工输入。除非某一步推理明确要求采样产物，否则它不消费每一帧实时动作或原始 Observation。

Agent 可以基于 Codex、Claude、Hermes、OpenAI Agents SDK 应用或其他模型运行时实现。不同 Agent 框架的细节放在 Agent Adapter seam 之后。

### Agent Gateway

Agent Gateway 把外部 Agent 的工具调用转换为有类型的 Skill Runtime 操作。Agent 身份、Skill allowlist、Mission 范围、风险等级、额度和审批 token 都在这里接受策略检查。

MCP 适合作为第一种 Agent 侧低频工具 Adapter，提供以下操作。

```text
robots.list
robots.inspect
skills.search
skills.inspect
skills.run
skills.watch
skills.cancel
missions.inspect
hub.snapshot
```

MCP 和 A2A 都不承担 Robot 实时动作传输。以后可以通过 A2A 把 Agent 暴露给外部智能体，同时保持 Skill 和控制语义不变。

### Skill Runtime

Skill Runtime 应设计成深模块。Provider 选择、参数校验、审批检查、Skill Run 状态、取消、超时、Hub 关联和结果归一化都隐藏在一个较小的 Interface 后面。

```python
class SkillRuntime:
    async def start(self, skill: str, request: SkillRequest) -> SkillRunRef: ...
    async def cancel(self, run: SkillRunRef) -> None: ...
    async def watch(self, run: SkillRunRef) -> AsyncIterator[SkillEvent]: ...
```

Agent 不能通过这个 Interface 调用 `take_over`、`publish`、`send_action` 或 `hand_over`。

### Skill Registry

Skill Registry 管理定义、不可变版本、包摘要、签名、Provider advertisement、生命周期状态、兼容性元数据和评测证据。它不掌握运动控制权。

Hub 可以缓存实时兼容性检查和可观测性所需的子集。Skill 包管理仍然位于 Hub 核心调度实现之外。

### Evolution Engine

Evolution Engine 消费已经完成的 Skill Run，以及训练或评测产物。它可以识别改进机会、创建候选版本、运行评测阶段，并建议晋级或回滚。它不能签发 Handle，也不能削弱安全策略。

## Skill 包格式

模型侧指令采用开放的 Agent Skills 目录结构，并增加机器人专用的机器可读 manifest。

```text
skills/pick-object/
├── SKILL.md
├── skill.yaml
├── schemas/
│   ├── input.json
│   └── output.json
├── adapters/
│   └── vla.py
├── evals/
│   ├── scenarios.yaml
│   └── metrics.py
├── references/
└── assets/
```

`SKILL.md` 保存发现元数据和程序性指导。`skill.yaml` 保存不能从自然语言推测的确定性系统元数据。

manifest 示例如下。

```yaml
id: pick-object
version: 1.3.0
kind: control

provider_selector:
  controller_kind: vla

input_schema: schemas/input.json
output_schema: schemas/output.json

requires:
  robot_capabilities:
    - cartesian-control
    - gripper
  observations:
    - wrist-camera
    - tcp-pose

control:
  mode: exclusive
  max_speed_scale: 0.2
  stale_timeout_ms: 100
  preemptible: true

risk:
  level: medium
  human_approval: canary-only

lifecycle:
  status: stable
  parent_version: 1.2.1

artifacts:
  digest: sha256:example
  signature: example
```

manifest schema 必须拒绝未知的安全相关字段、无效版本关系、缺失摘要、相互矛盾的控制要求和不受支持的能力标识。

## 自动发现

### 发现来源

Registry 负责协调四类来源。

1. 项目控制目录中的仓库级 Skill
2. 用户控制目录中的用户安装 Skill
3. 已注册 Provider 发出的在线 Skill advertisement
4. 配置好的远程 Registry 提供的签名包

自动生成的候选版本进入独立隔离来源，不能被当作已经安装的 Stable 包。

### Provider advertisement

Provider 通过注册信息和 heartbeat 元数据发布带版本的可执行能力。

```json
{
  "provider_id": "vla-piper-main",
  "provider_session_id": "session-123",
  "skills": [
    {
      "id": "pick-object",
      "versions": ["1.3.0"],
      "robot_types": ["piper"],
      "status": "available",
      "estimated_rate_hz": 20
    }
  ]
}
```

当所有 Provider 离线时，Skill 包仍然可以保持安装状态。定义是否存在与当前能否执行应分别表示。

### 渐进发现

Agent 最初只接收以下精简元数据。

```text
Skill ID 与版本
描述
类型
所需能力
风险等级
生命周期状态
实时可用性
```

Agent 选出少量候选项以后，再加载完整指令、schema、参考资料和 Provider 详情。这个过程遵循 Agent Skills 规范中的渐进加载模式，避免未使用的能力占满模型上下文。

### 搜索与排序

Skill 选择分成两个阶段。

第一阶段执行确定性硬过滤。遇到以下情况时排除 Skill。

- Robot 缺少必要能力
- 所需 Observation 不可用
- 没有兼容的在线 Provider
- 动作 schema 或控制模式不兼容
- 当前环境不允许该生命周期状态
- 包签名、摘要或依赖校验失败
- 风险超出 Agent 权限，或缺少必要审批
- Robot、Controller 或 Hub 状态禁止分配

第二阶段执行语义排序。排序依据包括任务相关性、场景标签、历史成功率、人工介入率、执行时间、资源成本、近期稳定性和操作者偏好。

LLM 只在合格候选项中选择。一个不合格 Skill 是否获准运行，不能由 LLM 决定。

## Mission 执行流程

```text
1. Agent 创建或修订有类型的 Plan
2. Agent 搜索并检查兼容 Skill
3. Agent 为一个 Step 启动 Skill Run
4. Skill Runtime 校验输入、风险和审批
5. Skill Runtime 解析一个在线 Provider
6. Hub 检查兼容性，并在需要时分配新的 Control Handle
7. 被选中的 Controller 收到 Handle，并调用 take_over(handle)
8. Controller 动作直接传给 Robot
9. Robot 独立执行 Handle、fencing、watchdog、工作空间与安全检查
10. Provider 发出低频结构化 Skill Event
11. 完成或失败后，Controller 调用 hand_over(handle)
12. Skill Runtime 返回归一化 Skill Result
13. Agent 前进、重试、重新规划或升级到 Human Teleop
```

Agent 看到的是 `skill_run_id`，不会得到可以复用的运动授权。

## Skill Run 模型

推荐使用以下生命周期。

```text
REQUESTED
  -> WAITING_FOR_APPROVAL
  -> WAITING_FOR_PROVIDER
  -> WAITING_FOR_CONTROL
  -> RUNNING
  -> SUCCEEDED / FAILED / CANCELLED / PREEMPTED
```

需要维持以下不变量。

- 一次 Skill Run 在明确恢复过程中可以顺序使用多个 Handle，每个 Handle 仍然只属于一次分配
- Handle 丢失时立即停止动作流，并让 Skill Run 离开 `RUNNING`
- 终态 Skill Run 不能恢复，恢复操作创建带有明确谱系的新 attempt
- 完成状态需要证据，只有 Provider 声明还不够
- 取消 Skill Run 时必须撤销或释放有效 Handle，并等待 Robot 确认 HOLD

## Control Skill Adapter

### Code-as-Policy

优先采用 Code-as-Plan。生成代码调用有类型的任务原语，并组合已有 Skill，不直接访问 Robot。

如果生成代码需要连续产生动作，它必须在受限 sandbox 中运行，并置于 Controller Adapter 后面。它只接收有界 Observation，只发布通过校验的动作，并遵循正常的 Handle 生命周期。它不能直接访问 CAN、Robot SDK、Hub 存储或不受限制的网络资源。

### VLA/WAM

VLA/WAM Provider 接收目标和已经声明的 Observation，完成推理后以 Controller 身份发布动作。它向 Skill Runtime 报告低频进度、置信度、可恢复性、阶段和终止证据。Agent 不处理高频动作流。

### Human Teleop

Human Teleop 是由 Teleop Controller 支持的异步 Control Skill。启动时请求操作者并等待接受，随后协调撤销当前自主 Handle，等待 Robot 进入 HOLD，再给 Teleop 分配新的 Handle。会话结束时执行 hand-over，并产生结构化结果或示教产物。

Human Teleop 既可以显式调用，也可以在自主执行置信度不足、跟踪异常或恢复失败时充当首选升级路径。

## 可观测性与关联

所有事件都保留以下身份链。

```text
mission_id
  -> plan_id 与 revision
     -> step_id
        -> skill_run_id 与 attempt
           -> provider_id 与 provider_session_id
              -> handle_id 与 fencing_token
                 -> stream_session_id
```

Hub UI 应显示当前 Mission、Agent、Skill、Provider、Controller、Robot、Handle 状态、Robot 控制状态、进度、风险、审批、指标和接管历史。

系统应持久化结构化 Plan、工具调用、Skill 选择依据摘要、审批、结果和评测证据。系统不依赖模型隐藏思维链，也不尝试持久化它。

## Episode 记录

每次 Skill Run 至少产生一份包含以下信息的 Episode。

- Mission、Plan、Step、Skill、版本、Provider、Controller 模型、Robot 和配置身份
- 输入、约束、前置条件和环境标签
- Handle 与动作流生命周期摘要
- 已启用情况下的 Observation 采样、轨迹、数据集或视频引用
- 请求控制量和实际应用控制量的统计信息
- 完成证据和归一化结果
- 失败类别与可恢复性
- 安全事件、watchdog 事件和限值介入
- 人工修正和接管区间
- 时长、延迟、平滑度、力、成功状态和任务专用指标

高频 payload 的保留方式由策略控制。Episode 可以引用外部数据集，不必在 Hub 存储中重复保存每一帧。

## Skill 进化

进化分成三层，风险依次提高。

### 运行时适应

正在运行的 Skill 可以在不可变 manifest 限值内调整已经声明的参数，例如速度、力、视觉阈值、重试次数或目标领先量。运行时适应不能创建新的可执行代码，也不能削弱安全机制。

### 程序性进化

Evolution Engine 可以改进 `SKILL.md`、Plan 模板、Skill 组合、恢复流程、参数选择逻辑或测试。程序性进化具备良好的可解释性，可以重放，也容易回滚，适合作为第一种自动进化能力。

### Policy 进化

训练流水线可以使用 Episode 和 Teleop 示教数据更新 VLA/WAM 权重、感知模型、Code-as-Policy 实现或控制参数。Policy 进化需要数据集谱系、训练来源、模型摘要、专用评测、Shadow 执行和实机 Canary 证据。

Robot 安全规则、硬工作空间限值、fencing 检查、陈旧动作 watchdog、急停和管理权限均处于进化机制之外。

## 候选版本生成

候选版本可以来自以下来源。

- 从重复成功序列中提取 Composite Skill
- 从反复失败的聚类中提出恢复流程
- 人工修正和 Teleop 示教
- 基于历史重放或仿真的参数优化
- Agent 提出的指令或流程修订
- 生成的 Code-as-Plan 或 sandboxed Code-as-Controller
- Policy 训练或蒸馏任务

每个候选版本记录以下信息。

```text
Skill ID 与不可变候选版本
父版本或父版本集合
创作机制与身份
来源 Episode 或数据集版本
内容与产物摘要
声明的行为变化
预期改进
评测套件版本
创建时间
```

候选版本不能原地替换正在使用的 Stable 版本。

## 晋级生命周期

```text
DRAFT
  -> STATIC_VALIDATED
  -> REPLAY_VALIDATED
  -> SIM_VALIDATED
  -> SHADOW
  -> CANARY
  -> STABLE
  -> DEPRECATED / REVOKED
```

### 静态验证

验证 schema、包结构、依赖、签名、摘要、版本谱系、允许使用的工具、代码质量、参数范围、能力标识和相互矛盾的控制声明。

### 历史重放验证

使用历史 Episode 运行候选版本。把它与当前 Stable 版本比较，检查成功率、失败类别、动作或决策差异、人工介入率、延迟和任务专用指标。

### 仿真验证

运行确定性场景和随机化环境。测试范围包括碰撞、可达性、陈旧 Observation、对象缺失、跟踪丢失、进程重启、Provider 断开、Handle 撤销、超时和恢复。

### Shadow 执行

候选版本接收真实 Observation 采样并生成决策或动作，但它拿不到 Control Handle，也不能向 Robot 发布动作。评测过程比较候选输出、Stable Controller 输出、人工动作和 Robot 实测状态。

### 实机 Canary

Canary 执行需要明确授权和严格限制。限制内容包括低速、缩小工作空间、限定对象和任务、更严格的 watchdog、已经清空的工作区域、可触达的急停、持续遥测，以及可以立即抢占的 Human Teleop 或管理员。

### Stable 晋级

晋级策略使用明确阈值，包括最低场景覆盖率、成功率、安全违规为零、人工介入率上限、延迟退化上限和足够的试验次数。自动评测可以给出晋级建议。系统初期仍由人或受控发布流程批准 Stable 状态。

## 回滚与撤销

- Stable alias 指向不可变版本，并且可以原子地切回旧版本
- 正在运行的 Skill Run 保持已经解析的版本，除非 Hub 因安全原因撤销控制权
- 新发现的严重故障可以立即撤销一个版本，并强制受影响 Robot 进入 HOLD
- 撤销结果传播到 Agent 搜索结果、Registry 解析、Hub 兼容性检查和 Provider 可用性
- 历史 Episode 继续引用被撤销的确切版本，供审计和重新训练使用

## 信任与供应链控制

- 远程包需要配置好的信任根和摘要校验
- 自动生成的候选版本在验证通过前保持隔离
- Skill 定义的可信度与在线 Provider 的可信度分别校验
- Provider advertisement 不能提升生命周期状态，也不能替换 Registry 证据
- 审批 token 绑定 actor、Mission、Skill Run、Robot、风险范围和到期时间
- Agent 生成的脚本使用最小化的文件系统、网络、进程和时间权限
- 模型级 guardrail 只能补充确定性的 Hub 与 Robot 检查，不能替代它们

## 控制权抢占

建议采用以下有效权限顺序。

```text
Robot 本地 SAFETY
  > 管理员急停或 force HOLD
  > Human Teleop
  > 已批准的恢复 Controller
  > VLA/WAM 或其他自主 Controller
```

抢占始终遵循以下流程。

```text
撤销旧 Handle
-> 停止旧动作流
-> Robot 确认 HOLD
-> 终止或抢占旧 Skill Run
-> 签发新 Handle
-> 新 Controller 接管
```

任何优先级规则都不能让一台 Robot 同时拥有两个有效 Controller。

## 评测指标

不同 Skill 使用各自的任务指标，同时共享一组通用指标。

- 任务成功状态和完成证据
- 安全违规次数
- 人工介入率与接管率
- 恢复成功率和重试次数
- 总时长与各阶段时长
- 感知与推理延迟
- 动作新鲜度和控制循环漏帧
- 条件允许时记录路径长度、jerk、平滑度、力和能耗
- 工作空间或限值介入次数
- 在对象、姿态、光照、Robot 和环境标签之间的泛化表现
- 相对于当前 Stable 版本的退化情况

评测阈值必须声明样本数量和场景覆盖率。狭窄场景中的高平均分不足以支持晋级。

## 故障处理

- Agent 丢失时，有效 Handle 到期或被撤销，Robot 进入 HOLD，Mission 进入 suspended
- Skill Runtime 丢失时，Hub 和 Robot 安全机制继续工作，重启后协调 Skill Run，运动不会自动恢复
- Registry 丢失时，已经解析的 Skill Run 可以按策略完成，但不能解析新版本或执行晋级
- Provider 丢失时，Robot 的陈旧动作 watchdog 触发 HOLD，Handle 终止，Skill Run 按策略失败或进入可恢复结果
- Hub 丢失时，Handle 无法续期，Robot 独立进入 HOLD
- Robot 故障时，Hub 撤销分配，Skill Run 记录 Robot failure，Agent 只能选择无运动诊断或已经批准的恢复流程
- Evolution Engine 丢失时，Stable 执行继续正常运行，候选工作暂停且不影响控制

## 测试策略

### 纯模型测试

- manifest 校验和版本谱系
- Skill Run 状态转换 reducer
- 候选版本晋级状态转换 reducer
- 兼容性硬过滤和排序特征
- 审批与风险策略
- 关联与审计模型

### 集成测试

- 本地与远程 Skill 发现结果协调
- Provider 注册、heartbeat、离线转换和替换 session
- Agent Gateway schema 校验与权限拒绝
- Skill Runtime 配合 MemoryRuntime 与 MockRobot
- Handle 分配、take-over、取消、hand-over、到期和撤销
- Stable 回滚和候选版本撤销

### 评测测试

- 历史 Episode 重放
- 确定性仿真场景
- 随机化仿真批次
- Shadow 输出比较
- 不激活真实硬件的 Canary 策略检查

自动化测试不得打开 Piper CAN，不得连接 CloudXR，也不得移动物理硬件。

## 推荐交付顺序

### 第一阶段 发现与执行

- 增加不可变 Skill 模型和包校验
- 增加本地 Registry 和 Provider advertisement
- 增加渐进式 `search`、`inspect`、`run`、`watch` 和 `cancel` 操作
- 增加 Agent Gateway 和与框架无关的 Agent Adapter
- 包装 Human Teleop、一条 VLA/Policy 路径和一个 Composite Skill
- 关联 Skill Run、Hub Handle 和 Robot 状态

这个阶段的所有 Skill 仍由人编写并手动发布。

### 第二阶段 Episode 与评测

- 记录归一化 Skill Episode
- 增加失败分类和任务专用指标
- 增加历史重放与仿真评测套件
- 增加版本比较、证据视图和回滚

### 第三阶段 候选版本生成

- 从重复成功序列中提取 Composite Skill 候选版本
- 提出参数和程序性修订
- 生成候选测试和预期行为变化
- 把带有完整来源的候选版本注册到隔离区

晋级继续由人决定。

### 第四阶段 受控进化

- 增加 Policy 训练和 Teleop 示教流水线
- 增加 Shadow 执行和比较
- 增加受限实机 Canary 流程
- 增加统计晋级建议和自动回滚触发条件

## 外部标准

- [Agent Skills 规范](https://agentskills.io/specification) 用于可移植的 `SKILL.md` 包和渐进加载
- [Model Context Protocol](https://modelcontextprotocol.io/specification/) 用于 Agent 侧低频工具和上下文
- [Agent2Agent Protocol](https://a2a-protocol.org/latest/specification) 用于可选的外部 Agent 发现和长任务交互

这些标准作用于 Agent 侧 seam。Lekit Runtime Adapter 和 Controller 到 Robot 的直接动作传输保持独立，因此以后替换通信中间件时，不需要改变 Skill、Hub 或 Robot 安全语义。

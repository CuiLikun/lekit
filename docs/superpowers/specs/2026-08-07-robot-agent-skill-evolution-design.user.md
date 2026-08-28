推荐采用“自动发现、自动提出候选、自动评测、分级晋升”的架构。

关键原则是：

> Agent 可以自动学习和生成新 Skill，但不能自动把未经验证的新 Skill 部署到实机稳定通道。

## 1. 先区分三种 Skill

“Skill”这个词很容易混淆，机器人系统中至少有三层：

### Agent Skill

告诉 Agent 如何完成任务的程序性知识，例如：

```text
如何完成抓取任务
如何处理抓取失败
什么时候请求人工接管
```

通常由 `SKILL.md`、参考资料和脚本组成。

### Control Skill

真正产生机器人动作的执行能力，例如：

```text
move-to-pose
visual-servo
grasp-object
human-teleop
vla-manipulation
```

它通常由一个 Controller 实现，执行时必须获得 Hub Control Handle。

### Composite Skill

由多个 Skill 组成的任务流程：

```text
pick-and-place =
    locate-object
    + approach-object
    + grasp-object
    + move-to-target
    + release-object
```

建议用统一的 `Skill Package` 把这三者关联起来，但不要混成同一个类。

## 2. 整体架构

```text
                     ┌──────────────────────┐
                     │     Robot Agent      │
                     │ 理解目标、规划、选Skill │
                     └──────────┬───────────┘
                                │ search/run
                                ▼
┌────────────────────────────────────────────────┐
│                 Skill Registry                 │
│ 定义、版本、状态、兼容性、评测结果、来源、签名     │
└───────────┬──────────────────────┬─────────────┘
            │                      │
     静态 Skill Package       动态 Skill Advertisement
            │                      │
            ▼                      ▼
      SKILL.md/manifest     Controller/Provider Nodes
                                   │
                                   ▼
┌────────────────────────────────────────────────┐
│                    Hub                         │
│ 在线状态、兼容性、Control Handle、接管和审计      │
└───────────────────────┬────────────────────────┘
                        ▼
              Controller → Robot
```

其中：

- Skill Registry 管理“系统知道哪些能力”；
- Hub 管理“当前谁能控制哪台机器人”；
- Agent 决定“当前任务应该使用什么能力”；
- Evolution Engine 决定“候选 Skill 是否足够好，可以晋级”。

不要把 Skill Registry 全部塞进 Hub。Hub 只保存与实时调度相关的 Skill 摘要和运行状态。

## 3. Skill Package

可以兼容开放的 [Agent Skills 规范](https://agentskills.io/specification)，用 `SKILL.md` 实现渐进式加载，再增加机器人专用 manifest：

```text
skills/
└── pick-object/
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

`SKILL.md` 面向模型：

```yaml
---
name: pick-object
description: Pick a visible object using an available manipulation controller.
---
```

`skill.yaml` 面向系统：

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
  digest: sha256:...
  signature: ...
```

Agent Skills 规范本身主要解决指令和资源的组织与渐进式加载，不足以表达机器人能力、风险、控制模式和评测证据，所以 `skill.yaml` 是必要的机器人扩展。

## 4. 自动发现

Skill 可以来自四个来源：

1. 项目内置 Skill

```text
$REPO/skills/
```

2. 用户安装 Skill

```text
~/.lekit/skills/
```

3. Controller 动态注册的 Skill

例如 VLA Controller 启动后向 Hub 发布：

```json
{
  "provider_id": "vla-piper-main",
  "skills": [
    {
      "id": "pick-object",
      "versions": ["1.3.0"],
      "robots": ["piper"],
      "status": "available",
      "estimated_rate_hz": 20
    }
  ]
}
```

4. 远程 Skill Registry

用于多机共享经过签名的 Skill Package。

发现流程建议是：

```text
扫描本地 Skill Package
        +
接收在线 Provider Advertisement
        +
同步受信任远程 Registry
        ↓
校验 manifest、签名和版本
        ↓
生成统一 Skill Catalog
        ↓
通过 AgentGateway 提供给 Agent
```

Agent 不应该一次加载所有 Skill 的完整内容。应先只看到：

```text
id
description
kind
required capabilities
risk level
availability
```

然后通过：

```text
skills.search(query, robot, constraints)
skills.inspect(skill_id, version)
skills.run(skill_id, request)
```

按需加载详细说明。这与 Codex、Claude Code 等 Agent 的渐进式 Skill 发现方式一致。

## 5. Skill 选择

不能只让 LLM 根据名称自由选择。建议采用“硬过滤 + 语义排序”。

### 硬过滤

先排除：

- Robot 不具备必要能力；
- Controller 不在线；
- 输入观察缺失；
- 控制模式不兼容；
- 风险等级超过当前授权；
- Skill 未通过相应环境晋级；
- Skill 版本或签名无效。

### 语义排序

再根据以下因素排序：

```text
任务语义匹配度
场景匹配度
历史成功率
人工接管率
执行时间
风险
资源消耗
最近稳定性
操作者偏好
```

最终 Agent 只看到少量候选：

```text
pick-object@1.3.0       score=0.91
human-teleop@2.0.0      score=0.72
code-policy-pick@0.8.0  score=0.68
```

模型负责在合格候选中决策，而不是决定什么能力“有资格”运行。

## 6. Skill 进化闭环

```text
执行 Skill
   ↓
记录 Episode
   ↓
评估结果
   ↓
发现失败模式/重复流程
   ↓
生成候选 Skill 或候选版本
   ↓
离线测试
   ↓
仿真
   ↓
Shadow
   ↓
Canary 实机
   ↓
晋升 Stable 或回滚
```

### 第一步：记录 Episode

每次 `SkillRun` 至少记录：

```text
mission_id
skill_id/version
controller/model version
robot/config version
输入和前置条件
关键观察摘要
计划和参数
Handle 生命周期
动作统计
执行结果
失败原因
安全事件
人工接管
耗时
环境标签
```

不一定持久化全部高频动作，但应保存可重放数据或数据集引用。

### 第二步：发现进化机会

Evolution Engine 可以识别：

- 多次出现的相同动作序列；
- 高频失败原因；
- 人工经常纠正的步骤；
- 成功率明显受某个参数影响；
- Agent 经常组合的 Skill 序列；
- VLA 在某类场景持续表现较差。

例如反复出现：

```text
locate → move-above → descend → close-gripper → lift
```

就可以提出一个新的 `pick-object` Composite Skill。

### 第三步：生成候选版本

候选来源包括：

- 从成功轨迹抽取工作流；
- 调整 Skill 参数；
- 修改 Skill 指令；
- 组合已有 Skill；
- 生成受限 Code-as-Policy；
- 用新数据微调 VLA/WAM；
- 从人工 Teleop 轨迹蒸馏 Policy。

新版本必须是不可变对象：

```text
pick-object@1.3.0
    ↓ parent
pick-object@1.4.0-candidate.1
```

不能原地修改正在运行的稳定 Skill。

## 7. 晋级状态机

```text
DRAFT
  ↓
STATIC_VALIDATED
  ↓
REPLAY_VALIDATED
  ↓
SIM_VALIDATED
  ↓
SHADOW
  ↓
CANARY
  ↓
STABLE
  ↓
DEPRECATED / REVOKED
```

### Static Validated

检查：

- Schema；
- 依赖；
- 版本；
- 签名；
- 禁止调用的工具；
- 代码静态分析；
- 参数边界。

### Replay Validated

在历史 Episode 上重放，确认候选版本没有明显退化。

### Sim Validated

在仿真或数字孪生中测试：

- 成功率；
- 碰撞；
- 越界；
- 超时；
- 恢复能力；
- 随机化场景。

### Shadow

候选 Controller 接收真实观察并生成动作，但没有 Handle，动作不会发给 Robot。

系统比较：

```text
candidate_action
stable_action
actual_robot_state
human_action
```

### Canary

在严格限制下进行少量实机测试：

- 人工批准；
- 低速；
- 缩小工作空间；
- 可触达急停；
- 限定对象和任务；
- 更短 watchdog；
- 随时由 Teleop 抢占。

### Stable

只有满足明确指标才能自动建议晋升，例如：

```text
成功率 >= 95%
安全违规 = 0
人工接管率 <= 2%
P95 执行时间无明显退化
至少覆盖 N 个场景和 M 次执行
```

正式晋升最好仍由人或受控发布流程确认。

## 8. 三种不同层次的进化

### 运行时适应

在单次 SkillRun 中调整参数：

```text
速度
抓取力度
目标领先量
视觉阈值
重试次数
```

必须受 manifest 中的硬范围限制。

### 程序性进化

改进：

```text
SKILL.md
任务分解
Skill 组合
错误恢复流程
参数选择逻辑
```

风险较低，适合首先实现。

### Policy 进化

更新：

```text
VLA/WAM 权重
Code-as-Policy
视觉模型
控制模型
```

必须经过数据集版本、训练记录、离线评测、Shadow 和 Canary。

机器人安全规则、工作空间硬限制、Handle 验证和急停逻辑不属于可自动进化内容。

## 9. 自动进化的权限边界

建议明确划分：

```text
Agent
  可以发现 Skill、选择 Skill、提出候选版本

Evolution Engine
  可以生成候选、评测、比较、推荐晋升

Skill Registry
  保存版本、证据、签名、状态和回滚点

Hub
  决定候选是否获得控制权

Robot
  永远执行本地安全限制

Human
  批准高风险 Canary、Stable 晋升和紧急接管
```

Agent 不应该拥有：

- 修改 Stable Skill 的权限；
- 自行将候选晋升 Stable 的权限；
- 绕过 Hub 获取 Handle 的权限；
- 修改 Robot Safety 的权限；
- 在实机上自由运行生成代码的权限。

## 10. 推荐实施顺序

### 第一阶段：自动发现

实现：

- Skill Package；
- Skill Registry；
- Controller Skill Advertisement；
- `skills.search/inspect/run`；
- Agent 渐进式加载。

所有 Skill 仍人工创建和发布。

### 第二阶段：可观测和评测

增加：

- SkillRun Episode；
- 统一结果和失败分类；
- Evaluation Suite；
- Skill 版本对比；
- Hub SkillRun 观测。

### 第三阶段：自动提出候选

支持：

- 从重复成功轨迹提取 Composite Skill；
- 自动修改参数；
- 自动改进 `SKILL.md`；
- 自动生成测试；
- 候选版本注册。

仍然人工决定晋升。

### 第四阶段：受控自动进化

增加：

- 仿真批量评测；
- Shadow；
- Canary；
- 自动回滚；
- 基于统计门槛的晋升建议；
- Teleop 演示到 Policy 训练流水线。

最推荐从“程序性 Skill 自动发现和组合”开始，不要一开始就做 VLA 权重在线自学习。前者容易解释、评测和回滚，也最能快速提高 Agent 的任务完成能力。
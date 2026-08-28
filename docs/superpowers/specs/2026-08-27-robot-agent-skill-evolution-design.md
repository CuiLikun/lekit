# Robot Agent and Evolving Skills Design

## Status

This document records the proposed architecture for adding a task-understanding Robot Agent, automatically discovered Skills, and a controlled Skill-evolution loop on top of Lekit's Hub-managed robot control system. It is a design record, not an implementation plan.

The design extends the existing [Control Hub design](2026-08-27-control-hub-design.md). The Hub remains the sole authority for motion control, Controllers remain the producers of real-time actions, and Robots retain local HOLD and SAFETY authority.

## Goal

Add an Agent that can:

- understand a high-level human goal;
- build and revise a structured Plan;
- discover compatible Skills automatically;
- invoke Code-as-Policy, VLA/WAM, and Human Teleop capabilities through a common task-level Interface;
- observe execution and recover, replan, or escalate to a human;
- learn reusable procedures from successful and failed executions;
- propose and evaluate new Skill versions without allowing unvalidated behavior to control hardware.

## Non-goals

- Placing an LLM in the real-time Robot action loop.
- Letting an Agent mint, forge, or bypass Control Handles.
- Letting generated code directly access a Robot SDK, CAN interface, or action socket.
- Automatically changing Robot safety limits, Handle validation, watchdogs, or emergency-stop behavior.
- Mutating a Stable Skill in place while it is running.
- Promoting a newly generated policy directly to unrestricted hardware execution.

## Design principles

1. **Separate cognition from control.** The Agent reasons at task time scales; Controllers and Robots operate the real-time loop.
2. **A Skill may be backed by a Controller, but it is not a Controller.** A Skill is a task-level capability. A Controller is an action-producing execution resource.
3. **The Hub remains the motion authority.** Agent and Skill decisions never replace Handle scheduling, fencing, expiry, or local Robot safety.
4. **Discovery is progressive.** The Agent initially sees compact Skill metadata and loads full instructions or schemas only after selecting candidates.
5. **Evolution produces immutable candidates.** New behavior always creates a new version with lineage and evidence.
6. **Promotion is evidence-based.** Static checks, replay, simulation, shadow execution, and restricted hardware canaries precede Stable status.
7. **Safety does not evolve automatically.** Learning may improve procedures and policies but cannot weaken deterministic safety enforcement.
8. **Every decision is observable.** A Mission must be traceable through its Plan, Skill Runs, Controllers, Handles, action stream sessions, and outcomes.

## Domain language

The following terms extend the existing Robot control language.

### Agent

A Node that interprets goals, constructs Plans, selects Skills, supervises execution, and replans. An Agent never publishes Robot actions and never holds a Control Handle.

### Mission

An end-to-end user objective, such as "place the red block in the left tray." A Mission owns one or more Plans and has a terminal outcome.

### Plan

A versioned, structured set of Steps with dependencies, preconditions, expected effects, failure policies, and completion criteria. Replanning creates a new Plan revision rather than rewriting execution history.

### Step

One planned invocation of a Skill within a Plan.

### Skill

An Agent-visible task capability with structured inputs, outputs, requirements, risk metadata, and execution semantics.

### Agent Skill

Procedural instructions that teach an Agent when and how to perform a class of work. It is normally represented by `SKILL.md`, optional references, and scripts.

### Control Skill

A Skill that requires exclusive Robot motion authority and is executed by a Controller while that Controller holds a valid Control Handle.

### Composite Skill

A Skill that coordinates other Skills, for example `locate-object -> approach-object -> grasp-object -> place-object`. A Composite Skill does not require its own Handle unless its implementation also publishes actions.

### Skill Provider

A Node or local Adapter that can execute one or more Skills. A Controller is the normal Skill Provider for a Control Skill. A non-motion perception or planning capability may use a provider that is not a Controller.

### Skill Run

One stateful execution of a Skill for a Mission Step. A Skill Run is distinct from the Control Handle and can exist before control is granted or after control is released.

### Skill Candidate

An immutable, non-Stable Skill version proposed by a human, an Agent, an extraction process, or a training pipeline.

### Promotion

The evidence-based transition of a Skill version to a higher trust stage. Promotion changes lifecycle status; it never changes the version's content.

## Two control loops

The system contains two loops with different time scales and authorities.

```text
Cognitive loop, typically seconds
Mission -> Plan -> Skill -> Skill Event -> Replan or Complete

Real-time loop, typically 10-100 Hz
Controller -> Action Envelope -> Robot -> Observation
```

The cognitive loop may pause, retry, query a model, or wait for a human. The real-time loop must remain bounded, latest-only, independently supervised, and safe if the Agent disappears.

## Architecture

```text
User / external Agent
        |
        v
+-------------------------------+
| Agent Node                    |
| goal understanding, planning, |
| Skill selection, supervision  |
+---------------+---------------+
                | task-level calls and events
                v
+-------------------------------+
| Agent Gateway + Skill Runtime |
| schemas, policy, approvals,   |
| Skill Runs, cancellation      |
+---------+------------+--------+
          |            |
          |            +-----------------------+
          v                                    v
+-------------------+               +----------------------+
| Skill Registry    |               | Evolution Engine     |
| packages, live    |               | episodes, candidates,|
| providers, trust, |               | evals, promotion     |
| versions, evidence|               +----------------------+
+---------+---------+
          |
          | assignment intent and execution correlation
          v
+-------------------+     Control Handle     +-------------------+
| Hub               |----------------------->| Controller        |
| scheduling, audit,|                         | Code/VLA/Teleop   |
| revoke, force HOLD|                         +---------+---------+
+-------------------+                                   |
                                                        | direct actions
                                                        v
                                              +-------------------+
                                              | Robot             |
                                              | fencing, limits,  |
                                              | HOLD and SAFETY   |
                                              +-------------------+
```

### Agent Node

The Agent Node owns Mission reasoning and Plan revisions. It consumes compact Robot state, Skill metadata, Skill Events, and human input. It does not consume every real-time action or raw observation unless a specific reasoning step requests a sampled artifact.

The Agent may be implemented with Codex, Claude, Hermes, an OpenAI Agents SDK application, or another model runtime. Agent-framework details stay behind an Agent Adapter seam.

### Agent Gateway

The Agent Gateway translates external Agent tool calls into typed Skill Runtime operations. It is the policy enforcement point for Agent identity, Skill allowlists, Mission scope, risk class, quotas, and approval tokens.

MCP is an appropriate first Adapter for Agent-facing, low-rate tools such as:

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

MCP and A2A are not real-time Robot action transports. A2A may later expose the Agent itself to external agents without changing Skill or control semantics.

### Skill Runtime

The Skill Runtime is a deep module that hides provider selection, parameter validation, approval checks, Skill Run state, cancellation, timeout behavior, Hub correlation, and result normalization behind a small Interface.

```python
class SkillRuntime:
    async def start(self, skill: str, request: SkillRequest) -> SkillRunRef: ...
    async def cancel(self, run: SkillRunRef) -> None: ...
    async def watch(self, run: SkillRunRef) -> AsyncIterator[SkillEvent]: ...
```

An Agent does not call `take_over`, `publish`, `send_action`, or `hand_over` through this Interface.

### Skill Registry

The Skill Registry owns definitions, immutable versions, package digests, signatures, provider advertisements, lifecycle status, compatibility metadata, and evaluation evidence. It is not the authority for motion.

The Hub may cache the subset needed for live compatibility and observability, but Skill package management remains outside the Hub's core scheduling implementation.

### Evolution Engine

The Evolution Engine consumes completed Skill Runs and training or evaluation artifacts. It can identify opportunities, create candidates, run evaluation stages, and recommend promotion or rollback. It cannot grant a Handle or weaken safety policy.

## Skill package format

Use the open Agent Skills layout for model-facing instructions and add a robot-specific machine manifest.

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

`SKILL.md` contains discovery metadata and procedural guidance. `skill.yaml` contains deterministic system metadata that must not be inferred from prose.

Example manifest:

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

The manifest schema must reject unknown safety-relevant fields, invalid version relationships, missing digests, contradictory control requirements, and unsupported capability identifiers.

## Automatic discovery

### Discovery sources

The Registry reconciles four sources:

1. Repository Skills under a project-controlled directory.
2. User-installed Skills under a user-controlled directory.
3. Live Skill advertisements from registered Providers.
4. Signed packages from configured remote registries.

Generated candidates enter a separate quarantine source and are never treated as installed Stable packages.

### Provider advertisement

A Provider advertises versioned executable capabilities through registration and heartbeat metadata:

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

A package can remain installed while all Providers are offline. Definition availability and execution availability are separate states.

### Progressive discovery

The Agent initially receives only compact metadata:

```text
skill ID and version
description
kind
required capabilities
risk level
lifecycle status
live availability
```

The Agent loads full instructions, schemas, references, and provider details only after selecting a small candidate set. This follows the progressive-disclosure pattern in the Agent Skills specification and avoids filling the model context with unused capabilities.

### Search and ranking

Skill selection uses two stages.

First, deterministic hard filtering excludes Skills when:

- required Robot capabilities are missing;
- required observations are unavailable;
- no compatible Provider is online;
- action schema or control mode is incompatible;
- lifecycle status is not permitted in the requested environment;
- package signature, digest, or dependency validation fails;
- risk exceeds the Agent's authority or no required approval exists;
- Robot, Controller, or Hub state prohibits assignment.

Second, semantic ranking orders the remaining Skills using task relevance, scene tags, historical success, intervention rate, execution time, resource cost, recent stability, and operator preference.

The LLM chooses among eligible candidates. It never decides whether an ineligible Skill is authorized to run.

## Mission execution

```text
1. Agent creates or revises a typed Plan.
2. Agent searches and inspects compatible Skills.
3. Agent starts a Skill Run for one Step.
4. Skill Runtime validates input, risk, and approvals.
5. Skill Runtime resolves a live Provider.
6. Hub checks compatibility and assigns a new Control Handle when required.
7. The selected Controller receives the Handle and invokes take_over(handle).
8. Controller actions travel directly to the Robot.
9. Robot independently enforces Handle, fencing, watchdog, workspace, and safety checks.
10. Provider emits low-rate structured Skill Events.
11. On completion or failure, Controller invokes hand_over(handle).
12. Skill Runtime returns a normalized Skill Result.
13. Agent advances, retries, replans, or escalates to Human Teleop.
```

The Agent sees a `skill_run_id`, not a reusable motion authorization.

## Skill Run model

Recommended lifecycle:

```text
REQUESTED
  -> WAITING_FOR_APPROVAL
  -> WAITING_FOR_PROVIDER
  -> WAITING_FOR_CONTROL
  -> RUNNING
  -> SUCCEEDED / FAILED / CANCELLED / PREEMPTED
```

Important invariants:

- one Skill Run may use multiple sequential Handles after explicit recovery, but a Handle belongs to one assignment only;
- losing a Handle immediately stops the action stream and changes the Skill Run out of `RUNNING`;
- terminal Skill Runs never resume; recovery creates a new attempt with explicit lineage;
- completion requires evidence, not only a Provider declaration;
- cancelling a Skill Run must revoke or release any active Handle and wait for Robot HOLD confirmation.

## Control Skill Adapters

### Code-as-Policy

Prefer Code-as-Plan first. Generated code calls typed task primitives and composes existing Skills without direct Robot access.

If generated code must continuously produce actions, it runs in a restricted sandbox behind a Controller Adapter. It receives bounded observations, publishes only validated actions, and follows the normal Handle lifecycle. It has no direct access to CAN, Robot SDKs, Hub storage, or unrestricted network resources.

### VLA/WAM

A VLA/WAM Provider receives a goal and declared observations, performs inference, and publishes actions as a Controller. It reports low-rate progress, confidence, recoverability, phase, and termination evidence to the Skill Runtime. The Agent does not process its high-rate action stream.

### Human Teleop

Human Teleop is an asynchronous Control Skill backed by a Teleop Controller. Starting it requests an operator, waits for acceptance, coordinates revocation of the current autonomous Handle, waits for Robot HOLD, and then grants a fresh Handle to Teleop. Ending the session performs hand-over and produces a structured result or demonstration artifact.

Human Teleop is both an explicit Skill and the preferred escalation path when autonomous confidence, tracking, or recovery is insufficient.

## Observability and correlation

The following identity chain is preserved across all events:

```text
mission_id
  -> plan_id and revision
     -> step_id
        -> skill_run_id and attempt
           -> provider_id and provider_session_id
              -> handle_id and fencing_token
                 -> stream_session_id
```

The Hub UI should expose current Mission, Agent, Skill, Provider, Controller, Robot, Handle state, Robot control state, progress, risk, approval, metrics, and takeover history.

Persist structured plans, tool calls, selected Skill rationale summaries, approvals, results, and evaluation evidence. Do not depend on or attempt to persist hidden model chain-of-thought.

## Episode recording

Every Skill Run produces an Episode record containing at least:

- Mission, Plan, Step, Skill, version, Provider, Controller model, Robot, and configuration identities;
- input, constraints, preconditions, and environment labels;
- Handle and stream lifecycle summaries;
- references to sampled observations, trajectories, datasets, or video when enabled;
- requested and applied control statistics;
- completion evidence and normalized outcome;
- failure category and recoverability;
- safety events, watchdog events, and limit interventions;
- human corrections and takeover intervals;
- duration, latency, smoothness, force, success, and task-specific metrics.

High-rate payload retention is policy-controlled. The Episode may reference an external dataset rather than duplicating every frame in Hub storage.

## Skill evolution

Evolution occurs in three increasingly risky layers.

### Runtime adaptation

A running Skill may adjust declared parameters such as speed, force, visual threshold, retry count, or target lead only within immutable manifest limits. Runtime adaptation does not create new executable code or weaken safety.

### Procedural evolution

The Evolution Engine may improve `SKILL.md`, Plan templates, Skill composition, recovery procedures, parameter-selection logic, or tests. This is the recommended first form of automatic evolution because it is explainable, replayable, and easy to roll back.

### Policy evolution

Training pipelines may update VLA/WAM weights, perception models, Code-as-Policy implementations, or control parameters using Episode and Teleop demonstration data. Policy evolution requires dataset lineage, training provenance, model digests, dedicated evaluation, shadow execution, and canary hardware evidence.

Robot safety rules, hard workspace limits, fencing checks, stale-action watchdogs, emergency stop, and administrative authority are outside all evolution mechanisms.

## Candidate generation

Candidates can originate from:

- repeated successful sequences promoted into a Composite Skill;
- clusters of recurrent failures that suggest a recovery procedure;
- human corrections and Teleop demonstrations;
- parameter optimization over replay or simulation;
- an Agent-proposed instruction or workflow revision;
- generated Code-as-Plan or sandboxed Code-as-Controller;
- a policy training or distillation job.

Each candidate records:

```text
skill ID and immutable candidate version
parent version or parent set
authoring mechanism and identity
source Episodes or dataset versions
content and artifact digests
declared behavior change
expected improvement
evaluation suite version
creation time
```

No candidate may replace an active Stable version in place.

## Promotion lifecycle

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

### Static validation

Validate schemas, package layout, dependencies, signatures, digests, version lineage, allowed tools, code quality, parameter ranges, capability identifiers, and contradictory control declarations.

### Replay validation

Run the candidate against historical Episodes. Compare success, failure categories, action or decision divergence, intervention rate, latency, and task-specific metrics with the current Stable version.

### Simulation validation

Exercise deterministic scenarios and randomized environments. Include collision, reachability, stale observations, missing objects, tracking loss, process restart, provider disconnect, Handle revoke, timeout, and recovery scenarios.

### Shadow execution

The candidate receives sampled live observations and generates decisions or actions but receives no Control Handle and cannot publish to the Robot. Compare candidate output with Stable Controller output, human actions, and measured Robot state.

### Canary hardware execution

Canary execution requires explicit authorization and restrictive limits: low speed, reduced workspace, limited objects and tasks, tighter watchdogs, cleared workspace, reachable emergency stop, active telemetry, and immediate Human Teleop or administrator preemption.

### Stable promotion

Promotion policy uses declared thresholds such as minimum scenario coverage, success rate, zero safety violations, maximum intervention rate, bounded latency regression, and sufficient trial count. Automatic evaluation may recommend promotion; the initial system requires a human or controlled release process to authorize Stable status.

## Rollback and revocation

- Stable aliases resolve to immutable versions and can atomically move back to a previous version.
- Active Runs retain their resolved version unless Hub revokes control for safety.
- A newly discovered critical fault can revoke a version immediately and force affected Robots to HOLD.
- Revocation propagates to Agent search results, Registry resolution, Hub compatibility, and Provider availability.
- Historical Episodes continue referencing the exact revoked version for audit and retraining.

## Trust and supply-chain controls

- Remote packages require configured trust roots and digest verification.
- Generated candidates remain quarantined until validation.
- Skill definition trust and live Provider trust are verified independently.
- Provider advertisements cannot upgrade lifecycle status or replace Registry evidence.
- Approval tokens bind actor, Mission, Skill Run, Robot, risk scope, and expiry.
- Agent-generated scripts run with minimal filesystem, network, process, and time permissions.
- Model-level guardrails supplement but never replace deterministic Hub and Robot checks.

## Control preemption

Recommended effective authority order:

```text
Robot local SAFETY
  > administrator emergency stop or force HOLD
  > Human Teleop
  > approved recovery Controller
  > VLA/WAM or other autonomous Controller
```

Preemption always follows:

```text
revoke old Handle
-> stop old action stream
-> Robot confirms HOLD
-> terminate or preempt old Skill Run
-> mint new Handle
-> new Controller takes over
```

No priority rule permits two concurrently valid Controllers for one Robot.

## Evaluation metrics

Metrics are Skill-specific but share a common envelope:

- task success and completion evidence;
- safety violation count;
- human intervention and takeover rate;
- recovery success and retry count;
- total duration and phase durations;
- perception and inference latency;
- action freshness and control-loop misses;
- path length, jerk, smoothness, force, and energy where available;
- workspace or limit intervention count;
- generalization across object, pose, lighting, Robot, and environment tags;
- regression relative to the current Stable version.

Evaluation thresholds must specify sample count and scenario coverage. A high average score from a narrow scenario set is not sufficient for promotion.

## Failure handling

- Agent loss: active Handle expires or is revoked; Robot enters HOLD; Mission becomes suspended.
- Skill Runtime loss: Hub and Robot safety remain active; Runs are reconciled after restart and motion does not resume automatically.
- Registry loss: already resolved Runs may finish under policy, but no new version resolution or promotion occurs.
- Provider loss: Robot stale watchdog enters HOLD; Handle terminates; Skill Run fails or becomes recoverable according to policy.
- Hub loss: no Handle renewal; Robot enters HOLD independently.
- Robot fault: Hub revokes assignment; Skill Run records a Robot failure and Agent may only choose non-motion diagnosis or approved recovery.
- Evolution Engine loss: normal Stable execution continues; candidate work pauses without affecting control.

## Testing strategy

### Pure model tests

- Manifest validation and version lineage.
- Skill Run transition reducer.
- Candidate promotion transition reducer.
- Compatibility hard filters and ranking features.
- Approval and risk policy.
- Correlation and audit models.

### Integration tests

- Local and remote Skill discovery reconciliation.
- Provider registration, heartbeat, offline transition, and replacement session.
- Agent Gateway schema validation and permission denial.
- Skill Runtime with MemoryRuntime and MockRobot.
- Handle assignment, take-over, cancellation, hand-over, expiry, and revoke.
- Stable rollback and candidate revocation.

### Evaluation tests

- Historical Episode replay.
- Deterministic simulation scenarios.
- Randomized simulation batches.
- Shadow output comparison.
- Canary policy checks without activating real hardware in automated CI.

Automated tests must never open Piper CAN, connect CloudXR, or move physical hardware.

## Recommended delivery sequence

### Phase 1: Discovery and execution

- Add immutable Skill models and package validation.
- Add local Registry and Provider advertisements.
- Add progressive `search`, `inspect`, `run`, `watch`, and `cancel` operations.
- Add Agent Gateway and a framework-neutral Agent Adapter.
- Wrap Human Teleop, one VLA/Policy path, and one Composite Skill.
- Correlate Skill Runs with Hub Handles and Robot state.

All Skills are still human-authored and manually published.

### Phase 2: Episodes and evaluation

- Record normalized Skill Episodes.
- Add failure taxonomy and task-specific metrics.
- Add replay and simulation evaluation suites.
- Add version comparison, evidence views, and rollback.

### Phase 3: Candidate generation

- Extract repeated successful sequences into Composite Skill candidates.
- Propose parameter and procedural revisions.
- Generate candidate tests and expected behavior changes.
- Register candidates in quarantine with complete provenance.

Promotion remains manual.

### Phase 4: Controlled evolution

- Add policy training and Teleop demonstration pipelines.
- Add shadow execution and comparison.
- Add restricted canary hardware workflow.
- Add statistical promotion recommendations and automatic rollback triggers.

## External standards

- [Agent Skills specification](https://agentskills.io/specification) for portable `SKILL.md` packages and progressive disclosure.
- [Model Context Protocol](https://modelcontextprotocol.io/specification/) for low-rate Agent-facing tools and context.
- [Agent2Agent Protocol](https://a2a-protocol.org/latest/specification) for optional external Agent discovery and long-running task interaction.

These standards adapt the Agent-facing seams. Lekit's Runtime Adapter and direct Controller-to-Robot action transport remain independent so the communication middleware can change without changing Skill, Hub, or Robot safety semantics.

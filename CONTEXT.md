# Robot Control Context

This context describes how independently running controllers obtain exclusive, observable control of robots through a central scheduler.

## Language

**Node**:
An independently running participant that is discoverable by the Hub and reports its current status.

**Controller**:
A Node that produces actions for a Robot while holding that Robot's valid Control Handle. Teleop and Policy are Controller types.
_Avoid_: Input source, control source

**Teleop**:
A human-driven Controller that turns operator input into Robot actions.
_Avoid_: Teleop node

**Policy**:
An autonomous Controller that turns observations or task state into Robot actions.
_Avoid_: Policy node

**Robot**:
A Node that owns a physical or simulated robot and accepts actions only from the Controller holding its valid Control Handle.
_Avoid_: Robot node, target

**Hub**:
The authority that discovers Nodes, distributes and reclaims Control Handles, and maintains the observable current control state.
_Avoid_: Control plane, broker, coordinator

**Control Handle**:
A short-lived, exclusive, and revocable authorization assigned by the Hub to one Controller for one Robot.
_Avoid_: Permanent robot reference, connection handle

**Take Over**:
The act of a Controller activating an assigned Control Handle to obtain control of its Robot.
_Avoid_: Connect, attach

**Hand Over**:
The act of a Controller relinquishing an active Control Handle so the Hub can reclaim it.
_Avoid_: Disconnect, detach

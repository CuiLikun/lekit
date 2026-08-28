"""Transactional SQLite persistence for Control Hub state."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .handles import (
    HandleRecord,
    HandleTransition,
    InvalidHandleTransition,
    StaleTransition,
    transition_handle,
)
from .model import (
    TERMINAL_HANDLE_STATES,
    ControlHandle,
    HandleState,
    HubSnapshot,
    NodeDescriptor,
    NodeReport,
    NodeRole,
)

_TERMINAL_SQL = ",".join(repr(state.value) for state in sorted(TERMINAL_HANDLE_STATES))


class HubStore:
    """Own SQLite state transitions with immediate, all-or-nothing transactions."""

    def __init__(self, path: Path | str) -> None:
        self._connection = sqlite3.connect(Path(path), isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        row = self._connection.execute(
            "SELECT epoch FROM epochs ORDER BY started_at_ns DESC, rowid DESC LIMIT 1"
        ).fetchone()
        self._current_epoch = row["epoch"] if row is not None else self.begin_epoch(started_at_ns=0)

    def begin_epoch(self, *, started_at_ns: int) -> str:
        """Persist and select a new Hub process epoch."""
        if isinstance(started_at_ns, bool) or started_at_ns < 0:
            raise ValueError("started_at_ns must be a non-negative integer")
        epoch = str(uuid.uuid4())
        with self._transaction():
            self._connection.execute(
                "INSERT INTO epochs(epoch, started_at_ns) VALUES (?, ?)", (epoch, started_at_ns)
            )
        self._current_epoch = epoch
        return epoch

    def upsert_node(self, descriptor: NodeDescriptor, *, seen_at_ns: int) -> None:
        """Persist the latest descriptor and liveness timestamp for a Node."""
        if isinstance(seen_at_ns, bool) or seen_at_ns < 0:
            raise ValueError("seen_at_ns must be a non-negative integer")
        with self._transaction():
            self._upsert_node(descriptor, seen_at_ns=seen_at_ns)

    def create_assignment(
        self,
        robot: NodeDescriptor,
        controller: NodeDescriptor,
        *,
        now_ns: int,
        ttl_ns: int,
        action_schema: str | None = None,
        control_mode: str = "teleop",
    ) -> ControlHandle:
        """Atomically mint one exclusive, fenced assignment Handle."""
        if robot.role is not NodeRole.ROBOT or controller.role is not NodeRole.CONTROLLER:
            raise ValueError("assignment requires a robot and controller descriptor")
        if not robot.administratively_enabled or not controller.administratively_enabled:
            raise ValueError("assignment requires administratively enabled nodes")
        if isinstance(now_ns, bool) or now_ns < 0 or isinstance(ttl_ns, bool) or ttl_ns <= 0:
            raise ValueError("assignment timing must be positive")
        selected_schema = self._select_schema(robot, controller, action_schema)
        if control_mode not in robot.control_modes or control_mode not in controller.control_modes:
            raise ValueError("control mode is not supported by both nodes")
        if controller.action_endpoint is None:
            raise ValueError("controller action endpoint is required")

        with self._transaction(immediate=True):
            self._upsert_node(robot, seen_at_ns=now_ns)
            self._upsert_node(controller, seen_at_ns=now_ns)
            if self._has_nonterminal_handle("robot_id", robot.node_id) or self._has_nonterminal_handle(
                "controller_id", controller.node_id
            ):
                raise ValueError("exclusive non-terminal Handle already exists")
            self._connection.execute(
                """
                INSERT INTO robot_fencing(robot_id, fencing_token) VALUES (?, 1)
                ON CONFLICT(robot_id) DO UPDATE SET fencing_token = robot_fencing.fencing_token + 1
                """,
                (robot.node_id,),
            )
            fencing_token = self._connection.execute(
                "SELECT fencing_token FROM robot_fencing WHERE robot_id = ?", (robot.node_id,)
            ).fetchone()["fencing_token"]
            handle = ControlHandle(
                handle_id=str(uuid.uuid4()),
                hub_epoch=self._current_epoch,
                robot_id=robot.node_id,
                robot_session_id=robot.session_id,
                controller_id=controller.node_id,
                controller_session_id=controller.session_id,
                controller_action_endpoint=controller.action_endpoint,
                action_schema=selected_schema,
                control_mode=control_mode,
                fencing_token=fencing_token,
                issued_at_ns=now_ns,
                expires_at_ns=now_ns + ttl_ns,
            )
            correlation_id = f"assignment:{handle.handle_id}"
            record = HandleRecord(
                handle=handle,
                state=HandleState.ASSIGNED,
                transition_sequence=0,
                correlation_id=correlation_id,
                updated_at_ns=now_ns,
            )
            self._insert_handle(record)
            self._insert_transition(
                HandleTransition(
                    handle_id=handle.handle_id,
                    fencing_token=handle.fencing_token,
                    state=record.state,
                    transition_sequence=0,
                    correlation_id=correlation_id,
                    at_ns=now_ns,
                )
            )
            self._append_audit(
                event="assignment_created",
                at_ns=now_ns,
                actor=None,
                correlation_id=correlation_id,
                details={"handle_id": handle.handle_id, "fencing_token": fencing_token},
            )
        return handle

    def get_handle(self, handle_id: str) -> HandleRecord:
        """Return the current record for a Handle."""
        row = self._connection.execute("SELECT * FROM handles WHERE handle_id = ?", (handle_id,)).fetchone()
        if row is None:
            raise KeyError(handle_id)
        return self._record_from_row(row)

    def renew_handle(self, handle_id: str, *, expires_at_ns: int, at_ns: int) -> ControlHandle:
        """Extend a non-terminal Handle without changing its identity or fencing."""
        if isinstance(expires_at_ns, bool) or isinstance(at_ns, bool):
            raise ValueError("renewal timestamps must be integers")
        with self._transaction(immediate=True):
            record = self.get_handle(handle_id)
            if record.state in TERMINAL_HANDLE_STATES:
                raise InvalidHandleTransition("terminal Handle cannot be renewed")
            if at_ns >= record.handle.expires_at_ns:
                raise ValueError("expired Handle cannot be renewed")
            if expires_at_ns <= at_ns:
                raise ValueError("expires_at_ns must be after renewal time")
            if expires_at_ns <= record.handle.issued_at_ns:
                raise ValueError("expires_at_ns must be after issuance")
            self._connection.execute(
                "UPDATE handles SET expires_at_ns = ?, updated_at_ns = ? WHERE handle_id = ?",
                (expires_at_ns, at_ns, handle_id),
            )
            self._append_audit(
                event="handle_renewed",
                at_ns=at_ns,
                actor=None,
                correlation_id=record.correlation_id,
                details={"expires_at_ns": expires_at_ns, "handle_id": handle_id},
            )
        return self.get_handle(handle_id).handle

    def transition(
        self,
        handle_id: str,
        target: HandleState,
        transition_sequence: int,
        correlation_id: str,
        at_ns: int,
        reason: str | None = None,
    ) -> HandleRecord:
        """Atomically persist one desired-state transition and its audit trail."""
        with self._transaction(immediate=True):
            current = self.get_handle(handle_id)
            updated = transition_handle(
                current,
                target,
                transition_sequence=transition_sequence,
                correlation_id=correlation_id,
                at_ns=at_ns,
                reason=reason,
            )
            if updated is current:
                return current
            self._connection.execute(
                """
                UPDATE handles
                SET state = ?, transition_sequence = ?, correlation_id = ?, updated_at_ns = ?, reason = ?
                WHERE handle_id = ?
                """,
                (
                    updated.state.value,
                    updated.transition_sequence,
                    updated.correlation_id,
                    updated.updated_at_ns,
                    updated.reason,
                    handle_id,
                ),
            )
            self._insert_transition(
                HandleTransition(
                    handle_id=handle_id,
                    fencing_token=updated.handle.fencing_token,
                    state=updated.state,
                    transition_sequence=updated.transition_sequence,
                    correlation_id=updated.correlation_id,
                    at_ns=at_ns,
                    reason=reason,
                )
            )
            self._append_audit(
                event="handle_transitioned",
                at_ns=at_ns,
                actor=None,
                correlation_id=correlation_id,
                details={"handle_id": handle_id, "state": target.value},
            )
        return updated

    def invalidate_previous_epochs(self, current_epoch: str, *, at_ns: int) -> tuple[str, ...]:
        """Expire every live Handle belonging to an earlier Hub process epoch."""
        with self._transaction(immediate=True):
            rows = self._connection.execute(
                f"SELECT * FROM handles WHERE hub_epoch != ? AND state NOT IN ({_TERMINAL_SQL}) ORDER BY handle_id",
                (current_epoch,),
            ).fetchall()
            invalidated: list[str] = []
            for row in rows:
                current = self._record_from_row(row)
                updated = self._force_terminal_transition(
                    current,
                    HandleState.EXPIRED,
                    current.transition_sequence + 1,
                    f"epoch-invalidated:{current_epoch}",
                    at_ns,
                    "hub epoch changed",
                )
                self._write_transition(updated)
                invalidated.append(updated.handle.handle_id)
        return tuple(invalidated)

    def append_audit(
        self,
        *,
        event: str,
        at_ns: int,
        actor: str | None,
        correlation_id: str,
        details: Mapping[str, Any],
    ) -> None:
        """Append a compact, deterministic audit event."""
        with self._transaction():
            self._append_audit(
                event=event,
                at_ns=at_ns,
                actor=actor,
                correlation_id=correlation_id,
                details=details,
            )

    def save_snapshot(self, snapshot: HubSnapshot) -> None:
        """Replace the one persisted low-rate diagnostic snapshot."""
        payload = self._json(self._snapshot_data(snapshot))
        with self._transaction():
            self._connection.execute(
                """
                INSERT INTO snapshots(snapshot_id, snapshot_json) VALUES (1, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET snapshot_json = excluded.snapshot_json
                """,
                (payload,),
            )

    def list_history(self, *, limit: int = 200) -> tuple[Mapping[str, Any], ...]:
        """Return newest audit events first, with decoded deterministic JSON details."""
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        rows = self._connection.execute(
            """
            SELECT event, at_ns, actor, correlation_id, details_json, hub_epoch
            FROM audit_events ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(
            {
                "event": row["event"],
                "at_ns": row["at_ns"],
                "actor": row["actor"],
                "correlation_id": row["correlation_id"],
                "details": json.loads(row["details_json"]),
                "hub_epoch": row["hub_epoch"],
            }
            for row in rows
        )

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _create_schema(self) -> None:
        self._connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS epochs (
                epoch TEXT PRIMARY KEY,
                started_at_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                last_seen_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS robot_fencing (
                robot_id TEXT PRIMARY KEY REFERENCES nodes(node_id),
                fencing_token INTEGER NOT NULL CHECK(fencing_token > 0)
            );
            CREATE TABLE IF NOT EXISTS handles (
                handle_id TEXT PRIMARY KEY,
                hub_epoch TEXT NOT NULL REFERENCES epochs(epoch),
                robot_id TEXT NOT NULL REFERENCES nodes(node_id),
                robot_session_id TEXT NOT NULL,
                controller_id TEXT NOT NULL REFERENCES nodes(node_id),
                controller_session_id TEXT NOT NULL,
                controller_action_endpoint TEXT NOT NULL,
                action_schema TEXT NOT NULL,
                control_mode TEXT NOT NULL,
                fencing_token INTEGER NOT NULL CHECK(fencing_token > 0),
                issued_at_ns INTEGER NOT NULL,
                expires_at_ns INTEGER NOT NULL,
                state TEXT NOT NULL,
                transition_sequence INTEGER NOT NULL,
                correlation_id TEXT NOT NULL,
                updated_at_ns INTEGER NOT NULL,
                reason TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS unique_live_robot_handle
                ON handles(robot_id) WHERE state NOT IN ({_TERMINAL_SQL});
            CREATE UNIQUE INDEX IF NOT EXISTS unique_live_controller_handle
                ON handles(controller_id) WHERE state NOT IN ({_TERMINAL_SQL});
            CREATE TABLE IF NOT EXISTS handle_transitions (
                id INTEGER PRIMARY KEY,
                handle_id TEXT NOT NULL REFERENCES handles(handle_id),
                fencing_token INTEGER NOT NULL,
                state TEXT NOT NULL,
                transition_sequence INTEGER NOT NULL,
                correlation_id TEXT NOT NULL,
                at_ns INTEGER NOT NULL,
                reason TEXT,
                UNIQUE(handle_id, transition_sequence)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY,
                event TEXT NOT NULL,
                at_ns INTEGER NOT NULL,
                actor TEXT,
                correlation_id TEXT NOT NULL,
                details_json TEXT NOT NULL,
                hub_epoch TEXT NOT NULL REFERENCES epochs(epoch)
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY CHECK(snapshot_id = 1),
                snapshot_json TEXT NOT NULL
            );
            """
        )

    def _upsert_node(self, descriptor: NodeDescriptor, *, seen_at_ns: int) -> None:
        self._connection.execute(
            """
            INSERT INTO nodes(node_id, session_id, role, descriptor_json, last_seen_ns)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                session_id = excluded.session_id,
                role = excluded.role,
                descriptor_json = excluded.descriptor_json,
                last_seen_ns = excluded.last_seen_ns
            """,
            (
                descriptor.node_id,
                descriptor.session_id,
                descriptor.role.value,
                self._json(self._descriptor_data(descriptor)),
                seen_at_ns,
            ),
        )

    def _has_nonterminal_handle(self, column: str, node_id: str) -> bool:
        row = self._connection.execute(
            f"SELECT 1 FROM handles WHERE {column} = ? AND state NOT IN ({_TERMINAL_SQL}) LIMIT 1",
            (node_id,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _select_schema(robot: NodeDescriptor, controller: NodeDescriptor, requested: str | None) -> str:
        shared = tuple(schema for schema in controller.action_schemas if schema in robot.action_schemas)
        if requested is not None:
            if requested not in shared:
                raise ValueError("action schema is not supported by both nodes")
            return requested
        if not shared:
            raise ValueError("nodes have no shared action schema")
        return shared[0]

    def _insert_handle(self, record: HandleRecord) -> None:
        handle = record.handle
        self._connection.execute(
            """
            INSERT INTO handles(
                handle_id, hub_epoch, robot_id, robot_session_id, controller_id, controller_session_id,
                controller_action_endpoint, action_schema, control_mode, fencing_token, issued_at_ns,
                expires_at_ns, state, transition_sequence, correlation_id, updated_at_ns, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                handle.handle_id,
                handle.hub_epoch,
                handle.robot_id,
                handle.robot_session_id,
                handle.controller_id,
                handle.controller_session_id,
                handle.controller_action_endpoint,
                handle.action_schema,
                handle.control_mode,
                handle.fencing_token,
                handle.issued_at_ns,
                handle.expires_at_ns,
                record.state.value,
                record.transition_sequence,
                record.correlation_id,
                record.updated_at_ns,
                record.reason,
            ),
        )

    def _insert_transition(self, transition: HandleTransition) -> None:
        self._connection.execute(
            """
            INSERT INTO handle_transitions(
                handle_id, fencing_token, state, transition_sequence, correlation_id, at_ns, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.handle_id,
                transition.fencing_token,
                transition.state.value,
                transition.transition_sequence,
                transition.correlation_id,
                transition.at_ns,
                transition.reason,
            ),
        )

    def _write_transition(self, updated: HandleRecord) -> None:
        self._connection.execute(
            """
            UPDATE handles
            SET state = ?, transition_sequence = ?, correlation_id = ?, updated_at_ns = ?, reason = ?
            WHERE handle_id = ?
            """,
            (
                updated.state.value,
                updated.transition_sequence,
                updated.correlation_id,
                updated.updated_at_ns,
                updated.reason,
                updated.handle.handle_id,
            ),
        )
        self._insert_transition(
            HandleTransition(
                handle_id=updated.handle.handle_id,
                fencing_token=updated.handle.fencing_token,
                state=updated.state,
                transition_sequence=updated.transition_sequence,
                correlation_id=updated.correlation_id,
                at_ns=updated.updated_at_ns,
                reason=updated.reason,
            )
        )
        self._append_audit(
            event="handle_transitioned",
            at_ns=updated.updated_at_ns,
            actor=None,
            correlation_id=updated.correlation_id,
            details={"handle_id": updated.handle.handle_id, "state": updated.state.value},
        )

    @staticmethod
    def _force_terminal_transition(
        current: HandleRecord,
        target: HandleState,
        transition_sequence: int,
        correlation_id: str,
        at_ns: int,
        reason: str | None,
    ) -> HandleRecord:
        if transition_sequence < current.transition_sequence:
            raise StaleTransition("transition sequence regressed")
        if transition_sequence == current.transition_sequence:
            if target is current.state and correlation_id == current.correlation_id:
                return current
            raise StaleTransition("sequence already used by another transition")
        if target not in TERMINAL_HANDLE_STATES:
            raise InvalidHandleTransition(f"{current.state} -> {target} is forbidden")
        return HandleRecord(
            handle=current.handle,
            state=target,
            transition_sequence=transition_sequence,
            correlation_id=correlation_id,
            updated_at_ns=at_ns,
            reason=reason,
        )

    def _append_audit(
        self,
        *,
        event: str,
        at_ns: int,
        actor: str | None,
        correlation_id: str,
        details: Mapping[str, Any],
    ) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("event must not be empty")
        if not isinstance(correlation_id, str) or not correlation_id:
            raise ValueError("correlation_id must not be empty")
        if actor is not None and (not isinstance(actor, str) or not actor):
            raise ValueError("actor must be non-empty when present")
        self._connection.execute(
            """
            INSERT INTO audit_events(event, at_ns, actor, correlation_id, details_json, hub_epoch)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event, at_ns, actor, correlation_id, self._json(details), self._current_epoch),
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> HandleRecord:
        return HandleRecord(
            handle=ControlHandle(
                handle_id=row["handle_id"],
                hub_epoch=row["hub_epoch"],
                robot_id=row["robot_id"],
                robot_session_id=row["robot_session_id"],
                controller_id=row["controller_id"],
                controller_session_id=row["controller_session_id"],
                controller_action_endpoint=row["controller_action_endpoint"],
                action_schema=row["action_schema"],
                control_mode=row["control_mode"],
                fencing_token=row["fencing_token"],
                issued_at_ns=row["issued_at_ns"],
                expires_at_ns=row["expires_at_ns"],
            ),
            state=HandleState(row["state"]),
            transition_sequence=row["transition_sequence"],
            correlation_id=row["correlation_id"],
            updated_at_ns=row["updated_at_ns"],
            reason=row["reason"],
        )

    @staticmethod
    def _descriptor_data(descriptor: NodeDescriptor) -> dict[str, Any]:
        return {field.name: getattr(descriptor, field.name) for field in fields(NodeDescriptor)}

    @classmethod
    def _snapshot_data(cls, snapshot: HubSnapshot) -> dict[str, Any]:
        return {
            "version": snapshot.version,
            "hub_epoch": snapshot.hub_epoch,
            "generated_at_ns": snapshot.generated_at_ns,
            "nodes": [
                {
                    "descriptor": cls._descriptor_data(node.descriptor),
                    "online": node.online,
                    "last_seen_ns": node.last_seen_ns,
                    "report": cls._report_data(node.report) if node.report is not None else None,
                }
                for node in snapshot.nodes
            ],
            "controls": [
                {field.name: getattr(control, field.name) for field in fields(control)}
                for control in snapshot.controls
            ],
            "alerts": snapshot.alerts,
        }

    @staticmethod
    def _report_data(report: NodeReport) -> dict[str, Any]:
        return {field.name: getattr(report, field.name) for field in fields(NodeReport)}

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value, default=HubStore._json_default, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, (HandleState, NodeRole)):
            return value.value
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

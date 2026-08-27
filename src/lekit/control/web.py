"""Browser-facing, synchronous Hub service.

This module deliberately knows only the public :class:`Hub` methods.  Action
transport and persistence remain owned by their respective adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib.resources import files
from typing import Annotated, Any

import anyio
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.websockets import WebSocketState

from .hub import ControlConflict, Hub, IncompatibleNode, NodeUnavailable

_MAX_ID_LENGTH = 128
_MAX_REASON_LENGTH = 512
_MAX_MODE_LENGTH = 64

Identifier = Annotated[str, Field(min_length=1, max_length=_MAX_ID_LENGTH)]
Reason = Annotated[str, Field(min_length=1, max_length=_MAX_REASON_LENGTH)]
OperatorHeader = Annotated[str | None, Header(max_length=_MAX_ID_LENGTH)]


class AssignRequest(BaseModel):
    """An operator's request to make one compatible assignment."""

    robot: Identifier
    controller: Identifier
    control_mode: Annotated[str, Field(min_length=1, max_length=_MAX_MODE_LENGTH)] = "teleop"

    @field_validator("robot", "controller", "control_mode")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ReasonRequest(BaseModel):
    """A human-readable reason required for disruptive operator actions."""

    reason: Reason

    @field_validator("reason")
    @classmethod
    def _reject_blank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def _operator(request: Request, supplied: str | None) -> str:
    """Use the explicit operator identity, falling back to the request peer."""
    if supplied is not None and supplied.strip():
        return supplied.strip()
    return request.client.host if request.client is not None else "unknown"


def _json(value: Any, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=jsonable_encoder(value), status_code=status_code)


def _hub_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, NodeUnavailable) and str(error).startswith(
        ("unknown Robot ", "unknown Controller ")
    ):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, ControlConflict):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (IncompatibleNode, NodeUnavailable, ValueError)):
        return HTTPException(status_code=422, detail=str(error))
    raise error


def _call(operation: Callable[[], Any]) -> Any:
    """Translate public Hub domain rejections without exposing internals."""
    try:
        return operation()
    except Exception as error:
        raise _hub_error(error) from error


def create_hub_app(hub: Hub) -> FastAPI:
    """Create the small HTTP/WebSocket facade for one already-running Hub."""
    app = FastAPI(title="Lekit Hub", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("lekit.control").joinpath("hub.html").read_text(encoding="utf-8")

    @app.get("/api/snapshot")
    def snapshot() -> JSONResponse:
        return _json(_call(hub.get_snapshot))

    @app.get("/api/nodes")
    def nodes() -> JSONResponse:
        return _json(_call(hub.list_nodes))

    @app.get("/api/history")
    def history(limit: Annotated[int, Query(ge=1, le=2_000)] = 200) -> JSONResponse:
        return _json(_call(lambda: hub.list_history(limit=limit)))

    @app.post("/api/assign", status_code=201)
    def assign(
        payload: AssignRequest,
        request: Request,
        x_operator: OperatorHeader = None,
    ) -> JSONResponse:
        value = _call(
            lambda: hub.assign(
                payload.robot,
                payload.controller,
                control_mode=payload.control_mode,
                actor=_operator(request, x_operator),
            )
        )
        return _json(value, status_code=201)

    @app.post("/api/handles/{handle_id}/take-over", status_code=204)
    def take_over(
        handle_id: Identifier,
        request: Request,
        x_operator: OperatorHeader = None,
    ) -> Response:
        _call(lambda: hub.request_take_over(handle_id, actor=_operator(request, x_operator)))
        return Response(status_code=204)

    @app.post("/api/handles/{handle_id}/hand-over", status_code=204)
    def hand_over(
        handle_id: Identifier,
        request: Request,
        x_operator: OperatorHeader = None,
    ) -> Response:
        _call(lambda: hub.request_hand_over(handle_id, actor=_operator(request, x_operator)))
        return Response(status_code=204)

    @app.post("/api/handles/{handle_id}/renew")
    def renew(
        handle_id: Identifier,
        request: Request,
        x_operator: OperatorHeader = None,
    ) -> JSONResponse:
        return _json(_call(lambda: hub.renew(handle_id, actor=_operator(request, x_operator))))

    @app.post("/api/handles/{handle_id}/revoke", status_code=204)
    def revoke(
        handle_id: Identifier,
        payload: ReasonRequest,
        request: Request,
        x_operator: OperatorHeader = None,
    ) -> Response:
        _call(lambda: hub.revoke(handle_id, reason=payload.reason, actor=_operator(request, x_operator)))
        return Response(status_code=204)

    @app.post("/api/robots/{robot_id}/force-hold", status_code=204)
    def force_hold(
        robot_id: Identifier,
        payload: ReasonRequest,
        request: Request,
        x_operator: OperatorHeader = None,
    ) -> Response:
        _call(lambda: hub.force_hold(robot_id, reason=payload.reason, actor=_operator(request, x_operator)))
        return Response(status_code=204)

    @app.websocket("/ws")
    async def watch(websocket: WebSocket) -> None:
        await websocket.accept()
        after_version = -1
        try:
            while websocket.client_state is WebSocketState.CONNECTED:
                snapshot_value = await anyio.to_thread.run_sync(
                    partial(hub.watch, after_version=after_version, timeout_s=1.0)
                )
                encoded = jsonable_encoder(snapshot_value)
                version = encoded.get("version") if isinstance(encoded, dict) else None
                if not isinstance(version, int) or version <= after_version:
                    continue
                await websocket.send_json(encoded)
                after_version = version
        except WebSocketDisconnect:
            return

    return app


__all__ = ["create_hub_app"]

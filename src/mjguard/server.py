"""MCP surface. Every acting tool goes through Executor, which plans and checks before anything moves."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecurityMiddleware
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from . import __version__
from .events import Event
from .executor import JOG_MAX_JOINT, JOG_MAX_MM, Executor

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

INSTRUCTIONS = """\
Controls the robot arm "{name}". Coordinates are millimetres in the world frame of its model; z is the height above
the floor or table. Angles are degrees. Call get_limits first and plan inside them. Every motion is checked in a
MuJoCo model before it runs; a refused motion returns ok=false with a reason code, a message and, when one exists,
a nearby point that would pass. Use that to correct yourself instead of retrying the same command. stop() halts
the current motion; it is a software stop.

Motion tools take preview=true to plan and check without moving: the result has executed=false and a plan_id, and
execute(plan_id) runs that plan once if the arm has not moved since. {confirmation}pause() halts and keeps the rest
of the motion; resume() re-checks and continues it.
"""
CONFIRMATION = """This server requires confirmation: every motion tool only previews (executed=false, nothing moved) \
until execute(plan_id) is called. """

SSE_HEARTBEAT_S = 15.0
SSE_QUEUE = 1000

Mm = Annotated[float, Field(allow_inf_nan=False, ge=-5000, le=5000)]
Speed = Annotated[
    float, Field(allow_inf_nan=False, ge=0.05, le=1.0, description="Fraction of the maximum joint speed.")
]
Preview = Annotated[
    bool,
    Field(description="Plan and check only; nothing moves. The result's plan_id can be passed to execute."),
]
JogMm = Annotated[float, Field(allow_inf_nan=False, ge=-JOG_MAX_MM, le=JOG_MAX_MM)]
PoseName = Annotated[str, Field(pattern=r"^[\w-]{1,40}$", description="1-40 letters, digits, '_' or '-'.")]


def tool_result(payload: dict[str, Any]) -> CallToolResult:
    """Structured result; refusals set isError so clients surface them to the model."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structuredContent=payload,
        isError=payload.get("ok") is not True,
    )


def create_server(
    executor: Executor, host: str = "127.0.0.1", port: int = 8000, log_level: LogLevel = "WARNING"
) -> FastMCP:
    name = executor.config.name
    server = FastMCP(
        f"mjguard-{name}",
        instructions=INSTRUCTIONS.format(name=name, confirmation=CONFIRMATION if executor.require_confirmation else ""),
        host=host,
        port=port,
        log_level=log_level,
    )
    server._mcp_server.version = __version__  # pyright: ignore[reportPrivateUsage]
    read = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    act = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)
    halt = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
    store = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    async def in_thread(fn: Any) -> CallToolResult:
        return tool_result(await anyio.to_thread.run_sync(fn))

    @server.tool(annotations=read)
    def get_state() -> CallToolResult:
        """Tool position, joint positions, gripper opening, lowest clearance above the floor, whether it is moving
        or paused."""
        return tool_result(executor.get_state())

    @server.tool(annotations=read)
    def get_limits() -> CallToolResult:
        """The safety envelope: joint ranges, clearance, workspace, speed, refusal codes. Fixed at startup."""
        return tool_result(executor.get_limits())

    @server.tool(annotations=act)
    async def move_to(
        x_mm: Mm,
        y_mm: Mm,
        z_mm: Annotated[Mm, Field(description="Height of the tool point above the floor or table.")],
        tool_down: Annotated[bool, Field(description="Point the tool straight down (true) or any direction.")] = True,
        speed: Speed = 0.5,
        path: Annotated[
            Literal["joint", "line"],
            Field(description="'joint': any checked path (default). 'line': the tool travels a straight line."),
        ] = "joint",
        preview: Preview = False,
    ) -> CallToolResult:
        """Move the tool point to (x, y, z). Returns when the motion ends or is refused."""
        return await in_thread(lambda: executor.move_to(x_mm, y_mm, z_mm, tool_down, speed, path, preview))

    @server.tool(annotations=act)
    async def move_joints(
        joints: Annotated[
            dict[str, float],
            Field(description="Joint name to target: degrees for hinge joints, millimetres for slide joints."),
        ],
        speed: Speed = 0.5,
        preview: Preview = False,
    ) -> CallToolResult:
        """Move the named joints; joints not named keep their position."""
        return await in_thread(lambda: executor.move_joints(joints, speed, preview))

    if executor.config.gripper is not None:

        @server.tool(annotations=act)
        async def gripper(
            opening: Annotated[float, Field(allow_inf_nan=False, ge=0.0, le=1.0, description="0 closed, 1 open.")],
            speed: Speed = 0.5,
            preview: Preview = False,
        ) -> CallToolResult:
            """Open or close the gripper."""
            return await in_thread(lambda: executor.gripper(opening, speed, preview))

    @server.tool(annotations=act)
    async def rest(speed: Speed = 0.5, preview: Preview = False) -> CallToolResult:
        """Return to the rest pose."""
        return await in_thread(lambda: executor.rest(speed, preview))

    @server.tool(annotations=act)
    async def jog(
        dx_mm: JogMm = 0.0,
        dy_mm: JogMm = 0.0,
        dz_mm: JogMm = 0.0,
        tool_down: Annotated[
            bool, Field(description="Point the tool straight down (true) or keep its current direction (false).")
        ] = False,
        speed: Speed = 0.5,
        preview: Preview = False,
    ) -> CallToolResult:
        """Move the tool point by (dx, dy, dz) mm in a straight line, at most 50 mm in total. Checked like any move;
        a refusal's suggestion is an absolute point for move_to."""
        return await in_thread(lambda: executor.jog(dx_mm, dy_mm, dz_mm, tool_down, speed, preview))

    @server.tool(annotations=act)
    async def jog_joint(
        joint: Annotated[str, Field(description="Joint name, as in get_state.")],
        delta: Annotated[
            float,
            Field(
                allow_inf_nan=False,
                ge=-JOG_MAX_JOINT,
                le=JOG_MAX_JOINT,
                description="Degrees for hinge joints, millimetres for slide joints; at most 10.",
            ),
        ],
        speed: Speed = 0.5,
        preview: Preview = False,
    ) -> CallToolResult:
        """Move one joint by a small amount relative to where it is."""
        return await in_thread(lambda: executor.jog_joint(joint, delta, speed, preview))

    @server.tool(annotations=act)
    async def execute(
        plan_id: Annotated[str, Field(max_length=64, description="plan_id from a preview.")],
    ) -> CallToolResult:
        """Run a previewed plan. Each plan runs at most once, within 60 s of the preview, and only if the arm is
        still where the plan starts; it is checked again against the envelope before it moves."""
        return await in_thread(lambda: executor.execute(plan_id))

    @server.tool(annotations=halt)
    def pause() -> CallToolResult:
        """Halt the running motion at the next waypoint and keep the rest of it for resume()."""
        return tool_result(executor.pause())

    @server.tool(annotations=act)
    async def resume() -> CallToolResult:
        """Continue a paused motion. The path from where the arm is now through the rest of the motion is checked
        again first; if the world changed, it is refused and stays paused until resume() or stop()."""
        return await in_thread(executor.resume)

    @server.tool(annotations=halt)
    def stop() -> CallToolResult:
        """Halt the running motion at the next waypoint; the arm holds where it is. Discards a paused motion.
        Software stop only."""
        return tool_result(executor.stop())

    @server.tool(annotations=store)
    def save_pose(name: PoseName) -> CallToolResult:
        """Remember where the arm is now (joints and gripper) under a name. Nothing moves. Replaces a pose of the
        same name."""
        return tool_result(executor.save_pose(name))

    @server.tool(annotations=act)
    async def go_to_pose(
        name: PoseName,
        speed: Speed = 0.5,
        include_gripper: Annotated[
            bool, Field(description="Also move the gripper to its saved opening (default: the gripper stays).")
        ] = False,
        preview: Preview = False,
    ) -> CallToolResult:
        """Move to a saved pose. Planned and checked like any other motion, against the envelope now."""
        return await in_thread(lambda: executor.go_to_pose(name, speed, include_gripper, preview))

    @server.tool(annotations=read)
    def list_poses() -> CallToolResult:
        """Saved poses: joint values (degrees or millimetres) and gripper opening."""
        return tool_result(executor.list_poses())

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False))
    def delete_pose(name: PoseName) -> CallToolResult:
        """Forget a saved pose."""
        return tool_result(executor.delete_pose(name))

    @server.resource("mjguard://limits", mime_type="application/json")
    def limits_resource() -> str:
        """The active safety envelope."""
        return json.dumps(executor.get_limits()["limits"])

    @server.resource("mjguard://trace/latest", mime_type="application/json")
    def trace_resource() -> str:
        """The last 50 calls with their verdicts."""
        return json.dumps(executor.trace.latest(50), ensure_ascii=False, default=str)

    @server.resource("mjguard://events/latest", mime_type="application/json")
    def events_resource() -> str:
        """The last 50 events: motion_started, progress, motion_finished, refused, stopped, paused, resumed,
        breach, pose_saved, pose_deleted."""
        return json.dumps(executor.events.latest(50), ensure_ascii=False, default=str)

    security = TransportSecurityMiddleware(server.settings.transport_security)

    @server.custom_route("/events", methods=["GET"])
    async def events_stream(request: Request) -> Response:
        """Server-sent events (HTTP only). `Last-Event-ID` or `?after=SEQ` replays newer events still in memory."""
        # Custom routes bypass the MCP endpoint's Host/Origin checks; apply the same ones here.
        rejected = await security.validate_request(request)
        if rejected is not None:
            return rejected
        raw_after = request.headers.get("last-event-id") or request.query_params.get("after")
        try:
            after = int(raw_after) if raw_after is not None else None
        except ValueError:
            return Response("after must be an integer", status_code=400)
        return StreamingResponse(
            sse(executor, after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return server


def sse_format(event: Event) -> str:
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n"


async def sse(executor: Executor, after: int | None, heartbeat_s: float = SSE_HEARTBEAT_S) -> AsyncIterator[str]:
    """Events as SSE text. Publishers are worker threads; they hand events to this loop without blocking."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Event] = asyncio.Queue(SSE_QUEUE)

    def put(event: Event) -> None:
        if not queue.full():  # a client that falls behind loses events rather than slowing the arm
            queue.put_nowait(event)

    def deliver(event: Event) -> None:
        with contextlib.suppress(RuntimeError):  # the loop is closed: the client is gone
            loop.call_soon_threadsafe(put, event)

    unsubscribe = executor.events.subscribe(deliver)  # subscribe before replaying, then skip duplicates by seq
    try:
        last = 0 if after is None else after
        if after is not None:
            for event in executor.events.latest(200, after):
                last = event["seq"]
                yield sse_format(event)
        else:
            yield ": connected\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), heartbeat_s)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if event["seq"] > last:
                last = event["seq"]
                yield sse_format(event)
    finally:
        unsubscribe()

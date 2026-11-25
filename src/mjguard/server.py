"""MCP surface. Every acting tool goes through Executor, which plans and checks before anything moves."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from . import __version__
from .executor import Executor

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

INSTRUCTIONS = """\
Controls the robot arm "{name}". Coordinates are millimetres in the world frame of its model; z is the height above
the floor or table. Angles are degrees. Call get_limits first and plan inside them. Every motion is checked in a
MuJoCo model before it runs; a refused motion returns ok=false with a reason code, a message and, when one exists,
a nearby point that would pass. Use that to correct yourself instead of retrying the same command. stop() halts
the current motion; it is a software stop.
"""

Mm = Annotated[float, Field(allow_inf_nan=False, ge=-5000, le=5000)]
Speed = Annotated[
    float, Field(allow_inf_nan=False, ge=0.05, le=1.0, description="Fraction of the maximum joint speed.")
]


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
        f"mjguard-{name}", instructions=INSTRUCTIONS.format(name=name), host=host, port=port, log_level=log_level
    )
    server._mcp_server.version = __version__  # pyright: ignore[reportPrivateUsage]
    read = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
    act = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)

    async def in_thread(fn: Any) -> CallToolResult:
        return tool_result(await anyio.to_thread.run_sync(fn))

    @server.tool(annotations=read)
    def get_state() -> CallToolResult:
        """Tool position, joint positions, gripper opening, lowest clearance above the floor, whether it is moving."""
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
    ) -> CallToolResult:
        """Move the tool point to (x, y, z). Returns when the motion ends or is refused."""
        return await in_thread(lambda: executor.move_to(x_mm, y_mm, z_mm, tool_down, speed))

    @server.tool(annotations=act)
    async def move_joints(
        joints: Annotated[
            dict[str, float],
            Field(description="Joint name to target: degrees for hinge joints, millimetres for slide joints."),
        ],
        speed: Speed = 0.5,
    ) -> CallToolResult:
        """Move the named joints; joints not named keep their position."""
        return await in_thread(lambda: executor.move_joints(joints, speed))

    if executor.config.gripper is not None:

        @server.tool(annotations=act)
        async def gripper(
            opening: Annotated[float, Field(allow_inf_nan=False, ge=0.0, le=1.0, description="0 closed, 1 open.")],
            speed: Speed = 0.5,
        ) -> CallToolResult:
            """Open or close the gripper."""
            return await in_thread(lambda: executor.gripper(opening, speed))

    @server.tool(annotations=act)
    async def rest(speed: Speed = 0.5) -> CallToolResult:
        """Return to the rest pose."""
        return await in_thread(lambda: executor.rest(speed))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    def stop() -> CallToolResult:
        """Halt the running motion at the next waypoint; the arm holds where it is. Software stop only."""
        return tool_result(executor.stop())

    @server.resource("mjguard://limits", mime_type="application/json")
    def limits_resource() -> str:
        """The active safety envelope."""
        return json.dumps(executor.get_limits()["limits"])

    @server.resource("mjguard://trace/latest", mime_type="application/json")
    def trace_resource() -> str:
        """The last 50 calls with their verdicts."""
        return json.dumps(executor.trace.latest(50), ensure_ascii=False, default=str)

    return server

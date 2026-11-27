from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextResourceContents

from mjguard import __version__
from mjguard.cli import main
from mjguard.config import RobotConfig
from mjguard.executor import Executor, SimTransport
from mjguard.server import create_server, sse

HERE = Path(__file__).parent


async def test_tools_and_version(executor: Executor) -> None:
    server = create_server(executor)
    async with create_connected_server_and_client_session(server._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        assert (await client.initialize()).serverInfo.version == __version__
        listed = {t.name: t for t in (await client.list_tools()).tools}
        assert set(listed) == {
            "get_state",
            "get_limits",
            "move_to",
            "move_joints",
            "gripper",
            "rest",
            "stop",
            "release_stop",
            "discard",
            "jog",
            "jog_joint",
            "execute",
            "pause",
            "resume",
            "save_pose",
            "go_to_pose",
            "list_poses",
            "delete_pose",
            "preview_sequence",
        }
        read_only = {n for n, t in listed.items() if t.annotations and t.annotations.readOnlyHint}
        assert read_only == {"get_state", "get_limits", "list_poses"}
        for name in ("move_to", "move_joints", "gripper", "rest", "jog", "jog_joint", "go_to_pose"):
            assert "preview" in listed[name].inputSchema["properties"], name
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        assert uris == {"mjguard://limits", "mjguard://trace/latest", "mjguard://events/latest"}


async def test_refusal_is_an_error_with_a_suggestion(executor: Executor) -> None:
    server = create_server(executor)
    async with create_connected_server_and_client_session(server._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        ok = await client.call_tool("move_to", {"x_mm": 300, "y_mm": 0, "z_mm": 150})
        assert ok.isError is False
        bad = await client.call_tool("move_to", {"x_mm": 300, "y_mm": 0, "z_mm": 0})
        assert bad.isError is True and bad.structuredContent is not None
        assert bad.structuredContent["refusal"]["code"] == "TABLE_CLEARANCE"
        assert "suggestion" in bad.structuredContent["refusal"]
        joints = await client.call_tool("move_joints", {"joints": {"yaw": 20}})
        assert joints.isError is False
        schema = await client.call_tool("gripper", {"opening": 3})
        assert schema.isError is True


def test_check_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(HERE / "three_link.toml")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["limits"]["robot"] == "three_link"


def test_check_reports_config_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "no-such-robot"]) == 2
    assert "config error" in capsys.readouterr().err


def test_http_refuses_non_loopback(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["serve", str(HERE / "three_link.toml"), "--http", "--host", "0.0.0.0"])
    assert "no authentication" in capsys.readouterr().err


def test_config_loading_never_writes_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A downloader that prints during config loading must not corrupt the stdio protocol."""
    import mjguard.config as config_module

    real_load = config_module.load

    def chatty_load(path: str) -> object:
        print("Cloning https://example.invalid/menagerie.git...")
        return real_load(path)

    monkeypatch.setattr(config_module, "load", chatty_load)
    assert main(["check", str(HERE / "three_link.toml")]) == 0
    out, err = capsys.readouterr()
    assert out.lstrip().startswith("{") and "Cloning" not in out and "Cloning" in err


async def test_preview_execute_jog_poses_over_mcp(executor: Executor) -> None:
    server = create_server(executor)
    async with create_connected_server_and_client_session(server._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        p = await client.call_tool("move_to", {"x_mm": 300, "y_mm": 0, "z_mm": 150, "preview": True})
        assert p.isError is False and p.structuredContent is not None
        assert p.structuredContent["executed"] is False
        r = await client.call_tool("execute", {"plan_id": p.structuredContent["plan_id"]})
        assert r.isError is False
        again = await client.call_tool("execute", {"plan_id": p.structuredContent["plan_id"]})
        assert again.isError is True and again.structuredContent is not None
        assert again.structuredContent["refusal"]["code"] == "PLAN_USED"
        assert (await client.call_tool("jog", {"dz_mm": 20})).isError is False
        for tool, args in (
            ("jog", {"dz_mm": 60}),
            ("jog_joint", {"joint": "yaw", "delta": 11}),
            ("save_pose", {"name": "no spaces"}),
        ):
            capped = await client.call_tool(tool, args)
            assert capped.isError is True and capped.structuredContent is not None, tool
            assert capped.structuredContent["refusal"]["code"] == "BAD_TARGET", tool
            assert capped.structuredContent["refusal"]["message_human"], tool
        assert (await client.call_tool("jog_joint", {"joint": "yaw", "delta": 5})).isError is False
        assert (await client.call_tool("save_pose", {"name": "here"})).isError is False
        listed = await client.call_tool("list_poses", {})
        assert listed.structuredContent is not None and list(listed.structuredContent["poses"]) == ["here"]
        assert (await client.call_tool("rest", {})).isError is False
        assert (await client.call_tool("go_to_pose", {"name": "here"})).isError is False
        assert (await client.call_tool("delete_pose", {"name": "here"})).isError is False
        assert (await client.call_tool("pause", {})).isError is False
        resume = await client.call_tool("resume", {})
        assert resume.structuredContent is not None
        assert resume.structuredContent["refusal"]["code"] == "NOTHING_TO_RESUME"
        assert "message_human" in resume.structuredContent["refusal"]
        events = await client.read_resource("mjguard://events/latest")  # pyright: ignore[reportArgumentType]
        content = events.contents[0]
        assert isinstance(content, TextResourceContents)
        types = [e["type"] for e in json.loads(content.text)]
        assert "motion_started" in types and "pose_saved" in types and types[-1] == "refused"


async def test_preview_sequence_over_mcp(executor: Executor) -> None:
    server = create_server(executor)
    async with create_connected_server_and_client_session(server._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        init = await client.initialize()
        assert init.instructions is not None and "preview_sequence(steps)" in init.instructions
        schema = {t.name: t for t in (await client.list_tools()).tools}["preview_sequence"].inputSchema
        assert schema["properties"]["steps"]["type"] == "array" and "preview" not in json.dumps(schema["$defs"])
        steps = [
            {"tool": "move_to", "x_mm": 300, "y_mm": 100, "z_mm": 120},
            {"tool": "move_to", "x_mm": 300, "y_mm": 100, "z_mm": 60, "path": "line", "speed": 0.8},
            {"tool": "gripper", "opening": 0},
            {"tool": "rest"},
        ]
        p = await client.call_tool("preview_sequence", {"steps": steps, "speed": 0.4})
        assert p.isError is False and p.structuredContent is not None
        assert [s["speed"] for s in p.structuredContent["steps"]] == [0.4, 0.8, 0.4, 0.4]
        r = await client.call_tool("execute", {"plan_id": p.structuredContent["plan_id"]})
        assert r.isError is False and r.structuredContent is not None
        assert r.structuredContent["sequence"]["completed"] == [0, 1, 2, 3]
        nested = await client.call_tool("preview_sequence", {"steps": [{"tool": "rest", "preview": True}]})
        assert nested.isError is True  # the schema has no nested preview flag
        jog = await client.call_tool("preview_sequence", {"steps": [{"tool": "jog", "dz_mm": 5}]})
        assert jog.isError is True
        low = {"tool": "move_to", "x_mm": 300, "y_mm": 0, "z_mm": 0}
        bad = await client.call_tool("preview_sequence", {"steps": [low]})
        assert bad.isError is True and bad.structuredContent is not None
        assert bad.structuredContent["refusal"]["details"]["step"] == 0
        assert bad.structuredContent["refusal"]["message_human"].startswith("Step 1: ")


async def test_require_confirmation_is_in_the_instructions(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config), require_confirmation=True)
    server = create_server(ex)
    async with create_connected_server_and_client_session(server._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        init = await client.initialize()
        assert init.instructions is not None and "requires confirmation" in init.instructions
        r = await client.call_tool("move_to", {"x_mm": 300, "y_mm": 0, "z_mm": 150})
        assert r.isError is False and r.structuredContent is not None and r.structuredContent["executed"] is False


async def _get(app: Any, path: str, headers: dict[str, str], until: str, max_s: float = 10.0) -> tuple[int, str]:
    """Drive the ASGI app by hand: read the streamed body until `until` appears, then disconnect."""
    status = [0]
    body: list[str] = []
    done = anyio.Event()
    path, _, query = path.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    sent_request = [False]

    async def receive() -> dict[str, Any]:
        if not sent_request[0]:
            sent_request[0] = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            status[0] = message["status"]
        elif message["type"] == "http.response.body":
            body.append(message.get("body", b"").decode())
            if until in "".join(body) or not message.get("more_body", False):
                done.set()

    with anyio.fail_after(max_s):
        await app(scope, receive, send)
    return status[0], "".join(body)


async def test_sse_delivers_events_and_checks_the_host(executor: Executor) -> None:
    server = create_server(executor)
    app = server.streamable_http_app()
    assert any(getattr(r, "path", None) == "/events" for r in app.routes)
    executor.save_pose("before")  # seq 1, replayed with after=0

    async def move_soon() -> None:
        await asyncio.sleep(0.2)
        await anyio.to_thread.run_sync(lambda: executor.move_to(300, 0, 150))

    async with anyio.create_task_group() as tg:
        tg.start_soon(move_soon)
        status, body = await _get(app, "/events?after=0", {"host": "127.0.0.1:8000"}, "event: motion_finished")
    assert status == 200
    names = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]
    assert names[0] == "pose_saved" and names[1] == "motion_started" and names[-1] == "motion_finished"
    data = [json.loads(line.removeprefix("data: ")) for line in body.splitlines() if line.startswith("data: ")]
    assert [d["seq"] for d in data] == sorted({d["seq"] for d in data})  # in order, no duplicates

    status, _ = await _get(app, "/events", {"host": "evil.example:8000"}, "")
    assert status == 421  # DNS rebinding protection, same as the MCP endpoint
    status, _ = await _get(app, "/events?after=x", {"host": "127.0.0.1:8000"}, "")
    assert status == 400


async def test_sse_heartbeat_and_live_connect(executor: Executor) -> None:
    stream = sse(executor, None, heartbeat_s=0.05)
    assert (await anext(stream)).startswith(": connected")
    assert (await anext(stream)).startswith(": keep-alive")
    executor.save_pose("p")
    assert (await anext(stream)).startswith("id: ")
    await stream.aclose()


def test_serve_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import mjguard.server as server_module

    built: dict[str, Any] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            built["transport"] = transport

    def fake_create_server(executor: Executor, *args: object) -> FakeServer:
        built["executor"] = executor
        return FakeServer()

    monkeypatch.setattr(server_module, "create_server", fake_create_server)
    poses = tmp_path / "poses.json"
    toml = str(HERE / "three_link.toml")
    assert main(["serve", toml, "--lang", "zh-TW", "--require-confirmation", "--poses", str(poses)]) == 0
    ex: Executor = built["executor"]
    assert ex.lang == "zh-TW" and ex.require_confirmation is True and built["transport"] == "stdio"
    assert ex.save_pose("p")["ok"] is True and "p" in json.loads(poses.read_text())["poses"]

    assert main(["serve", toml, "--http"]) == 0
    ex = built["executor"]
    assert ex.lang == "en" and ex.require_confirmation is False and built["transport"] == "streamable-http"

    poses.write_text("{broken")
    assert main(["serve", toml, "--poses", str(poses)]) == 2
    assert "poses file" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["serve", toml, "--lang", "fr"])

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mjguard import __version__
from mjguard.cli import main
from mjguard.executor import Executor
from mjguard.server import create_server

HERE = Path(__file__).parent


async def test_tools_and_version(executor: Executor) -> None:
    server = create_server(executor)
    async with create_connected_server_and_client_session(server._mcp_server) as client:  # pyright: ignore[reportPrivateUsage]
        assert (await client.initialize()).serverInfo.version == __version__
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"get_state", "get_limits", "move_to", "move_joints", "gripper", "rest", "stop"}
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        assert uris == {"mjguard://limits", "mjguard://trace/latest"}


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

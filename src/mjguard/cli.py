"""`mjguard serve CONFIG` runs the MCP server; `mjguard check CONFIG` validates a config and prints the envelope."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from . import __version__


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    from .config import presets

    parser = argparse.ArgumentParser(prog="mjguard", description="Guarded MCP control for MuJoCo-modelled arms.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    help_config = f"robot config TOML, or a built-in preset: {', '.join(presets())}"

    check = sub.add_parser("check", help="load a config, check the rest pose and print the envelope")
    check.add_argument("config", help=help_config)

    serve = sub.add_parser("serve", help="run the MCP server on a simulated arm")
    serve.add_argument("config", help=help_config)
    serve.add_argument("--trace", type=Path, help="append every call to this JSONL file")
    serve.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--allow-remote", action="store_true", help="allow a non-loopback --host (no authentication)")
    serve.add_argument("--realtime", action="store_true", help="run the simulation in real time")
    serve.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    from .config import ConfigError, load

    try:
        config = load(args.config)
        if args.command == "check":
            from .guard import Guard

            guard = Guard(config)
            refusal, clearance = guard.check_configuration(guard.model.home, guard.model.home_gripper)
            report = {
                "ok": refusal is None,
                "rest_pose": "passes" if refusal is None else refusal.to_dict(),
                "rest_clearance_mm": round(clearance, 1),
                "moving_collision_geoms": len(guard.model.moving_geoms),
                "physics": bool(config.actuators),
                "limits": guard.describe(),
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if refusal is None else 1
    except ConfigError as exc:
        print(f"mjguard: config error: {exc}", file=sys.stderr)
        return 2

    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    if args.http and not _is_loopback(args.host) and not args.allow_remote:
        parser.error(f"--host {args.host} is not loopback and the endpoint has no authentication; add --allow-remote")
    logging.basicConfig(
        level=args.log_level, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from .executor import Executor, SimTransport, Trace
    from .server import LogLevel, create_server

    executor = Executor(config, SimTransport(config, realtime=args.realtime), Trace(args.trace))
    server = create_server(executor, args.host, args.port, cast(LogLevel, args.log_level))
    try:
        server.run(transport="streamable-http" if args.http else "stdio")
    except KeyboardInterrupt:
        executor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

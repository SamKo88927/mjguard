from __future__ import annotations

import threading
import time

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mjguard.config import RobotConfig
from mjguard.executor import Executor, SimTransport


def test_move_reaches_target_pointing_down(executor: Executor) -> None:
    r = executor.move_to(300, 50, 120)
    assert r["ok"] is True
    tool = r["state"]["tool"]
    assert abs(tool["x_mm"] - 300) < 2 and abs(tool["y_mm"] - 50) < 2 and abs(tool["z_mm"] - 120) < 2
    assert tool["axis"][2] < -0.99


def test_refusal_leaves_arm_still_and_suggestion_is_accepted(executor: Executor) -> None:
    before = executor.get_state()["state"]["tool"]
    r = executor.move_to(300, 0, 5)
    assert r["refusal"]["code"] == "TABLE_CLEARANCE"
    assert executor.get_state()["state"]["tool"] == before
    s = r["refusal"]["suggestion"]
    assert executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"])["ok"] is True


def test_move_joints_by_name(executor: Executor) -> None:
    r = executor.move_joints({"yaw": 30})
    assert r["ok"] is True and abs(r["state"]["joints"]["yaw"] - 30) < 0.2
    assert executor.move_joints({"nope": 1})["refusal"]["code"] == "BAD_TARGET"


def test_gripper_and_rest(executor: Executor) -> None:
    assert executor.gripper(0.0)["state"]["gripper_opening"] < 0.01
    assert executor.gripper(1.0)["state"]["gripper_opening"] > 0.99
    assert executor.gripper(2.0)["refusal"]["code"] == "BAD_TARGET"
    assert executor.rest()["ok"] is True


def test_busy_and_stop(config: RobotConfig) -> None:
    executor = Executor(config, SimTransport(config, realtime=True))
    results: dict[str, dict[str, object]] = {}
    mover = threading.Thread(target=lambda: results.update(move=executor.move_joints({"yaw": -150}, speed=0.1)))
    mover.start()
    time.sleep(0.3)
    assert executor.move_joints({"yaw": 0})["refusal"]["code"] == "BUSY"
    assert executor.stop()["was_moving"] is True
    mover.join(5)
    assert results["move"]["refusal"]["code"] == "STOPPED"  # type: ignore[index]


def test_monitor_halts_on_what_actually_ran(config: RobotConfig) -> None:
    """A transport that drifts toward the floor while following a safe plan is halted mid-motion."""
    transport = SimTransport(config)
    real_follow = transport.follow

    def sagging_follow(waypoints, gripper, period, should_stop, on_waypoint):  # type: ignore[no-untyped-def]
        floor = np.radians([0, -95, -145, -35])  # touches the floor
        sag = [w + (floor - w) * min(1.0, i / 3) for i, w in enumerate(waypoints)]
        return real_follow(sag, gripper, period, should_stop, on_waypoint)

    transport.follow = sagging_follow  # type: ignore[method-assign]
    r = Executor(config, transport).move_joints({"yaw": 90})
    assert r["refusal"]["code"] == "ENVELOPE_BREACH"
    assert r["executed"]["waypoints"] < r["plan"]["waypoints"]


def test_transport_fault_is_a_traced_refusal(config: RobotConfig) -> None:
    transport = SimTransport(config)

    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("serial link dropped")

    transport.follow = broken  # type: ignore[method-assign]
    executor = Executor(config, transport)
    r = executor.move_joints({"yaw": 10})
    assert r["refusal"]["code"] == "INTERNAL_ERROR" and "serial link dropped" in r["refusal"]["message"]
    assert executor.trace.latest(1)[0]["result"]["refusal"]["code"] == "INTERNAL_ERROR"
    transport.follow = SimTransport(config).follow  # type: ignore[method-assign]
    assert executor.move_joints({"yaw": 10})["ok"] is True  # lock released


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    x=st.floats(-900, 900),
    y=st.floats(-900, 900),
    z=st.floats(-200, 900),
    down=st.booleans(),
)
def test_random_targets_never_breach_clearance(executor: Executor, x: float, y: float, z: float, down: bool) -> None:
    r = executor.move_to(x, y, z, tool_down=down, speed=1.0)
    limit = executor.config.limits.min_clearance_mm
    if r["ok"]:
        assert r["executed"]["min_clearance_mm"] >= limit - 0.5
    assert r["state"]["min_clearance_mm"] >= limit - 0.5
    sug = r.get("refusal", {}).get("suggestion")
    if sug:
        follow = executor.move_to(sug["x_mm"], sug["y_mm"], sug["z_mm"], tool_down=down, speed=1.0)
        assert follow["ok"] is True, follow

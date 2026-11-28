# ruff: noqa: RUF001  (zh-TW text uses full-width punctuation on purpose)
"""Sequences: preview several motions as one plan, run them in order, each step checked again before it runs."""

from __future__ import annotations

import itertools
import math
import threading
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from mjguard.config import RobotConfig, from_dict, load
from mjguard.executor import MAX_SEQUENCE_STEPS, Executor, SimTransport
from mjguard.guard import Guard

HERE = Path(__file__).parent

PICK_AND_PLACE: list[dict[str, Any]] = [
    {"tool": "move_to", "x_mm": 300, "y_mm": 100, "z_mm": 120},  # above the part
    {"tool": "move_to", "x_mm": 300, "y_mm": 100, "z_mm": 60, "path": "line"},  # down
    {"tool": "gripper", "opening": 0},  # grip
    {"tool": "move_to", "x_mm": 300, "y_mm": 100, "z_mm": 120, "path": "line"},  # up
    {"tool": "move_to", "x_mm": 250, "y_mm": -150, "z_mm": 120},  # above the bin
    {"tool": "gripper", "opening": 1},  # release
]
MOTION = {"sequence_started", "step_started", "motion_started", "motion_finished", "step_finished", "sequence_finished"}


def make_config(**extra: Any) -> RobotConfig:
    data = tomllib.loads((HERE / "three_link.toml").read_text())
    data.update(extra)
    return from_dict(data, HERE)


def tool_at(state: dict[str, Any], xyz: tuple[float, float, float], tol: float = 2.0) -> bool:
    t = state["tool"]
    return math.dist((t["x_mm"], t["y_mm"], t["z_mm"]), xyz) < tol


def record(ex: Executor) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    ex.events.subscribe(seen.append)
    return seen


def flow(ex: Executor, seen: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    assert ex.events.flush()
    return [(e["type"], e.get("index", e.get("step"))) for e in seen if e["type"] != "progress"]


def hook_after_step(transport: SimTransport, step: int, action: Callable[[], None]) -> None:
    """Call `action` right after the transport finished following step `step` (0-based)."""
    real = transport.follow
    calls = [0]

    def follow(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        if calls[0] == step:
            action()
        calls[0] += 1
        return out

    transport.follow = follow  # type: ignore[method-assign]


def nudge_yaw(transport: SimTransport, deg: float) -> Callable[[], None]:
    """Move the simulated arm's base by `deg` after a step, as if something pushed it."""

    def push() -> None:
        transport.d.qpos[transport.model.qadr[0]] += math.radians(deg)
        mujoco.mj_forward(transport.m, transport.d)

    return push


# ---------------------------------------------------------------------- preview, then run
def test_pick_and_place_previews_then_runs_every_step(executor: Executor) -> None:
    before = executor.get_state()["state"]
    p = executor.preview_sequence(PICK_AND_PLACE)
    assert p["ok"] is True and p["executed"] is False and p["expires_in_s"] == 60
    assert executor.get_state()["state"] == before  # nothing moved
    steps, totals = p["steps"], p["totals"]
    assert [s["index"] for s in steps] == list(range(6))
    assert [s["tool"] for s in steps] == [s["tool"] for s in PICK_AND_PLACE]
    assert steps[1]["arguments"]["path"] == "line" and steps[1]["arguments"]["speed"] == 0.5
    for s, nxt in itertools.pairwise(steps):
        assert s["end_tool"]["mm"] == nxt["start_tool"]["mm"]  # each step starts where the last one is predicted to end
        assert 2 <= len(s["path_mm"]) <= 16
    assert math.dist(steps[1]["end_tool"]["mm"], (300, 100, 60)) < 2
    assert steps[2]["gripper_opening_goal"] == 0.0 and steps[5]["gripper_opening_goal"] == 1.0
    assert totals["steps"] == 6 and totals["waypoints"] == sum(s["waypoints"] for s in steps)
    assert totals["duration_s"] == pytest.approx(sum(s["duration_s"] for s in steps), abs=0.02)
    assert totals["min_clearance_mm"] == min(s["min_clearance_mm"] for s in steps) >= 30
    assert totals["min_obstacle_gap_mm"] is None and totals["gripper_opening_end"] == 1.0
    assert totals["start_tool"] == steps[0]["start_tool"] and totals["end_tool"] == steps[-1]["end_tool"]

    seen = record(executor)
    r = executor.execute(p["plan_id"])
    assert r["ok"] is True, r.get("refusal")
    assert r["sequence"] == {"steps": 6, "completed": [0, 1, 2, 3, 4, 5], "halted_at": None}
    assert [s["index"] for s in r["steps"]] == list(range(6)) and all(s["ok"] for s in r["steps"])
    assert r["executed"]["waypoints"] == totals["waypoints"] and r["executed"]["min_clearance_mm"] >= 30
    assert tool_at(r["state"], (250, -150, 120)) and r["state"]["gripper_opening"] > 0.99
    assert r["plan"]["steps"] == 6
    assert flow(executor, seen) == [
        ("sequence_started", None),
        *[(t, i) for i in range(6) for t in ("step_started", "motion_started", "motion_finished", "step_finished")],
        ("sequence_finished", None),
    ]
    assert seen[0]["plan_id"] == p["plan_id"] and seen[0]["steps"] == 6
    assert seen[0]["tools"] == [s["tool"] for s in PICK_AND_PLACE]
    progress = [e for e in seen if e["type"] == "progress"]
    assert progress and all(e["steps"] == 6 and 0 <= e["step"] < 6 for e in progress)
    finished = [e for e in seen if e["type"] == "step_finished"]
    assert all(e["ok"] is True and e["of"] == 6 for e in finished)
    assert seen[-1]["ok"] is True and seen[-1]["completed"] == 6

    assert executor.execute(p["plan_id"])["refusal"]["code"] == "PLAN_USED"  # one shot, like a single plan


def test_every_step_tool_and_the_sequence_speed(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config))
    assert ex.move_to(300, 50, 150)["ok"] is True
    assert ex.save_pose("high")["ok"] is True
    assert ex.rest()["ok"] is True
    p = ex.preview_sequence(
        [
            {"tool": "move_joints", "joints": {"yaw": 30}, "speed": 1.0},
            {"tool": "gripper", "opening": 0.0},
            {"tool": "go_to_pose", "name": "high", "include_gripper": True},
            {"tool": "rest"},
        ],
        speed=0.2,
    )
    assert p["ok"] is True, p.get("refusal")
    assert [s["speed"] for s in p["steps"]] == [1.0, 0.2, 0.2, 0.2]
    assert p["steps"][2]["gripper_opening_goal"] == 1.0  # the pose was saved with the gripper open
    assert math.dist(p["steps"][2]["end_tool"]["mm"], (300, 50, 150)) < 2
    r = ex.execute(p["plan_id"])
    assert r["ok"] is True and r["sequence"]["completed"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------- refused at preview
def test_an_unsafe_third_step_is_refused_naming_step_2_and_nothing_is_stored(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config), lang="zh-TW")
    steps = [dict(s) for s in PICK_AND_PLACE]
    steps[2] = {"tool": "move_to", "x_mm": 300, "y_mm": 100, "z_mm": 5}  # into the table
    seen = record(ex)
    r = ex.preview_sequence(steps)
    assert r["ok"] is False and "plan_id" not in r and not ex._plans  # pyright: ignore[reportPrivateUsage]
    refusal = r["refusal"]
    assert refusal["code"] == "TABLE_CLEARANCE"
    assert refusal["details"]["step"] == 2 and refusal["details"]["step_tool"] == "move_to"
    assert refusal["message"].startswith("step 2 (move_to): ")
    assert refusal["message_human"] == "第 3 步：會太靠近桌面（還差 25 mm），已停止規劃"
    s = refusal["suggestion"]
    assert s["step"] == 2 and s["z_mm"] >= 30  # planned from where step 1 is predicted to end, not from the arm now
    steps[2] = {"tool": "move_to", "x_mm": s["x_mm"], "y_mm": s["y_mm"], "z_mm": s["z_mm"], "path": s["path"]}
    fixed = ex.preview_sequence(steps)
    assert fixed["ok"] is True, fixed.get("refusal")
    assert ex.execute(fixed["plan_id"])["ok"] is True
    assert flow(ex, seen)[0] == ("refused", None) and seen[0]["refusal"]["details"]["step"] == 2


@pytest.mark.parametrize(
    ("steps", "index", "fragment"),
    [
        ([], None, "1 to 10 steps"),
        ([{"tool": "rest"}] * (MAX_SEQUENCE_STEPS + 1), None, "1 to 10 steps"),
        ([{"tool": "rest"}, {"tool": "jog", "dz_mm": 10}], 1, "cannot be a step"),
        ([{"tool": "rest", "preview": True}], 0, "no 'preview'"),
        ([{"tool": "move_to", "x_mm": 1}], 0, "missing ['y_mm', 'z_mm']"),
        ([{"tool": "rest", "colour": "red"}], 0, "unknown ['colour']"),
        ([{"tool": "rest", "speed": 2}], 0, "speed"),
        ([{"tool": "rest", "speed": True}], 0, "speed"),
        (["rest"], 0, "an object"),
        ([{"x_mm": 1}], 0, "an object"),
        ([{"tool": "move_to", "x_mm": "a", "y_mm": 0, "z_mm": 0}], 0, "numbers"),
        ([{"tool": "move_to", "x_mm": 300, "y_mm": 0, "z_mm": 100, "path": "curve"}], 0, "path"),
        ([{"tool": "move_joints", "joints": {"yaw": "x"}}], 0, "joint names to numbers"),
        ([{"tool": "move_joints", "joints": {}}], 0, "joint names to numbers"),
        ([{"tool": "move_joints", "joints": {"nope": 1}}], 0, "unknown joints"),
        ([{"tool": "gripper", "opening": 3}], 0, "opening"),
        ([{"tool": "gripper", "opening": None}], 0, "opening"),
        ([{"tool": "go_to_pose", "name": 3}], 0, "name is text"),
    ],
)
def test_malformed_sequences_are_bad_targets(
    executor: Executor, steps: list[Any], index: int | None, fragment: str
) -> None:
    r = executor.preview_sequence(steps)
    assert r["ok"] is False and r["refusal"]["code"] == "BAD_TARGET"
    assert fragment in r["refusal"]["message"]
    assert r["refusal"]["details"].get("step") == index
    assert r["refusal"]["message_human"]


def test_other_preview_refusals(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config))
    assert ex.preview_sequence("rest")["refusal"]["code"] == "BAD_TARGET"  # type: ignore[arg-type]
    assert ex.preview_sequence([{"tool": "rest"}], speed=0)["refusal"]["code"] == "BAD_TARGET"
    unknown = ex.preview_sequence([{"tool": "rest"}, {"tool": "go_to_pose", "name": "nope"}])
    assert unknown["refusal"]["code"] == "POSE_UNKNOWN" and unknown["refusal"]["details"]["step"] == 1
    data = tomllib.loads((HERE / "three_link.toml").read_text())
    del data["gripper"]
    no_gripper = from_dict(data, HERE)
    bare = Executor(no_gripper, SimTransport(no_gripper))
    assert "no gripper" in bare.preview_sequence([{"tool": "gripper", "opening": 0}])["refusal"]["message"]


# ---------------------------------------------------------------------- checked again while it runs
def test_a_world_change_refuses_a_later_step_mid_run(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config), lang="zh-TW")
    p = ex.preview_sequence(PICK_AND_PLACE)
    assert p["ok"] is True
    # After the preview, a box is put where step 4 goes (above the bin); steps 0-3 are clear of it.
    box = {"name": "box", "shape": "sphere", "size_mm": [40], "center_mm": [250, -150, 120]}
    ex.guard = Guard(make_config(obstacles=[box]))
    seen = record(ex)
    r = ex.execute(p["plan_id"])
    assert r["ok"] is False and r["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"}
    assert r["refusal"]["details"]["step"] == 4 and r["refusal"]["message_human"].startswith("第 5 步：")
    assert r["sequence"] == {"steps": 6, "completed": [0, 1, 2, 3], "halted_at": 4}
    assert [s["index"] for s in r["steps"]] == [0, 1, 2, 3] and "executed" in r
    assert tool_at(r["state"], (300, 100, 120))  # held where step 3 ended
    events = flow(ex, seen)
    assert events[-3:] == [("step_finished", 4), ("refused", 4), ("sequence_finished", None)]
    refused = next(e for e in seen if e["type"] == "refused")
    assert refused["refusal"]["details"]["step"] == 4 and refused["step"] == 4
    assert seen[-1]["ok"] is False and seen[-1]["halted_at"] == 4 and seen[-1]["completed"] == 4


def test_stale_start_and_stale_later_step(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config))
    p = ex.preview_sequence(PICK_AND_PLACE)
    assert ex.jog_joint("shoulder", 2)["ok"] is True  # the arm moves after the preview
    stale = ex.execute(p["plan_id"])
    assert stale["refusal"]["code"] == "PLAN_STALE" and stale["refusal"]["details"]["step"] == 0
    assert "executed" not in stale and ex.execute(p["plan_id"])["refusal"]["code"] == "PLAN_USED"

    # Something pushes the arm 5 deg at the base after step 1 (about 28 mm at the tool, the margin is 10 mm).
    transport = SimTransport(config)
    hook_after_step(transport, 1, nudge_yaw(transport, 5))
    ex = Executor(config, transport)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    # Step 1 ran but the arm did not end where it was planned: caught at the end of step 1 (TARGET_MISSED), before
    # step 2 could start from the wrong place.
    assert r["refusal"]["code"] == "TARGET_MISSED", r["refusal"]
    assert r["refusal"]["details"]["miss_mm"] > 20
    assert r["sequence"]["completed"] == [0] and r["sequence"]["halted_at"] == 1


def test_tracking_error_inside_the_planning_margin_is_not_stale(config: RobotConfig) -> None:
    """1 deg at the base is about 5 mm at the tool: inside the 10 mm margin, so the next step runs, from the measured
    arm (the path to its first waypoint is filled in and checked)."""
    transport = SimTransport(config)
    hook_after_step(transport, 1, nudge_yaw(transport, 1))
    ex = Executor(config, transport)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    assert r["ok"] is True, r.get("refusal")
    assert tool_at(r["state"], (250, -150, 120))


def test_a_breach_mid_sequence_halts_the_rest(config: RobotConfig) -> None:
    transport = SimTransport(config)
    real = transport.follow
    calls = [0]

    def sagging(waypoints, gripper, period, should_stop, on_waypoint):  # type: ignore[no-untyped-def]
        calls[0] += 1
        if calls[0] == 2:  # step 1 sags into the table
            floor = np.radians([0, -95, -145, -35])
            waypoints = [w + (floor - w) * min(1.0, i / 3) for i, w in enumerate(waypoints)]
        return real(waypoints, gripper, period, should_stop, on_waypoint)

    transport.follow = sagging  # type: ignore[method-assign]
    ex = Executor(config, transport)
    seen = record(ex)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    assert r["refusal"]["code"] == "ENVELOPE_BREACH" and r["refusal"]["details"]["step"] == 1
    assert r["sequence"]["completed"] == [0] and [s["ok"] for s in r["steps"]] == [True, False]
    assert ("breach", 1) in flow(ex, seen)


def test_a_transport_fault_mid_sequence_finishes_the_sequence_events(config: RobotConfig) -> None:
    transport = SimTransport(config)
    real = transport.follow
    calls = [0]

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("serial link dropped")
        return real(*args, **kwargs)

    transport.follow = flaky  # type: ignore[method-assign]
    ex = Executor(config, transport)
    seen = record(ex)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    assert r["refusal"]["code"] == "INTERNAL_ERROR"
    events = flow(ex, seen)
    assert events[-3:] == [("step_finished", 1), ("sequence_finished", None), ("refused", None)]


# ---------------------------------------------------------------------- pause, resume, stop
def run_slowly(ex: Executor, plan_id: str) -> tuple[threading.Thread, dict[str, Any]]:
    out: dict[str, Any] = {}
    mover = threading.Thread(target=lambda: out.update(r=ex.execute(plan_id)))
    mover.start()
    return mover, out


def wait_for_motion(seen: list[dict[str, Any]], timeout: float = 10.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout and not any(e["type"] == "progress" for e in list(seen)):
        time.sleep(0.01)


def wait_for_step(seen: list[dict[str, Any]], step: int, timeout: float = 10.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if any(e["type"] == "progress" and e["step"] == step for e in list(seen)):
            return
        time.sleep(0.01)
    raise AssertionError(f"step {step} never moved")


def test_pause_mid_sequence_then_resume_finishes(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    seen = record(ex)
    p = ex.preview_sequence(PICK_AND_PLACE, speed=0.3)
    mover, out = run_slowly(ex, p["plan_id"])
    wait_for_step(seen, 1)
    assert ex.pause()["was_moving"] is True
    mover.join(10)
    r = out["r"]
    assert r["refusal"]["code"] == "PAUSED" and r["paused"] is True and r["refusal"]["details"]["step"] == 1
    assert r["sequence"]["completed"] == [0] and r["sequence"]["remaining_steps"] == 5
    paused = ex.get_state()["paused_motion"]
    assert paused["sequence"] == {"step": 1, "steps": 6, "remaining_steps": 5} and paused["remaining_waypoints"] > 0

    ex.transport.realtime = False  # type: ignore[attr-defined]
    done = ex.resume()
    assert done["ok"] is True, done.get("refusal")
    assert done["sequence"]["completed"] == [0, 1, 2, 3, 4, 5]
    assert [s["index"] for s in done["steps"]] == list(range(6))  # step 1 once, both halves together
    assert tool_at(done["state"], (250, -150, 120)) and ex.get_state()["state"]["paused"] is False
    events = flow(ex, seen)
    assert events[:6] == [
        ("sequence_started", None),
        ("step_started", 0),
        ("motion_started", 0),
        ("motion_finished", 0),
        ("step_finished", 0),
        ("step_started", 1),
    ]
    i = events.index(("paused", 1))
    assert events[i - 1 : i + 5] == [
        ("motion_started", 1),
        ("paused", 1),
        ("motion_finished", 1),
        ("resumed", 1),
        ("motion_started", 1),
        ("motion_finished", 1),
    ]
    assert events[i + 5] == ("step_finished", 1) and events[-1] == ("sequence_finished", None)
    assert [t for t, _ in events].count("sequence_finished") == 1


def test_pause_between_steps_then_resume(config: RobotConfig) -> None:
    """A pause that lands after a step's last waypoint holds before the next step; resume checks that step."""
    transport = SimTransport(config)
    ex = Executor(config, transport)
    hook_after_step(transport, 2, ex.pause)
    seen = record(ex)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    assert r["refusal"]["code"] == "PAUSED" and r["refusal"]["details"]["step"] == 3
    assert r["sequence"]["completed"] == [0, 1, 2] and r["sequence"]["remaining_steps"] == 3
    assert ex.get_state()["paused_motion"]["sequence"]["step"] == 3
    done = ex.resume()
    assert done["ok"] is True and done["sequence"]["completed"] == [0, 1, 2, 3, 4, 5]
    events = flow(ex, seen)
    assert ("paused", 3) in events and events[events.index(("paused", 3)) + 1 : events.index(("paused", 3)) + 3] == [
        ("resumed", 3),
        ("step_started", 3),
    ]


def test_resume_mid_sequence_after_the_world_changed_stays_paused(config: RobotConfig) -> None:
    transport = SimTransport(config)
    ex = Executor(config, transport)
    hook_after_step(transport, 3, ex.pause)
    assert ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])["refusal"]["code"] == "PAUSED"
    box = {"name": "box", "shape": "sphere", "size_mm": [40], "center_mm": [250, -150, 120]}
    ex.guard = Guard(make_config(obstacles=[box]))
    r = ex.resume()
    assert r["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"} and r["refusal"]["details"]["step"] == 4
    assert ex.get_state()["paused_motion"]["sequence"]["step"] == 4  # kept: a person decides


def test_stop_mid_sequence_discards_the_rest(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    seen = record(ex)
    p = ex.preview_sequence(PICK_AND_PLACE, speed=0.3)
    mover, out = run_slowly(ex, p["plan_id"])
    wait_for_step(seen, 1)
    ex.stop()
    mover.join(10)
    r = out["r"]
    assert r["refusal"]["code"] == "STOPPED" and r["refusal"]["details"]["step"] == 1
    assert r["sequence"]["completed"] == [0] and r["sequence"]["halted_at"] == 1
    assert ex.get_state()["state"]["paused"] is False
    events = flow(ex, seen)
    assert ("stopped", 1) in events and events[-2:] == [("step_finished", 1), ("sequence_finished", None)]
    assert [t for t, _ in events].count("step_started") == 2
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"


def test_stop_between_steps_and_stop_while_paused(config: RobotConfig) -> None:
    transport = SimTransport(config)
    ex = Executor(config, transport)
    hook_after_step(transport, 0, lambda: ex.stop())
    seen = record(ex)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    assert r["refusal"]["code"] == "STOPPED" and r["refusal"]["details"]["step"] == 1
    assert r["sequence"]["completed"] == [0]
    assert ("stopped", 1) in flow(ex, seen)

    transport = SimTransport(config)
    ex = Executor(config, transport)
    hook_after_step(transport, 1, ex.pause)
    assert ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])["refusal"]["code"] == "PAUSED"
    seen = record(ex)
    stopped = ex.stop()
    assert stopped["discarded_paused_motion"] is True and stopped["discarded_sequence_steps"] == 4
    assert flow(ex, seen) == [("stopped", None)] and seen[0]["discarded_sequence_steps"] == 4
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"


def test_a_stop_that_races_a_pause_between_steps_wins(config: RobotConfig) -> None:
    transport = SimTransport(config)
    ex = Executor(config, transport)

    def both() -> None:
        ex.pause()
        ex.stop()

    hook_after_step(transport, 0, both)
    r = ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])
    assert r["refusal"]["code"] == "STOPPED" and ex.get_state()["state"]["paused"] is False


def test_a_new_motion_replaces_a_paused_sequence(config: RobotConfig) -> None:
    transport = SimTransport(config)
    ex = Executor(config, transport)
    hook_after_step(transport, 1, ex.pause)
    assert ex.execute(ex.preview_sequence(PICK_AND_PLACE)["plan_id"])["refusal"]["code"] == "PAUSED"
    r = ex.move_joints({"yaw": 0})
    assert r["ok"] is True and r["discarded_paused_motion"] is True
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"

    # and a sequence replaces a paused single motion, saying so
    slow = Executor(config, SimTransport(config, realtime=True))
    seen = record(slow)
    mover = threading.Thread(target=lambda: slow.move_joints({"yaw": -150}, speed=0.1))
    mover.start()
    wait_for_motion(seen)
    slow.pause()
    mover.join(10)
    slow.transport.realtime = False  # type: ignore[attr-defined]
    r = slow.execute(slow.preview_sequence([{"tool": "rest"}])["plan_id"])
    assert r["ok"] is True and r["discarded_paused_motion"] is True


def test_paused_twice_in_one_step_keeps_one_record(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    seen = record(ex)
    mover, out = run_slowly(ex, ex.preview_sequence([{"tool": "move_joints", "joints": {"yaw": -150}}], 0.1)["plan_id"])
    wait_for_step(seen, 0)
    ex.pause()
    mover.join(10)
    assert out["r"]["refusal"]["code"] == "PAUSED"
    resumer: dict[str, Any] = {}
    thread = threading.Thread(target=lambda: resumer.update(r=ex.resume()))
    thread.start()
    time.sleep(0.3)
    ex.pause()
    thread.join(10)
    assert resumer["r"]["refusal"]["code"] == "PAUSED" and len(resumer["r"]["steps"]) == 1
    ex.transport.realtime = False  # type: ignore[attr-defined]
    done = ex.resume()
    assert done["ok"] is True and len(done["steps"]) == 1
    assert done["steps"][0]["executed"]["waypoints"] == done["plan"]["waypoints"]


# ---------------------------------------------------------------------- same rules as single plans
def test_ttl_discard_latch_busy_and_confirmation(config: RobotConfig) -> None:
    now = [1000.0]
    ex = Executor(config, SimTransport(config), clock=lambda: now[0], require_confirmation=True)
    p = ex.preview_sequence(PICK_AND_PLACE)
    assert p["ok"] is True  # require_confirmation: a sequence is a preview anyway
    now[0] += 61
    assert ex.execute(p["plan_id"])["refusal"]["code"] == "PLAN_EXPIRED"
    p = ex.preview_sequence(PICK_AND_PLACE)
    assert ex.discard(p["plan_id"])["discarded"] is True
    assert ex.execute(p["plan_id"])["refusal"]["code"] == "PLAN_UNKNOWN"

    ex.stop(latch=True)
    p = ex.preview_sequence(PICK_AND_PLACE)
    assert p["ok"] is True  # previews still work while a stop is latched
    assert ex.execute(p["plan_id"])["refusal"]["code"] == "STOP_LATCHED"
    ex.release_stop()
    assert ex.execute(p["plan_id"])["ok"] is True

    slow = Executor(config, SimTransport(config, realtime=True))
    seen = record(slow)
    mover, _ = run_slowly(slow, slow.preview_sequence(PICK_AND_PLACE, speed=0.3)["plan_id"])
    try:
        wait_for_step(seen, 0)
        assert slow.preview_sequence(PICK_AND_PLACE)["refusal"]["code"] == "BUSY"
    finally:
        slow.stop()
        mover.join(10)


def test_a_sequence_cannot_widen_the_envelope(config: RobotConfig) -> None:
    """Every step meets the same limits as the tool on its own: one that would be refused alone is refused here."""
    ex = Executor(config, SimTransport(config))
    for step in (
        {"tool": "move_to", "x_mm": 300, "y_mm": 0, "z_mm": 10},
        {"tool": "move_to", "x_mm": 900, "y_mm": 0, "z_mm": 100},
        {"tool": "move_joints", "joints": {"yaw": 175}},
    ):
        alone = ex.preview_sequence([step])["refusal"]["code"]
        single = ex.move_to(**{k: v for k, v in step.items() if k != "tool"}) if step["tool"] == "move_to" else None
        assert alone == (single["refusal"]["code"] if single else "JOINT_LIMIT")


# ---------------------------------------------------------------------- physics
@pytest.mark.menagerie
def test_pick_and_place_on_the_panda_under_physics() -> None:
    """The Panda settles about 6 mm from each planned end; its planning margin is 10 mm, so no step is stale."""
    config = load("panda")
    ex = Executor(config, SimTransport(config))
    steps: list[dict[str, Any]] = [
        {"tool": "gripper", "opening": 1},
        {"tool": "move_to", "x_mm": 450, "y_mm": 100, "z_mm": 250},
        {"tool": "move_to", "x_mm": 450, "y_mm": 100, "z_mm": 150, "path": "line"},
        {"tool": "gripper", "opening": 0},
        {"tool": "move_to", "x_mm": 450, "y_mm": 100, "z_mm": 250, "path": "line"},
        {"tool": "move_to", "x_mm": 450, "y_mm": -150, "z_mm": 250},
    ]
    p = ex.preview_sequence(steps)
    assert p["ok"] is True, p.get("refusal")
    r = ex.execute(p["plan_id"])
    assert r["ok"] is True, r.get("refusal")
    assert r["sequence"]["completed"] == list(range(6))
    assert r["executed"]["min_clearance_mm"] >= config.limits.min_clearance_mm
    assert all(s["executed"]["tool_tracking_error_mm"] < config.limits.planning_margin_mm for s in r["steps"])
    tool = r["state"]["tool"]
    assert math.dist((tool["x_mm"], tool["y_mm"], tool["z_mm"]), (450, -150, 250)) < 20
    assert r["state"]["gripper_opening"] < 0.1

# ruff: noqa: RUF001  (zh-TW text uses full-width punctuation on purpose)
"""Human-robot interaction: preview then confirm, jog, named poses, pause/resume, events, human messages."""

from __future__ import annotations

import json
import math
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mjguard.config import ConfigError, RobotConfig, from_dict
from mjguard.events import EventBus
from mjguard.executor import Executor, SimTransport
from mjguard.guard import CODES, Guard
from mjguard.messages import LANGS, TEMPLATES, humanize, refusal_text
from mjguard.poses import PoseStore

HERE = Path(__file__).parent
BIN = {"name": "bin", "shape": "box", "size_mm": [120, 160, 140], "center_mm": [300, 0, 70]}


def make_config(**extra: Any) -> RobotConfig:
    data = tomllib.loads((HERE / "three_link.toml").read_text())
    data.update(extra)
    return from_dict(data, HERE)


def tool_at(r: dict[str, Any], x: float, y: float, z: float, tol: float = 2.0) -> bool:
    t = r["state"]["tool"]
    return math.dist((t["x_mm"], t["y_mm"], t["z_mm"]), (x, y, z)) < tol


def start_slow_sweep(executor: Executor) -> tuple[threading.Thread, dict[str, Any]]:
    out: dict[str, Any] = {}
    mover = threading.Thread(target=lambda: out.update(r=executor.move_joints({"yaw": -150}, speed=0.1)))
    mover.start()
    time.sleep(0.3)
    return mover, out


# ---------------------------------------------------------------------- preview, then confirm
def test_preview_moves_nothing_and_execute_runs_it(executor: Executor) -> None:
    before = executor.get_state()["state"]
    p = executor.move_to(300, 50, 120, preview=True)
    assert p["ok"] is True and p["executed"] is False and p["expires_in_s"] == 60
    assert executor.get_state()["state"] == before
    plan = p["plan"]
    assert plan["waypoints"] > 1 and plan["duration_s"] > 0 and plan["min_clearance_mm"] >= 30
    assert plan["min_obstacle_gap_mm"] is None  # no obstacles configured
    assert 2 <= len(plan["path_mm"]) <= 50
    assert plan["path_mm"][0] == plan["start_tool"]["mm"]
    assert math.dist(plan["path_mm"][-1], (300, 50, 120)) < 2
    assert math.dist(plan["end_tool"]["mm"], (300, 50, 120)) < 2

    r = executor.execute(p["plan_id"])
    assert r["ok"] is True and tool_at(r, 300, 50, 120)
    again = executor.execute(p["plan_id"])
    assert again["refusal"]["code"] == "PLAN_USED"
    assert executor.execute("nope")["refusal"]["code"] == "PLAN_UNKNOWN"


def test_execute_refuses_expired_and_stale_plans(config: RobotConfig) -> None:
    now = [1000.0]
    ex = Executor(config, SimTransport(config), clock=lambda: now[0])
    p = ex.move_joints({"yaw": 20}, preview=True)
    now[0] += 61
    assert ex.execute(p["plan_id"])["refusal"]["code"] == "PLAN_EXPIRED"
    assert ex.execute(p["plan_id"])["refusal"]["code"] == "PLAN_UNKNOWN"  # expired plans are dropped

    p = ex.move_joints({"yaw": 20}, preview=True)
    assert ex.jog_joint("shoulder", 2)["ok"] is True  # the arm moves after the preview
    stale = ex.execute(p["plan_id"])
    assert stale["refusal"]["code"] == "PLAN_STALE" and stale["refusal"]["details"]["joint"] == "shoulder"
    assert ex.execute(p["plan_id"])["refusal"]["code"] == "PLAN_USED"  # one shot, even when refused


def test_preview_of_an_unsafe_move_is_a_refusal_with_a_suggestion(executor: Executor) -> None:
    r = executor.move_to(300, 0, 5, preview=True)
    assert r["ok"] is False and "plan_id" not in r
    assert r["refusal"]["code"] == "TABLE_CLEARANCE"
    s = r["refusal"]["suggestion"]
    follow = executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"], tool_down=s["tool_down"], preview=True)
    assert follow["ok"] is True and executor.execute(follow["plan_id"])["ok"] is True


def test_execute_rechecks_the_plan_against_the_world_now(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config))
    p = ex.move_to(300, 0, 60, preview=True)  # fine without the bin
    assert p["ok"] is True
    ex.guard = Guard(make_config(obstacles=[BIN]))  # the bin is put on the table after the preview
    r = ex.execute(p["plan_id"])
    assert r["ok"] is False and r["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"}
    assert "executed" not in r


def test_require_confirmation_turns_every_motion_into_a_preview(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config), require_confirmation=True)
    before = ex.get_state()["state"]
    for r in (
        ex.move_to(300, 50, 120),
        ex.move_joints({"yaw": 10}),
        ex.gripper(0.0),
        ex.rest(),
        ex.jog(0, 0, 10),
        ex.jog_joint("yaw", 5),
    ):
        assert r["ok"] is True and r["executed"] is False and "plan_id" in r
    assert ex.get_state()["state"] == before
    r = ex.execute(ex.move_to(300, 50, 120)["plan_id"])
    assert r["ok"] is True and tool_at(r, 300, 50, 120)
    from_file = Executor(make_config(ui={"require_confirmation": True}), SimTransport(config))
    assert from_file.require_confirmation is True and from_file.rest()["executed"] is False


def test_previewed_gripper_plan_runs(executor: Executor) -> None:
    p = executor.gripper(0.0, preview=True)
    assert p["plan"]["gripper_opening_goal"] == 0.0
    assert executor.execute(p["plan_id"])["state"]["gripper_opening"] < 0.01


def test_plan_store_is_bounded(executor: Executor) -> None:
    first = executor.rest(preview=True)["plan_id"]
    for _ in range(40):
        executor.rest(preview=True)
    assert executor.execute(first)["refusal"]["code"] == "PLAN_UNKNOWN"


# ---------------------------------------------------------------------- jog
def test_jog_moves_relative_in_a_straight_line(executor: Executor) -> None:
    assert executor.move_to(300, 0, 150)["ok"] is True
    r = executor.jog(10, -20, 30)
    assert r["ok"] is True and tool_at(r, 310, -20, 180)
    assert r["state"]["tool"]["axis"][2] < -0.99  # tool_down=False keeps the direction it had


def test_jog_caps_and_bad_values(executor: Executor) -> None:
    assert executor.jog(40, 40, 0)["refusal"]["code"] == "BAD_TARGET"  # 56.6 mm total
    assert executor.jog(math.nan, 0, 0)["refusal"]["code"] == "BAD_TARGET"
    assert executor.jog_joint("yaw", 10.5)["refusal"]["code"] == "BAD_TARGET"
    assert executor.jog_joint("nope", 1)["refusal"]["code"] == "BAD_TARGET"
    assert executor.jog_joint("yaw", math.inf)["refusal"]["code"] == "BAD_TARGET"
    r = executor.jog_joint("yaw", -10)
    assert r["ok"] is True and abs(r["state"]["joints"]["yaw"] + 10) < 0.2


def test_jog_is_checked_and_its_suggestion_passes(executor: Executor) -> None:
    assert executor.move_to(300, 0, 60)["ok"] is True
    r = executor.jog(0, 0, -50)
    assert r["refusal"]["code"] == "TABLE_CLEARANCE"
    s = r["refusal"]["suggestion"]
    assert s["tool"] == "move_to" and s["path"] == "line"
    follow = executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"], tool_down=s["tool_down"], path=s["path"])
    assert follow["ok"] is True, follow.get("refusal")


def test_jog_joint_respects_joint_limits(executor: Executor) -> None:
    assert executor.move_joints({"yaw": 166})["ok"] is True
    assert executor.jog_joint("yaw", 5)["refusal"]["code"] == "JOINT_LIMIT"


# ---------------------------------------------------------------------- named poses
def test_poses_round_trip_and_persist(config: RobotConfig, tmp_path: Path) -> None:
    path = tmp_path / "poses.json"
    ex = Executor(config, SimTransport(config), poses=PoseStore(config.joints, path))
    assert ex.move_to(300, 50, 120)["ok"] is True
    saved = ex.save_pose("pick_1")
    assert saved["ok"] is True and saved["replaced"] is False and saved["pose"]["gripper_opening"] == 1.0
    assert ex.save_pose("pick_1")["replaced"] is True
    assert ex.save_pose("bad name!")["refusal"]["code"] == "BAD_TARGET"
    assert ex.rest()["ok"] is True
    p = ex.go_to_pose("pick_1", preview=True)
    assert p["ok"] is True and math.dist(p["plan"]["end_tool"]["mm"], (300, 50, 120)) < 2
    assert tool_at(ex.execute(p["plan_id"]), 300, 50, 120)
    assert list(ex.list_poses()["poses"]) == ["pick_1"]

    reloaded = Executor(config, SimTransport(config), poses=PoseStore(config.joints, path))
    assert tool_at(reloaded.go_to_pose("pick_1"), 300, 50, 120)
    assert reloaded.delete_pose("pick_1")["ok"] is True
    assert reloaded.delete_pose("pick_1")["refusal"]["code"] == "POSE_UNKNOWN"
    assert reloaded.go_to_pose("pick_1")["refusal"]["code"] == "POSE_UNKNOWN"
    assert json.loads(path.read_text()) == {"poses": {}}


def test_gripper_follows_a_pose_only_when_asked(executor: Executor) -> None:
    assert executor.save_pose("open")["pose"]["gripper_opening"] == 1.0
    assert executor.gripper(0.0)["ok"] is True
    assert executor.go_to_pose("open")["state"]["gripper_opening"] < 0.01  # holding an object stays holding it
    assert executor.go_to_pose("open", include_gripper=True)["state"]["gripper_opening"] > 0.99


def test_a_saved_pose_that_became_unsafe_is_refused(config: RobotConfig, tmp_path: Path) -> None:
    path = tmp_path / "poses.json"
    ex = Executor(config, SimTransport(config), poses=PoseStore(config.joints, path))
    assert ex.move_to(300, 0, 60)["ok"] is True
    assert ex.save_pose("low")["ok"] is True

    with_bin = make_config(obstacles=[BIN])
    later = Executor(with_bin, SimTransport(with_bin), poses=PoseStore(with_bin.joints, path))
    for r in (later.go_to_pose("low"), later.go_to_pose("low", preview=True)):
        assert r["ok"] is False and r["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"}


def test_pose_file_is_validated(config: RobotConfig, tmp_path: Path) -> None:
    path = tmp_path / "poses.json"
    joints = {"yaw": 0, "shoulder": 0, "elbow": 0, "wrist": 0}
    bad_files: list[Any] = [
        "not json",
        {"nope": {}},
        {"poses": {"a b": {"joints": joints}}},
        {"poses": {"p": {"joints": {"yaw": 0}}}},
        {"poses": {"p": {"joints": {**joints, "yaw": "x"}}}},
        {"poses": {"p": {"joints": joints, "gripper_opening": 2}}},
        {"poses": {"p": []}},
    ]
    for bad in bad_files:
        path.write_text(bad if isinstance(bad, str) else json.dumps(bad))
        with pytest.raises(ConfigError):
            PoseStore(config.joints, path)
    # Out-of-range joints load (a pose is only a target) but going there is refused like any other target.
    path.write_text(json.dumps({"poses": {"far": {"joints": {**joints, "yaw": 400}}}}))
    ex = Executor(config, SimTransport(config), poses=PoseStore(config.joints, path))
    assert ex.go_to_pose("far")["refusal"]["code"] == "JOINT_LIMIT"


# ---------------------------------------------------------------------- pause / resume
def test_pause_then_resume_completes(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    events: list[str] = []
    ex.events.subscribe(lambda e: events.append(e["type"]))
    mover, out = start_slow_sweep(ex)
    assert ex.pause()["was_moving"] is True
    mover.join(5)
    r = out["r"]
    assert r["refusal"]["code"] == "PAUSED" and r["paused"] is True
    state = ex.get_state()
    assert state["state"]["paused"] is True and state["paused_motion"]["remaining_waypoints"] > 0
    ex.transport.realtime = False  # type: ignore[attr-defined]
    done = ex.resume()
    assert done["ok"] is True and abs(done["state"]["joints"]["yaw"] + 150) < 0.2
    assert ex.get_state()["state"]["paused"] is False
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"
    assert ex.events.flush()
    flow = [t for t in events if t != "progress"]
    assert flow == [
        "motion_started",
        "paused",
        "motion_finished",
        "resumed",
        "motion_started",
        "motion_finished",
        "refused",  # the second resume
    ]


def test_resume_after_the_world_changed_is_refused(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    mover, out = start_slow_sweep(ex)
    ex.pause()
    mover.join(5)
    assert out["r"]["refusal"]["code"] == "PAUSED"
    # Someone puts a box where the rest of the sweep goes.
    end = ex.guard.model.tool(ex.guard.model.home + np.radians([-150, 0, 0, 0]))[0]
    box = {
        "name": "box",
        "shape": "sphere",
        "size_mm": [80],
        "center_mm": [end[0] * 1000, end[1] * 1000, end[2] * 1000],
    }
    ex.guard = Guard(make_config(obstacles=[box]))
    r = ex.resume()
    assert r["ok"] is False and r["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"}
    assert ex.get_state()["state"]["paused"] is True  # kept, so a person can decide: stop() discards it


def test_stop_discards_a_paused_motion(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    mover, _ = start_slow_sweep(ex)
    ex.pause()
    mover.join(5)
    stopped = ex.stop()
    assert stopped["discarded_paused_motion"] is True and stopped["was_moving"] is False
    assert ex.get_state()["state"]["paused"] is False
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"


def test_stop_wins_over_pause_and_pause_when_idle_is_harmless(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    assert ex.pause()["was_moving"] is False
    mover, out = start_slow_sweep(ex)
    ex.pause()
    ex.stop()
    mover.join(5)
    assert out["r"]["refusal"]["code"] == "STOPPED"
    assert ex.get_state()["state"]["paused"] is False


def test_a_new_motion_replaces_a_paused_one(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    mover, _ = start_slow_sweep(ex)
    ex.pause()
    mover.join(5)
    ex.transport.realtime = False  # type: ignore[attr-defined]
    assert ex.rest()["ok"] is True
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"


def test_busy_applies_to_every_new_entry_point(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    mover, _ = start_slow_sweep(ex)
    try:
        assert ex.move_to(300, 0, 150, preview=True)["refusal"]["code"] == "BUSY"
        assert ex.execute("x")["refusal"]["code"] == "BUSY"
        assert ex.resume()["refusal"]["code"] == "BUSY"
        assert ex.save_pose("p")["refusal"]["code"] == "BUSY"
    finally:
        ex.stop()
        mover.join(5)


# ---------------------------------------------------------------------- events
def test_event_order_for_a_normal_and_a_refused_move(executor: Executor) -> None:
    seen: list[dict[str, Any]] = []
    executor.events.subscribe(seen.append)
    assert executor.move_to(300, 50, 120)["ok"] is True
    assert executor.events.flush()
    types = [e["type"] for e in seen]
    assert types[0] == "motion_started" and types[-1] == "motion_finished" and "progress" in types
    progress = next(e for e in seen if e["type"] == "progress")
    assert {"tool", "joints", "min_clearance_mm"} <= set(progress["state"])
    assert seen[-1]["ok"] is True
    assert [e["seq"] for e in seen] == sorted(e["seq"] for e in seen)

    seen.clear()
    assert executor.move_to(300, 0, 5)["ok"] is False
    assert executor.events.flush()
    assert [e["type"] for e in seen] == ["refused"]
    assert seen[0]["refusal"]["code"] == "TABLE_CLEARANCE" and "message_human" in seen[0]["refusal"]

    seen.clear()
    executor.save_pose("a")
    executor.delete_pose("a")
    assert executor.events.flush()
    assert [e["type"] for e in seen] == ["pose_saved", "pose_deleted"]


def test_breach_and_stop_events(config: RobotConfig) -> None:
    transport = SimTransport(config)
    real_follow = transport.follow

    def sagging_follow(waypoints, gripper, period, should_stop, on_waypoint):  # type: ignore[no-untyped-def]
        floor = np.radians([0, -95, -145, -35])
        sag = [w + (floor - w) * min(1.0, i / 3) for i, w in enumerate(waypoints)]
        return real_follow(sag, gripper, period, should_stop, on_waypoint)

    transport.follow = sagging_follow  # type: ignore[method-assign]
    ex = Executor(config, transport)
    seen: list[str] = []
    ex.events.subscribe(lambda e: seen.append(e["type"]))
    assert ex.move_joints({"yaw": 90})["refusal"]["code"] == "ENVELOPE_BREACH"
    assert ex.events.flush()
    assert [t for t in seen if t != "progress"] == ["motion_started", "breach", "motion_finished"]

    ex = Executor(config, SimTransport(config, realtime=True))
    seen = []
    ex.events.subscribe(lambda e: seen.append(e["type"]))
    mover, _ = start_slow_sweep(ex)
    ex.stop()
    mover.join(5)
    assert ex.events.flush()
    assert [t for t in seen if t != "progress"] == ["motion_started", "stopped", "motion_finished"]


def test_transport_fault_still_finishes_the_motion_event(config: RobotConfig) -> None:
    transport = SimTransport(config)

    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("serial link dropped")

    transport.follow = broken  # type: ignore[method-assign]
    ex = Executor(config, transport)
    seen: list[dict[str, Any]] = []
    ex.events.subscribe(seen.append)
    assert ex.move_joints({"yaw": 10})["refusal"]["code"] == "INTERNAL_ERROR"
    assert ex.events.flush()
    assert [e["type"] for e in seen] == ["motion_started", "motion_finished", "refused"]
    assert seen[1]["code"] == "INTERNAL_ERROR"


def test_event_bus_queues_callbacks_and_history() -> None:
    bus = EventBus(keep=3)
    sub = bus.subscribe_queue(maxsize=2)
    calls: list[int] = []
    unsubscribe = bus.subscribe(lambda e: calls.append(e["seq"]))

    def boom(e: dict[str, Any]) -> None:
        raise RuntimeError("bad subscriber")

    bus.subscribe(boom)  # a failing subscriber never breaks the publisher
    for i in range(4):
        bus.publish("progress", i=i)
    assert bus.flush()
    assert calls == [1, 2, 3, 4]
    assert [e["seq"] for e in bus.latest()] == [2, 3, 4]
    assert [e["seq"] for e in bus.latest(10, after=3)] == [4]
    assert bus.latest(0) == []
    first = sub.get(0)
    assert first is not None and first["seq"] == 1
    assert sub.get(0) is not None and sub.get(0) is None
    assert sub.dropped == 2
    unsubscribe()
    unsubscribe()
    sub.close()
    sub.close()
    bus.publish("stopped")
    assert calls == [1, 2, 3, 4] and sub.get(0) is None


def test_events_cross_threads() -> None:
    bus = EventBus()
    sub = bus.subscribe_queue()
    threading.Thread(target=lambda: bus.publish("pose_saved", name="x")).start()
    event = sub.get(2)
    assert event is not None and event["name"] == "x"


# ---------------------------------------------------------------------- human messages
@pytest.mark.parametrize("lang", LANGS)
def test_every_refusal_code_has_a_human_message(lang: str) -> None:
    for code in CODES:
        assert code in TEMPLATES[lang], f"{lang}: no template for {code}"
        assert refusal_text(code, {}, lang).strip()
    assert set(TEMPLATES[lang]) == set(CODES)


def test_zh_tw_messages_are_plain_and_specific(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config), lang="zh-TW")
    r = ex.move_to(300, 0, 5)["refusal"]
    assert r["message_human"] == "會太靠近桌面（還差 25 mm），已停止規劃"
    assert r["message"].startswith("target z is")  # the model still gets English with numbers
    assert "可以改去途中安全的位置" in r["suggestion"]["human"]

    with_bin = make_config(obstacles=[BIN], ui={"lang": "zh-TW"})
    ex = Executor(with_bin, SimTransport(with_bin))
    assert ex.lang == "zh-TW"
    r = ex.move_to(300, 0, 150)["refusal"]
    assert r["code"] == "OBSTACLE_CLEARANCE" and r["message_human"] == "會太靠近 bin（還差 10 mm），已停止規劃"
    en = humanize(
        {"code": "OBSTACLE_CLEARANCE", "details": {"obstacle": "obstacle 'bin'", "gap_mm": 5, "limit_mm": 20}}, "en"
    )
    assert en["message_human"] == "Too close to bin (short by 15 mm). Not moving."
    assert refusal_text("NO_SUCH_CODE", {}, "zh-TW") == "已停止"
    assert refusal_text("NO_SUCH_CODE", {}, "en") == "Not moving."
    assert refusal_text("TABLE_CLEARANCE", {}, "en") == "Too close to the table. Not moving."
    assert (
        refusal_text("TABLE_CLEARANCE", {"clearance_mm": 10, "limit_mm": 30}, "zh-TW")
        == "會太靠近桌面（還差 20 mm），已停止規劃"
    )


def test_ui_config_is_validated(config: RobotConfig) -> None:
    with pytest.raises(ConfigError, match=r"ui\.lang"):
        make_config(ui={"lang": "fr"})
    with pytest.raises(ConfigError, match=r"\[ui\]"):
        make_config(ui={"colour": "red"})
    with pytest.raises(ConfigError, match="require_confirmation"):
        make_config(ui={"require_confirmation": "yes"})
    with pytest.raises(ValueError, match="lang"):
        Executor(config, SimTransport(config), lang="fr")


def test_a_slow_callback_never_delays_a_stop(config: RobotConfig) -> None:
    """Callbacks run on their own thread: one that blocks must not slow the follow loop or a stop."""
    import time as _time

    ex = Executor(config, SimTransport(config, realtime=True))
    ex.events.subscribe(lambda e: _time.sleep(1.0))  # would add ~1 s per event on the publisher's thread
    t0 = _time.monotonic()
    mover, _ = start_slow_sweep(ex)
    ex.stop()
    mover.join(5)
    assert _time.monotonic() - t0 < 2.0


def test_a_new_motion_reports_that_it_replaced_a_paused_one(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config, realtime=True))
    mover, out = start_slow_sweep(ex)
    ex.pause()
    mover.join(5)
    assert out["r"]["refusal"]["code"] == "PAUSED"
    r = ex.move_joints({"yaw": 0})
    assert r["ok"] is True and r["discarded_paused_motion"] is True
    assert ex.resume()["refusal"]["code"] == "NOTHING_TO_RESUME"


def test_zh_tw_obstacle_names_are_spaced_only_when_latin() -> None:
    from mjguard.messages import refusal_text

    d = {"gap_mm": 5.0, "limit_mm": 15.0}
    assert refusal_text("OBSTACLE_CLEARANCE", {**d, "obstacle": "obstacle '零件盒'"}, "zh-TW").startswith(
        "會太靠近零件盒（"
    )
    assert refusal_text("OBSTACLE_CLEARANCE", {**d, "obstacle": "obstacle 'bin'"}, "zh-TW").startswith("會太靠近 bin（")


def test_latched_stop_refuses_motion_until_released(config: RobotConfig) -> None:
    ex = Executor(config, SimTransport(config), latch_stop=True)
    seen: list[dict[str, Any]] = []
    ex.events.subscribe(seen.append)
    r = ex.stop()  # idle stop
    assert r["latched"] is True
    assert ex.get_state()["state"]["stop_latched"] is True
    assert ex.move_joints({"yaw": 10})["refusal"]["code"] == "STOP_LATCHED"
    preview = ex.move_joints({"yaw": 10}, preview=True)  # previews still work: nothing moves
    assert preview["ok"] is True
    assert ex.execute(preview["plan_id"])["refusal"]["code"] == "STOP_LATCHED"
    assert ex.resume()["refusal"]["code"] == "STOP_LATCHED"
    assert ex.release_stop()["was_latched"] is True
    assert ex.move_joints({"yaw": 10})["ok"] is True
    assert ex.events.flush()
    types = [e["type"] for e in seen]
    assert types[0] == "stopped" and "stop_released" in types  # an idle stop is visible to every client


def test_unlatched_stop_is_the_default_and_idle_stop_publishes(executor: Executor) -> None:
    seen: list[str] = []
    executor.events.subscribe(lambda e: seen.append(e["type"]))
    assert executor.stop()["latched"] is False
    assert executor.move_joints({"yaw": 5})["ok"] is True
    assert executor.events.flush()
    assert seen[0] == "stopped"


def test_discard_makes_a_preview_unusable(executor: Executor) -> None:
    preview = executor.move_joints({"yaw": 10}, preview=True)
    assert executor.discard(preview["plan_id"])["discarded"] is True
    assert executor.execute(preview["plan_id"])["refusal"]["code"] == "PLAN_UNKNOWN"
    assert executor.discard("nope")["discarded"] is False


def test_sse_ids_carry_the_server_run() -> None:
    from mjguard.server import parse_after

    assert parse_after(None, "abc") is None
    assert parse_after("7", "abc") == 7
    assert parse_after("abc:7", "abc") == 7
    assert parse_after("old:900", "abc") == 0  # a restarted server: replay from the start, don't skip new events

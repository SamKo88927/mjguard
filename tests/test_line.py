from __future__ import annotations

import numpy as np

from mjguard.executor import Executor


def tool_positions(executor: Executor, waypoints: list[np.ndarray]) -> np.ndarray:
    return np.array([executor.guard.model.tool(q)[0] for q in waypoints])


def test_line_keeps_the_tool_on_the_segment(executor: Executor) -> None:
    assert executor.move_to(280, 60, 220)["ok"] is True
    q, g = executor.transport.read()
    start = executor.guard.model.tool(q)[0]
    target = np.array([0.28, 0.06, 0.09])  # straight down 130 mm
    plan = executor.guard.plan_line(q, target, np.array([0.0, 0.0, -1.0]), g)
    assert not hasattr(plan, "code"), plan
    pts = tool_positions(executor, plan.waypoints)  # type: ignore[union-attr]
    seg = target - start
    t = np.clip((pts - start) @ seg / (seg @ seg), 0, 1)
    off_line_mm = np.linalg.norm(pts - (start + np.outer(t, seg)), axis=1) * 1000
    assert off_line_mm.max() < 2.0
    r = executor.move_to(280, 60, 90, path="line")
    assert r["ok"] is True and abs(r["state"]["tool"]["z_mm"] - 90) < 2


def test_line_into_the_table_is_refused_with_a_line_suggestion(executor: Executor) -> None:
    assert executor.move_to(280, 0, 200)["ok"] is True
    r = executor.move_to(280, 0, 0, path="line")
    assert r["refusal"]["code"] == "TABLE_CLEARANCE"
    s = r["refusal"]["suggestion"]
    assert s["path"] == "line"
    follow = executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"], tool_down=s["tool_down"], path=s["path"])
    assert follow["ok"] is True


def test_line_out_of_reach_is_refused(executor: Executor) -> None:
    r = executor.move_to(580, 0, 200, path="line")
    assert r["refusal"]["code"] == "UNREACHABLE"


def test_bad_path_value(executor: Executor) -> None:
    assert executor.move_to(280, 0, 200, path="curvy")["refusal"]["code"] == "BAD_TARGET"

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from mjguard.config import ConfigError, RobotConfig, from_dict
from mjguard.executor import Executor, SimTransport

HERE = Path(__file__).parent


def with_obstacles(*obstacles: dict[str, object]) -> RobotConfig:
    data = tomllib.loads((HERE / "three_link.toml").read_text())
    data["obstacles"] = list(obstacles)
    return from_dict(data, HERE)


BIN = {"name": "bin", "shape": "box", "size_mm": [120, 160, 140], "center_mm": [300, 0, 70]}


def test_target_inside_an_obstacle_is_refused_with_a_suggestion_outside_it() -> None:
    config = with_obstacles(BIN)
    ex = Executor(config, SimTransport(config))
    r = ex.move_to(300, 0, 60)
    assert r["ok"] is False
    assert r["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"}
    assert "bin" in r["refusal"]["message"]
    s = r["refusal"]["suggestion"]
    follow = ex.move_to(s["x_mm"], s["y_mm"], s["z_mm"], tool_down=s["tool_down"])
    assert follow["ok"] is True


def test_above_the_obstacle_is_fine_and_clearance_is_kept() -> None:
    config = with_obstacles(BIN)
    ex = Executor(config, SimTransport(config))
    r = ex.move_to(300, 0, 260)
    assert r["ok"] is True, r.get("refusal")
    gap, _, _ = ex.guard.model.obstacle_clearance(*ex.transport.read())
    assert gap >= config.limits.obstacle_clearance_mm


def test_the_same_target_passes_without_the_obstacle() -> None:
    config = with_obstacles()
    ex = Executor(config, SimTransport(config))
    assert ex.move_to(300, 0, 60)["ok"] is True


def test_obstacles_appear_in_limits_and_shapes_are_validated() -> None:
    config = with_obstacles(
        BIN, {"name": "post", "shape": "cylinder", "size_mm": [40, 200], "center_mm": [0, 300, 100]}
    )
    limits = Executor(config, SimTransport(config)).get_limits()["limits"]
    assert [o["name"] for o in limits["obstacles"]] == ["bin", "post"]
    with pytest.raises(ConfigError, match="needs 3 positive"):
        with_obstacles({"name": "bad", "shape": "box", "size_mm": [10, 10], "center_mm": [0, 0, 0]})
    with pytest.raises(ConfigError, match="unique"):
        with_obstacles(BIN, BIN)


def test_touching_an_obstacle_at_rest_is_not_allowed_as_a_design_contact() -> None:
    """Rest-pose contacts are allowed as design features, but never contacts with a configured obstacle."""
    config = with_obstacles({"name": "wall", "shape": "box", "size_mm": [40, 400, 600], "center_mm": [330, 0, 300]})
    ex = Executor(config, SimTransport(config))
    refusal, _ = ex.guard.check_configuration(ex.guard.model.home, ex.guard.model.home_gripper)
    assert refusal is not None and "wall" in refusal.message
    assert ex.guard.model.describe_rest_contacts() == []


def test_arm_that_settled_inside_the_margin_can_move_away_but_not_closer() -> None:
    """Servo sag can leave the measured arm inside the planning margin. It must not be stuck there."""
    import numpy as np

    config = with_obstacles(BIN)
    transport = SimTransport(config)
    ex = Executor(config, transport)
    assert ex.move_to(300, 0, 260)["ok"] is True
    model = ex.guard.model
    q, g = transport.read()
    for z_mm in range(258, 140, -2):  # lower the tool, unchecked, until it sits inside the 10 mm margin
        target = np.array([0.3, 0.0, model.floor_z + z_mm / 1000])
        q, _, _ = model.solve_ik(target, np.array([0.0, 0.0, -1.0]), q, ex.guard.lower, ex.guard.upper)
        gap = model.obstacle_clearance(q, g)[0]
        if 1 < gap < 4:
            break
    else:
        raise AssertionError("could not place the arm inside the margin")
    transport.d.qpos[model.qadr] = q
    import mujoco

    mujoco.mj_forward(transport.m, transport.d)
    closer = ex.move_to(300, 0, 140)
    assert closer["refusal"]["code"] in {"OBSTACLE_CLEARANCE", "COLLISION"}
    # Straight up in 5 mm steps: the first steps are still inside the margin, but each is farther than the start.
    away = ex.move_to(300, 0, 320, path="line")
    assert away["ok"] is True, away.get("refusal")

from __future__ import annotations

import math

import numpy as np
import pytest

from mjguard.config import RobotConfig
from mjguard.guard import Guard, Refusal

Q = {  # found by a grid search over the test arm; each triggers one refusal
    "COLLISION": (0, -95, -145, -115),
    "TABLE_COLLISION": (0, -95, -145, -35),
    "TABLE_CLEARANCE": (0, -95, -30, 30),
}


@pytest.fixture
def guard(config: RobotConfig) -> Guard:
    return Guard(config)


@pytest.mark.parametrize("code", sorted(Q))
def test_configuration_refusals(guard: Guard, code: str) -> None:
    refusal, _ = guard.check_configuration(np.radians(Q[code]), guard.model.home_gripper)
    assert refusal is not None and refusal.code == code


def test_joint_limit_uses_margin(guard: Guard) -> None:
    q = guard.model.home.copy()
    q[1] = math.radians(99.5)  # inside the model range (100), outside the 2 degree margin
    refusal = guard.check_joints(q)
    assert refusal is not None and refusal.code == "JOINT_LIMIT" and refusal.details["joint"] == "shoulder"


def test_path_is_checked_at_every_waypoint(guard: Guard) -> None:
    # Both ends pass on their own; the straight joint-space line between them dips toward the floor.
    start, goal = np.radians((0, -90, -20, 0)), np.radians((0, -45, -140, 0))
    g = guard.model.home_gripper
    assert guard.check_configuration(start, g)[0] is None and guard.check_configuration(goal, g)[0] is None
    plan = guard.check_path(start, goal, g)
    assert isinstance(plan, Refusal) and plan.code == "TABLE_CLEARANCE"
    done, total = plan.details["waypoint"].split("/")
    assert int(done) < int(total)


def test_unreachable_and_bad_target(guard: Guard) -> None:
    home = guard.model.home
    far = guard.plan_pose(home, np.array([0.0, 0.0, 0.95]), None)
    assert getattr(far, "code", None) == "UNREACHABLE"
    bad = guard.plan_pose(home, np.array([math.nan, 0, 0.1]), None)
    assert getattr(bad, "code", None) == "BAD_TARGET"


def test_workspace(guard: Guard) -> None:
    refusal = guard.plan_pose(guard.model.home, np.array([0.7, 0.0, 0.2]), None)
    assert getattr(refusal, "code", None) == "WORKSPACE"


def test_describe_lists_joints_and_codes(guard: Guard) -> None:
    d = guard.describe()
    assert set(d["joints"]) == {"yaw", "shoulder", "elbow", "wrist"}
    assert "ENVELOPE_BREACH" in d["refusal_codes"]


def test_an_arm_halted_in_contact_may_move_away_but_not_deeper(guard: Guard) -> None:
    """The monitor can halt an arm that is already touching itself; it must not be stuck there."""
    g = guard.model.home_gripper
    touching = np.radians(Q["COLLISION"])
    assert guard.check_configuration(touching, g)[0] is not None  # normally refused
    limits = guard.thresholds(touching, g)
    assert limits[2], "the start contact is recorded"
    away = touching + np.radians([0, 10, 0, 0])  # shoulder up: every start contact gets no deeper
    deeper = touching + np.radians([0, 0, 0, 10])  # wrist in: the hand digs into the base
    assert guard.check_configuration(away, g, limits)[0] is None
    refusal = guard.check_configuration(deeper, g, limits)[0]
    assert refusal is not None and refusal.code == "COLLISION"
    assert guard.check_configuration(away, g)[0] is not None  # without the start contacts it is still refused

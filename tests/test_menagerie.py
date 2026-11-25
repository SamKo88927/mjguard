"""Every built-in preset, on the real robot model from MuJoCo Menagerie, simulated with its own position servos.

Each preset must: pass its rest pose, reach a target near its own rest pose, refuse a target in the table and one out
of reach, keep the measured arm above the hard clearance limit, and accept the suggestion it returns.
"""

from __future__ import annotations

import math

import pytest

from mjguard.config import load, presets
from mjguard.executor import Executor, SimTransport

pytestmark = pytest.mark.menagerie

# A reachable target for each robot: below and to the side of its rest pose, along the direction the arm faces.
TARGETS: dict[str, tuple[float, float, float]] = {
    "fr3": (430, 110, 312),
    "gen3": (354, 91, 217),
    "iiwa14": (518, 132, 142),
    "low_cost_robot_arm": (-115, -30, 84),
    "panda": (450, 100, 250),
    "piper": (287, 74, 191),
    "so_arm100": (47, -185, 150),
    "so_arm101": (90, -287, 130),
    "ur10e": (-272, 501, 347),
    "ur5e": (400, 150, 300),
    "xarm7": (310, 79, 174),
    "yam": (188, 48, 201),
    "z1": (69, 18, 146),
}


def test_every_preset_has_a_target() -> None:
    assert set(TARGETS) == set(presets())


@pytest.mark.parametrize("preset", sorted(TARGETS))
def test_preset(preset: str) -> None:
    config = load(preset)
    executor = Executor(config, SimTransport(config))
    hard = config.limits.min_clearance_mm
    guard = executor.guard
    assert guard.check_configuration(guard.model.home, guard.model.home_gripper)[0] is None
    if config.gripper is not None:  # at the rest pose, well clear of the table
        for opening in (1.0, 0.0, 0.5):
            g = executor.gripper(opening)
            assert g["ok"] is True, g["refusal"]
            assert abs(g["state"]["gripper_opening"] - opening) < 0.05

    target = TARGETS[preset]
    r = executor.move_to(*target)
    if not r["ok"]:
        r = executor.move_to(*target, tool_down=False)
    assert r["ok"] is True, r["refusal"]
    tool = r["state"]["tool"]
    assert math.dist((tool["x_mm"], tool["y_mm"], tool["z_mm"]), target) < 50  # physics: servos lag and sag
    assert r["executed"]["min_clearance_mm"] >= hard

    assert executor.move_to(target[0], target[1], 0)["refusal"]["code"] == "TABLE_CLEARANCE"
    far = executor.move_to(5000, 0, 200)
    assert far["refusal"]["code"] == "UNREACHABLE"
    s = far["refusal"].get("suggestion")
    if s is not None:
        follow = executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"], tool_down=s["tool_down"])
        assert follow["ok"] is True, follow["refusal"]
        assert follow["executed"]["min_clearance_mm"] >= hard
    line = executor.move_to(target[0], target[1], target[2] + 40, tool_down=r["state"]["tool"]["axis"][2] < -0.99)
    if line["ok"]:
        down = executor.move_to(
            target[0], target[1], target[2], tool_down=line["state"]["tool"]["axis"][2] < -0.99, path="line"
        )
        # A straight line is a stricter path than any-path; some arms (the Z1 folds its wrist into its upper arm)
        # cannot take it here, and must say why in geometric terms.
        assert down["ok"] is True or down["refusal"]["code"] in {"COLLISION", "TABLE_CLEARANCE", "UNREACHABLE"}
    assert executor.rest()["ok"] is True


def test_grippers_configured() -> None:
    with_gripper = {p for p in presets() if load(p).gripper is not None}
    assert with_gripper == {"panda", "so_arm100", "so_arm101", "low_cost_robot_arm", "piper", "yam"}


@pytest.mark.parametrize(("preset", "xy"), [("ur5e", (400, 150)), ("panda", (450, 100))])
def test_servo_sag_stays_above_the_hard_limit(preset: str, xy: tuple[float, float]) -> None:
    """Under physics the arm settles a few mm below the plan; the planning margin must absorb that."""
    config = load(preset)
    executor = Executor(config, SimTransport(config))
    s = executor.move_to(*xy, 0)["refusal"]["suggestion"]
    r = executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"], tool_down=s["tool_down"])
    assert r["ok"] is True, r
    assert r["executed"]["min_clearance_mm"] >= config.limits.min_clearance_mm
    assert r["state"]["min_clearance_mm"] >= config.limits.min_clearance_mm


def test_check_lists_contacts_allowed_from_the_rest_pose(capsys: pytest.CaptureFixture[str]) -> None:
    """Gen3's base and shoulder hulls overlap by design; check must show that, not hide it."""
    import json

    from mjguard.cli import main

    assert main(["check", "gen3"]) == 0
    report = json.loads(capsys.readouterr().out)
    pairs = [c["bodies"] for c in report["contacts_allowed_from_rest_pose"]]
    assert ["base_link", "shoulder_link"] in pairs

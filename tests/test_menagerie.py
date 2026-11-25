"""The same guard on real robot models from MuJoCo Menagerie, simulated with their own position servos."""

from __future__ import annotations

import pytest

pytest.importorskip("robot_descriptions")

from mjguard.config import load
from mjguard.executor import Executor, SimTransport

pytestmark = pytest.mark.menagerie


@pytest.mark.parametrize(("preset", "target"), [("ur5e", (400, 150, 300)), ("panda", (450, 100, 250))])
def test_preset_moves_refuses_and_recovers(preset: str, target: tuple[float, float, float]) -> None:
    config = load(preset)
    executor = Executor(config, SimTransport(config))
    r = executor.move_to(*target)
    assert r["ok"] is True, r
    assert r["executed"]["tool_tracking_error_mm"] < 10  # physics with the model's own servos, under gravity
    below = executor.move_to(target[0], target[1], 0)
    assert below["refusal"]["code"] == "TABLE_CLEARANCE"
    far = executor.move_to(3000, 0, 300)
    assert far["refusal"]["code"] == "UNREACHABLE"
    s = far["refusal"]["suggestion"]
    assert executor.move_to(s["x_mm"], s["y_mm"], s["z_mm"])["ok"] is True
    assert executor.rest()["ok"] is True


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

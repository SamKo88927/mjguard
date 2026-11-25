from __future__ import annotations

from pathlib import Path

import pytest

from mjguard.config import ConfigError, Limits, from_dict, load, presets
from mjguard.model import ArmModel

HERE = Path(__file__).parent
BASE = {
    "model": {"mjcf": "three_link.xml", "floor_geom": "floor"},
    "arm": {"joints": ["yaw", "shoulder", "elbow", "wrist"], "tcp_site": "tcp"},
}


def cfg(**overrides: dict[str, object]) -> dict[str, object]:
    data = {k: dict(v) for k, v in BASE.items()}
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return data


def test_presets_are_listed() -> None:
    assert {"panda", "ur5e"} <= set(presets())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"arm": {"joints": ["yaw", "nope"]}}, "joint 'nope'"),
        ({"arm": {"tcp_site": "nope"}}, "site 'nope'"),
        ({"model": {"floor_geom": "nope"}}, "floor geom 'nope'"),
        ({"arm": {"tcp_body": "hand"}}, "exactly one"),
        ({"arm": {"tool_axis": "w"}}, "tool_axis"),
        ({"arm": {"actuators": ["a_yaw"]}}, "one entry per joint"),
        ({"limits": {"min_clearance_mm": -1}}, "positive"),
        ({"limits": {"workspace_mm": {"x": [5, 1]}}}, "low < high"),
        ({"limits": {"bogus": 1}}, "[limits]"),
    ],
)
def test_bad_configs_name_the_problem(overrides: dict[str, dict[str, object]], message: str) -> None:
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        ArmModel(from_dict(cfg(**overrides), HERE))


def test_missing_file_and_unknown_preset() -> None:
    with pytest.raises(ConfigError, match="not a built-in preset"):
        load("no-such-robot")
    with pytest.raises(ConfigError, match="does not exist"):
        from_dict(cfg(model={"mjcf": "missing.xml"}), HERE)


def test_limits_are_frozen() -> None:
    limits = Limits()
    with pytest.raises(AttributeError):
        limits.min_clearance_mm = 0  # type: ignore[misc]


def test_turret_is_exempt_from_clearance_but_not_from_contacts() -> None:
    model = ArmModel(load(HERE / "three_link.toml"))
    turret = [g for g in model.moving_geoms if model.m.geom_bodyid[g] == model.m.body("turret").id]
    assert turret and not set(turret) & set(model.clearance_geoms)

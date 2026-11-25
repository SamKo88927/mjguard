"""Robot configuration: which MJCF, which joints, where the tool is, and the safety limits.

A config is a small TOML file. Paths are relative to the file. `mjcf` may also be `menagerie:<name>`, which resolves
to a MuJoCo Menagerie scene through the optional `robot_descriptions` package.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """The robot config is missing a field or names something the model does not have."""


@dataclass(frozen=True)
class Limits:
    """The safety envelope. Frozen at startup; the MCP surface has no way to change it."""

    min_clearance_mm: float = 20.0
    """Every moving collision geom and the tool point stay at least this far above the floor."""
    joint_margin_deg: float = 2.0
    """Planning keeps this far inside each joint's range."""
    max_step_deg: float = 3.0
    """Waypoint spacing: no joint moves more than this between two checked waypoints."""
    max_joint_speed_dps: float = 60.0
    workspace_mm: dict[str, tuple[float, float]] = field(default_factory=dict[str, tuple[float, float]])
    """Optional box for the tool point, e.g. {"x": (-400, 600)}. Missing axes are unbounded."""
    position_tolerance_mm: float = 1.0
    angle_tolerance_deg: float = 1.5

    def __post_init__(self) -> None:
        for name in ("min_clearance_mm", "max_step_deg", "max_joint_speed_dps", "position_tolerance_mm"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ConfigError(f"limits.{name} must be a positive number, got {value!r}")
        if not (math.isfinite(self.joint_margin_deg) and self.joint_margin_deg >= 0):
            raise ConfigError("limits.joint_margin_deg must be >= 0")
        for axis, bounds in self.workspace_mm.items():
            if axis not in ("x", "y", "z") or len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ConfigError(f"limits.workspace_mm.{axis} must be [low, high] with low < high")


@dataclass(frozen=True)
class Gripper:
    joints: tuple[str, ...]
    closed: tuple[float, ...]
    open: tuple[float, ...]
    actuator: str | None = None
    ctrl_closed: float | None = None
    ctrl_open: float | None = None


@dataclass(frozen=True)
class RobotConfig:
    name: str
    mjcf: Path
    joints: tuple[str, ...]
    floor_geom: str
    tcp_site: str | None = None
    tcp_body: str | None = None
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_axis: str = "z"
    """Axis of the tool frame that points out of the tool; `move_to(tool_down=True)` points it at the floor."""
    keyframe: str | None = None
    actuators: tuple[str, ...] = ()
    """Position actuators for the arm joints, in the same order. Empty: the simulator moves joints kinematically."""
    gripper: Gripper | None = None
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        if not self.joints:
            raise ConfigError("arm.joints must list at least one joint")
        if (self.tcp_site is None) == (self.tcp_body is None):
            raise ConfigError("set exactly one of arm.tcp_site or arm.tcp_body")
        if self.tool_axis not in ("x", "y", "z", "-x", "-y", "-z"):
            raise ConfigError("arm.tool_axis must be one of x, y, z, -x, -y, -z")
        if self.actuators and len(self.actuators) != len(self.joints):
            raise ConfigError("arm.actuators must have one entry per joint, or be empty")


def _menagerie(name: str) -> Path:
    try:
        import importlib

        module = importlib.import_module(f"robot_descriptions.{name}_mj_description")
    except ImportError as exc:
        raise ConfigError(
            f"menagerie:{name} needs the robot_descriptions package (pip install 'mjguard[menagerie]') "
            f"and a known Menagerie name: {exc}"
        ) from exc
    mjcf = Path(module.MJCF_PATH)
    scene = mjcf.parent / "scene.xml"
    return scene if scene.exists() else mjcf


def resolve_mjcf(value: str, base: Path) -> Path:
    if value.startswith("menagerie:"):
        return _menagerie(value.removeprefix("menagerie:"))
    path = (base / value).resolve()
    if not path.exists():
        raise ConfigError(f"model.mjcf: {path} does not exist")
    return path


def _require(table: dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"{where}.{key} is required")
    return table[key]


def from_dict(data: dict[str, Any], base: Path) -> RobotConfig:
    model = data.get("model", {})
    arm = data.get("arm", {})
    raw_limits = dict(data.get("limits", {}))
    if "workspace_mm" in raw_limits:
        raw_limits["workspace_mm"] = {k: (float(v[0]), float(v[1])) for k, v in raw_limits["workspace_mm"].items()}
    try:
        limits = Limits(**raw_limits)
    except TypeError as exc:
        raise ConfigError(f"[limits]: {exc}") from exc
    gripper = None
    if "gripper" in data:
        g = data["gripper"]
        gripper = Gripper(
            joints=tuple(_require(g, "joints", "gripper")),
            closed=tuple(float(v) for v in _require(g, "closed", "gripper")),
            open=tuple(float(v) for v in _require(g, "open", "gripper")),
            actuator=g.get("actuator"),
            ctrl_closed=g.get("ctrl_closed"),
            ctrl_open=g.get("ctrl_open"),
        )
        if not (len(gripper.joints) == len(gripper.closed) == len(gripper.open)):
            raise ConfigError("gripper.joints, gripper.closed and gripper.open must have the same length")
    return RobotConfig(
        name=str(data.get("name", "robot")),
        mjcf=resolve_mjcf(str(_require(model, "mjcf", "model")), base),
        floor_geom=str(_require(model, "floor_geom", "model")),
        keyframe=model.get("keyframe"),
        joints=tuple(_require(arm, "joints", "arm")),
        tcp_site=arm.get("tcp_site"),
        tcp_body=arm.get("tcp_body"),
        tcp_offset_m=tuple(float(v) for v in arm.get("tcp_offset_m", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        tool_axis=str(arm.get("tool_axis", "z")),
        actuators=tuple(arm.get("actuators", ())),
        gripper=gripper,
        limits=limits,
    )


def load(path: str | Path) -> RobotConfig:
    path = Path(path)
    if not path.exists():
        preset = Path(__file__).parent / "presets" / f"{path.name.removesuffix('.toml')}.toml"
        if preset.exists():
            path = preset
        else:
            raise ConfigError(f"{path} does not exist and is not a built-in preset ({', '.join(presets())})")
    with path.open("rb") as f:
        return from_dict(tomllib.load(f), path.parent)


def presets() -> list[str]:
    return sorted(p.stem for p in (Path(__file__).parent / "presets").glob("*.toml"))

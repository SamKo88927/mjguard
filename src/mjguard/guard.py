"""The guard: plans motions, checks every waypoint, and explains refusals in a way a model can act on."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RobotConfig
from .model import ArmModel, Vec, interpolate

CODES = {
    "JOINT_LIMIT": "a joint would leave its allowed range",
    "TABLE_CLEARANCE": "a moving part would come too close to the floor or table",
    "TABLE_COLLISION": "a moving part would touch the floor or table",
    "COLLISION": "the arm would touch itself or something in the scene",
    "WORKSPACE": "the tool point would leave the configured workspace box",
    "UNREACHABLE": "no joint solution reaches the target within tolerance",
    "BAD_TARGET": "the request is malformed",
    "BUSY": "another motion is running",
    "STOPPED": "the motion was stopped before it finished",
    "ENVELOPE_BREACH": "the measured arm left the envelope during the motion and was halted",
    "INTERNAL_ERROR": "a fault inside the server; the arm is free for the next call",
}


@dataclass
class Refusal:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict[str, Any])
    suggestion: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message, "details": self.details}
        if self.suggestion is not None:
            out["suggestion"] = self.suggestion
        return out


@dataclass(frozen=True)
class Plan:
    waypoints: list[Vec]
    min_clearance_mm: float


def _mm(v: float) -> float:
    return round(v * 1000.0, 1)


class Guard:
    def __init__(self, config: RobotConfig, model: ArmModel | None = None) -> None:
        self.config = config
        self.limits = config.limits
        self.model = model or ArmModel(config)
        margin = np.where(
            self.model.hinge, math.radians(self.limits.joint_margin_deg), self.limits.joint_margin_deg / 1000
        )
        self.lower = self.model.lower + margin
        self.upper = self.model.upper - margin
        step = math.radians(self.limits.max_step_deg)
        self.max_step = np.where(self.model.hinge, step, step * 0.05)  # slide joints: 3 deg ~ 2.6 mm at default

    # ------------------------------------------------------------------ single configurations
    def check_joints(self, q: Vec) -> Refusal | None:
        if len(q) != self.model.n or not np.all(np.isfinite(q)):
            return Refusal("BAD_TARGET", f"give {self.model.n} finite joint values")
        for i, name in enumerate(self.config.joints):
            if not (self.lower[i] - 1e-9 <= q[i] <= self.upper[i] + 1e-9):
                unit = "deg" if self.model.hinge[i] else "mm"
                factor = 180 / math.pi if self.model.hinge[i] else 1000.0

                def scale(v: float, factor: float = factor) -> float:
                    return v * factor

                return Refusal(
                    "JOINT_LIMIT",
                    f"{name} would be {scale(float(q[i])):.1f} {unit}; allowed "
                    f"{scale(float(self.lower[i])):.1f} to {scale(float(self.upper[i])):.1f} {unit}",
                    {"joint": name},
                )
        return None

    def check_configuration(self, q: Vec, gripper: Vec | None = None) -> tuple[Refusal | None, float]:
        refusal = self.check_joints(q)
        if refusal is not None:
            return refusal, math.nan
        for c in self.model.contacts(q, gripper):
            if c.floor:
                return Refusal("TABLE_COLLISION", f"{c.a if c.b == 'world' else c.b} would touch the floor"), 0.0
            return Refusal("COLLISION", f"{c.a} would touch {c.b} ({c.depth_mm} mm deep)", {"a": c.a, "b": c.b}), 0.0
        clearance = self.model.clearance_mm(q, gripper)
        if clearance < self.limits.min_clearance_mm:
            return (
                Refusal(
                    "TABLE_CLEARANCE",
                    f"a moving part would be {clearance:.1f} mm above the floor; "
                    f"the minimum is {self.limits.min_clearance_mm:g} mm",
                    {"clearance_mm": round(clearance, 1)},
                ),
                clearance,
            )
        pos, _ = self.model.tool(q)
        refusal = self.check_workspace(pos)
        return refusal, clearance

    def check_workspace(self, pos: Vec) -> Refusal | None:
        for axis, (low, high) in self.limits.workspace_mm.items():
            v = float(pos["xyz".index(axis)]) * 1000
            if axis == "z":
                v -= self.model.floor_z * 1000  # z is the height above the floor everywhere on the surface
            if not low <= v <= high:
                return Refusal(
                    "WORKSPACE",
                    f"tool {axis} would be {v:.1f} mm; the workspace allows {low:g} to {high:g} mm",
                    {"axis": axis, "value_mm": round(v, 1)},
                )
        return None

    # ------------------------------------------------------------------ paths
    def check_path(self, start: Vec, goal: Vec, gripper: Vec | None = None) -> Plan | Refusal:
        refusal = self.check_joints(goal)
        if refusal is not None:
            return refusal
        lowest = math.inf
        waypoints = interpolate(np.asarray(start, float), np.asarray(goal, float), float(np.min(self.max_step)))
        for i, q in enumerate(waypoints):
            refusal, clearance = self.check_configuration(q, gripper)
            if refusal is not None:
                refusal.details["waypoint"] = f"{i + 1}/{len(waypoints)}"
                return refusal
            lowest = min(lowest, clearance)
        return Plan(waypoints, lowest)

    def solve(self, target: Vec, direction: Vec | None, current: Vec) -> Vec | Refusal:
        tol_mm = self.limits.position_tolerance_mm
        tol_deg = self.limits.angle_tolerance_deg
        seeds = [current, self.model.home]
        span = np.where(np.isfinite(self.upper - self.lower), self.upper - self.lower, 2 * math.pi)
        low = np.where(np.isfinite(self.lower), self.lower, -math.pi)
        rng = np.random.default_rng(0)  # same request, same answer: a suggestion must re-plan identically
        seeds += [low + rng.random(self.model.n) * span for _ in range(6)]
        best: tuple[float, Vec] | None = None
        closest = (math.inf, math.inf)
        for seed in seeds:
            q, pos_err, ang_err = self.model.solve_ik(target, direction, seed, self.lower, self.upper)
            if pos_err < closest[0]:
                closest = (pos_err, ang_err)
            if pos_err <= tol_mm and ang_err <= tol_deg:
                cost = float(np.max(np.abs(q - current)))
                if best is None or cost < best[0]:
                    best = (cost, q)
        if best is None:
            return Refusal(
                "UNREACHABLE",
                f"no joint solution reaches the target; the closest misses by {closest[0]:.1f} mm "
                f"and {closest[1]:.1f} deg",
                {"miss_mm": round(closest[0], 1), "miss_deg": round(closest[1], 1)},
            )
        return best[1]

    def plan_pose(self, current: Vec, target: Vec, direction: Vec | None, gripper: Vec | None = None) -> Plan | Refusal:
        if not np.all(np.isfinite(target)):
            return Refusal("BAD_TARGET", "target coordinates must be finite numbers")
        refusal = self.check_workspace(target)
        if refusal is not None:
            return refusal
        height = (float(target[2]) - self.model.floor_z) * 1000
        if height < self.limits.min_clearance_mm:
            return Refusal(
                "TABLE_CLEARANCE",
                f"target z is {height:.1f} mm above the floor; the tool must stay at least "
                f"{self.limits.min_clearance_mm:g} mm above it",
                {"z_mm": round(height, 1)},
            )
        q = self.solve(target, direction, current)
        if isinstance(q, Refusal):
            return q
        return self.check_path(current, q, gripper)

    def plan_pose_with_fallback(
        self, current: Vec, target: Vec, direction: Vec | None, gripper: Vec | None = None
    ) -> Plan | Refusal:
        """Plan to target; on refusal, attach the farthest checked-safe pose on the straight line toward it."""
        plan = self.plan_pose(current, target, direction, gripper)
        if not isinstance(plan, Refusal) or plan.code == "BAD_TARGET":
            return plan
        start, _ = self.model.tool(current)
        good: Vec | None = None
        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = (lo + hi) / 2
            point = np.round((start + (target - start) * mid) * 10000) / 10000  # 0.1 mm, what the model sends back
            if isinstance(self.plan_pose(current, point, direction, gripper), Plan):
                lo, good = mid, point
            else:
                hi = mid
        if good is not None and lo > 0.02:
            plan.suggestion = {
                "x_mm": _mm(float(good[0])),
                "y_mm": _mm(float(good[1])),
                "z_mm": _mm(float(good[2]) - self.model.floor_z),
                "note": f"farthest checked-safe point on the straight line toward the target ({lo:.0%} of the way)",
            }
        return plan

    def describe(self) -> dict[str, Any]:
        lim = self.limits
        joints: dict[str, Any] = {}
        for i, name in enumerate(self.config.joints):
            if self.model.hinge[i]:
                joints[name] = {
                    "min_deg": round(math.degrees(float(self.lower[i])), 1) if math.isfinite(self.lower[i]) else None,
                    "max_deg": round(math.degrees(float(self.upper[i])), 1) if math.isfinite(self.upper[i]) else None,
                }
            else:
                joints[name] = {"min_mm": _mm(float(self.lower[i])), "max_mm": _mm(float(self.upper[i]))}
        return {
            "robot": self.config.name,
            "frame": "world frame of the MJCF; millimetres; z is the height above the floor plane",
            "min_clearance_mm": lim.min_clearance_mm,
            "workspace_mm": {k: list(v) for k, v in lim.workspace_mm.items()},
            "max_joint_speed_dps": lim.max_joint_speed_dps,
            "joints": joints,
            "has_gripper": self.config.gripper is not None,
            "refusal_codes": CODES,
        }

"""The guard: plans motions, checks every waypoint, and explains refusals in a way a model can act on."""

from __future__ import annotations

import itertools
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
    "COLLISION": "the arm would touch itself, an obstacle or something else in the scene",
    "OBSTACLE_CLEARANCE": "a moving part would come too close to a configured obstacle",
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


LINE_STEP_M = 0.005
RELAX_TOLERANCE_MM = 0.5


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

    def thresholds(self, start: Vec, gripper: Vec | None = None) -> tuple[float, float]:
        """Clearance a plan must keep from the floor and from obstacles, given where the arm starts.

        Normally the planning values. If the measured arm already sits inside the planning margin (servo sag or
        tracking error after a move that was planned right at the margin), a plan may not get any closer than where
        it starts, but it may move: otherwise every motion, including moving away, would be refused and the arm
        would be stuck. The hard floor limit and contacts are never relaxed.
        """
        floor = self.limits.planning_clearance_mm
        start_clearance = self.model.clearance_mm(start, gripper)
        if start_clearance < floor:
            floor = max(self.limits.min_clearance_mm, start_clearance - RELAX_TOLERANCE_MM)
        obstacle = self.limits.obstacle_clearance_mm
        start_gap = self.model.obstacle_clearance(start, gripper)[0]
        if start_gap < obstacle:
            obstacle = max(0.0, start_gap - RELAX_TOLERANCE_MM)
        return floor, obstacle

    def check_configuration(
        self, q: Vec, gripper: Vec | None = None, thresholds: tuple[float, float] | None = None
    ) -> tuple[Refusal | None, float]:
        floor_mm, obstacle_mm = thresholds or (self.limits.planning_clearance_mm, self.limits.obstacle_clearance_mm)
        refusal = self.check_joints(q)
        if refusal is not None:
            return refusal, math.nan
        for c in self.model.contacts(q, gripper):
            if c.floor:
                part = c.b if c.a == self.config.floor_geom else c.a
                return Refusal("TABLE_COLLISION", f"{part} would touch the floor"), 0.0
            return Refusal("COLLISION", f"{c.a} would touch {c.b} ({c.depth_mm} mm deep)", {"a": c.a, "b": c.b}), 0.0
        clearance = self.model.clearance_mm(q, gripper)
        if clearance < floor_mm:
            return (
                Refusal(
                    "TABLE_CLEARANCE",
                    f"a moving part would be {clearance:.1f} mm above the floor; plans keep at least {floor_mm:.4g} mm",
                    {"clearance_mm": round(clearance, 1)},
                ),
                clearance,
            )
        gap, part, obstacle = self.model.obstacle_clearance(q, gripper)
        if gap < obstacle_mm:
            return (
                Refusal(
                    "OBSTACLE_CLEARANCE",
                    f"{part} would be {max(gap, 0):.1f} mm from {obstacle}; plans keep at least {obstacle_mm:.4g} mm",
                    {"part": part, "obstacle": obstacle, "gap_mm": round(max(gap, 0), 1)},
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
        limits = self.thresholds(np.asarray(start, float), gripper)
        waypoints = interpolate(np.asarray(start, float), np.asarray(goal, float), float(np.min(self.max_step)))
        for i, q in enumerate(waypoints):
            refusal, clearance = self.check_configuration(q, gripper, limits)
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

    def plan_pose(
        self, current: Vec, target: Vec, direction: Vec | None, gripper: Vec | None = None, line: bool = False
    ) -> Plan | Refusal:
        if not np.all(np.isfinite(target)):
            return Refusal("BAD_TARGET", "target coordinates must be finite numbers")
        refusal = self.check_workspace(target)
        if refusal is not None:
            return refusal
        height = (float(target[2]) - self.model.floor_z) * 1000
        if height < self.limits.planning_clearance_mm:
            return Refusal(
                "TABLE_CLEARANCE",
                f"target z is {height:.1f} mm above the floor; the tool must stay at least "
                f"{self.limits.planning_clearance_mm:g} mm above it",
                {"z_mm": round(height, 1)},
            )
        if line:
            return self.plan_line(current, target, direction, gripper)
        q = self.solve(target, direction, current)
        if isinstance(q, Refusal):
            return q
        return self.check_path(current, q, gripper)

    def plan_line(self, current: Vec, target: Vec, direction: Vec | None, gripper: Vec | None = None) -> Plan | Refusal:
        """The tool point follows the straight line to the target, sampled every LINE_STEP_M.

        The tool axis turns smoothly from where it points now to `direction` (or keeps pointing where it points now).
        IK is solved point by point from the previous solution, so the arm cannot flip configuration mid-line; a
        jump bigger than a few waypoint steps means the line crosses a singularity or leaves reach, and is refused.
        """
        start, axis0 = self.model.tool(current)
        axis1 = axis0 if direction is None else direction
        if float(axis0 @ axis1) < -0.95:
            return Refusal("UNREACHABLE", "the tool would have to flip over along the line; move with path='joint'")
        span = float(np.linalg.norm(target - start))
        steps = max(1, math.ceil(span / LINE_STEP_M))
        q = current.astype(float).copy()
        sampled = [q]
        jump_limit = 4 * float(np.min(self.max_step)) * max(1.0, LINE_STEP_M / 0.005)
        for i in range(1, steps + 1):
            t = i / steps
            point = start + (target - start) * t
            axis = axis0 * (1 - t) + axis1 * t
            axis /= np.linalg.norm(axis)
            nxt, pos_err, ang_err = self.model.solve_ik(point, axis, q, self.lower, self.upper, iters=60)
            if pos_err > self.limits.position_tolerance_mm or ang_err > self.limits.angle_tolerance_deg:
                return Refusal(
                    "UNREACHABLE",
                    f"the straight line leaves the arm's reach {t:.0%} of the way (misses by {pos_err:.1f} mm)",
                    {"fraction": round(t, 2)},
                )
            if float(np.max(np.abs(nxt - q))) > jump_limit:
                return Refusal(
                    "UNREACHABLE",
                    f"the straight line crosses a singularity {t:.0%} of the way; the joints would jump",
                    {"fraction": round(t, 2)},
                )
            q = nxt
            sampled.append(q)
        waypoints: list[Vec] = []
        for a, b in itertools.pairwise(sampled):
            waypoints += interpolate(a, b, float(np.min(self.max_step)))
        lowest = math.inf
        limits = self.thresholds(current, gripper)
        for i, w in enumerate(waypoints):
            refusal, clearance = self.check_configuration(w, gripper, limits)
            if refusal is not None:
                refusal.details["waypoint"] = f"{i + 1}/{len(waypoints)}"
                return refusal
            lowest = min(lowest, clearance)
        return Plan(waypoints, lowest)

    def plan_pose_with_fallback(
        self, current: Vec, target: Vec, direction: Vec | None, gripper: Vec | None = None, line: bool = False
    ) -> Plan | Refusal:
        """Plan to target; on refusal, attach the farthest checked-safe pose on the straight line toward it."""
        plan = self.plan_pose(current, target, direction, gripper, line)
        if not isinstance(plan, Refusal) or plan.code == "BAD_TARGET":
            return plan
        start, _ = self.model.tool(current)
        good: Vec | None = None
        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = (lo + hi) / 2
            point = np.round((start + (target - start) * mid) * 10000) / 10000  # 0.1 mm, what the model sends back
            if isinstance(self.plan_pose(current, point, direction, gripper, line), Plan):
                lo, good = mid, point
            else:
                hi = mid
        if good is not None and lo > 0.02:
            plan.suggestion = {
                "x_mm": _mm(float(good[0])),
                "y_mm": _mm(float(good[1])),
                "z_mm": _mm(float(good[2]) - self.model.floor_z),
                "tool_down": direction is not None,
                "path": "line" if line else "joint",
                "note": f"farthest checked-safe point on the straight line toward the target ({lo:.0%} of the way); "
                "send it with the same tool_down and path",
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
            "planning_clearance_mm": lim.planning_clearance_mm,
            "workspace_mm": {k: list(v) for k, v in lim.workspace_mm.items()},
            "max_joint_speed_dps": lim.max_joint_speed_dps,
            "joints": joints,
            "has_gripper": self.config.gripper is not None,
            "obstacle_clearance_mm": lim.obstacle_clearance_mm,
            "obstacles": [
                {"name": o.name, "shape": o.shape, "size_mm": list(o.size_mm), "center_mm": list(o.center_mm)}
                for o in self.config.obstacles
            ],
            "refusal_codes": CODES,
        }

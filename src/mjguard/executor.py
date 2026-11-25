"""The only path from a request to a transport: plan, check every waypoint, run, and monitor what actually ran."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco as _mujoco
import numpy as np

from .config import RobotConfig
from .guard import Guard, Plan, Refusal
from .model import ArmModel, Vec

# MuJoCo's compiled bindings ship no type stubs; keep the untyped surface behind one explicit name.
mj: Any = _mujoco

log = logging.getLogger("mjguard")
Result = dict[str, Any]


@dataclass(frozen=True)
class FollowResult:
    stopped: bool
    waypoints_done: int


class Transport(Protocol):
    """Something that can report joint positions and follow checked waypoints: a simulator or a real arm."""

    name: str

    def read(self) -> tuple[Vec, Vec]:
        """Measured (arm joints, gripper joints)."""
        ...

    def follow(
        self,
        waypoints: Sequence[Vec],
        gripper: Vec | None,
        period_s: float,
        should_stop: Callable[[], bool],
        on_waypoint: Callable[[Vec, Vec], None],
    ) -> FollowResult: ...


class SimTransport:
    """MuJoCo simulation of the arm. With actuators configured it runs physics; otherwise it sets joints directly."""

    def __init__(self, config: RobotConfig, realtime: bool = False) -> None:
        self.name = "mujoco"
        self.model = ArmModel(config)
        self.m, self.d = self.model.m, self.model.d
        self.realtime = realtime
        self.physics = bool(config.actuators)
        self.gripper_cfg = config.gripper
        self._lock = threading.Lock()
        self.d.qpos[:] = self.model.qpos0
        if self.physics:
            for a, q in zip(self.model.actuator_ids, self.model.home, strict=True):
                self.d.ctrl[a] = q
        mj.mj_forward(self.m, self.d)

    def read(self) -> tuple[Vec, Vec]:
        with self._lock:
            return self.d.qpos[self.model.qadr].copy(), self.d.qpos[self.model.gripper_qadr].copy()

    def _gripper_ctrl(self, gripper: Vec) -> None:
        g = self.gripper_cfg
        if g is None or self.model.gripper_actuator < 0 or g.ctrl_closed is None or g.ctrl_open is None:
            self.d.qpos[self.model.gripper_qadr] = gripper
            return
        closed, opened = np.array(g.closed), np.array(g.open)
        span = opened - closed
        frac = float(np.mean(np.divide(gripper - closed, span, out=np.zeros_like(span), where=span != 0)))
        self.d.ctrl[self.model.gripper_actuator] = g.ctrl_closed + frac * (g.ctrl_open - g.ctrl_closed)

    def follow(
        self,
        waypoints: Sequence[Vec],
        gripper: Vec | None,
        period_s: float,
        should_stop: Callable[[], bool],
        on_waypoint: Callable[[Vec, Vec], None],
    ) -> FollowResult:
        steps = max(1, round(period_s / self.m.opt.timestep))
        start_g = self.read()[1]
        for i, q in enumerate(waypoints):
            if should_stop():
                return FollowResult(True, i)
            t0 = time.monotonic()
            g = None if gripper is None else start_g + (gripper - start_g) * ((i + 1) / len(waypoints))
            with self._lock:
                if self.physics:
                    for a, v in zip(self.model.actuator_ids, q, strict=True):
                        self.d.ctrl[a] = v
                    if g is not None:
                        self._gripper_ctrl(g)
                    for _ in range(steps):
                        mj.mj_step(self.m, self.d)
                else:
                    self.d.qpos[self.model.qadr] = q
                    if g is not None:
                        self.d.qpos[self.model.gripper_qadr] = g
                    self.d.qvel[:] = 0
                    mj.mj_forward(self.m, self.d)
                measured = self.d.qpos[self.model.qadr].copy(), self.d.qpos[self.model.gripper_qadr].copy()
            on_waypoint(*measured)
            if self.realtime:
                time.sleep(max(0.0, period_s - (time.monotonic() - t0)))
        if self.physics:  # let the servos settle on the last waypoint
            with self._lock:
                for _ in range(round(0.3 / self.m.opt.timestep)):
                    mj.mj_step(self.m, self.d)
        return FollowResult(False, len(waypoints))


class Trace:
    """Last `keep` calls in memory; optionally appended to a JSONL file."""

    def __init__(self, path: Path | None = None, keep: int = 200) -> None:
        self.path = path
        self.events: deque[Result] = deque(maxlen=keep)
        self._lock = threading.Lock()

    def record(self, tool: str, arguments: Result, result: Result) -> None:
        event = {"t": round(time.time(), 3), "tool": tool, "arguments": arguments, "result": result}
        with self._lock:
            self.events.append(event)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def latest(self, n: int = 50) -> list[Result]:
        with self._lock:
            return list(self.events)[-n:]


def _deg(v: float) -> float:
    return round(math.degrees(v), 1)


class Executor:
    def __init__(self, config: RobotConfig, transport: Transport, trace: Trace | None = None) -> None:
        self.config = config
        self.transport = transport
        self.guard = Guard(config)
        self.trace = trace or Trace()
        self._monitor = ArmModel(config)  # only used inside the motion lock
        self._view = ArmModel(config)
        self._view_lock = threading.Lock()
        self._motion = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ reading
    def _state(self, q: Vec, g: Vec) -> Result:
        with self._view_lock:
            pos, axis = self._view.tool(q)
            clearance = self._view.clearance_mm(q, g)
        floor = self._view.floor_z
        state: Result = {
            "backend": self.transport.name,
            "moving": self._motion.locked(),
            "tool": {
                "x_mm": round(float(pos[0]) * 1000, 1),
                "y_mm": round(float(pos[1]) * 1000, 1),
                "z_mm": round((float(pos[2]) - floor) * 1000, 1),
                "axis": [round(float(v), 3) for v in axis],
            },
            "joints": {
                name: (_deg(float(q[i])) if self._view.hinge[i] else round(float(q[i]) * 1000, 1))
                for i, name in enumerate(self.config.joints)
            },
            "min_clearance_mm": round(clearance, 1),
        }
        if self.config.gripper is not None:
            closed, opened = np.array(self.config.gripper.closed), np.array(self.config.gripper.open)
            span = opened - closed
            frac = np.divide(g - closed, span, out=np.zeros_like(span), where=span != 0)
            state["gripper_opening"] = round(float(np.clip(np.mean(frac), 0, 1)), 3)
        return state

    def get_state(self) -> Result:
        return {"ok": True, "state": self._state(*self.transport.read())}

    def get_limits(self) -> Result:
        return {"ok": True, "limits": self.guard.describe()}

    # ------------------------------------------------------------------ acting
    def stop(self) -> Result:
        was_moving = self._motion.locked()
        self._stop.set()
        result: Result = {
            "ok": True,
            "was_moving": was_moving,
            "note": "software stop; on real hardware a physical e-stop is the only power cut",
        }
        self.trace.record("stop", {}, result)
        return result

    def _run(
        self,
        tool: str,
        arguments: Result,
        speed: float,
        planner: Callable[[Vec, Vec], tuple[Plan | Refusal, Vec | None]],
    ) -> Result:
        if not self._motion.acquire(blocking=False):
            result: Result = {"ok": False, "refusal": Refusal("BUSY", "another motion is running").to_dict()}
            self.trace.record(tool, arguments, result)
            return result
        try:
            self._stop.clear()
            q, g = self.transport.read()
            plan, gripper_goal = planner(q, g)
            if isinstance(plan, Refusal):
                result = {"ok": False, "refusal": plan.to_dict(), "state": self._state(q, g)}
            else:
                result = self._execute(plan, gripper_goal, speed)
        except Exception as exc:
            log.exception("%s failed", tool)
            refusal = Refusal("INTERNAL_ERROR", f"{tool} failed inside the server: {type(exc).__name__}: {exc}")
            result = {"ok": False, "refusal": refusal.to_dict()}
        finally:
            self._motion.release()
        self.trace.record(tool, arguments, result)
        return result

    def _execute(self, plan: Plan, gripper: Vec | None, speed: float) -> Result:
        lim = self.config.limits
        speed = float(np.clip(speed, 0.05, 1.0))
        period = lim.max_step_deg / (lim.max_joint_speed_dps * speed)
        lowest = [math.inf]
        breach: list[str] = []

        def monitor(q: Vec, g: Vec) -> None:
            clearance = self._monitor.clearance_mm(q, g)
            lowest[0] = min(lowest[0], clearance)
            if clearance < lim.min_clearance_mm:
                breach.append(f"measured clearance {clearance:.1f} mm, hard limit {lim.min_clearance_mm:g} mm")
            elif contacts := self._monitor.contacts(q, g):
                breach.append(f"measured contact {contacts[0].a} / {contacts[0].b}")

        followed = self.transport.follow(
            plan.waypoints, gripper, period, lambda: self._stop.is_set() or bool(breach), monitor
        )
        final = self.transport.read()
        if not breach:
            monitor(*final)  # after the servos settle: what the arm holds now must be inside the envelope too
        with self._view_lock:
            planned_tool = self._view.tool(plan.waypoints[-1])[0]
            measured_tool = self._view.tool(final[0])[0]
        tracking_mm = round(float(np.linalg.norm(planned_tool - measured_tool)) * 1000, 1)
        if breach:
            refusal: Refusal | None = Refusal("ENVELOPE_BREACH", f"halted: {breach[0]}")
        elif followed.stopped:
            refusal = Refusal("STOPPED", "motion was stopped before it finished")
        else:
            refusal = None
        return {
            "ok": refusal is None,
            "plan": {
                "waypoints": len(plan.waypoints),
                "duration_s": round(len(plan.waypoints) * period, 2),
                "planned_min_clearance_mm": round(plan.min_clearance_mm, 1),
            },
            "executed": {
                "waypoints": followed.waypoints_done,
                "min_clearance_mm": round(lowest[0], 1),
                "tool_tracking_error_mm": tracking_mm,
            },
            "state": self._state(*final),
            **({"refusal": refusal.to_dict()} if refusal is not None else {}),
        }

    def move_to(self, x_mm: float, y_mm: float, z_mm: float, tool_down: bool = True, speed: float = 0.5) -> Result:
        floor = self.guard.model.floor_z
        target = np.array([x_mm / 1000, y_mm / 1000, floor + z_mm / 1000])
        direction = np.array([0.0, 0.0, -1.0]) if tool_down else None
        args = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "tool_down": tool_down, "speed": speed}
        return self._run(
            "move_to",
            args,
            speed,
            lambda q, g: (self.guard.plan_pose_with_fallback(q, target, direction, g), None),
        )

    def move_joints(self, joints: dict[str, float], speed: float = 0.5) -> Result:
        args = {"joints": joints, "speed": speed}
        unknown = sorted(set(joints) - set(self.config.joints))
        if unknown:
            refusal = Refusal("BAD_TARGET", f"unknown joints {unknown}; known: {list(self.config.joints)}")
            result: Result = {"ok": False, "refusal": refusal.to_dict()}
            self.trace.record("move_joints", args, result)
            return result

        def planner(q: Vec, g: Vec) -> tuple[Plan | Refusal, Vec | None]:
            goal = q.copy()
            for i, name in enumerate(self.config.joints):
                if name in joints:
                    v = joints[name]
                    goal[i] = math.radians(v) if self.guard.model.hinge[i] else v / 1000
            return self.guard.check_path(q, goal, g), None

        return self._run("move_joints", args, speed, planner)

    def gripper(self, opening: float, speed: float = 0.5) -> Result:
        args = {"opening": opening, "speed": speed}
        cfg = self.config.gripper

        def planner(q: Vec, g: Vec) -> tuple[Plan | Refusal, Vec | None]:
            if cfg is None:
                return Refusal("BAD_TARGET", "this robot has no gripper configured"), None
            if not (math.isfinite(opening) and 0.0 <= opening <= 1.0):
                return Refusal("BAD_TARGET", "opening must be between 0 (closed) and 1 (open)"), None
            goal = np.array(cfg.closed) + (np.array(cfg.open) - np.array(cfg.closed)) * opening
            steps = max(2, int(np.ceil(np.max(np.abs(goal - g)) / 0.01)))
            for i in range(1, steps + 1):
                refusal, _ = self.guard.check_configuration(q, g + (goal - g) * (i / steps))
                if refusal is not None:
                    return refusal, None
            return Plan([q.copy() for _ in range(steps)], self.guard.model.clearance_mm(q, goal)), goal

        return self._run("gripper", args, speed, planner)

    def rest(self, speed: float = 0.5) -> Result:
        home = self.guard.model.home
        return self._run("rest", {"speed": speed}, speed, lambda q, g: (self.guard.check_path(q, home, g), None))

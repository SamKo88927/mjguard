"""The only path from a request to a transport: plan, check every waypoint, run, and monitor what actually ran."""

from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco as _mujoco
import numpy as np

from .config import LANGS, ConfigError, RobotConfig
from .events import EventBus
from .guard import Guard, Plan, Refusal
from .messages import humanize
from .model import ArmModel, Vec, interpolate
from .poses import PoseStore, valid_name

# MuJoCo's compiled bindings ship no type stubs; keep the untyped surface behind one explicit name.
mj: Any = _mujoco

log = logging.getLogger("mjguard")
MONITOR_TOLERANCE_MM = 1.0
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


PLAN_TTL_S = 60.0
"""A previewed plan can be executed for this long."""
MAX_PLANS = 32
START_TOLERANCE = 0.5
"""Degrees (hinge) or millimetres (slide): how far the measured arm may be from a previewed plan's start."""
JOG_MAX_MM = 50.0
JOG_MAX_JOINT = 10.0
"""jog_joint cap: degrees for hinge joints, millimetres for slide joints."""
PROGRESS_PERIOD_S = 0.1
PATH_POINTS = 50
GRIPPER_STEP = 0.01

Planner = Callable[[Vec, Vec], tuple[Plan | Refusal, Vec | None]]


@dataclass
class PendingMotion:
    """A checked plan that has not run yet (a preview), or the remainder of one that was paused."""

    tool: str
    plan: Plan
    gripper_goal: Vec | None
    speed: float
    start: Vec
    created: float
    used: bool = False


class Executor:
    """The only way to move: plan, check every waypoint, run, monitor. Previews, jogs, poses and resumes included."""

    def __init__(
        self,
        config: RobotConfig,
        transport: Transport,
        trace: Trace | None = None,
        events: EventBus | None = None,
        poses: PoseStore | None = None,
        *,
        lang: str | None = None,
        require_confirmation: bool | None = None,
        latch_stop: bool | None = None,
        plan_ttl_s: float = PLAN_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.transport = transport
        self.guard = Guard(config)
        self.trace = trace or Trace()
        self.events = events or EventBus()
        self.poses = poses or PoseStore(config.joints)
        self.lang = lang or config.ui.lang
        if self.lang not in LANGS:
            raise ValueError(f"lang must be one of {', '.join(LANGS)}")
        self.require_confirmation = (
            config.ui.require_confirmation if require_confirmation is None else require_confirmation
        )
        self.latch_stop = config.ui.latch_stop if latch_stop is None else latch_stop
        self._latched = threading.Event()
        self.plan_ttl_s = plan_ttl_s
        self.clock = clock
        self._monitor = ArmModel(config)  # only used inside the motion lock
        self._view = ArmModel(config)
        self._view_lock = threading.Lock()
        self._motion = threading.Lock()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._plans: dict[str, PendingMotion] = {}  # only touched inside the motion lock
        self._paused: PendingMotion | None = None
        self._paused_lock = threading.Lock()

    # ------------------------------------------------------------------ reading
    def _tool_mm(self, pos: Vec) -> list[float]:
        return [
            round(float(pos[0]) * 1000, 1),
            round(float(pos[1]) * 1000, 1),
            round((float(pos[2]) - self._view.floor_z) * 1000, 1),
        ]

    def _joints_out(self, q: Vec) -> dict[str, float]:
        return {
            name: (_deg(float(q[i])) if self._view.hinge[i] else round(float(q[i]) * 1000, 1))
            for i, name in enumerate(self.config.joints)
        }

    def _opening(self, g: Vec) -> float | None:
        if self.config.gripper is None:
            return None
        closed, opened = np.array(self.config.gripper.closed), np.array(self.config.gripper.open)
        span = opened - closed
        frac = np.divide(g - closed, span, out=np.zeros_like(span), where=span != 0)
        return round(float(np.clip(np.mean(frac), 0, 1)), 3)

    def _state(self, q: Vec, g: Vec) -> Result:
        with self._view_lock:
            pos, axis = self._view.tool(q)
            clearance = self._view.clearance_mm(q, g)
        x, y, z = self._tool_mm(pos)
        state: Result = {
            "backend": self.transport.name,
            "moving": self._motion.locked(),
            "paused": self._paused is not None,
            "stop_latched": self._latched.is_set(),
            "tool": {"x_mm": x, "y_mm": y, "z_mm": z, "axis": [round(float(v), 3) for v in axis]},
            "joints": self._joints_out(q),
            "min_clearance_mm": round(clearance, 1),
        }
        opening = self._opening(g)
        if opening is not None:
            state["gripper_opening"] = opening
        return state

    def get_state(self) -> Result:
        result: Result = {"ok": True, "state": self._state(*self.transport.read())}
        with self._paused_lock:
            paused = self._paused
        if paused is not None:
            result["paused_motion"] = {"tool": paused.tool, "remaining_waypoints": len(paused.plan.waypoints)}
        return result

    def get_limits(self) -> Result:
        return {"ok": True, "limits": self.guard.describe()}

    def list_poses(self) -> Result:
        return {"ok": True, "poses": self.poses.all()}

    # ------------------------------------------------------------------ results
    def _finish(self, tool: str, arguments: Result, result: Result) -> Result:
        """Every call ends here: human message, trace, and a `refused` event for a request that never moved."""
        refusal = result.get("refusal")
        if isinstance(refusal, dict):
            humanize(refusal, self.lang)  # pyright: ignore[reportUnknownArgumentType]
        self.trace.record(tool, arguments, result)
        if result.get("ok") is not True and "executed" not in result and isinstance(refusal, dict):
            self.events.publish("refused", tool=tool, refusal=refusal)
        return result

    def _refuse(self, tool: str, arguments: Result, code: str, message: str, **details: Any) -> Result:
        return self._finish(tool, arguments, {"ok": False, "refusal": Refusal(code, message, details).to_dict()})

    def _period(self, speed: float) -> float:
        lim = self.config.limits
        return lim.max_step_deg / (lim.max_joint_speed_dps * float(np.clip(speed, 0.05, 1.0)))

    def _summary(self, plan: Plan, gripper_goal: Vec | None, speed: float, start: Vec) -> Result:
        """What a UI needs to show a plan before it runs, including a downsampled tool path to draw."""
        n = len(plan.waypoints)
        picks = sorted({round(i) for i in np.linspace(0, n - 1, min(n, PATH_POINTS - 1))})
        with self._view_lock:
            start_pos, start_axis = self._view.tool(start)
            end_pos, end_axis = self._view.tool(plan.waypoints[-1])
            path = [self._tool_mm(start_pos)] + [self._tool_mm(self._view.tool(plan.waypoints[i])[0]) for i in picks]
        summary: Result = {
            "waypoints": n,
            "duration_s": round(n * self._period(speed), 2),
            "speed": speed,
            "min_clearance_mm": round(plan.min_clearance_mm, 1),
            "min_obstacle_gap_mm": round(plan.min_obstacle_gap_mm, 1)
            if math.isfinite(plan.min_obstacle_gap_mm)
            else None,
            "start_tool": {"mm": self._tool_mm(start_pos), "axis": [round(float(v), 3) for v in start_axis]},
            "end_tool": {"mm": self._tool_mm(end_pos), "axis": [round(float(v), 3) for v in end_axis]},
            "end_joints": self._joints_out(plan.waypoints[-1]),
            "path_mm": path,
        }
        if gripper_goal is not None:
            summary["gripper_opening_goal"] = self._opening(gripper_goal)
        return summary

    # ------------------------------------------------------------------ stopping
    def stop(self, latch: bool | None = None) -> Result:
        """Halt at the next waypoint; discard a paused motion. `latch` (default: the server's latch_stop setting)
        also refuses every motion until release_stop(): an e-stop button latches, a plain stop button does not."""
        was_moving = self._motion.locked()
        self._stop.set()
        with self._paused_lock:
            discarded = self._paused is not None
            self._paused = None
        if self.latch_stop if latch is None else latch:
            self._latched.set()
        result: Result = {
            "ok": True,
            "was_moving": was_moving,
            "discarded_paused_motion": discarded,
            "latched": self._latched.is_set(),
            "note": "software stop; on real hardware a physical e-stop is the only power cut",
        }
        self.trace.record("stop", {}, result)
        if not was_moving:  # a moving arm's own motion publishes "stopped"; an idle stop must be visible too
            self.events.publish(
                "stopped", tool="stop", was_moving=False, discarded_paused_motion=discarded, latched=result["latched"]
            )
        return result

    def release_stop(self) -> Result:
        """Clear a latched stop so motions are accepted again. Nothing moves."""
        was_latched = self._latched.is_set()
        self._latched.clear()
        result: Result = {"ok": True, "was_latched": was_latched}
        self.trace.record("release_stop", {}, result)
        if was_latched:
            self.events.publish("stop_released")
        return result

    def discard(self, plan_id: str) -> Result:
        """Forget a previewed plan so it can no longer run (the UI's reject button). Nothing moves."""
        tool, arguments = "discard", {"plan_id": plan_id}
        if not self._motion.acquire(timeout=1.0):
            return self._refuse(tool, arguments, "BUSY", "another motion is running; discard once it ends")
        try:
            existed = self._plans.pop(plan_id, None) is not None
        finally:
            self._motion.release()
        result: Result = {"ok": True, "discarded": existed}
        self.trace.record(tool, arguments, result)
        return result

    def _latched_refusal(self, tool: str, arguments: Result) -> Result | None:
        if self._latched.is_set():
            return self._refuse(tool, arguments, "STOP_LATCHED", "a latched stop is in force; call release_stop()")
        return None

    def pause(self) -> Result:
        """Halt at the next waypoint and keep the rest of the plan for resume()."""
        was_moving = self._motion.locked()
        if was_moving:
            self._pause.set()
        result: Result = {
            "ok": True,
            "was_moving": was_moving,
            "note": "the arm halts at the next waypoint and holds; resume() re-checks and continues, stop() discards",
        }
        self.trace.record("pause", {}, result)
        return result

    # ------------------------------------------------------------------ the one way to move
    def _request(self, tool: str, arguments: Result, speed: float, planner: Planner, preview: bool) -> Result:
        """Plan under the motion lock; then preview (store the plan) or execute it."""
        preview = preview or self.require_confirmation
        arguments = {**arguments, "preview": preview}
        if not preview and (latched := self._latched_refusal(tool, arguments)) is not None:
            return latched
        if not self._motion.acquire(blocking=False):
            return self._refuse(tool, arguments, "BUSY", "another motion is running")
        try:
            self._stop.clear()
            self._pause.clear()
            q, g = self.transport.read()
            plan, gripper_goal = planner(q, g)
            if isinstance(plan, Refusal):
                result: Result = {"ok": False, "refusal": plan.to_dict(), "state": self._state(q, g)}
            elif preview:
                result = self._store(tool, plan, gripper_goal, speed, q, g)
            else:
                result = self._execute(tool, plan, gripper_goal, speed)
        except Exception as exc:
            result = self._internal_error(tool, exc)
        finally:
            self._motion.release()
        return self._finish(tool, arguments, result)

    def _internal_error(self, tool: str, exc: Exception) -> Result:
        log.exception("%s failed", tool)
        refusal = Refusal("INTERNAL_ERROR", f"{tool} failed inside the server: {type(exc).__name__}: {exc}")
        return {"ok": False, "refusal": refusal.to_dict()}

    def _store(self, tool: str, plan: Plan, gripper_goal: Vec | None, speed: float, q: Vec, g: Vec) -> Result:
        now = self.clock()
        for pid in [k for k, v in self._plans.items() if now - v.created > self.plan_ttl_s]:
            del self._plans[pid]
        while len(self._plans) >= MAX_PLANS:
            del self._plans[next(iter(self._plans))]  # oldest first
        plan_id = secrets.token_hex(8)
        self._plans[plan_id] = PendingMotion(tool, plan, gripper_goal, speed, q.copy(), now)
        return {
            "ok": True,
            "executed": False,
            "plan_id": plan_id,
            "expires_in_s": self.plan_ttl_s,
            "plan": self._summary(plan, gripper_goal, speed, q),
            "state": self._state(q, g),
            "note": "nothing moved: this is a checked preview; execute(plan_id) runs it once, if the arm has not moved",
        }

    def execute(self, plan_id: str) -> Result:
        """Run a previewed plan once: unknown, expired, used or stale plans are refused; the rest is re-checked."""
        tool, arguments = "execute", {"plan_id": plan_id}
        if (latched := self._latched_refusal(tool, arguments)) is not None:
            return latched
        if not self._motion.acquire(blocking=False):
            return self._refuse(tool, arguments, "BUSY", "another motion is running")
        try:
            self._stop.clear()
            self._pause.clear()
            pending = self._plans.get(plan_id)
            q, g = self.transport.read()
            refusal: Refusal | None = None
            if pending is None:
                refusal = Refusal("PLAN_UNKNOWN", f"no previewed plan {plan_id!r}; preview the motion first")
            elif pending.used:
                refusal = Refusal("PLAN_USED", "this plan already ran; preview again to move again")
            elif self.clock() - pending.created > self.plan_ttl_s:
                del self._plans[plan_id]
                refusal = Refusal("PLAN_EXPIRED", f"plans expire {self.plan_ttl_s:g} s after the preview")
            else:
                pending.used = True
                moved = np.abs(q - pending.start) * np.where(self.guard.model.hinge, 180 / math.pi, 1000.0)
                if float(np.max(moved)) > START_TOLERANCE:
                    i = int(np.argmax(moved))
                    refusal = Refusal(
                        "PLAN_STALE",
                        f"{self.config.joints[i]} moved {float(moved[i]):.1f} since the preview; preview again",
                        {"joint": self.config.joints[i]},
                    )
            if refusal is not None:
                result: Result = {"ok": False, "refusal": refusal.to_dict(), "state": self._state(q, g)}
            else:
                assert pending is not None
                # The world may have changed since the preview: check again, from where the arm is now.
                plan = self.guard.check_waypoints(q, pending.plan.waypoints, g, pending.gripper_goal)
                if isinstance(plan, Refusal):
                    result = {"ok": False, "refusal": plan.to_dict(), "state": self._state(q, g)}
                else:
                    arguments["tool"] = pending.tool
                    result = self._execute(pending.tool, plan, pending.gripper_goal, pending.speed)
        except Exception as exc:
            result = self._internal_error(tool, exc)
        finally:
            self._motion.release()
        return self._finish(tool, arguments, result)

    def resume(self) -> Result:
        """Continue a paused motion: the path from the measured arm through the remaining waypoints is re-checked."""
        tool, arguments = "resume", {}
        if (latched := self._latched_refusal(tool, arguments)) is not None:
            return latched
        if not self._motion.acquire(blocking=False):
            return self._refuse(tool, arguments, "BUSY", "another motion is running")
        try:
            self._stop.clear()
            self._pause.clear()
            with self._paused_lock:
                paused = self._paused
            q, g = self.transport.read()
            if paused is None:
                result: Result = {
                    "ok": False,
                    "refusal": Refusal("NOTHING_TO_RESUME", "no motion is paused").to_dict(),
                    "state": self._state(q, g),
                }
            else:
                remaining = paused.plan.waypoints
                bridge = interpolate(q, remaining[0], float(np.min(self.guard.max_step)))
                plan = self.guard.check_waypoints(q, bridge + remaining[1:], g, paused.gripper_goal)
                if isinstance(plan, Refusal):
                    plan.details["note"] = "the paused motion is kept; stop() discards it"
                    result = {"ok": False, "refusal": plan.to_dict(), "state": self._state(q, g)}
                else:
                    with self._paused_lock:
                        self._paused = None
                    arguments = {"tool": paused.tool}
                    self.events.publish("resumed", tool=paused.tool, remaining_waypoints=len(plan.waypoints))
                    result = self._execute(paused.tool, plan, paused.gripper_goal, paused.speed, resumed=True)
        except Exception as exc:
            result = self._internal_error(tool, exc)
        finally:
            self._motion.release()
        return self._finish(tool, arguments, result)

    def _execute(self, tool: str, plan: Plan, gripper: Vec | None, speed: float, resumed: bool = False) -> Result:
        lim = self.config.limits
        speed = float(np.clip(speed, 0.05, 1.0))
        period = self._period(speed)
        with self._paused_lock:
            discarded = self._paused is not None and not resumed
            self._paused = None  # a new motion replaces whatever was paused; the result says so
        lowest = [math.inf]
        closest = [math.inf]
        q0, g0 = self.transport.read()
        start_clear = self._monitor.clearance_mm(q0, g0)
        start_gap = self._monitor.obstacle_clearance(q0, g0)[0]
        breach: list[str] = []
        count = [0]
        last_progress = [-math.inf]
        n = len(plan.waypoints)

        def monitor(q: Vec, g: Vec) -> None:
            clearance = self._monitor.clearance_mm(q, g)
            lowest[0] = min(lowest[0], clearance)
            gap, part, obstacle = self._monitor.obstacle_clearance(q, g)
            closest[0] = min(closest[0], gap)
            # Below a hard limit is a breach, except for an arm that started below it and is not getting closer
            # (moving away after a halt).
            if clearance < lim.min_clearance_mm and clearance < start_clear - MONITOR_TOLERANCE_MM:
                breach.append(f"measured clearance {clearance:.1f} mm, hard limit {lim.min_clearance_mm:g} mm")
            elif gap < lim.obstacle_clearance_mm and gap < start_gap - MONITOR_TOLERANCE_MM:
                breach.append(
                    f"measured {part} {max(gap, 0):.1f} mm from {obstacle}, hard limit {lim.obstacle_clearance_mm:g} mm"
                )
            elif contacts := self._monitor.contacts(q, g):
                breach.append(f"measured contact {contacts[0].a} / {contacts[0].b}")
            count[0] += 1
            now = self.clock()
            if not breach and now - last_progress[0] >= PROGRESS_PERIOD_S:
                last_progress[0] = now
                self.events.publish("progress", tool=tool, waypoint=min(count[0], n), of=n, state=self._state(q, g))

        self.events.publish("motion_started", tool=tool, waypoints=n, duration_s=round(n * period, 2), resumed=resumed)
        try:
            followed = self.transport.follow(
                plan.waypoints,
                gripper,
                period,
                lambda: self._stop.is_set() or self._pause.is_set() or bool(breach),
                monitor,
            )
            final = self.transport.read()
            if not breach:
                monitor(*final)  # after the servos settle: what the arm holds now must be inside the envelope too
        except Exception as exc:
            self.events.publish("motion_finished", tool=tool, ok=False, code="INTERNAL_ERROR", message=str(exc))
            raise
        with self._view_lock:
            planned_tool = self._view.tool(plan.waypoints[-1])[0]
            measured_tool = self._view.tool(final[0])[0]
        tracking_mm = round(float(np.linalg.norm(planned_tool - measured_tool)) * 1000, 1)
        paused = False
        if breach:
            refusal: Refusal | None = Refusal("ENVELOPE_BREACH", f"halted: {breach[0]}")
        elif followed.stopped and not self._stop.is_set() and self._pause.is_set():
            remainder = plan.waypoints[followed.waypoints_done :]
            with self._paused_lock:
                if not self._stop.is_set():  # a stop that raced the pause wins
                    self._paused = PendingMotion(
                        tool, Plan(remainder, plan.min_clearance_mm), gripper, speed, final[0], 0
                    )
                    paused = True
            if paused:
                refusal = Refusal(
                    "PAUSED",
                    f"paused on request with {len(remainder)} of {n} waypoints left; do not resume or start another "
                    "motion unless asked",
                    {"remaining_waypoints": len(remainder)},
                )
            else:
                refusal = Refusal("STOPPED", "motion was stopped before it finished")
        elif followed.stopped:
            refusal = Refusal("STOPPED", "motion was stopped before it finished")
        else:
            refusal = None
        executed: Result = {
            "waypoints": followed.waypoints_done,
            "min_clearance_mm": round(lowest[0], 1),
            **({"min_obstacle_gap_mm": round(closest[0], 1)} if math.isfinite(closest[0]) else {}),
            "tool_tracking_error_mm": tracking_mm,
        }
        state = self._state(*final)
        result: Result = {
            "ok": refusal is None,
            "plan": {
                "waypoints": n,
                "duration_s": round(n * period, 2),
                "planned_min_clearance_mm": round(plan.min_clearance_mm, 1),
            },
            "executed": executed,
            "state": state,
        }
        if discarded:
            result["discarded_paused_motion"] = True
        if refusal is not None:
            result["refusal"] = humanize(refusal.to_dict(), self.lang)
        if paused:
            result["paused"] = True
        if breach:
            self.events.publish("breach", tool=tool, refusal=result["refusal"], state=state)
        elif paused:
            self.events.publish("paused", tool=tool, remaining_waypoints=len(plan.waypoints) - followed.waypoints_done)
        elif followed.stopped:
            self.events.publish("stopped", tool=tool, waypoints_done=followed.waypoints_done)
        self.events.publish(
            "motion_finished",
            tool=tool,
            ok=refusal is None,
            **({"code": refusal.code} if refusal is not None else {}),
            executed=executed,
            state=state,
        )
        return result

    # ------------------------------------------------------------------ motions
    def _bad(self, tool: str, arguments: Result, message: str) -> Result:
        arguments = {**arguments, "preview": arguments.get("preview", False) or self.require_confirmation}
        return self._refuse(tool, arguments, "BAD_TARGET", message)

    def move_to(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        tool_down: bool = True,
        speed: float = 0.5,
        path: str = "joint",
        preview: bool = False,
    ) -> Result:
        floor = self.guard.model.floor_z
        target = np.array([x_mm / 1000, y_mm / 1000, floor + z_mm / 1000])
        direction = np.array([0.0, 0.0, -1.0]) if tool_down else None
        args = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "tool_down": tool_down, "speed": speed, "path": path}
        if path not in ("joint", "line"):
            return self._bad("move_to", args, "path must be 'joint' or 'line'")
        return self._request(
            "move_to",
            args,
            speed,
            lambda q, g: (self.guard.plan_pose_with_fallback(q, target, direction, g, path == "line"), None),
            preview,
        )

    def _goal(self, q: Vec, joints: dict[str, float], relative: bool = False) -> Vec:
        goal = q.copy()
        for i, name in enumerate(self.config.joints):
            if name in joints:
                v = joints[name]
                v = math.radians(v) if self.guard.model.hinge[i] else v / 1000
                goal[i] = goal[i] + v if relative else v
        return goal

    def move_joints(self, joints: dict[str, float], speed: float = 0.5, preview: bool = False) -> Result:
        args = {"joints": joints, "speed": speed}
        unknown = sorted(set(joints) - set(self.config.joints))
        if unknown:
            return self._bad("move_joints", args, f"unknown joints {unknown}; known: {list(self.config.joints)}")
        return self._request(
            "move_joints", args, speed, lambda q, g: (self.guard.check_path(q, self._goal(q, joints), g), None), preview
        )

    def _gripper_goal(self, opening: float) -> Vec:
        cfg = self.config.gripper
        assert cfg is not None
        return np.array(cfg.closed) + (np.array(cfg.open) - np.array(cfg.closed)) * opening

    def _joint_path(self, q: Vec, g: Vec, goal: Vec, gripper_goal: Vec | None) -> Plan | Refusal:
        """Joint-space path to `goal`, with the gripper moving along when `gripper_goal` is given."""
        refusal = self.guard.check_joints(goal)
        if refusal is not None:
            return refusal
        waypoints = interpolate(q, goal, float(np.min(self.guard.max_step)))
        if gripper_goal is not None:
            steps = max(2, math.ceil(float(np.max(np.abs(gripper_goal - g), initial=0.0)) / GRIPPER_STEP))
            if len(waypoints) < steps:
                waypoints = [q + (goal - q) * (i / steps) for i in range(1, steps + 1)]
        return self.guard.check_waypoints(q, waypoints, g, gripper_goal)

    def gripper(self, opening: float, speed: float = 0.5, preview: bool = False) -> Result:
        args = {"opening": opening, "speed": speed}
        if self.config.gripper is None:
            return self._bad("gripper", args, "this robot has no gripper configured")
        if not (math.isfinite(opening) and 0.0 <= opening <= 1.0):
            return self._bad("gripper", args, "opening must be between 0 (closed) and 1 (open)")
        goal = self._gripper_goal(opening)
        return self._request("gripper", args, speed, lambda q, g: (self._joint_path(q, g, q, goal), goal), preview)

    def rest(self, speed: float = 0.5, preview: bool = False) -> Result:
        home = self.guard.model.home
        return self._request(
            "rest", {"speed": speed}, speed, lambda q, g: (self.guard.check_path(q, home, g), None), preview
        )

    def jog(
        self,
        dx_mm: float = 0.0,
        dy_mm: float = 0.0,
        dz_mm: float = 0.0,
        tool_down: bool = False,
        speed: float = 0.5,
        preview: bool = False,
    ) -> Result:
        """Move the tool point by (dx, dy, dz) along a straight line. tool_down=False keeps the tool's direction."""
        args = {"dx_mm": dx_mm, "dy_mm": dy_mm, "dz_mm": dz_mm, "tool_down": tool_down, "speed": speed}
        delta = np.array([dx_mm, dy_mm, dz_mm], dtype=float)
        if not np.all(np.isfinite(delta)):
            return self._bad("jog", args, "dx_mm, dy_mm and dz_mm must be finite numbers")
        if float(np.linalg.norm(delta)) > JOG_MAX_MM + 1e-9:
            return self._bad("jog", args, f"a jog moves at most {JOG_MAX_MM:g} mm; use move_to for longer moves")
        direction = np.array([0.0, 0.0, -1.0]) if tool_down else None

        def planner(q: Vec, g: Vec) -> tuple[Plan | Refusal, Vec | None]:
            start = self.guard.model.tool(q)[0]
            plan = self.guard.plan_pose_with_fallback(q, start + delta / 1000, direction, g, line=True)
            if isinstance(plan, Refusal) and plan.suggestion is not None:
                plan.suggestion["tool"] = "move_to"
            return plan, None

        return self._request("jog", args, speed, planner, preview)

    def jog_joint(self, joint: str, delta: float, speed: float = 0.5, preview: bool = False) -> Result:
        """Move one joint by `delta` degrees (hinge) or millimetres (slide), at most JOG_MAX_JOINT."""
        args = {"joint": joint, "delta": delta, "speed": speed}
        if joint not in self.config.joints:
            return self._bad("jog_joint", args, f"unknown joint {joint!r}; known: {list(self.config.joints)}")
        if not math.isfinite(delta) or abs(delta) > JOG_MAX_JOINT:
            unit = "deg" if self.guard.model.hinge[self.config.joints.index(joint)] else "mm"
            return self._bad("jog_joint", args, f"delta must be finite and at most {JOG_MAX_JOINT:g} {unit}")
        return self._request(
            "jog_joint",
            args,
            speed,
            lambda q, g: (self.guard.check_path(q, self._goal(q, {joint: delta}, relative=True), g), None),
            preview,
        )

    # ------------------------------------------------------------------ named poses
    def save_pose(self, name: str) -> Result:
        """Store where the arm is now (joints and gripper) under `name`. Nothing moves."""
        tool, args = "save_pose", {"name": name}
        if not valid_name(name):
            return self._refuse(
                tool, args, "BAD_TARGET", "pose names are 1-40 letters (any script), digits, '_' or '-'"
            )
        if not self._motion.acquire(blocking=False):
            return self._refuse(tool, args, "BUSY", "the arm is moving; save the pose once it holds still")
        try:
            q, g = self.transport.read()
        finally:
            self._motion.release()
        hinge = self.guard.model.hinge
        pose: Result = {
            "joints": {
                n: (math.degrees(float(q[i])) if hinge[i] else float(q[i]) * 1000)
                for i, n in enumerate(self.config.joints)
            }
        }
        opening = self._opening(g)
        if opening is not None:
            pose["gripper_opening"] = opening
        try:
            replaced = self.poses.put(name, pose)
        except ConfigError as exc:
            return self._refuse(tool, args, "BAD_TARGET", str(exc))
        result: Result = {"ok": True, "name": name, "replaced": replaced, "pose": self.poses.get(name)}
        self.trace.record(tool, args, result)
        self.events.publish("pose_saved", name=name, replaced=replaced)
        return result

    def delete_pose(self, name: str) -> Result:
        tool, args = "delete_pose", {"name": name}
        if not self.poses.delete(name):
            return self._refuse(tool, args, "POSE_UNKNOWN", f"no pose named {name!r}", name=name)
        result: Result = {"ok": True, "name": name}
        self.trace.record(tool, args, result)
        self.events.publish("pose_deleted", name=name)
        return result

    def go_to_pose(self, name: str, speed: float = 0.5, include_gripper: bool = False, preview: bool = False) -> Result:
        """Move to a saved pose through the normal checked path. The gripper moves only with include_gripper."""
        tool, args = "go_to_pose", {"name": name, "speed": speed, "include_gripper": include_gripper}
        pose = self.poses.get(name)
        if pose is None:
            args["preview"] = preview or self.require_confirmation
            return self._refuse(tool, args, "POSE_UNKNOWN", f"no pose named {name!r}; see list_poses", name=name)
        gripper_goal = None
        if include_gripper and self.config.gripper is not None and "gripper_opening" in pose:
            gripper_goal = self._gripper_goal(float(pose["gripper_opening"]))

        def planner(q: Vec, g: Vec) -> tuple[Plan | Refusal, Vec | None]:
            return self._joint_path(q, g, self._goal(q, pose["joints"]), gripper_goal), gripper_goal

        return self._request(tool, args, speed, planner, preview)

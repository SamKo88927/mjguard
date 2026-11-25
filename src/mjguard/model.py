"""Kinematic and geometric queries on the robot's MuJoCo model: tool pose, contacts, floor clearance, IK.

One ArmModel owns one MjData. It is not thread-safe; give each thread its own instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco as _mujoco
import numpy as np
from numpy.typing import NDArray

from .config import ConfigError, RobotConfig

# MuJoCo's compiled bindings ship no type stubs; keep the untyped surface behind one explicit name.
mj: Any = _mujoco

Vec = NDArray[np.float64]

_AXES = {"x": 0, "y": 1, "z": 2}
REST_CONTACT_TOLERANCE_M = 0.002


@dataclass(frozen=True)
class Contact:
    a: str
    b: str
    depth_mm: float
    floor: bool


def _name(m: Any, kind: Any, i: int) -> str:
    return mj.mj_id2name(m, kind, i) or f"#{i}"


OBSTACLE_PREFIX = "mjguard_obstacle:"


class ArmModel:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.m = self._compile(config)
        self.d = mj.MjData(self.m)
        m = self.m

        def lookup(kind: Any, name: str, what: str) -> int:
            i = mj.mj_name2id(m, kind, name)
            if i < 0:
                raise ConfigError(f"{what} {name!r} is not in {config.mjcf.name}")
            return i

        self.joint_ids = [lookup(mj.mjtObj.mjOBJ_JOINT, j, "joint") for j in config.joints]
        for j, jid in zip(config.joints, self.joint_ids, strict=True):
            if int(m.jnt_type[jid]) not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
                raise ConfigError(f"joint {j!r} must be a hinge or slide joint")
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.hinge = np.array([int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_HINGE) for j in self.joint_ids])
        self.floor = lookup(mj.mjtObj.mjOBJ_GEOM, config.floor_geom, "floor geom")
        if int(m.geom_type[self.floor]) != int(mj.mjtGeom.mjGEOM_PLANE):
            raise ConfigError(f"floor geom {config.floor_geom!r} must be a plane")
        self.floor_z = float(m.geom_pos[self.floor][2])
        self.obstacle_geoms = [
            g for g in range(m.ngeom) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "").startswith(OBSTACLE_PREFIX)
        ]
        self.site = lookup(mj.mjtObj.mjOBJ_SITE, config.tcp_site, "site") if config.tcp_site else -1
        self.body = lookup(mj.mjtObj.mjOBJ_BODY, config.tcp_body, "body") if config.tcp_body else -1
        self.offset = np.array(config.tcp_offset_m, dtype=float)
        self.axis = _AXES[config.tool_axis[-1]]
        self.axis_sign = -1.0 if config.tool_axis.startswith("-") else 1.0

        self.gripper_qadr = np.array([], dtype=int)
        if config.gripper is not None:
            gids = [lookup(mj.mjtObj.mjOBJ_JOINT, j, "gripper joint") for j in config.gripper.joints]
            self.gripper_qadr = np.array([m.jnt_qposadr[g] for g in gids])
        self.actuator_ids = [lookup(mj.mjtObj.mjOBJ_ACTUATOR, a, "actuator") for a in config.actuators]
        self.gripper_actuator = (
            lookup(mj.mjtObj.mjOBJ_ACTUATOR, config.gripper.actuator, "gripper actuator")
            if config.gripper is not None and config.gripper.actuator
            else -1
        )

        # Bodies that move when an arm or gripper joint moves. Only their geoms count for clearance and contacts:
        # a base bolted to the floor touches it by design.
        moving_joints = set(self.joint_ids)
        if config.gripper is not None:
            moving_joints |= {mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j) for j in config.gripper.joints}
        first = {int(m.jnt_bodyid[j]) for j in moving_joints}
        self.moving_body = np.zeros(m.nbody, dtype=bool)
        for b in range(m.nbody):
            k = b
            while k > 0:
                if k in first:
                    self.moving_body[b] = True
                    break
                k = int(m.body_parentid[k])
        collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.moving_geoms = [
            g for g in range(m.ngeom) if collides[g] and self.moving_body[m.geom_bodyid[g]] and g != self.floor
        ]
        if not self.moving_geoms:
            raise ConfigError("no collision geoms move with the arm joints; check arm.joints")

        lo = m.jnt_range[self.joint_ids, 0].astype(float)
        hi = m.jnt_range[self.joint_ids, 1].astype(float)
        limited = m.jnt_limited[self.joint_ids].astype(bool)
        self.lower = np.where(limited, lo, -np.inf)
        self.upper = np.where(limited, hi, np.inf)

        if config.keyframe is not None:
            key = lookup(mj.mjtObj.mjOBJ_KEY, config.keyframe, "keyframe")
            mj.mj_resetDataKeyframe(m, self.d, key)
        else:
            mj.mj_resetData(m, self.d)
        mj.mj_forward(m, self.d)
        self.qpos0 = self.d.qpos.copy()
        self.home = self.d.qpos[self.qadr].copy()
        self.home_gripper = self.d.qpos[self.gripper_qadr].copy()
        self._ft = np.zeros(6)

        # A link whose moving joints are all hinges about a vertical axis keeps its height whatever the joints do
        # (a turret on its base). It cannot approach the floor, so the clearance check skips it; contacts still count.
        vertical = {
            j
            for j in moving_joints
            if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_HINGE) and abs(float(self.d.xaxis[j][2])) > 0.999
        }

        def height_fixed(body: int) -> bool:
            k = body
            while k > 0:
                for j in range(m.body_jntadr[k], m.body_jntadr[k] + m.body_jntnum[k]):
                    if j in moving_joints and j not in vertical:
                        return False
                k = int(m.body_parentid[k])
            return True

        self.clearance_geoms = [g for g in self.moving_geoms if not height_fixed(int(m.geom_bodyid[g]))]

        # Two bodies moved by exactly the same arm joints keep their relative pose whatever the arm does (two fingers
        # of a gripper, say). Their contact cannot be caused or cured by an arm motion, so it is not a collision.
        def chain(body: int) -> frozenset[int]:
            found: set[int] = set()
            k = body
            while k > 0:
                found |= {j for j in range(m.body_jntadr[k], m.body_jntadr[k] + m.body_jntnum[k]) if j in moving_joints}
                k = int(m.body_parentid[k])
            return frozenset(found)

        # Fingers move against each other by design: bodies whose relative pose depends only on gripper joints are
        # not checked against each other either. Fingers against the arm, the table or obstacles still are.
        gripper_joints = moving_joints - set(self.joint_ids)
        self._chain = [chain(b) - gripper_joints for b in range(m.nbody)]

        # Contacts the model already has in its rest pose are part of its design: collision hulls of a turret that
        # overlap its base, closed fingers that touch, a shoulder that sits on the floor. They are allowed up to the
        # depth they have at rest (plus a small tolerance); anything deeper, or any other pair, is a collision.
        # `mjguard check` lists them so they stay visible.
        self.rest_contacts: dict[frozenset[int], float] = {}
        obstacles = set(self.obstacle_geoms)
        for c in self._raw_contacts(self.home, self.home_gripper):
            if c.geom1 in obstacles or c.geom2 in obstacles:
                continue  # touching an obstacle at rest is a config error, not a design feature
            key = self._pair(c.geom1, c.geom2)
            self.rest_contacts[key] = max(self.rest_contacts.get(key, 0.0), -float(c.dist))

    def _raw_contacts(self, q: Sequence[float] | Vec, gripper: Sequence[float] | Vec | None) -> list[Any]:
        self.set(q, gripper)
        mj.mj_collision(self.m, self.d)
        moving = set(self.moving_geoms)
        return [
            c
            for c in self.d.contact[: self.d.ncon]
            if c.dist < 0
            and (c.geom1 in moving or c.geom2 in moving)
            and self._chain[int(self.m.geom_bodyid[c.geom1])] != self._chain[int(self.m.geom_bodyid[c.geom2])]
        ]

    def _pair(self, g1: int, g2: int) -> frozenset[int]:
        """Contact identity: the body for arm parts, the geom itself (negative id) for static scene geoms."""

        def key(g: int) -> int:
            body = int(self.m.geom_bodyid[g])
            return body if body > 0 else -(g + 1)

        return frozenset((key(g1), key(g2)))

    def _pair_label(self, k: int) -> str:
        return self.geom_label(-k - 1) if k < 0 else _name(self.m, mj.mjtObj.mjOBJ_BODY, k)

    def describe_rest_contacts(self) -> list[dict[str, Any]]:
        return [
            {"bodies": sorted(self._pair_label(k) for k in pair), "depth_mm_at_rest": round(depth * 1000, 1)}
            for pair, depth in self.rest_contacts.items()
        ]

    @staticmethod
    def _compile(config: RobotConfig) -> Any:
        if not config.obstacles:
            return mj.MjModel.from_xml_path(str(config.mjcf))
        spec = mj.MjSpec.from_file(str(config.mjcf))
        floor_z = 0.0
        for g in spec.geoms:
            if g.name == config.floor_geom:
                floor_z = float(g.pos[2])
        for o in config.obstacles:
            geom = spec.worldbody.add_geom()
            geom.name = OBSTACLE_PREFIX + o.name
            size = [v / 1000 for v in o.size_mm]
            if o.shape == "box":
                geom.type, geom.size = mj.mjtGeom.mjGEOM_BOX, [size[0] / 2, size[1] / 2, size[2] / 2]
            elif o.shape == "cylinder":
                geom.type, geom.size = mj.mjtGeom.mjGEOM_CYLINDER, [size[0] / 2, size[1] / 2, 0]
            else:
                geom.type, geom.size = mj.mjtGeom.mjGEOM_SPHERE, [size[0] / 2, 0, 0]
            x, y, z = (v / 1000 for v in o.center_mm)
            geom.pos = [x, y, floor_z + z]
            geom.rgba = [0.95, 0.55, 0.2, 0.55]
            geom.contype, geom.conaffinity = 1, 1
        return spec.compile()

    def geom_label(self, g: int) -> str:
        """Body name for arm parts; `obstacle 'name'` for configured obstacles; geom name for other scene geoms."""
        name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(OBSTACLE_PREFIX):
            return f"obstacle '{name.removeprefix(OBSTACLE_PREFIX)}'"
        body = int(self.m.geom_bodyid[g])
        if body == 0 and name:
            return name
        return _name(self.m, mj.mjtObj.mjOBJ_BODY, body)

    def obstacle_clearance(
        self, q: Sequence[float] | Vec, gripper: Sequence[float] | Vec | None = None
    ) -> tuple[float, str, str]:
        """(mm, arm part, obstacle) for the closest pair; (inf, "", "") without obstacles."""
        best = (float("inf"), "", "")
        if not self.obstacle_geoms:
            return best
        self.set(q, gripper)
        for g in self.moving_geoms:
            for o in self.obstacle_geoms:
                dist = float(mj.mj_geomDistance(self.m, self.d, g, o, 0.5, self._ft)) * 1000
                if dist < best[0]:
                    best = (dist, self.geom_label(g), self.geom_label(o))
        return best

    @property
    def n(self) -> int:
        return len(self.joint_ids)

    # ------------------------------------------------------------------ state
    def set(self, q: Sequence[float] | Vec, gripper: Sequence[float] | Vec | None = None) -> None:
        self.d.qpos[:] = self.qpos0
        self.d.qpos[self.qadr] = q
        if gripper is not None and len(self.gripper_qadr):
            self.d.qpos[self.gripper_qadr] = gripper
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)

    def _frame(self) -> tuple[Vec, Vec]:
        if self.site >= 0:
            return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()
        rot = self.d.xmat[self.body].reshape(3, 3).copy()
        return self.d.xpos[self.body] + rot @ self.offset, rot

    def tool(self, q: Sequence[float] | Vec) -> tuple[Vec, Vec]:
        """Tool point (metres, world frame) and the unit tool axis."""
        self.set(q)
        pos, rot = self._frame()
        return pos, rot[:, self.axis] * self.axis_sign

    # ------------------------------------------------------------------ geometry
    def contacts(self, q: Sequence[float] | Vec, gripper: Sequence[float] | Vec | None = None) -> list[Contact]:
        """Penetrating contacts that involve at least one moving geom."""
        out: list[Contact] = []
        for c in self._raw_contacts(q, gripper):
            allowed = self.rest_contacts.get(self._pair(c.geom1, c.geom2))
            if allowed is not None and -float(c.dist) <= allowed + REST_CONTACT_TOLERANCE_M:
                continue
            out.append(
                Contact(
                    self.geom_label(c.geom1),
                    self.geom_label(c.geom2),
                    round(-float(c.dist) * 1000, 1),
                    self.floor in (c.geom1, c.geom2),
                )
            )
        return out

    def clearance_mm(self, q: Sequence[float] | Vec, gripper: Sequence[float] | Vec | None = None) -> float:
        """Lowest distance from any moving collision geom, or the tool point, to the floor plane."""
        self.set(q, gripper)
        lowest = float(self._frame()[0][2] - self.floor_z)
        for g in self.clearance_geoms:
            lowest = min(lowest, float(mj.mj_geomDistance(self.m, self.d, g, self.floor, 1.0, self._ft)))
        return lowest * 1000.0

    # ------------------------------------------------------------------ inverse kinematics
    def solve_ik(
        self,
        target: Vec,
        direction: Vec | None,
        seed: Sequence[float] | Vec,
        lower: Vec,
        upper: Vec,
        iters: int = 200,
    ) -> tuple[Vec, float, float]:
        """Damped least squares on the tool point (and tool axis when `direction` is given).

        Returns (q, position error in mm, axis error in degrees). The caller decides whether that is converged.
        """
        q = np.clip(np.asarray(seed, dtype=float).copy(), lower, upper)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        damping = 1e-3
        pos_err = ang_err = np.inf
        for _ in range(iters):
            self.set(q)
            pos, rot = self._frame()
            axis = rot[:, self.axis] * self.axis_sign
            e_pos = target - pos
            pos_err = float(np.linalg.norm(e_pos)) * 1000
            if direction is not None:
                e_rot = np.cross(axis, direction)
                ang_err = float(np.degrees(np.arccos(np.clip(axis @ direction, -1, 1))))
            else:
                e_rot = np.zeros(3)
                ang_err = 0.0
            if pos_err < 0.05 and ang_err < 0.05:
                break
            if self.site >= 0:
                mj.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            else:
                mj.mj_jac(self.m, self.d, jacp, jacr, pos, self.body)
            if direction is not None:
                jac = np.vstack([jacp[:, self.dadr], 0.2 * jacr[:, self.dadr]])
                err = np.concatenate([e_pos, 0.2 * e_rot])
            else:
                jac, err = jacp[:, self.dadr], e_pos
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(len(err)), err)
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, lower, upper)
        return q, pos_err, ang_err


def interpolate(start: Vec, goal: Vec, max_step: float) -> list[Vec]:
    """Joint-space straight line, sampled so no joint moves more than max_step between waypoints."""
    span = float(np.max(np.abs(goal - start))) if len(goal) else 0.0
    n = max(1, int(np.ceil(span / max_step)))
    return [start + (goal - start) * (i / n) for i in range(1, n + 1)]

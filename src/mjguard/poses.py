"""Named poses: stored joint targets, optionally persisted to a JSON file.

A stored pose is only a target. Going to one is planned and checked like any other motion, with the envelope that
is in force now, so a pose file (edited by hand, or saved before an obstacle was added) cannot move the arm anywhere
a direct request could not.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from .config import ConfigError

NAME = re.compile(r"[\w-]{1,40}")
MAX_POSES = 200
Pose = dict[str, Any]
"""{"joints": {name: degrees or millimetres}, "gripper_opening": 0..1 (optional)}"""


def valid_name(name: str) -> bool:
    return isinstance(name, str) and NAME.fullmatch(name) is not None  # pyright: ignore[reportUnnecessaryIsInstance]


def _validate(name: str, pose: Any, joints: tuple[str, ...]) -> Pose:
    where = f"pose {name!r}"
    if not valid_name(name):
        raise ConfigError(f"{where}: names are 1-40 letters, digits, '_' or '-'")
    if not isinstance(pose, dict) or not isinstance(pose.get("joints"), dict):
        raise ConfigError(f"{where}: needs a 'joints' table")
    values: dict[str, Any] = pose["joints"]
    if set(values) != set(joints):
        raise ConfigError(f"{where}: joints must be exactly {list(joints)}")
    out: Pose = {"joints": {}}
    for j in joints:
        v = values[j]
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ConfigError(f"{where}: joint {j} must be a finite number")
        out["joints"][j] = float(v)
    if "gripper_opening" in pose:
        g = pose["gripper_opening"]
        if isinstance(g, bool) or not isinstance(g, int | float) or not 0.0 <= g <= 1.0:
            raise ConfigError(f"{where}: gripper_opening must be between 0 and 1")
        out["gripper_opening"] = float(g)
    return out


class PoseStore:
    """Thread-safe name -> pose map. With a path, loads it at start and rewrites it atomically on every change."""

    def __init__(self, joints: tuple[str, ...], path: Path | None = None) -> None:
        self.joints = joints
        self.path = path
        self._lock = threading.Lock()
        self._poses: dict[str, Pose] = {}
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(f"poses file {path}: {exc}") from exc
            if not isinstance(data, dict) or not isinstance(data.get("poses"), dict):
                raise ConfigError(f'poses file {path}: expected {{"poses": {{name: pose}}}}')
            raw: dict[str, Any] = data["poses"]
            if len(raw) > MAX_POSES:
                raise ConfigError(f"poses file {path}: more than {MAX_POSES} poses")
            self._poses = {name: _validate(name, pose, joints) for name, pose in raw.items()}

    def _write(self) -> None:
        if self.path is None:
            return
        payload = json.dumps({"poses": self._poses}, indent=2, ensure_ascii=False, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".poses-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, name: str) -> Pose | None:
        with self._lock:
            pose = self._poses.get(name)
            return None if pose is None else {**pose, "joints": dict(pose["joints"])}

    def put(self, name: str, pose: Pose) -> bool:
        """Store a pose; returns True if it replaced one. Raises ConfigError on a bad name or a full store."""
        checked = _validate(name, pose, self.joints)
        with self._lock:
            replaced = name in self._poses
            if not replaced and len(self._poses) >= MAX_POSES:
                raise ConfigError(f"at most {MAX_POSES} poses; delete one first")
            self._poses[name] = checked
            self._write()
            return replaced

    def delete(self, name: str) -> bool:
        with self._lock:
            if self._poses.pop(name, None) is None:
                return False
            self._write()
            return True

    def all(self) -> dict[str, Pose]:
        with self._lock:
            return {k: {**v, "joints": dict(v["joints"])} for k, v in sorted(self._poses.items())}

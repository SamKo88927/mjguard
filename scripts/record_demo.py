"""Record the README GIF from the real code path: Executor + Guard + SimTransport on a Menagerie arm.

uv run --group demo python scripts/record_demo.py docs/media/demo.gif [--robot panda]
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjguard.config import from_dict
from mjguard.executor import Executor, FollowResult, SimTransport

W, H, FPS = 576, 360, 12
GREEN, RED, WHITE = (90, 220, 140), (240, 90, 90), (235, 235, 235)


class Recorder(SimTransport):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = (
            max(W, self.m.vis.global_.offwidth),
            max(H, self.m.vis.global_.offheight),
        )
        self.renderer = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (0.36, -0.03, 0.32)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.95, 60, -20
        self.frames: list[np.ndarray] = []
        self.caption: tuple[str, str, tuple[int, int, int]] = ("", "", WHITE)
        self.marker: tuple[np.ndarray, tuple[float, float, float, float]] | None = None
        self.part: np.ndarray | None = None  # a small cube drawn on the table: the thing being picked

    def snap(self, n: int = 1) -> None:
        self.renderer.update_scene(self.d, self.cam)
        if self.marker is not None:
            scene = self.renderer.scene
            pos, rgba = self.marker
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]),
                pos,
                np.eye(3).flatten(),
                np.array(rgba, dtype=np.float32),
            )
            scene.ngeom += 1
        if self.part is not None:
            scene = self.renderer.scene
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([0.02, 0.02, 0.02]),
                self.part,
                np.eye(3).flatten(),
                np.array((0.25, 0.55, 0.95, 1.0), dtype=np.float32),
            )
            scene.ngeom += 1
        self.frames.extend([annotate(self.renderer.render(), *self.caption)] * n)

    def follow(
        self,
        waypoints: Sequence[np.ndarray],
        gripper: Any,
        period_s: float,
        should_stop: Callable[[], bool],
        on_waypoint: Callable[[Any, Any], None],
    ) -> FollowResult:
        every = max(1, round(1 / FPS / period_s))

        def watch(q: Any, g: Any) -> None:
            on_waypoint(q, g)
            watch.i += 1  # type: ignore[attr-defined]
            if watch.i % every == 0:  # type: ignore[attr-defined]
                self.snap()

        watch.i = 0  # type: ignore[attr-defined]
        result = super().follow(waypoints, gripper, period_s, should_stop, watch)
        self.snap(FPS // 2)
        return result


def font(size: int) -> Any:
    for path in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(rgb: np.ndarray, call: str, result: str, colour: tuple[int, int, int]) -> np.ndarray:
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 0, W, 62), fill=(10, 12, 20, 200))
    d.text((12, 8), call, font=font(15), fill=WHITE)
    d.text((12, 34), result, font=font(13), fill=colour)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    preset = Path(__file__).resolve().parents[1] / "src/mjguard/presets/panda.toml"
    data = tomllib.loads(preset.read_text())
    data["obstacles"] = [{"name": "bin", "shape": "box", "size_mm": [160, 200, 160], "center_mm": [450, -200, 80]}]
    config = from_dict(data, preset.parent)
    sim = Recorder(config)
    ex = Executor(config, sim)
    floor = sim.model.floor_z

    short = {
        "TABLE_CLEARANCE": "too close to the table",
        "UNREACHABLE": "no joint solution reaches it",
        "OBSTACLE_CLEARANCE": "too close to the bin",
        "COLLISION": "would hit something",
    }

    def step(call: str, run: Callable[[], dict[str, Any]], xyz: tuple[float, float, float] | None) -> dict[str, Any]:
        sim.marker = None
        if xyz is not None:
            sim.marker = (np.array([xyz[0] / 1000, xyz[1] / 1000, floor + xyz[2] / 1000]), (0.9, 0.9, 0.9, 0.6))
        sim.caption = (call, "checking every waypoint in the model, then moving...", WHITE)
        sim.snap(FPS // 3)
        r = run()
        print(f"{call:60s} -> {'ok' if r['ok'] else r['refusal']['code'] + ': ' + r['refusal']['message'][:70]}")
        if r["ok"]:
            sim.caption = (call, "ok", GREEN)
            if sim.marker is not None:
                sim.marker = (sim.marker[0], (0.3, 0.9, 0.5, 0.8))
        else:
            ref = r["refusal"]
            hint = ""
            if "suggestion" in ref:
                sg = ref["suggestion"]
                hint = f"; suggests ({sg['x_mm']:.0f}, {sg['y_mm']:.0f}, {sg['z_mm']:.0f})"
            sim.caption = (call, f"REFUSED {ref['code']}: {short.get(ref['code'], '')}{hint}", RED)
            if sim.marker is not None:
                sim.marker = (sim.marker[0], (0.95, 0.3, 0.3, 0.9))
        sim.snap(int(FPS * (2.6 if not r["ok"] else 0.5)))
        return r

    sim.part = np.array([0.45, 0.15, floor + 0.02])
    step("gripper(opening=1)", lambda: ex.gripper(1.0), None)
    r = step("move_to(x=450, y=-200, z=60)   # into the bin", lambda: ex.move_to(450, -200, 60), (450, -200, 60))
    s = r["refusal"]["suggestion"]
    step(
        f"move_to(x={s['x_mm']:.0f}, y={s['y_mm']:.0f}, z={s['z_mm']:.0f})   # the suggestion",
        lambda: ex.move_to(s["x_mm"], s["y_mm"], s["z_mm"]),
        (s["x_mm"], s["y_mm"], s["z_mm"]),
    )
    step(
        'move_to(x=477, y=-148, z=320, path="line")   # back off upward',
        lambda: ex.move_to(477, -148, 320, path="line"),
        (477, -148, 320),
    )
    step("move_to(x=450, y=150, z=150)", lambda: ex.move_to(450, 150, 150), (450, 150, 150))
    step('move_to(x=450, y=150, z=45, path="line")', lambda: ex.move_to(450, 150, 45, path="line"), (450, 150, 45))
    step("gripper(opening=0)", lambda: ex.gripper(0.0), None)
    sim.part = None  # carried from here on; drawn in the hand is beyond this demo
    step('move_to(x=450, y=150, z=220, path="line")', lambda: ex.move_to(450, 150, 220, path="line"), (450, 150, 220))
    step(
        'move_to(x=450, y=150, z=0, path="line")   # into the table',
        lambda: ex.move_to(450, 150, 0, path="line"),
        (450, 150, 0),
    )
    step("rest()", ex.rest, None)
    iio.imwrite(args.out, np.stack(sim.frames), duration=1000 / FPS, loop=0)
    print(f"{args.out}: {len(sim.frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

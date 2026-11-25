# mjguard

Let an LLM agent drive **any MuJoCo-modelled robot arm** over [MCP](https://modelcontextprotocol.io), with every
motion checked in the model before it runs.

Language models refuse unsafe requests in chat, then carry them out when they hold a robot's tools
(["Refuses in Chat, Complies with Arms"](https://arxiv.org/abs/2610.14043)). mjguard puts the check below the model, where a prompt cannot
reach it: the agent asks for a pose, mjguard plans it, checks every waypoint against the robot's own MuJoCo model,
and either moves or returns a refusal the model can act on.

```sh
uvx --from 'mjguard[menagerie]' mjguard check panda    # first run downloads MuJoCo Menagerie (about a minute)
uvx --from 'mjguard[menagerie]' mjguard serve panda    # MCP over stdio; or ur5e, or your own robot.toml
```

<p align="center">
  <img src="docs/media/demo.gif" alt="A simulated Franka Panda moves to a checked target; a move into the table and an out-of-reach move are refused with a suggested safe point, and the arm then follows the suggestion" width="576">
</p>

<sub>Franka Panda from MuJoCo Menagerie, driven through the same executor and guard the MCP server uses
(`scripts/record_demo.py`). A target into the table and one out of reach are refused; the refusal carries the
farthest checked-safe point, and the next call follows it.</sub>

## What it checks

Every motion is planned into joint-space waypoints, and **every waypoint** is checked before anything moves:

| Check | Refusal code |
| --- | --- |
| Joint ranges from the model, minus a margin | `JOINT_LIMIT` |
| Contacts between moving links and the rest of the scene, from MuJoCo's own collision pipeline | `COLLISION`, `TABLE_COLLISION` |
| Distance from every moving collision geom (meshes included) and the tool point to the floor or table, with a planning margin above the hard limit | `TABLE_CLEARANCE` |
| An optional workspace box for the tool point | `WORKSPACE` |
| IK convergence: a miss beyond 1 mm or 1.5° is a refusal, never a "close enough" plan | `UNREACHABLE` |

While the motion runs, and again after the servos settle, a monitor checks what the robot **actually** did and
halts it (`ENVELOPE_BREACH`) if the measured arm crosses the hard limit or touches anything. Plans keep a margin
(10 mm by default) above that limit: under physics, the UR5e and Panda presets settle about 6 mm below their plan,
and a test pins that they stay above the hard limit.

## Refusals a model can use

```json
{
  "ok": false,
  "refusal": {
    "code": "UNREACHABLE",
    "message": "no joint solution reaches the target; the closest misses by 2032.0 mm and 40.7 deg",
    "suggestion": {"x_mm": 890.7, "y_mm": 125.3, "z_mm": 277.5,
                   "note": "farthest checked-safe point on the straight line toward the target (19% of the way)"}
  }
}
```

The suggestion is the farthest point toward the target that passed every check, rounded the way a model will send
it back, and re-planned deterministically, so following it (with the `tool_down` it names) is accepted. A property test sends random targets and
follows every suggestion to keep it that way.

## Your own robot

Write a small TOML file next to your MJCF:

```toml
name = "my_arm"

[model]
mjcf = "scene.xml"          # or "menagerie:<name>" with the [menagerie] extra
floor_geom = "floor"        # a plane; clearance is measured against it
keyframe = "home"           # start and rest pose

[arm]
joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
tcp_site = "tool"           # or tcp_body = "hand" with tcp_offset_m = [0, 0, 0.1]
tool_axis = "z"             # the axis that points out of the tool
actuators = ["a1", "a2", "a3", "a4", "a5", "a6"]   # position servos: simulate with physics; omit for kinematic

[limits]
min_clearance_mm = 20       # hard limit, enforced on the measured arm
planning_margin_mm = 10     # plans keep 30 mm
workspace_mm = { x = [-600, 600], y = [-600, 600] }
```

```sh
mjguard check robot.toml     # loads it, checks the rest pose, prints the envelope
mjguard serve robot.toml     # MCP over stdio
```

Links whose moving joints all spin about a vertical axis (a turret on its base) cannot change height, so they are
exempt from the clearance check; they are still collision-checked.

### Claude Code

```sh
claude mcp add ur5e -- uvx --from 'mjguard[menagerie]' mjguard serve ur5e
```

## Tools

`get_state`, `get_limits`, `move_to(x_mm, y_mm, z_mm, tool_down=true, speed)`, `move_joints({name: value})`,
`gripper(opening)` (when configured), `rest`, `stop`. Resources: `mjguard://limits`, `mjguard://trace/latest`.
Millimetres and degrees; z is the height above the floor plane.

## Real hardware

mjguard drives a `Transport`: anything that can report joint positions and follow a list of checked waypoints. It
ships the MuJoCo simulator transport. A real arm needs a transport for its driver; the guard, the monitor and the
MCP surface stay the same. [roarm-mcp](https://github.com/SamKo88927/roarm-mcp) is the RoArm-M3 application this
library was extracted from.

`stop()` is a software stop. On real hardware a physical e-stop is the only real emergency stop.

## Limits of the method

- The check is only as good as the model. Objects that are not in the MJCF scene are invisible to it.
- Joint-space straight lines between waypoints are checked at a fixed spacing (3° by default), not continuously.
- Dynamics (speed, torque, payload) are not checked; the envelope is geometric.

## Development

```sh
uv sync --extra menagerie
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

## License

Apache-2.0. Robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under
their own licenses and are downloaded at runtime, not redistributed.

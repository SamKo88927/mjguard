# Changelog

## 0.1.0 (unreleased)

- Guard for any MuJoCo-modelled arm from a small TOML config: joint limits with margin, MuJoCo contacts, floor
  clearance of every moving collision geom (meshes included), optional workspace box, IK convergence check.
- Refusals with reason codes and a checked-safe suggestion that re-plans deterministically.
- Executor: one motion at a time, software stop, runtime monitor on measured joints, fault refusals, trace.
- MuJoCo simulator transport: physics with the model's own position servos, or kinematic.
- MCP server (`mjguard serve`) and config checker (`mjguard check`).
- Presets: UR5e and Franka Panda from MuJoCo Menagerie (`[menagerie]` extra).

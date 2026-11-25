# Changelog

## 0.3.2 (2026-11-25)

- Fix (safety): the obstacle gap was enforced only at plan time. Under physics the Z1 settled 2.7 mm from a box with
  `ok: true` and the iiwa touched it. `obstacle_clearance_mm` is now a hard limit the monitor enforces on the
  measured arm; plans keep it plus `planning_margin_mm`. Results report `executed.min_obstacle_gap_mm`.
- An arm that starts below a hard limit (after a halt) may move, but not closer; the monitor allows that and nothing
  else. Contacts are never relaxed.
- Physics test: four arms, three approach cycles beside a box, measured gap checked each time.

## 0.3.1 (2026-11-25)

- Fix: an arm that settled inside the planning margin (servo sag after a move planned right at the margin) could not
  move at all, not even away. A plan that starts inside the margin may now move as long as no waypoint gets closer
  than the start; the hard limit and contacts are never relaxed. Found while recording the demo; regression test
  fails without the fix.
- README demo: parts bin as an obstacle, straight-line descent and grasp.
- docs/why-below-the-model.md.

## 0.3.0 (2026-11-25)

- Obstacles: boxes, cylinders and spheres in the config are added to the model; plans keep `obstacle_clearance_mm`
  from them (`OBSTACLE_CLEARANCE`), touching one is a `COLLISION`, and touching one at rest is never allowed.
- Straight-line moves: `move_to(..., path="line")`, sampled every 5 mm, IK solved point to point; singularities and
  reach limits along the line are refused. Suggestions carry the `path` they were checked with.
- Grippers for SO-ARM100, SO-ARM101, low-cost arm, PiPER and YAM; CI opens and closes each under physics.
- Finger-to-finger contact is not a collision; fingers against the arm, table and obstacles still are.

## 0.2.0 (2026-11-25)

- Eleven more Menagerie presets: SO-ARM100, SO-ARM101, low-cost robot arm, PiPER, YAM, Z1, xArm7, Gen3, iiwa 14, FR3,
  UR10e. CI runs all thirteen end to end under physics.
- Contacts present in the rest pose are allowed up to their rest depth (+2 mm) and listed by `mjguard check`.
- Contacts between bodies moved by exactly the same commanded joints are not counted as motion collisions.
- Presets use a 20 mm hard clearance limit with a per-robot planning margin sized to the simulated tracking error.

## 0.1.0 (2026-11-25)

- Guard for any MuJoCo-modelled arm from a small TOML config: joint limits with margin, MuJoCo contacts, floor
  clearance of every moving collision geom (meshes included), optional workspace box, IK convergence check.
- Refusals with reason codes and a checked-safe suggestion that re-plans deterministically.
- Floor clearance has a hard limit and a planning margin (10 mm by default); the monitor halts on the measured arm,
  during the motion and after the servos settle. Pinned by a physics test on UR5e and Panda.
- Executor: one motion at a time, software stop, runtime monitor on measured joints, fault refusals, trace.
- Suggestions carry the `tool_down` they were checked with.
- Config loading never writes to stdout (the Menagerie downloader prints), so stdio stays a clean protocol.
- MuJoCo simulator transport: physics with the model's own position servos, or kinematic.
- MCP server (`mjguard serve`) and config checker (`mjguard check`).
- Presets: UR5e and Franka Panda from MuJoCo Menagerie, downloaded on first use.

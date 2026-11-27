# Changelog

## 0.4.2 (2026-11-27)

- `stop(latch=...)`: latching is chosen per call, so one UI can have a plain stop button and a latching e-stop.
  `--latch-stop` / `[ui] latch_stop` now only set the default. Found by the console's end-to-end test: with 0.4.1's
  server-wide latch, the plain stop button also latched and refused every later motion.

## 0.4.1 (2026-11-27)

Found while wiring a real UI (a web console) to 0.4.0:

- `discard(plan_id)`: a rejected preview can no longer run. Before, rejecting in the UI left the plan executable for
  60 s.
- `--latch-stop` / `[ui] latch_stop`: `stop()` latches; every motion, `execute` and `resume` is refused with
  `STOP_LATCHED` until `release_stop()`. Previews still work. `get_state` shows `stop_latched`. Off by default.
- A stop while idle now publishes a `stopped` event, so every client sees an e-stop pressed elsewhere.
- SSE ids are `BOOT:SEQ`. After a server restart `seq` starts at 1 again; an id from an earlier run now replays from
  the start instead of hiding every new event.
- zh-TW obstacle message: a space only between Chinese and a Latin name (「會太靠近 bin」, 「會太靠近零件盒」).
- Pose names: any script's letters are accepted (as before); the message and docs now say so.

## 0.4.0 (2026-11-26)

Human-robot interaction: a UI, phone app or voice front end can sit on top of mjguard. Everything below goes through
the same plan, check, follow and monitor path; nothing widens the envelope.

- Preview, then confirm: `preview=true` on every motion tool returns a checked plan (`plan_id`, duration, clearance,
  obstacle gap, start and end tool pose, a downsampled tool path) without moving. `execute(plan_id)` runs it once,
  within 60 s, if the measured arm is still at its start, after checking the waypoints again from the measured arm.
  New codes `PLAN_UNKNOWN`, `PLAN_USED`, `PLAN_EXPIRED`, `PLAN_STALE`. `--require-confirmation` / `[ui]
  require_confirmation` makes every motion tool preview only.
- Jog: `jog(dx, dy, dz)` (straight line, at most 50 mm) and `jog_joint(joint, delta)` (at most 10° or 10 mm).
- Named poses: `save_pose`, `go_to_pose` (gripper only with `include_gripper=true`), `list_poses`, `delete_pose`,
  `--poses FILE`. A stored pose is only a target, checked against the envelope in force when you go there.
  `POSE_UNKNOWN`.
- Pause and resume: `pause()` keeps the remaining waypoints, `resume()` re-checks from the measured arm and continues
  or refuses and stays paused (`PAUSED`, `NOTHING_TO_RESUME`); `stop()` discards a paused motion.
- Events: an in-process bus (`motion_started`, `progress`, `motion_finished`, `refused`, `stopped`, `paused`,
  `resumed`, `breach`, `pose_saved`, `pose_deleted`), `mjguard://events/latest`, and `GET /events` server-sent
  events over HTTP, with the MCP endpoint's Host/Origin checks.
- `message_human` on every refusal and a `human` sentence on suggestions, in `--lang en` or `zh-TW` (`[ui] lang`).
- Refusal details carry `limit_mm` for clearance refusals; suggestions carry `fraction`.
- Callback subscribers run on their own thread, so a slow or failing callback can never delay a stop or a breach
  halt (test: a callback that sleeps 1 s per event does not slow a stop). `events.flush()` waits for them.
- A new motion that replaces a paused one says so: `discarded_paused_motion: true` in its result.
- Confirmation and pause are protocol steps, not authorization: any client of the server can call `execute` or
  `resume`. Put authentication in front of the server if people and agents share it.

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

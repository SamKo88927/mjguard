# Why the safety check has to live below the model

A language model that refuses a harmful request in chat will often carry out the same request when it holds a
robot's tools. In "Refuses in Chat, Complies with Arms" (Sun et al., 2026, [arXiv:2610.14043](https://arxiv.org/abs/2610.14043)),
refusal fell from 73% of text requests to 18-25% once the model controlled the arm. The authors' conclusion:
safety alignment learned in text does not carry over to action.

Whatever the cause, the engineering answer is the same one we use everywhere else a component can be wrong: do not
ask the component to police itself. Put the check where the component cannot reach it, and make it deterministic.

mjguard is that check for robot arms driven over MCP. This note explains the design, and then lists the bugs that
its own tests and demos found, because they are a better argument than the design.

## The shape of the check

```
model ──► MCP tool ──► plan (IK, joint-space or straight-line waypoints)
                         │
                         ▼
               check every waypoint in the robot's own MuJoCo model
               (joint limits, contacts, floor clearance, obstacle clearance, workspace)
                         │                         │
                         ▼                         ▼
               transport.follow             refusal: code + message +
                         │                  farthest checked-safe point
                         ▼
               monitor what actually ran ── halt on breach
```

Four decisions carry most of the weight.

1. **The envelope is fixed at startup.** It is a frozen value built from a config file. No MCP tool can widen it.
   A model that talks its way past the instructions still meets the same numbers.
2. **The model is the robot's own model.** mjguard does not approximate the arm with spheres; it asks MuJoCo for
   contacts and for the distance from every moving collision mesh to the floor and to each obstacle.
3. **Every waypoint, not just the target.** A safe start and a safe goal can have an unsafe straight line between
   them. The check runs at a fixed joint spacing (3° by default) and, for straight-line moves, every 5 mm.
4. **Measure, then trust.** The plan is a prediction. While the arm moves, and again after the servos settle, the
   monitor checks the measured joints against the hard limit.

## Refusals are for the model

A bare "no" teaches a model nothing; it retries the same command or gives up. Every refusal carries a reason code
and, when one exists, the farthest point along the straight line toward the target that passed every check. The
point is rounded the way a model will send it back and re-planned before it is returned, so following it works.

That last sentence took two bugs to make true. Both are below.

## What the tests found

Each of these was found by a test or a demo recording, not by reading the code. Each has a regression test that
fails without the fix.

**A suggestion that was refused when followed.** IK tried random seeds from a shared generator, so planning the
same target twice could choose two different joint solutions, one of whose paths collided. A property test that
follows every suggestion caught it. Fix: the same request always gives the same answer (seeds are derived per call).

**Servo sag ate the safety margin.** Under physics with the models' own position servos, the UR5e and Panda settle
about 6 mm below their plan. A move planned to exactly 30 mm above the table ended at 26 mm and reported success.
Fix: plans keep a margin above a hard limit, and the monitor enforces the hard limit on the measured arm, including
after settling. A physics test on both arms pins it.

**The arm could get stuck.** The same sag, next to an obstacle: after following a suggestion, the arm settled 6 mm
from a bin with a 10 mm planning gap. Every later plan failed at its first waypoint, including moving away. Found
while recording the README GIF. Fix: a plan that starts inside the margin may move as long as no waypoint gets closer
than where it started.

**The obstacle gap was only a plan-time promise.** Checking the previous fix on heavier arms showed it: beside a
box, the Z1 settled 2.7 mm from it with `ok: true` (10 mm configured), and the iiwa touched it. The floor had a hard
limit enforced on the measured arm; obstacles did not. Fix: the configured obstacle gap is now a hard limit on the
measured arm too, and plans keep it plus each robot's planning margin. A physics test on four arms runs three
approach cycles each and checks the measured gap every time. An arm that starts below a hard limit (after a halt)
may only move away from it; contacts are never relaxed.

**Designed-in contacts looked like collisions.** Bringing up thirteen arms from MuJoCo Menagerie, three failed their
own rest pose: Kinova Gen3's base and shoulder hulls overlap by 12 mm, YAM's closed fingers touch, and a UR10e
shoulder sits on the floor. Fix: contacts present at rest are part of the model, allowed up to their rest depth,
and listed by `mjguard check` so they stay visible. Contacts with a configured obstacle are never allowed this way.

**A gripper that could not close.** PiPER's fingers meet at fully closed, which the guard counted as a collision.
Fix: bodies whose relative pose depends only on gripper joints are not checked against each other. Fingers against
the arm, the table and obstacles still are.

**The first start corrupted the protocol.** On a fresh machine, the Menagerie downloader printed a line to stdout,
which in stdio mode is the MCP protocol stream. Fix: config loading runs with stdout redirected to stderr; a test
injects a chatty loader and checks stdout stays clean.

## Limits

- The check is as good as the model. Anything not in the MJCF scene or the obstacle list is invisible to it.
- Waypoints are checked at a fixed spacing, not continuously.
- The envelope is geometric. Speed, torque and payload are not checked.
- `stop()` is a software stop. On real hardware, a physical e-stop is the only real emergency stop.

None of this replaces a hardware e-stop or a risk assessment. It removes one failure mode: a model that was talked,
or talked itself, into moving somewhere it should not.

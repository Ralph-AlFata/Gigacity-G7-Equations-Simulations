# T011 — Reference format + recorded-path reference

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T007

## Goal
Define one file format for reference trajectories (any source), and produce the recorded-path
reference from the scene: at each tick, the path the car actually drove afterwards.

## Inputs → Outputs
- Inputs: `data/scene/<clip_id>/<ego_source>/ego.parquet`.
- Outputs: `data/reference/<clip_id>/<ego_source>/recorded/reference.parquet` and `meta.json`.

## Specification
**Reference format.** The same format is used for every source. One row per (tick, sample, step).

| Column | Unit | Meaning |
|---|---|---|
| `tick` | int | scene tick the plan was made at (t) |
| `sample` | int | sample index (0 for single-trajectory sources) |
| `step` | int | 0, 1, 2, … |
| `tau_s` | s | time after t (step × 0.1) |
| `x, y, z` | m | rear-axle position, world frame (same frame as the scene) |
| `yaw` | rad | heading, world frame |
| `speed` | m/s | reference speed at this step |

`meta.json` holds: source name, step size, number of steps, frame, ego source of the scene, and the
input SHA-256s.

**Recorded path.**
- At tick t, the reference is the ego's own future poses from the scene: steps τ = 0, 0.1, … up to
  the end of the clip.
- Speed comes from the scene's derived speed.
- One sample.
- Every tick of the clip has a reference. It gets shorter near the end, and its `T_rem` is shorter
  accordingly.

## Assumptions
- 0.1 s steps (same as the scene grid and Alpamayo).

## Subtasks
- [ ] Reference reader/writer with validation (monotonic steps, no NaN, frame recorded)
- [ ] Recorded-path builder, CLI `reference recorded --clips …`

## Acceptance criteria
- [ ] Test: the recorded reference at tick t, step k equals the scene ego pose at tick t+k.
- [ ] Test: the reader rejects missing columns, NaNs and non-monotonic steps.
- [ ] Rerun check (after T017): the recorded reference at any tick lies exactly on the ego's future
      track.

## Outcome
<Filled in when done.>

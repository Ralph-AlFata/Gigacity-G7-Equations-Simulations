# T008 — Lidar observability and `in_scene`

**Type:** Task
**Epic:** E02
**Status:** todo
**Depends on:** T005, T007

## Goal
Fill `lidar_points`, `observed` and `in_scene` in `objects.parquet`, so an object is in the scene
only when it exists and the lidar actually hits it.

## Inputs → Outputs
- Inputs:
  - the lidar spins of the clip (format from T005);
  - `objects.parquet` and `ego.parquet` from T007;
  - lidar extrinsics if T005 says they are needed.
- Outputs: the same `objects.parquet` with the three columns filled; rule parameters in `meta.json`.

## Specification
- For each tick, use the lidar spin whose timestamp is nearest to the tick. Record the time
  difference.
- Put the spin's points in the same frame as the object boxes (world or ego frame at the tick,
  as T005 decides). Correct for ego motion between the spin time and the tick.
- **Point count.** For every existing object at the tick, count points inside its 3D box enlarged
  by `lidar_box_margin_m` on each side. Exclude points within `ground_clearance_m` of the box
  bottom, so road points under the box do not count.
- `observed = lidar_points ≥ lidar_min_points`, and `in_scene = observed` (the row already exists).
- **Defaults**, all in `meta.json`. They are starting values and will be tuned by looking at the
  distribution and at Rerun:
  - `lidar_min_points` = 5;
  - `lidar_box_margin_m` = 0.2;
  - `ground_clearance_m` = 0.2.
- The points are not stored in the scene. Only the counts are stored; Rerun reads the raw spins.

## Assumptions
- One spin per tick is enough. A full 360° spin is about 0.1 s, so parts of a spin are up to
  ~0.05 s from the tick. Recorded per tick so it can be checked.
- No occlusion modeling beyond what the lidar itself provides (hidden objects get no points).

## Subtasks
- [ ] Spin reader (from T005)
- [ ] Points-in-box count (vectorized)
- [ ] Fill the columns, record parameters
- [ ] Report per clip: share of existing rows observed, by class group and by distance band

## Acceptance criteria
- [ ] Test: a synthetic box with known points inside and outside, including the ground band,
      gives the exact count.
- [ ] Test: a rotated box counts correctly (points near its corners).
- [ ] Check: objects near the ego (< 20 m) and not behind other objects are observed at almost
      every tick.
- [ ] Rerun check (after T010): in-scene boxes contain visible lidar points; boxes that switch to
      not-in-scene are visibly hidden or far away at that moment.

## Outcome
<Filled in when done.>

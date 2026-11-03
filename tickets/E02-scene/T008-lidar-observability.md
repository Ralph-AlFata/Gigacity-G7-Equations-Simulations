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
  - the lidar extrinsic (`sensor_extrinsics.offline`, row `lidar_top_360fov`): points are in the
    sensor frame (T005).
- Outputs: the same `objects.parquet` with the three columns filled; rule parameters in `meta.json`.

## Specification
- **Point placement (T005).** Points are raw: each is in the lidar sensor frame at its own
  `timestamp` (absolute µs, clip clock). Place every point with
  `p_world = Pose_offline(point.timestamp) · (R_ext · p + t_ext)`, interpolating the pose per point.
  Then express it in the ego frame at the tick. One pose per spin misplaces points by up to v × 100 ms.
- **Which spin, per object.** The sweep crosses an object at one moment per spin, and the
  object's label is stamped with that moment (T005: within 5 ms for 99% of sightings). For each
  (tick, object), use the spin whose sweep crossed the object nearest to the tick. Record that time
  difference per row (at most ~50 ms).
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
- One sweep crossing per (tick, object) is enough. Crossings are 0.1 s apart, so the nearest one is at
  most ~0.05 s from the tick. Recorded per row so it can be checked.
- The lidar has a blind zone: no point is closer than ~3.2 m (horizontal) to the sensor (T005).
  An object entirely inside it cannot be observed.
- No occlusion modeling beyond what the lidar itself provides (hidden objects get no points).

## Subtasks
- [ ] Spin reader (format in T005: DracoPy, attributes `intensity`, `timestamp`, `model_element`)
- [ ] Points-in-box count (vectorized)
- [ ] Fill the columns, record parameters
- [ ] Report per clip: share of existing rows observed, by class group and by distance band

## Acceptance criteria
- [ ] Test: a synthetic box with known points inside and outside, including the ground band,
      gives the exact count.
- [ ] Test: a rotated box counts correctly (points near its corners).
- [ ] Check: objects near the ego (< 20 m, but outside the ~3.2 m blind zone) and not behind
      other objects are observed at almost every tick.
- [ ] Rerun check (after T010): in-scene boxes contain visible lidar points; boxes that switch to
      not-in-scene are visibly hidden or far away at that moment.

## Outcome
<Filled in when done.>

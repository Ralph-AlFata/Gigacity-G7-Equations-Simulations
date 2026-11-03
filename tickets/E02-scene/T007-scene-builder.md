# T007 — Scene builder: grid, ego, objects, frames, derived motion

**Type:** Task
**Epic:** E02
**Status:** todo
**Depends on:** T002, T006

## Goal
From a clip's raw files and an ego source, write `meta.json`, `ego.parquet` and
`objects.parquet` (schema in the E02 README). Every column is filled here except
`lidar_points`, `observed` and `in_scene`, which come from T008.

## Inputs → Outputs
- Inputs, from `data/raw/<clip_id>/`:
  - `egomotion.parquet` or `egomotion.offline.parquet`;
  - `obstacle.offline.parquet`;
  - `vehicle_dimensions.parquet`.
- Outputs: `data/scene/<clip_id>/<ego_source>/` (schema in E02).

## Specification
Steps run in this order, because the frame transform must happen at each sighting's own time,
before anything is resampled.

1. **Clip time.** `t0_us` = timestamp of the first ego pose. Grid: `t_s = 0.0, 0.1, …` up to the
   last time covered by ego poses. Ticks are never extrapolated past the data.

2. **Ego pose at any time.** Interpolate position linearly and orientation by slerp between the
   two ego poses around the time. The pose is already the rear-axle point: the rig origin is the
   rear-axle midpoint on the ground (T006), so there is no offset.

3. **Clean sightings, per track.**
   - Sort each track's rows by time.
   - Merge rows of the same track less than 20 ms apart, keeping the later row. Before merging,
     assert that both rows have the same class and size and positions within 0.2 m. If not, stop
     and report the pair.

4. **Sightings to world frame.** Each sighting's box is in the rig frame at its own
   `timestamp_us`. Compose it with the ego pose at that exact time (step 2) to get a world-frame
   pose. Size stays as is.

5. **Existence.** A track exists at tick t when:
   - the first sighting is at or before t, and t is at or before the last sighting; and
   - the two sightings around t are at most `gap_bridge_s` = 0.5 s apart.

   No timeouts, no distance cut.

6. **Object state at a tick.** Interpolate between the two sightings around t:
   - position linearly;
   - orientation by slerp;
   - size linearly.

   A tick exactly on a sighting takes it as is. `last_sighting_age_s` = t minus the latest sighting
   at or before t. `gap_s` = the spacing of the two sightings if it is above 0.15 s, else 0.

7. **Derived motion.** Computed from the resampled 10 Hz states, the same way for ego and objects:
   - velocity is the central difference over ±0.1 s (one-sided at the ends of a lifetime);
   - acceleration is the central difference of velocity;
   - yaw rate is the difference of unwrapped yaw.

   No smoothing in this version; T009 decides whether it is needed. A track that exists at a single
   tick gets NaN motion.

8. **Ego-frame columns.** `rel_*` = object pose expressed in the ego frame at the same tick (origin
   at the rear axle). `distance` = planar norm of (rel_x, rel_y).

9. **Classes.** `class_group` from the mapping in `meta.json`. An unknown class becomes `unknown`
   and is logged, never dropped.

10. **`has_history`** = t_s ≥ `history_s` (1.6 s), on the ego table.

## Assumptions
- Both ego sources cover the whole clip. Checked in T004: offline covers 0–20 s and online's dense
  part covers −0.19–20.2 s (beyond that, online is a sparse tail to ignore).
- Label `size_x, size_y, size_z` are length, width, height along the box's own x, y, z axes.
- Linear interpolation over 0.1–0.5 s is accurate enough. The Rerun check below will show if it
  is not.

## Subtasks
- [ ] Pose utilities (interpolate, compose, invert, yaw) with tests
- [ ] Steps 1–10
- [ ] CLI: `scene build --clips … --ego-source online|offline|both`
- [ ] Build both clips × both sources

## Acceptance criteria
- [ ] Test: a synthetic object moving at constant velocity, sighted on an offset clock, is
      reconstructed exactly at the ticks and its derived velocity is exact.
- [ ] Test: the frame round trip (world → ego frame → world) gives back the input.
- [ ] Test: a skip of 0.4 s is bridged; a skip of 0.6 s makes the object not exist in between.
- [ ] Test: duplicate rows that fail the consistency checks stop the build.
- [ ] Test: no object row before its first or after its last sighting.
- [ ] Check: parked objects stay still in the world frame (spread < 0.5 m) with offline poses
      (T004 floor: median RMS 0.16–0.21 m).
- [ ] Rerun check (after T010): boxes move smoothly; parked cars stay still; ego footprint sits
      at the right place relative to the lane.

## Outcome
<Filled in when done.>

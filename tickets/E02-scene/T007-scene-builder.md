# T007 — Scene builder: grid, ego, objects, frames, derived motion

**Type:** Task
**Epic:** E02
**Status:** done
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

1. **Clip time.** `t0_us` = clip time 0 on the dataset clock (or the first 0.1 s multiple after the
   first ego pose, if that is later). Grid: `t_s = 0.0, 0.1, …` up to the last time covered by ego
   poses. Ticks are never extrapolated past the data. *(Changed in T007 from "first ego pose": online
   starts at −0.19 s, which would shift its grid 0.0047 s off the offline one. Clip time 0 is where
   both anchor frames are defined, so online and offline scenes now share their ticks.)* "Covered"
   means the dense part: online's sparse tail after ~20.2 s is cut.

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

7. **Derived motion.** Computed from the resampled 10 Hz states, the same way for ego and objects,
   per run of ticks. It is **causal**: only the tick itself and earlier ticks of the run are used
   (changed by T009, 2026-11-04; it was central differences).
   - velocity is the backward difference over 0.1 s: (x(t) − x(t−0.1)) / 0.1;
   - yaw rate is the backward difference of unwrapped yaw over 0.1 s;
   - acceleration is the change of velocity over the last `accel_span_s` (0.4 s), over the run's
     whole velocity history while that is shorter.

   NaN where nothing earlier exists: velocity and yaw rate at a run's first tick, acceleration at its
   first two. No smoothing: T009 compared smoothers, and none does better without looking ahead.

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
- [x] Pose utilities (interpolate, compose, invert, yaw) with tests
- [x] Steps 1–10
- [x] CLI: `scene build --clips … --ego-source online|offline|both`
- [x] Build both clips × both sources

## Acceptance criteria
- [x] Test: a synthetic object moving at constant velocity, sighted on an offset clock, is
      reconstructed exactly at the ticks and its derived velocity is exact.
- [x] Test: the frame round trip (world → ego frame → world) gives back the input.
- [x] Test: a skip of 0.4 s is bridged; a skip of 0.6 s makes the object not exist in between.
- [x] Test: duplicate rows that fail the consistency checks stop the build.
- [x] Test: no object row before its first or after its last sighting.
- [x] Check: parked objects stay still in the world frame (spread < 0.5 m) with offline poses
      (T004 floor: median RMS 0.16–0.21 m).
- [ ] Rerun check (after T010): boxes move smoothly; parked cars stay still; ego footprint sits
      at the right place relative to the lane.

## Outcome
Code: `horizon_pipeline/scene/` (`frames.py` poses and transforms, `build.py` steps 1–10,
`checks.py`, `cli.py`); CLI `horizon-scene build`. Tests: `tests/test_scene.py` (synthetic,
exact) and `tests/test_scene_real.py` (both clips × both sources).

**Built** (`uv run horizon-scene build --clips-file clips_selected_train.txt --ego-source both`, ~4 s):
| | b789fab8 | b4d979af |
|---|---|---|
| ticks (offline / online) | 201 / 202 (0–20.0 / 20.1 s) | 201 / 202 |
| sightings, duplicates merged | 7 140, 236 | 53 109, 129 |
| tracks, object rows | 110, 7 048 | 652 of 653, 52 447 |
| sightings outside the ego poses, unknown classes | 0, none | 0, none |
| parked tracks: position RMS median (offline / online) | **0.106 / 0.165 m** | **0.174 / 0.243 m** |
| parked tracks: 95th percentile, max (offline) | 0.29, 0.40 m | 0.38, 1.01 m |

One b4d979af track exists at no tick: its sightings fall between two ticks.

**Verification.**
- *Transforms against an independent implementation:* `tests/test_scene.py` checks `to_local`, `yaw_of` and every
  ego-frame column (`rel_*`, `rel_yaw`, `distance`) against rotation matrices written out by hand from the
  quaternion formula, with no scipy.
- *Synthetic clip:* the ego translates while yawing at 0.2 rad/s, so linear and slerp interpolation are both
  exact for it. A constant-velocity object is sighted on an offset clock (37 ms + k × 100 ms), and the
  test writes its boxes the way the dataset does (in the rig frame at each sighting time). Positions,
  velocities, yaw and ego-frame columns come back exact to 1e-9.
- *Real data:* parked objects reproduce T004's floor (median 0.16–0.21 m from raw sightings). The online scenes
  are ~0.06 m worse, as T004 predicts. No object moves more than 3 m between ticks.
- *Parked selection:* T004's span-only filter (< 3 m) also admits slow walkers and creeping cars,
  so `parked_spread` adds a median speed under 0.3 m/s. The 5 b4d979af tracks still above 0.5 m are
  objects that wait and then move (e.g. track 302: still for most of the clip, then 3 m at the end). Their
  paths are smooth, with at most 0.1–0.3 m per tick. The test therefore asserts median < 0.25 m (offline) and
  95th percentile < 0.5 m, not a per-track maximum.

**Choices made while implementing** (all in the code and `meta.json`):
- `speed` is the 3D norm of the derived velocity. Ego `yaw` is unwrapped over the clip, as the E02 table
  says. Object `yaw` is wrapped to [−π, π), and is unwrapped per run of ticks only to compute `yaw_rate`.
- `rel_yaw` is the yaw of `R_egoᵀ · R_obj`, the heading in the ego frame, not a difference of world yaws.
- A track's class at a tick is the class of its latest sighting at or before it.
- Duplicate check: same class, sizes equal within 1 mm, rig-frame centers within 0.2 m. Real data: 365
  pairs, all consistent.
- `meta.json` records a `build_report` (counts above) and the SHA-256 of each raw input, computed from the
  files themselves.
- The Rerun check waits for T010.

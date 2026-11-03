# T008 — Lidar observability and `in_scene`

**Type:** Task
**Epic:** E02
**Status:** review
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
- **Point count.** For every existing object at the tick, count that spin's points inside its 3D box,
  placed at the crossing time (interpolated from the track's sightings, not the box at the tick: a
  moving object can be 50 ms = 0.75 m at 15 m/s from it), enlarged
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
- [x] Spin reader (format in T005: DracoPy, attributes `intensity`, `timestamp`, `model_element`)
- [x] Points-in-box count (vectorized)
- [x] Fill the columns, record parameters
- [x] Report per clip: share of existing rows observed, by class group and by distance band

## Acceptance criteria
- [x] Test: a synthetic box with known points inside and outside, including the ground band,
      gives the exact count.
- [x] Test: a rotated box counts correctly (points near its corners).
- [x] Check: objects near the ego (< 20 m, but outside the ~3.2 m blind zone) and not behind
      other objects are observed at almost every tick.
- [ ] Rerun check (after T010): in-scene boxes contain visible lidar points; boxes that switch to
      not-in-scene are visibly hidden or far away at that moment.

## Outcome
Code: `horizon_pipeline/scene/lidar.py` (spin reader, per-point placement, sweep timing, box count,
`observe`), called by `build_scene` (`lidar=True` by default; `horizon-scene build --no-lidar` skips
it). New columns `lidar_spin` and `lidar_dt_s` record the spin used and its crossing time minus the
tick (E02 schema updated). Tests: `tests/test_lidar.py` and the lidar part of `tests/test_scene_real.py`.

**How a row is decided:**
1. The object's azimuth from the sensor at the tick, and the clip's sweep profile (median point time
   per 1° azimuth bin, measured on one spin), give when each spin crosses it. Take the crossing nearest
   the tick.
2. Interpolate the object's box at that crossing time from its own sightings.
3. Place that spin's points with the ego pose at each point's time, in 0.1 ms buckets (≤ 0.5 mm from
   exact placement at 10 m/s). Count the points inside the box enlarged by 0.2 m, ignoring the 0.2 m above
   its bottom.

**Results** (offline; online is the same to 0.1%):
| | b789fab8 | b4d979af |
|---|---|---|
| object rows observed | 90.6% | 89.5% |
| 0–10 m (vehicle / VRU) | 100 / 100% | 100 / 100% |
| 10–20 m | 100 / 99.4% | 97.3 / 97.1% |
| 20–40 m | 96.7 / 97.7% | 92.7 / 92.0% |
| 40–80 m | 91.6 / 98.8% | 87.1 / 91.1% |
| 80 m + | 78.0 / 100% | 82.8 / — |
| points in the box when observed (median) | 56 | 64 |
| unobserved rows with 1–4 points / with 0 | 460 / 205 | 2 237 / 3 250 |
| \|lidar_dt_s\| median, max | 23 ms, 85 ms | 30 ms, 97 ms |

- *Near objects (4–20 m):* 99.9% / 97.7% observed. On b4d979af, the 126 near rows that aren't observed belong to
  17 tracks (parked cars and pedestrians in a crowded lot). Those tracks are observed 60–98% of their life
  with 40–500 points, and the misses come in short runs with mostly 0 points: occlusion. To confirm in Rerun (T010).
- *`lidar_dt_s` above 50 ms* happens only within 0.1 s of the first spin start or the last spin end (e.g.
  b4d979af's first spin starts at 99.5 ms). There, a tick has a spin on one side only. Asserted in the tests.
- *Threshold sensitivity:* many unobserved rows have 1–4 points, so `lidar_min_points` = 5 matters at
  long range. Leave it at 5 until the Rerun check.
- *Speed:* ~20 s per clip and ego source, mostly decoding and building the point tree. Per-point slerp
  was 10× slower, which is why poses are bucketed.

**Tests:** exact counts with margin and ground band; a rotated box at its corners; raw points coming back
to one static point only with per-point poses (one pose for the spin smears them by 1 m); the sweep
profile; nearest-crossing assignment. And end to end: a fake lidar sweeping like the real one, with
40 points on a *moving* car at its crossing times. Every row of that car counts exactly 40, a hidden
object counts 0, and every crossing is within 50 ms.

The Rerun check waits for T010.

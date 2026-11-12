# T021 — Causal lidar observability

**Type:** Task
**Epic:** E02
**Status:** done
**Depends on:** T008, T009

## Goal
Make `observed` / `in_scene` a causal sensor reading: at tick t, use only lidar data recorded by t.

## Inputs → Outputs
- Inputs: as in T008.
- Outputs: the same columns, now causal: `lidar_points`, `observed`, `in_scene`, `lidar_spin` and
  `lidar_dt_s`. The build report gets a new `rows_without_reading`.

## Specification
The rule decided in the T009 review (2026-11-04): building the environment may use the future;
calculations and sensor readings may not. `observed` is a sensor reading. T008 used the sweep
crossing *nearest* the tick, which can be up to ~50 ms after it.
- **Spin:** the latest spin whose sweep crossed the object at or before the tick. `lidar_dt_s` is
  then in [−0.1, 0] s.
- **Points:** only that spin's points with a timestamp ≤ the tick. When the sweep is still passing
  over the object, or the object sits on the seam between two spins, its later points don't
  count yet.
- **Box (environment, unchanged):** the object's true box at the crossing time, interpolated from
  its sightings.
- **Before the first crossing:** no reading. `lidar_points`, `lidar_spin` and `lidar_dt_s` are
  NA/NaN, and `observed = in_scene = false`.
- **Unchanged:** the sweep profile (when the sweep passes each azimuth) is still measured from one
  spin in the middle of the clip. It is a fixed property of the sensor, i.e. calibration known in
  advance, not a reading.

## Assumptions
- No processing latency: a point is usable at its own timestamp. A latency can be added later, in
  the phase that models perception and timing delays (E02 scope).

## Subtasks
- [x] `assign_spins`: latest crossing at or before the tick; −1 where none
- [x] `observe`: point cut at the tick, no-reading rows, report count
- [x] CLI summary, E02 schema and principles, T008 spec, GUIDE.md

## Acceptance criteria
- [x] Test: `assign_spins` picks the latest crossing ≤ t, not the nearest, and −1 before any crossing.
- [x] Test: removing every point recorded after 2.0 s changes no reading at ticks ≤ 2.0 s. There are
      12 objects at every bearing, so some crossings fall just after a tick. This fails on the T008
      code.
- [x] Test (end to end): a track that exists at tick 0, before the first crossing, has no reading and
      is not observed; every other row is as before.
- [x] Check, real data: `lidar_dt_s` ∈ [−0.1, 0]; rows without a reading only in the first spin;
      objects at 4–20 m still observed > 95% of the time.

## Outcome
Code: `horizon_pipeline/scene/lidar.py` (`assign_spins`, `observe`). Tests: `tests/test_lidar.py`
and `tests/test_scene_real.py`. 113 tests pass. All 5 clips were rebuilt (offline).

**Observed share of object rows, offline:**
| Clip | T008 (nearest crossing) | T021 (causal) | Rows without a reading |
|---|---|---|---|
| b789fab8 | 90.6% | 89.8% | 2 |
| b4d979af | 89.5% | 89.2% | 203 |
| b6cb8b12 | 96.3% | 95.5% | 43 |
| ca823746 | 98.2% | 97.6% | 8 |
| 23958e60 | 89.4% | 87.8% | 26 |

**Where the drop comes from:** almost all of it is the point cut. I rebuilt b789fab8 and 23958e60
without the cut to separate it out:
- **Rows affected:** 170 rows lose their reading. 144 of them are beyond 80 m, 22 at 40–80 m, 4 at
  20–40 m, and none within 20 m (4–20 m stays at 99.9% / 99.6%).
- **What they are:** objects whose points are partly recorded after the tick. Either the sweep is
  still passing over them (23958e60, median 10 ms before the tick), or they straddle the seam
  between two spins (b789fab8, median 85 ms). At long range they are already close to the 5-point
  threshold, so losing part of the points tips them under it.
- **Why the cut stays:** it is physically right. At tick t those points haven't been measured yet.

**Open:**
- **Rerun:** the viewer still draws the spin whose *middle* is nearest the tick. That is a display
  choice, not a calculation, so it is unchanged.
- **If the far-range partial reads matter for H:** the alternative is a rolling window, i.e. the
  points of the last 100 ms before the tick across two spins. It would need its own decision.

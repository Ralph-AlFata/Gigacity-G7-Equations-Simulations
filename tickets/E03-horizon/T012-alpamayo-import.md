# T012 — Alpamayo trajectory import

**Type:** Task
**Epic:** E03
**Status:** done
**Depends on:** T011

## Goal
Convert the Alpamayo plans of Ralph's snapshot dataset into the reference format of T011. These are
the real plans that H is computed for (T023). Generating the dataset and running Alpamayo are not
part of this ticket. It is built and tested on local data only; no GPU work.

## Inputs → Outputs
- Inputs: Alpamayo output files for a clip, one per snapshot tick it was run at. They come from a
  separate process run by Ralph that builds the snapshot dataset (e.g. every 0.3 s, T023). Ralph runs
  these properly:
  - 4 cameras;
  - online egomotion;
  - 1.6 s of history;
  - no future leak.
- Outputs: `data/reference/<clip_id>/online/alpamayo/reference.parquet` and `meta.json`. The ego
  source comes from `run.json`; it is `online` because Alpamayo's inputs come from online egomotion.

## Specification
- **Input contract** (agreed 2026-11-12; the snapshot process writes it):

  ```
  <dir>/<clip_id>/run.json
  <dir>/<clip_id>/t_<t0_us:09d>.npz        one per snapshot
  ```

  Each `.npz`, straight from Alpamayo 1.5 and NVIDIA's loader:

  | Key | Shape | Meaning |
  |---|---|---|
  | `clip_id` | str | the clip |
  | `t0_us` | int | input time on the clip's clock (= the scene's `timestamp_us`); must be ≥ 1.6 s |
  | `pred_xyz` | (1, 1, S, 64, 3) | positions, rig frame at t0, steps t0 + 0.1 … t0 + 6.4 s |
  | `pred_rot` | (1, 1, S, 64, 3, 3) | rotation matrices, same frame |

  `run.json` holds at least `model`, `ego_source`, `snapshot_interval_s`, `n_cameras`, `history_s`;
  anything else (seed, sampling settings, code version) is kept as is in `meta.json`.
- **Match to a tick.** Each output is matched to the scene tick with the same timestamp, within
  1 ms. An output that matches no tick is an error, never rounded silently.
- **Frame.** The loader writes `xyz_local = R_t0⁻¹ (xyz − xyz_t0)` and `rot_local = R_t0⁻¹ R`. The
  importer undoes it with the scene's ego pose at that tick: `xyz_world = R_t0 xyz_local + xyz_t0`,
  `R_world = R_t0 rot_local`. The scene's world frame differs from the egomotion frame only by a
  fixed rigid transform, so this is exact. The rig origin is the rear axle (T006) and Alpamayo is
  trained on rig egomotion, so the offset is zero.
- **Step 0.** Alpamayo outputs start at t0 + 0.1 s, with no step 0. The importer adds step 0 as the
  ego pose at t0, so every plan starts exactly at the ego.
- **Speed.** Steps 1–64: the distance from the previous step over 0.1 s. Step 0: the scene's speed
  at t0 (causal, known at t0).
- **Check value.** `meta.json` records `step1_gap_m`: the length of the first step minus 0.1 s at
  the ego's speed at t0 (median and max |·|). Near 0 means the plan starts where and as fast as the
  car is. This replaces the "step-0 offset" check, which is 0 by construction now.
- **Every sample is kept,** as its own `sample` index. Each gets its own H (T023).
- Ticks Alpamayo was not run at simply have no Alpamayo reference.
- `meta.json` also records the snapshot interval, the number of samples per snapshot and `run.json`.
- **Refused:** a snapshot with less than 1.6 s of history, for another clip, with another ego
  source than the scene, of the wrong shape, with non-finite values, whose file name does not
  match its `t0_us`, or two snapshots on the same tick.

## Assumptions
- Confirmed from NVIDIA's loader (handoff 2026-11-12): outputs are in the rig frame at t0, with
  steps starting at t0 + 0.1 s (no step 0).

## Subtasks
- [x] Write the input contract into the ticket and `docs/GUIDE.md` (for the run script)
- [x] Importer + CLI `horizon-reference alpamayo --clips … --from <dir>`
- [ ] Import the real runs: waits for the snapshot dataset (model weights still to download on the
      GPU machine).

## Acceptance criteria
- [x] Test: a synthetic trajectory in the ego frame, imported at a known ego pose, lands at the
      expected world coordinates.
- [x] Test: an output with a timestamp not on the grid is rejected with a clear message.
- [x] Test: a snapshot with 3 samples gives 3 plans at that tick, in sample order.
- [x] Test: Alpamayo-shaped files made from the online scene's own future, with the loader's
      transform, import back exactly onto the ego track.
- [ ] Check: `step1_gap_m` is small on all imported real ticks (with the real runs).
- [ ] Rerun check (after T017): Alpamayo trajectories start at the ego and follow the lane.

## Outcome
**Code:** `horizon_pipeline/reference/alpamayo.py`, CLI `horizon-reference alpamayo`.
`write_snapshot` and `loader_frame` produce contract files (the second from a scene's own future,
for tests).

**Tests:**
- `tests/test_alpamayo.py`: 14 synthetic tests (known pose, exact inverse, 3 samples, 1 ms
  matching, 6 kinds of bad input, run.json, CLI).
- `tests/test_alpamayo_real.py`: on b789fab8 and b4d979af, snapshots every 0.3 s from 1.6 s made
  from the online ego's future import back onto the ego track within 1e-6.
- Full suite: 174 passed.

**Round trip on all 5 clips** (online scenes, snapshots every 0.3 s from 1.6 s to the last tick
with 6.4 s of future):

| Clip | Snapshots | Rows | Largest error (x, y, z, yaw, speed) | step1_gap_m max \|·\| |
|---|---|---|---|---|
| 23958e60 | 41 | 2 665 | 2.9e-13 | 0.003 m |
| b4d979af | 41 | 2 665 | 1.1e-13 | 0.009 m |
| b6cb8b12 | 41 | 2 665 | 2.8e-13 | 0.009 m |
| b789fab8 | 41 | 2 665 | 2.3e-13 | 0.015 m |
| ca823746 | 441 | 28 665 | 2.5e-12 | 0.019 m |

ca823746 has 441 snapshots because its online scene spans 140 s (the T011 finding, still to be
filed as a bug).

**Left open:** the real import and its check wait for the snapshot dataset; the Rerun check waits
for T017.

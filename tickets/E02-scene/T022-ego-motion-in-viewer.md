# T022 — Ego speed and acceleration in the viewer

**Type:** Task
**Epic:** E02
**Status:** review
**Depends on:** T010

## Goal
See the ego's speed and acceleration inside the Rerun viewer, at the current tick and over the
whole clip. Objects already had speed labels (T010); the ego had none.

## Inputs → Outputs
- Inputs: `ego.parquet` (`speed`, `a_long`, `a_lat`, `yaw_rate`; causal since T009).
- Outputs: in the recording,
  - a per-tick label on `world/ego/box`;
  - time series `plots/<ego>/<quantity>`;
  - a layout with three plots under the 3D view.

## Specification
- **Label:** `ego <speed> m/s, a_long <±a> m/s²` at every tick, always shown. `?` where the value is
  NaN, i.e. the first tick(s), because the motion is causal. With `--compare` the other source's
  box reads `ego <source> …`.
- **Plots:** `speed` (m/s), `a_long` and `a_lat` (m/s²), and `yaw_rate` (converted to °/s), under
  `plots/ego/`. NaN ticks are left out. With `--compare`, the other source goes under
  `plots/ego_<source>/` in red, overlaid on the same plots.
- **Layout:** the 3D scene on top and three time-series views below it (speed, acceleration, yaw
  rate), next to the cameras as before.
- **Cache:** `RECORDING_VERSION` is part of the recording file's hash, so recordings built before
  this change are rebuilt instead of reused.

## Assumptions
- One value per 0.1 s tick. Between ticks the viewer shows the latest tick at or before the
  playhead.

## Subtasks
- [x] `ego_label`, per-tick label, `log_ego_plots`
- [x] Blueprint with the plots; `RECORDING_VERSION` in `recording_path`
- [x] GUIDE.md (panels, labels, entity table, reuse)

## Acceptance criteria
- [x] Test: the label text, including `?` for a NaN value.
- [x] Test: the recording holds `plots/ego/*` and `plots/ego_online/*` (with `--compare`), with
      exactly one row per tick that has a value (speed from tick 1, accelerations from tick 2), and
      one label per tick on the ego box.
- [x] Test: changing `RECORDING_VERSION` changes the recording path.
- [x] Check, real data: in the b6cb8b12 recording at 14.3 s, the speed plot holds 7.099 m/s and
      the label reads `ego 7.1 m/s, a_long +0.7 m/s²`, the same as `ego.parquet` at tick 143.
- [ ] Rerun check (by eye): the plots sit under the 3D view and the playhead follows the timeline.

## Outcome
Code: `horizon_pipeline/scene/view.py` (`ego_label`, `log_ego`, `log_ego_plots`, `blueprint`,
`recording_path`). Tests: `tests/test_view.py`.

# T014 — Horizon runner: one H per plan

**Type:** Task
**Epic:** E03
**Status:** done
**Depends on:** T008, T012, T013

## Goal
For every plan of a reference source, compute H once, at the tick the plan is received, from what is
known at that tick (T023). One plan is one sample of the reference at one tick.

## Inputs → Outputs
- Inputs:
  - scene: `ego.parquet`, `objects.parquet` and `meta.json` (ego dimensions);
  - one reference source (`recorded` or `alpamayo`);
  - `configs/horizon.yaml`.
- Outputs: `results/<run_name>/<clip_id>/<ego_source>/<reference_source>/horizons.parquet`, plus
  `config_used.yaml` and `meta.json` (input SHA-256s, code version, date).

## Specification
- **Plans:** every (tick, sample) present in the reference. Ticks without a plan (Alpamayo not run
  there) produce no row; how many there are is recorded in `meta.json`.
- **What the solver sees for a plan received at tick t:** only rows of `objects.parquet` at tick t
  with `in_scene = true`, i.e. their position, velocity and size at t. Nothing from later ticks.
- **Once per plan:** each plan is solved once. No row is ever updated from later ticks.
- **Every sample is solved separately.**
- **One row per (tick, sample):**
  - `tick`, `t_s`, `sample`, `reference_source`;
  - every solver output (`H_s`, `capped`, `limiting_track_id`, `T_cap_s`, …);
  - `n_in_scene`;
  - `ego_speed_mps` at t, for breaking results down by speed.
- Reference sources are run separately and written to separate folders. They are never combined.
- **Objects with no velocity yet** (in the scene at t, but at their track's first tick, T009): set by
  `objects.no_velocity` in the config. Default `exclude`: left out of H, but counted
  (`n_no_velocity`) and listed (`no_velocity_track_ids`), so T015 can label contacts they cause as
  "no velocity yet". The other choice is `static` (velocity 0).
- **The solver is picked by `solver.method`** in the config (`baseline` = the PDF's equations), from a
  registry in the runner, so a new method can be swapped in without touching the runner.

## Assumptions
- None beyond the solver's.

## Subtasks
- [x] Runner + CLI `horizon-run --clips … --ego-source … --reference recorded|alpamayo --name …`
- [x] Results writer

## Acceptance criteria
- [x] Test (causality): changing an object at tick t+1 does not change H at tick t. Also on the
      real clips: deleting every object row after t leaves H at t unchanged.
- [x] Test: an object with `in_scene = false` at t has no effect on H at t.
- [x] Test: a reference with plans at ticks 16, 19, 22 only gives exactly those rows, and two
      different samples at one tick give two rows with their own H.
- [x] Check: runs for the recorded stand-in on all built clips and both ego sources.
- [ ] Rerun check: see T017.

## Outcome
**Code:** `horizon_pipeline/horizon/run.py` (runner, results writer), `horizon_pipeline/horizon/cli.py`
(`horizon-run`). Two config settings added: `solver.method` and `objects.no_velocity`.

**Tests:**
- `tests/test_run.py`: 11 synthetic tests (causality, `in_scene`, plan ticks, samples, both
  `no_velocity` settings, the method registry, output columns, a plan at a tick the scene lacks, a
  capped row's missing limiting track, the CLI).
- `tests/test_run_real.py`: on b789fab8 and b4d979af, one row per plan with consistent values, and
  H at ticks 20, 80, 150 unchanged when every later object row is deleted.
- Full suite: 193 passed.
- Found and fixed while testing: a capped plan's missing `limiting_track_id` was written as the
  string `"nan"`.

**Real data** (`horizon-run --reference recorded --ego-source both --name t014-baseline`, all 5 clips,
under 1 s per clip and source):

| Clip | Source | Plans | H = 0 | Median H | Capped | Plans with a no-velocity object |
|---|---|---|---|---|---|---|
| 23958e60 | offline | 201 | 92% | 0.00 s | 1% | 14% |
| b4d979af | offline | 201 | 59% | 0.00 s | 9% | 81% |
| b6cb8b12 | offline | 201 | 62% | 0.00 s | 1% | 47% |
| b789fab8 | offline | 201 | 42% | 0.29 s | 9% | 28% |
| ca823746 | offline | 201 | 4% | 2.14 s | 6% | 13% |

Online gives nearly the same numbers on the first four clips. Pooled over the 5 offline runs
(1 005 plans), H = 0 at 52% and H < 2.0 s (the inference time) at 87%.

- As expected: the baseline disks need ~6.5 m to the next-lane car at 3.5 m, so H is 0 in dense
  traffic. This is the baseline to improve from (later epic), not a bug.
- 23958e60's 92% is partly the known quirk: the ego is labelled as an object (track 41) at ticks
  1–9, which gives H = 0 there.
- ca823746 online has 1 402 plans and is 86% capped: its online scene runs 140 s while objects stop at
  20 s, so most of its plans see no objects. This comes from the open egomotion-span bug, not the
  runner. Use the offline run until that is fixed.

**Rerun check:** see T017.

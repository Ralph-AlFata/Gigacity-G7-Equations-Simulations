# T014 — Horizon runner: one H per plan

**Type:** Task
**Epic:** E03
**Status:** todo
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

## Assumptions
- None beyond the solver's.

## Subtasks
- [ ] Runner + CLI `horizon-run --clips … --ego-source … --reference recorded|alpamayo --name …`
- [ ] Results writer

## Acceptance criteria
- [ ] Test (causality): changing an object at tick t+1 does not change H at tick t.
- [ ] Test: an object with `in_scene = false` at t has no effect on H at t.
- [ ] Test: a reference with plans at ticks 16, 19, 22 only gives exactly those rows, and two
      different samples at one tick give two rows with their own H.
- [ ] Check: runs for the recorded stand-in on all built clips and both ego sources.
- [ ] Rerun check: see T017.

## Outcome
<Filled in when done.>

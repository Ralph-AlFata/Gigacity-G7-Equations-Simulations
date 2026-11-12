# T014 — Live replay runner

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T008, T012, T013

## Goal
Play a scene tick by tick. At every tick where a plan is received (a tick that has a reference),
give the solver what is known at that moment, and store H. As decided in T023, H is computed once
per plan, on receipt, and never updated while the plan is followed.

## Inputs → Outputs
- Inputs:
  - scene: `ego.parquet` and `objects.parquet`;
  - one reference source;
  - `configs/horizon.yaml`.
- Outputs: `results/<run_name>/<clip_id>/<ego_source>/<reference_source>/horizons.parquet`, plus
  `config_used.yaml` and `meta.json` (input SHA-256s, code version, date).

## Specification
- Ticks are processed in time order.
- **What the solver sees at tick t:** only rows of `objects.parquet` at tick t with
  `in_scene = true`, i.e. their position, velocity and size at t. Nothing from later ticks.
- Every sample of the reference at tick t is solved separately.
- **One row per (tick, sample):**
  - `tick`, `t_s`, `sample`;
  - every solver output;
  - `n_in_scene`;
  - `reference_source`.
- Reference sources are run separately and written to separate folders. They are never combined.
- Ticks with no reference (e.g. Alpamayo not run there) produce no row; they are counted in
  `meta.json`.

## Assumptions
- None beyond the solver's.

## Subtasks
- [ ] Runner + CLI `horizon run --clips … --ego-source … --reference recorded|alpamayo --name …`
- [ ] Results writer

## Acceptance criteria
- [ ] Test: changing an object at tick t+1 does not change H at tick t (causality).
- [ ] Test: an object with `in_scene = false` at t has no effect on H at t.
- [ ] Check: runs for both clips, both ego sources, recorded reference.
- [ ] Rerun check: see T017.

## Outcome
<Filled in when done.>

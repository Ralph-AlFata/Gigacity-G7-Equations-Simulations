# T011 — Reference format + recorded-path reference

**Type:** Task
**Epic:** E03
**Status:** review
**Depends on:** T007

## Goal
Define one file format for reference trajectories (any source), and produce the recorded-path
reference from the scene: at each tick, the path the car actually drove afterwards.

## Inputs → Outputs
- Inputs: `data/scene/<clip_id>/<ego_source>/ego.parquet`.
- Outputs: `data/reference/<clip_id>/<ego_source>/recorded/reference.parquet` and `meta.json`.

## Specification
**Reference format.** The same format is used for every source. One row per (tick, sample, step).

| Column | Unit | Meaning |
|---|---|---|
| `tick` | int | scene tick the plan was made at (t) |
| `sample` | int | sample index (0 for single-trajectory sources) |
| `step` | int | 0, 1, 2, … |
| `tau_s` | s | time after t (step × 0.1) |
| `x, y, z` | m | rear-axle position, world frame (same frame as the scene) |
| `yaw` | rad | heading, world frame |
| `speed` | m/s | reference speed at this step |

`meta.json` holds: source name, step size, number of steps, frame, ego source of the scene, and the
input SHA-256s.

**Recorded path.**
- At tick t, the reference is the ego's own future poses from the scene: steps τ = 0, 0.1, … up to
  the end of the clip.
- Speed comes from the scene's derived speed. The scene has none at tick 0 (causal differences, T009),
  so that one value is tick 1's speed: the distance from tick 0 to tick 1 over 0.1 s. This is allowed
  because the recorded path is hindsight, so it is environment, not a sensor reading.
- One sample.
- Every tick of the clip has a reference. It gets shorter near the end, and its `T_rem` is shorter
  accordingly.

## Assumptions
- 0.1 s steps (same as the scene grid and Alpamayo).

## Subtasks
- [x] Reference reader/writer with validation (monotonic steps, no NaN, frame recorded)
- [x] Recorded-path builder, CLI `horizon-reference recorded --clips …`

## Acceptance criteria
- [x] Test: the recorded reference at tick t, step k equals the scene ego pose at tick t+k.
- [x] Test: the reader rejects missing columns, NaNs and non-monotonic steps.
- [ ] Rerun check (after T017): the recorded reference at any tick lies exactly on the ego's future
      track.

## Outcome
**Code:** `horizon_pipeline/reference/` (`format.py`, `recorded.py`, `cli.py`), CLI `horizon-reference`.

**Format details settled here:**
- `meta.json` keys: `source`, `clip_id`, `ego_source`, `frame`, `step_s`, `n_steps` (`{min, max}`,
  because the recorded plans get shorter), `n_ticks`, `n_samples`, `speed_filled_ticks`, and
  `inputs` (SHA-256 of the scene's `ego.parquet`, and the code version).
- The reader also refuses non-integer `tick`/`sample`/`step`, a (tick, sample) that appears twice,
  and `tau_s` ≠ `step × step_s`. Each of these has its own test.

**Tests:**
- `tests/test_reference.py`: 14 synthetic tests.
- `tests/test_reference_real.py`: on b789fab8 and b4d979af, both ego sources, every step of every
  plan equals the ego row at t + k, exactly.
- Full suite: 133 passed.

**Real data** (`horizon-reference recorded --clips <all 5> --ego-source both`):

| Clip | Offline plans / rows | Online plans / rows | Max speed (offline) |
|---|---|---|---|
| 23958e60 | 201 / 20 301 | 202 / 20 503 | 8.6 m/s |
| b4d979af | 201 / 20 301 | 202 / 20 503 | 5.1 m/s |
| b6cb8b12 | 201 / 20 301 | 202 / 20 503 | 11.7 m/s |
| b789fab8 | 201 / 20 301 | 202 / 20 503 | 9.2 m/s |
| ca823746 | 201 / 20 301 | 1 402 / 983 503 | 34.8 m/s |

Only tick 0's speed is filled, in every case.

**Found, not fixed here:** ca823746's online `egomotion` file spans −0.2 s to 140.2 s (5 881 rows),
while the offline one spans 0 to 20 s. The scene builder grids the whole span, so that online scene
has 1 402 ticks. The reference follows the scene faithfully. Cutting the grid to the clip is a scene
question, for a separate bug ticket.

**Rerun check:** left open until T017 draws references.

# T012 — Alpamayo trajectory import

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T011

## Goal
Convert the trajectories from Ralph's Alpamayo runs into the reference format of T011. Running
Alpamayo is not part of this ticket.

## Inputs → Outputs
- Inputs: Alpamayo output files for a clip, one per tick it was run at. Ralph runs these properly:
  - 4 cameras;
  - online egomotion;
  - 1.6 s of history;
  - no future leak.
- Outputs: `data/reference/<clip_id>/<ego_source>/alpamayo/reference.parquet` and `meta.json`.

## Specification
- **Input contract.** Agreed with Ralph before he runs. Each output records:
  - clip ID and the input timestamp t (µs, same clock as the scene);
  - the trajectory for every sample: positions (x, y, z) and rotations per step, in the ego frame at
    time t, at 0.1 s steps (64 steps);
  - model version and run settings.
- **Match to a tick.** Each output is matched to the scene tick with the same timestamp, within
  1 ms. An output that matches no tick is an error, never rounded silently.
- **Frame.** Ego frame at t → world frame, using the scene ego pose at that tick. First confirm
  which point on the car Alpamayo's trajectory describes and apply an offset if needed. The rig
  origin is the rear axle (T006), and Alpamayo is trained on egomotion of the rig origin, so the
  offset is expected to be zero.
- **Speed.** Per step, from the difference between consecutive positions.
- **Step 0.** Its offset from the ego position is stored in `meta.json` as a check value. It should
  be ~0, since the trajectory starts at the ego.
- Ticks Alpamayo was not run at simply have no Alpamayo reference.

## Assumptions
- Alpamayo outputs are in the ego frame at the input time, with steps starting at +0.1 s or at 0.
  To confirm from the model card and loader, and record which.

## Subtasks
- [ ] Write the input contract into the repo README (for the run script)
- [ ] Importer + CLI `reference alpamayo --clip … --from <dir>`
- [ ] Import the new runs

## Acceptance criteria
- [ ] Test: a synthetic trajectory in the ego frame, imported at a known ego pose, lands at the
      expected world coordinates.
- [ ] Test: an output with a timestamp not on the grid is rejected with a clear message.
- [ ] Check: step-0 offset from the ego is below 0.1 m on all imported ticks.
- [ ] Rerun check (after T017): Alpamayo trajectories start at the ego and follow the lane.

## Outcome
<Filled in when done.>

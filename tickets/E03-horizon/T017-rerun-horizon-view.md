# T017 — Rerun view of the horizon

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T010, T014

## Goal
Add the horizon to the scene viewer of T010, so the result at any tick can be checked by eye.

## Inputs → Outputs
- Inputs: a scene, a reference, a run (and its truth results, if present).
- Outputs: a live Rerun viewer or a saved `.rrd` under `artifacts/`.

## Specification
Everything is on the `clip_time` timeline.
- `world/reference/<source>`: the trajectory planned at the current tick (all samples; sample 0
  highlighted).
- `world/horizon/ego_disk`: disks of radius r_veh + ρ(τ) at reference points, every 0.1 s up to H,
  with the point at H marked.
- `world/horizon/objects/<track_id>`: each in-scene object's predicted disk at the same τ values.
  The limiting object is highlighted.
- `plots/H`: H over time as a scalar plot. `plots/H_true` is shown too, if truth exists.
- Toggle: show the true future objects over [t, t+H] to see why a row was invalid.

## Assumptions
- None.

## Subtasks
- [ ] Logger additions
- [ ] CLI flag `scene view … --run …`

## Acceptance criteria
- [ ] Test: logging a synthetic run writes the expected entity paths.
- [ ] Rerun check: at a tick with small H, the growing ego disk visibly meets the limiting object's
      disk at τ = H.

## Outcome
<Filled in when done.>

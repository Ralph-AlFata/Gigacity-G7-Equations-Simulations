# T017 — Rerun view of the horizon

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T010, T014, T015

## Goal
Add the horizon to the scene viewer of T010, so any plan's H, and its safety, can be checked by eye.

## Inputs → Outputs
- Inputs: a scene, a reference, a run (and its safety results, if present).
- Outputs: a live Rerun viewer or a saved `.rrd` under `artifacts/`.

## Specification
Everything is on the `clip_time` timeline. A plan is shown from the tick it is received until the
next plan arrives, because that is the plan the car is following.
- `world/reference/<source>`: the current plan (all samples; the sample being inspected
  highlighted).
- `world/horizon/ego`: the ego's disk of radius r_veh + ρ(τ) at plan points every 0.1 s up to H,
  with the point at H marked.
- `world/horizon/objects/<track_id>`: each known object's predicted disk at the same τ values. The
  limiting object is highlighted.
- `world/safety/contact` (if the plan was unsafe): the ego box and the object box at H\*, and the
  object's group (`known` / `unobserved` / `new_object`).
- Toggle: the true future objects over [t, t+H], to see why a plan was unsafe.
- `plots/H`: H of each plan over time, and H\* when safety results exist. Unsafe plans are marked.

## Assumptions
- None.

## Subtasks
- [ ] Logger additions
- [ ] CLI flag `horizon-scene view … --run …`

## Acceptance criteria
- [ ] Test: logging a synthetic run writes the expected entity paths.
- [ ] Rerun check: at a plan with small H, the growing ego disk visibly meets the limiting object's
      disk at τ = H.
- [ ] Rerun check: for an unsafe plan (synthetic if none is real), the contact boxes appear at H\*.

## Outcome
<Filled in when done.>

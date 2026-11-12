# T015 — Truth check: was H actually safe?

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T014

## Goal
For every H computed live, look at what really happened afterwards and say whether H was safe,
and how much room was left.

**Unsafe means contact (T023):** a car following the plan exactly would have touched a real object
before H. That is `H_true_box` below, the main result. The disk truth is a diagnostic: it tells
whether the predictions held.

## Inputs → Outputs
- Inputs: a run from T014; the same scene and reference.
- Outputs: `truth.parquet` next to `horizons.parquet`, with one row per (tick, sample).

## Specification
- **Scope (T018).** This validates the object side only. The ego is assumed to track the
  reference within ρ(τ), which is not checked. Every output and summary labels this check
  "object-side validation".
- **True margin.** Over τ ∈ [0, T_cap], with the same ego disk r_veh + ρ0 + q̄τ around p_ref(t+τ),
  compute the true margin against the actual future scene:
  - **Objects:** every object that exists at tick t+τ, observed or not, including objects that were
    not in the scene at t.
  - **Positions:** their true positions at t+τ (scene ticks, interpolated between ticks), with no β
    growth.
  - **Formula:** m_true(τ) = minₖ (‖p_ref(t+τ) − oₖ(t+τ)‖ − r_veh − rₖ − ρ0 − q̄τ).
- **Two truths, both reported:**
  - `H_true_disk`: the true margin above, with the same disk shapes as the live H. It answers
    "did the predictions hold?".
  - `H_true_box`: the same, but with real shapes. The ego footprint is the box around the rear
    axle at the plan's pose, not enlarged (the car follows the plan exactly, T023); each object is
    its labeled 3D box, at its true pose. The planar distance
    between the two oriented boxes must be > 0. This answers "would there have been real contact?".
    The difference H_true_box − H_true_disk shows how much room the disks waste.
  - Either one is the first τ with margin ≤ 0, or T_cap.
- **Columns:**
  - `H_true_disk_s`, `H_true_box_s`;
  - `valid_disk` (H ≤ H_true_disk), `valid_box` (H ≤ H_true_box);
  - `slack_disk_s`, `slack_box_s` (H_true − H; how conservative H was);
  - `true_limiting_track_id` (for each truth);
  - `cause`, filled when H was not valid:
    - `new_object`: the true limiting object did not exist at t;
    - `unobserved`: it existed at t but was not observed (`in_scene = false`). This tests the
      lidar rule (T008), not the equations;
    - `prediction`: it was in the scene at t but moved outside its predicted region.
- **Scene end.** The scene ends at the clip end, so for τ beyond the clip there is no truth.
  `truth_until_s` records where the truth stops. Rows where H lies beyond it are marked `unknown`,
  not valid.

## Assumptions
- The scene labels are the truth (T001).
- For Alpamayo references, the car did not actually drive that trajectory. The check asks whether
  a car following it would have stayed clear of the real objects.

## Subtasks
- [ ] True-margin computation (reusing the solver's geometry)
- [ ] Cause classification
- [ ] CLI `horizon truth --run …`

## Acceptance criteria
- [ ] Test: an object that appears after t and cuts across the reference gives `valid_disk = false`,
      `cause = new_object`.
- [ ] Test: an object that exists at t with `in_scene = false` and cuts across gives
      `cause = unobserved`.
- [ ] Test: two boxes side by side that are clear but whose disks overlap give
      `H_true_box > H_true_disk`.
- [ ] Test: H_true_box ≥ H_true_disk always (the disks contain the boxes).
- [ ] Test: an object that accelerates harder than ā gives `cause = prediction` when it breaks H.
- [ ] Test: with all objects at constant velocity and β ≥ 0, H ≤ H_true always.
- [ ] Rerun check: see T017.

## Outcome
<Filled in when done.>

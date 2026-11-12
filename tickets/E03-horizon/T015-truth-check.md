# T015 — Safety check: was H actually safe?

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T014

## Goal
For every plan's H (T014), play the true future and answer: if the car had followed this plan
exactly, would it have touched a real object before H? Also measure how much time H gave away,
and which prediction broke when it failed.

This check may use the future: it is the environment, not what the car knew (E02 principles).

## Inputs → Outputs
- Inputs: a run from T014; the same scene and reference.
- Outputs: `safety.parquet` next to `horizons.parquet`, with one row per (tick, sample).

## Specification
**The ego follows the plan exactly (T023).** At time t+τ its box is the scene's ego footprint
(`meta.json`, around the rear axle) placed at the plan's pose: position p_ref(t+τ), heading
yaw_ref(t+τ), linearly interpolated between plan steps. It is not enlarged by ρ(τ), because there is
no tracking error to cover (T018).

**Objects are the true ones:** every object that exists at t+τ, whether or not it was in the scene
at t, with its labeled box at its true pose (scene ticks, interpolated between ticks). Each one is
put in one of three groups by its state at t:
- `known`: `in_scene = true` at t. H took it into account.
- `unobserved`: it existed at t but `in_scene = false`. This is about the lidar rule (T008), not
  the equations.
- `new_object`: it did not exist at t.

The last two together are **unknown at t**.

**Contact** means the planar distance between the ego box and an object box is ≤ 0. It is evaluated
on a 0.01 s grid over [0, T_cap], and bisected to 1 ms at the first contact, as the solver does.

**Per plan:**
- `H_contact_known_s`, `H_contact_unknown_s`: the first contact with a known object and with an
  unknown object. Empty if there is none before T_cap or before the truth runs out.
- `H_contact_s` (H\*): the earlier of the two. This is the plan's real time to contact.
- `outcome`:
  - `unsafe`: some contact at τ ≤ H;
  - `safe`: no contact over [0, H];
  - `unknown`: the clip ends before H and there was no contact before it. `truth_until_s` records
    where the truth stops.
- `unsafe_known`, `unsafe_unknown`: the same test against each group alone. They are what T016
  reports separately; `unsafe` is the total.
- `contact_track_id`, `contact_group` (`known` / `unobserved` / `new_object`) of the first contact.
- `slack_s` = H\* − H when a contact exists after H. When no contact exists, `slack_s` is a lower
  bound, T_cap − H, flagged by `slack_censored = true`.
- **Prediction check, a diagnostic.** For each known object, did its true center stay inside its
  predicted disk ôₖ(τ) ± βₖ(τ) over [0, H]? It records `n_prediction_breaks`, and the first
  break's `track_id` and τ. This explains unsafe-known plans, and also flags lucky plans where a
  prediction broke but nothing was hit.

## Assumptions
- The scene labels are the truth (T001).
- **Objects do not react.** They replay what they did around the real car. When a plan differs from
  what the real car did, a car behind may "hit" the plan because it braked for the real car. Such
  cases are found when unsafe plans are inspected, and are stated with the results.
- **The recorded stand-in is safe by construction:** the real car drove it without contact. For it,
  this check tests the code only. The prediction check is still meaningful.

## Subtasks
- [ ] Oriented-box contact along a plan (planar)
- [ ] Group objects at t; per-group first contact; outcome and slack
- [ ] Prediction check
- [ ] CLI `horizon-safety --run …`

## Acceptance criteria
- [ ] Test: a straight plan toward a static box gives H\* equal to the analytic contact time, within
      1 ms.
- [ ] Test: two boxes passing side by side, clear by 0.2 m but with overlapping disks, give no
      contact.
- [ ] Test: an object that appears after t and cuts across the plan before H gives `unsafe`,
      `unsafe_unknown`, `contact_group = new_object`, and not `unsafe_known`.
- [ ] Test: an object that exists at t with `in_scene = false` and cuts across gives
      `contact_group = unobserved`.
- [ ] Test: a known object accelerating harder than ā breaks H: `unsafe_known`, and a prediction
      break recorded for it.
- [ ] Test: with only known objects, moving at constant velocity, and with disks that contain the
      boxes, `unsafe_known` is never true.
- [ ] Test: a clip ending before H, with no contact before it, gives `outcome = unknown`.
- [ ] Rerun check: see T017.

## Outcome
<Filled in when done.>

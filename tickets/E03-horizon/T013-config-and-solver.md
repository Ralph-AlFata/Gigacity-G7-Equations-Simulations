# T013 — Horizon config + baseline H solver

**Type:** Task
**Epic:** E03
**Status:** todo
**Depends on:** T011

## Goal
A config file Ralph edits, and a pure function that computes H for one tick from numbers only:
a reference, the ego dimensions, the in-scene objects and the config.

## Inputs → Outputs
- Inputs:
  - one tick's reference (one sample);
  - the in-scene objects at that tick (planar position, velocity, length, width);
  - ego dimensions;
  - the config.
- Outputs:
  - `H_s`;
  - `capped`;
  - `limiting_track_id`;
  - `tau_s` of the crossing;
  - `margin_at_0_m`;
  - `T_cap_s`;
  - `q_bar`;
  - `r_veh`.

## Specification
**Config** `configs/horizon.yaml`. Every value is in SI units, with a comment explaining it. The
starting values are placeholders, not validated:

```yaml
ego:
  r_veh_m: auto          # auto = disk around rear axle covering the footprint; or a number
  rho0_m: 0.2            # initial tracking allowance, ρ0 ≥ ‖e_p(t)‖
tracking:
  e_v_bar_mps: 0.3       # bound on speed error
  e_theta_bar_rad: 0.05  # bound on heading error
  # q_bar = e_v_bar + 2·v_ref_bar·sin(e_theta_bar/2), v_ref_bar = max ref speed over [0, T_cap]
objects:
  eps_p_m: 0.3           # position-estimate error bound
  eps_v_mps: 0.5         # velocity-estimate error bound
  a_bar_mps2: 2.0        # acceleration bound
  radius: half_diagonal  # r_k = ½·sqrt(length² + width²)
horizon:
  t_valid_s: null        # optional extra cap; null = T_cap is the reference duration
solver:
  time_tolerance_s: 0.001
```

**Solver**, exactly the equations in the E03 README.
- Planar (x, y) only, as in the PDF; z is ignored here.
- p_ref between reference steps is linear interpolation.
- Evaluate minₖ mₖ(τ) on a 0.01 s grid over [0, T_cap]. At the first grid point with a value ≤ 0,
  bisect between it and the previous point to the time tolerance.
- If mₖ(0) ≤ 0 for some k, then H = 0.
- If no crossing is found, then H = T_cap and `capped` is true.
- Zero in-scene objects: H = T_cap, `capped` is true, and `limiting_track_id` is null.

The function reads no files and keeps no state, so it can be tested in isolation.

## Assumptions
- A 0.01 s grid does not skip a crossing. mₖ changes by at most (relative speed + growth rate) ×
  0.01 s between grid points; the test below checks a fast case.

## Subtasks
- [ ] Config loader with validation (unknown keys, negative values, θ̄ > π rejected)
- [ ] r_veh auto from vehicle dimensions and rear-axle offset
- [ ] Solver
- [ ] Tests

## Acceptance criteria
- [ ] Test: a straight reference and a static object ahead give H equal to the analytic solution of
      Dₖ(τ) = Sₖ(τ), within 1 ms.
- [ ] Test: an object approaching head-on at 30 m/s is caught, not skipped.
- [ ] Test: no objects → capped at T_cap; an overlap at τ = 0 → H = 0.
- [ ] Test, hand-checked: one fixed example (ego pose, one reference, two objects, config values)
      worked out by hand step by step in `tests/hand_example.md`, then asserted:
  - r_veh, rₖ, q̄;
  - Dₖ(τ), Sₖ(τ), mₖ(τ) at τ = 0, 0.5, 1.0;
  - H and the limiting object.

  Ralph checks the hand calculation once; after that, any solver change is checked against it.
- [ ] Test: q̄ matches the formula; the old hand-checked radius 3.790 m is reproduced for the same
      dimensions.
- [ ] Rerun check: see T017.

## Outcome
<Filled in when done.>

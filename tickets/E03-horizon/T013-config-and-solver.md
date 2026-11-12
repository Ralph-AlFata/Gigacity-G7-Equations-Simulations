# T013 — Horizon config + baseline H solver

**Type:** Task
**Epic:** E03
**Status:** review
**Depends on:** T011

## Goal
A config file Ralph edits, and a pure function that computes H for one plan from numbers only:
the plan (one sample, received at tick t), the ego dimensions, the objects in the scene at t, and
the config. It is called once per plan (T023).

## Inputs → Outputs
- Inputs:
  - one plan: one sample of the reference at tick t;
  - the in-scene objects at t (planar position, velocity, length, width);
  - ego dimensions;
  - the config.
- Outputs:
  - `H_s`;
  - `capped`;
  - `limiting_track_id`;
  - `margin_at_0_m`;
  - `T_cap_s`;
  - `q_bar_mps`, `v_ref_bar_mps`;
  - `r_veh_m`.

## Specification
**Config** `configs/horizon.yaml`. Every value is in SI units, with a comment explaining it. The
starting values are placeholders, not validated:

The file as committed is `configs/horizon.yaml`. Its layout:

```yaml
ego:
  r_veh_m: auto          # disk around the rear axle covering the footprint (auto), or a number
  rho0_m: 0.2            # initial tracking allowance, ρ0 ≥ ‖e_p(t)‖
tracking:
  e_v_bar_mps: 0.3       # bound on the speed error ē_v
  e_theta_bar_rad: 0.05  # bound on the heading error ē_θ (≤ π)
objects:
  eps_p_m: 0.3           # bound on the position-estimate error ε_p
  eps_v_mps: 0.5         # bound on the velocity-estimate error ε_v
  a_bar_mps2: 2.0        # bound on the object's acceleration ā
horizon:
  t_valid_s: 6.4         # the PDF's T_valid: T_cap = min(plan's remaining duration, t_valid)
solver:
  grid_s: 0.01           # spacing of the τ grid searched for the first crossing
  time_tolerance_s: 0.001
```

rₖ is always half the box's planar diagonal (the PDF's choice), so it has no key.

**Solver**, exactly the equations in the E03 README.
- Planar (x, y) only, as in the PDF; z is ignored here.
- p_ref between reference steps is linear interpolation.
- Evaluate minₖ mₖ(τ) on a 0.01 s grid over [0, T_cap]. At the first grid point with a value ≤ 0,
  bisect between it and the previous point to the time tolerance.
- If mₖ(0) ≤ 0 for some k, then H = 0.
- T_cap = min(the plan's last `tau_s`, `t_valid_s`), as in the PDF (p. 9).
- If no crossing is found, then H = T_cap and `capped` is true.
- Zero in-scene objects: H = T_cap, `capped` is true, and `limiting_track_id` is null.
- H is the last time known clear: never later than the exact crossing, at most the tolerance earlier.
- **Non-finite inputs are refused.** A NaN margin compares false with 0, so an object with a NaN
  velocity would otherwise be silently ignored. The solver raises an error naming the objects.

The function reads no files and keeps no state, so it can be tested in isolation.

## Assumptions
- A 0.01 s grid does not skip a crossing. mₖ changes by at most (relative speed + growth rate) ×
  0.01 s between grid points; the test below checks a fast case.

## Subtasks
- [x] Config loader with validation (unknown keys, negative values, θ̄ > π rejected)
- [x] r_veh auto from vehicle dimensions and rear-axle offset
- [x] Solver
- [x] Tests

## Acceptance criteria
- [x] Test: a straight reference and a static object ahead give H equal to the analytic solution of
      Dₖ(τ) = Sₖ(τ), within 1 ms.
- [x] Test: an object approaching head-on at 30 m/s is caught, not skipped.
- [x] Test: no objects → capped at T_cap; an overlap at τ = 0 → H = 0.
- [x] Test: a 20 s recorded plan is capped at `t_valid_s`, and a 3 s plan at 3 s.
- [x] Test, hand-checked: one fixed example (ego pose, one reference, two objects, config values)
      worked out by hand step by step in `tests/hand_example.md`, then asserted:
  - r_veh, rₖ, q̄;
  - Dₖ(τ), Sₖ(τ), mₖ(τ) at τ = 0, 0.5, 1.0;
  - H and the limiting object.

  **Still open:** Ralph checks the hand calculation once; after that, any solver change is checked
  against it.
- [x] Test: q̄ matches the formula; the old hand-checked radius 3.790 m is reproduced for the same
      dimensions.
- [ ] Rerun check: see T017.

## Outcome
**Code:** `horizon_pipeline/horizon/` (`config.py`, `solver.py`), `configs/horizon.yaml`.

**Checked against the source:** the equations match `Deadline_Aware_Multi_Vehicle_Serving.pdf`
pp. 4–9. The cap is named `t_valid_s` after the PDF's T_valid (T_cap = min(T_rem, T_valid)).

**Tests:**
- `tests/test_horizon.py`, 23 tests: config validation, r_veh = 3.790 m, q̄, analytic crossings,
  caps, refusal of non-finite inputs, and the hand example.
- Deliberately broken solvers are caught: dropping the ½ on ā fails 5 tests; reporting the later end
  of the bisection bracket fails 3.
- `tests/test_horizon_real.py`: every recorded plan of b789fab8 and b4d979af solves, with consistent
  outputs.
- Full suite: 158 passed.

**First look on real data** (recorded stand-in, offline ego, every tick, current placeholder config):

| Clip | r_veh | H median | p90 | capped | H = 0 |
|---|---|---|---|---|---|
| 23958e60 | 3.910 m | 0.00 s | 0.00 s | 1% | 185 / 201 |
| b4d979af | 3.790 m | 0.00 s | 1.18 s | 9% | 118 / 201 |
| b6cb8b12 | 4.211 m | 0.00 s | 1.76 s | 1% | 124 / 201 |
| b789fab8 | 3.790 m | 0.29 s | 1.32 s | 9% | 85 / 201 |
| ca823746 | 3.790 m | 2.14 s | 3.61 s | 6% | 9 / 201 |

(H = 0 counts include the last tick, whose plan has no length.)

**Why H is so often 0. This is the baseline model, not a bug:**
- In b789fab8, every H = 0 is limited by a vehicle in the next lane, about 3 m to the side and 3–6 m
  from the rear axle.
- The disks then need S(0) ≈ 6.5 m between centres: r_veh 3.79 m (the disk is centred on the rear axle,
  so it must reach the front corner), plus about 2.2 m for the other car's disk, plus ρ0 + ε_p = 0.5 m.
- Two cars in neighbouring lanes are about 3.5 m apart, so the baseline cannot certify any plan next to
  one.
- This is what the minimal equations give. Tighter shapes are a later epic (E03 scope).

**Found in the data, for later tickets:**
- **Objects with no velocity yet:** a track's first tick has no causal velocity (T009). This is 0.5–2%
  of in-scene rows: 45 / 7 202, 439 / 46 794, 151 / 7 760, 99 / 6 328, 36 / 1 701. The solver refuses
  them, so T014 must decide how to treat them. The test above leaves them out.
- **The ego car labelled as an object:** in 23958e60, track 41 ("automobile", 4.87 × 2.06 m) sits 1.5 m
  ahead of the rear axle (our own box centre) at ticks 1–9. No other clip has this.

**Rerun check:** left open until T017.

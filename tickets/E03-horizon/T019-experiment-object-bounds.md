# T019 — Do the object bounds hold on our scenes?

**Type:** Experiment
**Epic:** E03
**Status:** todo
**Depends on:** T008, T009, T013

## Question
Do the object assumptions in the config hold on the scene data, independently of H? Specifically:
- **Acceleration bound:** is ‖aₖ‖ ≤ ā?
- **Prediction region:** does the true position stay inside the predicted region,
  ‖oₖ(t+τ) − (ôₖ + ŵₖτ)‖ ≤ βₖ(τ)?

If not, which parameter is wrong, by how much, and for which class groups?

## Setup
- Data: both clips, both ego sources; objects with `in_scene = true` at the prediction time t.
- References: none. This does not use trajectories.
- Parameters: `eps_p_m`, `eps_v_mps`, `a_bar_mps2` from `configs/horizon.yaml`.
- Code version: <commit hash>

## Method
- **Acceleration:** the distribution of ‖aₖ‖ from the scene, by class group. Report the share of
  object-ticks above ā, and the 95th, 99th percentile and max.
- **Prediction error:** for every in-scene object at every tick t, and τ = 0.1 … 6.4 s, while the
  object still exists:
  - error(τ) = ‖true position at t+τ − (ôₖ + ŵₖτ)‖;
  - compare it with βₖ(τ);
  - report the share of (object, t, τ) with error > βₖ(τ), as a function of τ and by class group.
- **Fitted values:** for each class group, the smallest ε_p, ε_v, ā that would make β cover 95%
  and 99% of the observed errors. These are measured values to compare with the config. They do
  not change the config automatically.
- **Caveat:** ŵₖ and aₖ are derived from the same labels (T007). T009 says how far to trust derived
  motion.

## Command
```bash
<exact command to reproduce>
```

## Results
<…>

## Interpretation
<Which parameter is too small or too loose, for which class groups and which τ range.>

## Rerun check
- [ ] Objects exceeding β visualized at a few ticks: true path versus predicted disk.

## Follow-ups
<Config changes proposed to Ralph; possible class-specific bounds (later epic).>

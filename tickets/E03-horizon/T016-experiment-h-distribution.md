# T016 — Distribution of H on the two clips

**Type:** Experiment
**Epic:** E03
**Status:** todo
**Depends on:** T014, T015

## Question
What does the distribution of H look like with the baseline equations and the current config, for
each reference source?

## Setup
- **Clips:** b789fab8, b4d979af.
- **Ego source:** `egomotion.offline` (T004).
- **References:** recorded path and Alpamayo, reported separately, never pooled.
- **Config:** `configs/horizon.yaml` as committed; its SHA-256 is recorded.

## Metrics
Per (clip, ego source, reference) and pooled over the two clips for one reference:
- **Count and coverage:** number of ticks; share `capped`.
- **Central values and spread:** mean, std, min, max.
- **Percentiles:** 5, 10, 25, 50, 75, 90, 95.
- **Plots:** histogram and empirical CDF of H; H over clip time.
- **Alpamayo only:** the spread of H across samples at the same tick (mean and max of per-tick std).
- **Truth check (object-side only, T018):** for both the disk and the box truths: share valid,
  share invalid (by cause: `new_object` / `unobserved` / `prediction`), share unknown; distribution
  of slack.

The headline mean is reported both with and without capped rows. A capped H means "no crossing
found before T_cap", not a measured value, so it pulls the mean around.

## Outputs
- `results/<run_name>/summary.md`: tables and figures.
- `summary.json`: the same numbers, machine-readable.

## Results
<…>

## Interpretation
<…>

Caveats to state with the results:
- **Correlation:** ticks 0.1 s apart are nearly the same situation, so the ~400 ticks of 2 clips
  are not independent samples. The mean, std and percentiles describe these two scenes, not
  driving in general.
- **Ego half not validated:** the tracking bounds (ē_v, ē_θ, q̄, ρ0) are assumptions (T018).
- **Placeholder values:** the object parameters are placeholders until T019 measures them.

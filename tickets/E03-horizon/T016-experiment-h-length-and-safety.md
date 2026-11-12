# T016 — H: how long, and was it safe?

**Type:** Experiment
**Epic:** E03
**Status:** todo
**Depends on:** T014, T015

## Question
With the baseline equations and the current config, for each reference source:
- **A. How long is H?** The distribution of H over plans.
- **B. Was H right?** How often would following the plan for H have caused contact, and how much
  time did H give away?

## Setup
- **Plans:**
  - recorded stand-in: every tick of every built clip;
  - Alpamayo: the plans of Ralph's snapshot dataset (T012, T023).
- **Ego source:** `egomotion.offline` (T004).
- **References:** reported separately, never pooled.
- **Config:** `configs/horizon.yaml` as committed; its SHA-256 is recorded.

## Metrics
Per (clip, reference) and pooled over clips for one reference.

**A. Length of H**
- **Count:** number of plans; share `capped`.
- **Central values and spread:** median, mean, std, min, max. The mean is given with and without
  capped plans: a capped H means "safe for the whole plan", not a measured value.
- **Low tail, the headline next to the median:** percentiles 5, 10, 25, 50, 75, 90, 95. Short H is
  what loads the scheduler.
- **Share of plans with H below the inference time** (~1.2 s on an A40). Even a request served at
  once would arrive too late for these.
- **Plots:** histogram and empirical CDF of H; H over clip time.
- **Breakdowns:** by ego speed band, by clip, and by the limiting object's class group and
  distance.
- **Alpamayo only:** the spread of H across samples of one snapshot.

**B. Safety (T015, contact with the ego following the plan exactly)**
- **Unsafe rate,** each with a 95% upper bound (Clopper-Pearson; about 3/n when there are 0 failures
  in n plans):
  - by objects known at t;
  - by objects unknown at t (`new_object`, `unobserved`);
  - total.
- **Share `unknown`** (truth ended before H).
- **Slack H\* − H:** its distribution, with censored plans counted as lower bounds.
- **Prediction breaks:** the share of plans with a known object outside its predicted disk before H,
  even without contact.
- **Every unsafe plan is listed** with its tick, sample, object and cause, and opened in Rerun.

## Outputs
- `results/<run_name>/summary.md`: tables and figures.
- `summary.json`: the same numbers, machine-readable.

## Results
<…>

## Interpretation
<…>

Caveats to state with the results:
- **Correlation:** plans 0.1–0.3 s apart are nearly the same situation, so they are not independent
  samples. The confidence bound is therefore also given per clip, and the results describe these
  scenes, not driving in general. Claiming an unsafe rate below 0.1% needs thousands of independent
  plans.
- **Objects do not react** to a plan that differs from what the real car did (T015).
- **The recorded stand-in is safe by construction,** so B is meaningful only for Alpamayo plans.
- **Ego half not validated:** the car follows the plan exactly, so ρ0 and q̄ are not tested (T018).
- **Placeholder values:** the object parameters are placeholders until T019 measures them.

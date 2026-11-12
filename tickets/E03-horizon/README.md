# E03 — Horizon replay

**Type:** Epic
**Status:** in-progress
**Depends on:** E02

## Goal
**What H is for (T023):** in a real system, the car computes H once, when it receives a plan from
the Alpamayo server. H is how long it is safe to keep following that plan. The next inference
request carries H, and the server's priority scheduler serves the shortest H (earliest deadline)
first. So H must be both useful (not too short) and safe.

On the dataset, for every plan, at the tick it is received, compute H once with the PDF's baseline
equations. Then answer two questions:
- **A. How long is H?** Its distribution.
- **B. Was H right?** Would a car following the plan have touched a real object before H?

**Plans** come from a separate process, run by Ralph, that generates the snapshots and their
Alpamayo plans (e.g. every 0.3 s). Until then, the recorded path (T011) is a stand-in.

## Principles (decided 2026-10-31)
- **Minimal equations:** exactly the PDF baseline.
  - ego and objects as disks;
  - constant-velocity object centers with growth βₖ(τ);
  - ego growth ρ(τ) = ρ0 + q̄τ.

  Improvements come later, as their own epic.
- **Two reference sources, kept separate:** Alpamayo trajectories (run by Ralph, imported here) and
  the car's recorded future path. Every output says which source it used; they are never mixed.
- **Live H uses only what is known at t:** in-scene objects at tick t, predicted forward. Objects that
  appear later are unknown, as in reality.
- **H is computed once per plan, on receipt** (T023). It is never updated while the plan is followed.
  Every sample of a snapshot gets its own H.
- **Truth check, kept separate:** compares each H with the true future scene, to measure if H was
  valid and how conservative it was.
  - **Unsafe means contact** (T023): the ego's box on the plan touches an object's true box before H.
  - Failures caused by objects not known at t (not yet existing, or not observed) are reported
    separately from failures by objects known at t, with the total shown too.
  - It is done with disks (did the predictions hold?) and with real boxes (would there have been
    contact?).
  - It validates the object side only. The ego tracking bounds are assumptions (T018).
  - The object bounds are also measured directly (T019).
- **All parameters in one config file** that Ralph edits. Every output records the config it used.
- **Outputs:** the distribution of H (A) and its safety (B), with a confidence bound on the failure
  rate. The share of plans with H below the inference time is reported, but no latency or scheduler
  is simulated.

## Equations (PDF, baseline)
At tick t, for τ ∈ [0, T_cap]:

**Reference and ego**
- p_ref(t+τ): reference rear-axle position.
- ρ(τ) = ρ0 + q̄τ.
- q̄ = ē_v + 2·v̄_ref·sin(ē_θ/2), where v̄_ref = max reference speed over [0, T_cap].

**Objects**
- ôₖ(τ) = ôₖ + ŵₖτ: center, from scene position and derived velocity.
- βₖ(τ) = ε_p + ε_v·τ + ½·ā·τ².
- rₖ = half of the box's planar diagonal.

**Margin and horizon**
- Dₖ(τ) = ‖p_ref(t+τ) − ôₖ(τ)‖.
- Sₖ(τ) = r_veh + rₖ + ρ0 + ε_p + (q̄ + ε_v)·τ + ½·ā·τ².
- mₖ(τ) = Dₖ(τ) − Sₖ(τ).
- H = inf{τ ∈ [0, T_cap] : minₖ mₖ(τ) ≤ 0}. If there is no such τ, then H = T_cap and the row is
  marked `capped`.

**T_cap** = the remaining duration of the reference trajectory (6.4 s for Alpamayo; until the clip
ends for the recorded path), optionally lowered by `t_valid_s` in the config (off by default).

## Scope
- In scope:
  - the reference format;
  - the recorded-path reference;
  - the Alpamayo import contract;
  - the config;
  - the solver;
  - the replay runner;
  - the truth check;
  - statistics;
  - Rerun view.
- Out of scope:
  - running Alpamayo (Ralph does it);
  - latency or serving-loop simulation;
  - the tighter shapes and bounds researched earlier;
  - sensitivity sweeps over the config (later experiment);
  - validating the ego tracking bounds (needs a controller or simulator; T018).

## Exit criteria
- [ ] For both clips and each reference source: a per-tick results table and an H summary (mean,
      std, min, percentiles, histogram, share capped).
- [ ] Truth check results for the same rows: share of plans where H was unsafe (contact before H), by
      cause (known vs unknown objects), with a 95% bound, and the slack H* − H.
- [ ] Rerun: at any tick, the trajectory, the ego disk growing along it until H, each object's
      predicted disk, and the limiting object are visible.

## Tickets
| ID | Type | Title | Status |
|---|---|---|---|
| T011 | Task | Reference format + recorded-path reference | review |
| T012 | Task | Alpamayo trajectory import | todo |
| T013 | Task | Horizon config + baseline H solver | todo |
| T014 | Task | Live replay runner | todo |
| T015 | Task | Truth check: was H actually safe? | todo |
| T016 | Experiment | Distribution of H on the two clips | todo |
| T017 | Task | Rerun view of the horizon | todo |
| T018 | Decision | What the truth check validates | done |
| T019 | Experiment | Do the object bounds hold on our scenes? | todo |
| T023 | Decision | What H means: once per plan on receipt, checked against the true future | done |

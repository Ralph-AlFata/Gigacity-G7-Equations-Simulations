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
- **B. Was H right?** Would a car following the plan exactly have touched a real object before H?

**Plans** come from a separate process, run by Ralph, that generates snapshots of each clip and
their Alpamayo plans (e.g. every 0.3 s) and simulates them. This epic takes those plans as given,
in the T011 format. Until they exist, the recorded path (T011) is a stand-in.

## Principles
Decided 2026-10-31, revised by T023 on 2026-11-12.
- **Minimal equations:** exactly the PDF baseline.
  - ego and objects as disks;
  - constant-velocity object centers with growth βₖ(τ);
  - ego growth ρ(τ) = ρ0 + q̄τ.

  Improvements come later, as their own epic.
- **One H per plan, computed once on receipt.** A plan is one sample of the reference received at
  tick t. Its H is never updated while the plan is followed. Every sample of a snapshot gets its own
  H.
- **H uses only what is known at t:** the objects in the scene at tick t (`in_scene = true`),
  predicted forward. Objects that appear later, or are not observed, are unknown, as in reality.
- **Reference sources kept separate:** Alpamayo plans and the recorded stand-in. Every output says
  which source it used; they are never mixed.
- **Safety check, kept separate from H (T015).** It may use the future, because it is the
  environment, not the car's knowledge.
  - **Unsafe means contact:** the ego's box, following the plan exactly, touches an object's true
    box at some τ ≤ H.
  - **Failures are split by cause:** objects known at t (the equations or bounds were wrong) and
    objects unknown at t (not yet existing, or not observed). The total is shown too.
  - **Scope (T018):** the car follows the plan exactly, so the ego tracking allowance ρ(τ) is not
    tested. The object bounds are also measured directly (T019).
- **All parameters in one config file** that Ralph edits. Every output records the config it used.
- **Results:** the distribution of H (A) and its safety with a 95% confidence bound (B). The share
  of plans with H below the inference time is reported, but no latency or scheduler is simulated.

## Equations (PDF, baseline)
For a plan received at tick t, for τ ∈ [0, T_cap]:

**Reference and ego**
- p_ref(t+τ): the plan's rear-axle position.
- ρ(τ) = ρ0 + q̄τ.
- q̄ = ē_v + 2·v̄_ref·sin(ē_θ/2), where v̄_ref = max plan speed over [0, T_cap].

**Objects** (in the scene at t)
- ôₖ(τ) = ôₖ + ŵₖτ: center, from scene position and derived velocity.
- βₖ(τ) = ε_p + ε_v·τ + ½·ā·τ².
- rₖ = half of the box's planar diagonal.

**Margin and horizon**
- Dₖ(τ) = ‖p_ref(t+τ) − ôₖ(τ)‖.
- Sₖ(τ) = r_veh + rₖ + ρ0 + ε_p + (q̄ + ε_v)·τ + ½·ā·τ².
- mₖ(τ) = Dₖ(τ) − Sₖ(τ).
- H = inf{τ ∈ [0, T_cap] : minₖ mₖ(τ) ≤ 0}. If there is no such τ, then H = T_cap and the row is
  marked `capped`.

**T_cap** = the plan's remaining duration, at most `t_valid_s` in the config (the PDF's T_valid; default 6.4 s,
Alpamayo's plan length). The recorded stand-in runs to the end of the clip, so the cap makes its H
comparable with Alpamayo's.

## Scope
- In scope:
  - the reference format;
  - the recorded-path stand-in;
  - the Alpamayo import contract;
  - the config;
  - the solver;
  - the runner (one H per plan);
  - the safety check;
  - statistics;
  - Rerun view.
- Out of scope:
  - generating the snapshot dataset and running Alpamayo (Ralph's separate process);
  - simulating latency, the request queue or the scheduler (H's use is described in T023, not
    built);
  - updating H while a plan is followed;
  - the tighter shapes and bounds researched earlier;
  - sensitivity sweeps over the config (a later experiment: the unsafe rate against the median H);
  - validating the ego tracking bounds (needs a controller; T018).

## Exit criteria
- [ ] For each reference source: a per-plan results table and the answer to A: median, mean
      (with and without capped plans), std, min, percentiles, histogram, share capped, share below
      the inference time.
- [ ] For the same plans, the answer to B: the share unsafe (contact before H), by cause (objects
      known at t, objects unknown at t, total), each with a 95% upper bound; and the slack H* − H.
- [ ] Rerun: at any plan, the trajectory, the ego growing along it until H, each object's predicted
      disk, the limiting object, and where contact happened if the plan was unsafe.

## Tickets
| ID | Type | Title | Status |
|---|---|---|---|
| T011 | Task | Reference format + recorded-path reference | done |
| T012 | Task | Alpamayo trajectory import | todo |
| T013 | Task | Horizon config + baseline H solver | review |
| T014 | Task | Horizon runner: one H per plan | todo |
| T015 | Task | Safety check: was H actually safe? | todo |
| T016 | Experiment | H: how long, and was it safe? | todo |
| T017 | Task | Rerun view of the horizon | todo |
| T018 | Decision | What the truth check validates | done |
| T019 | Experiment | Do the object bounds hold on our scenes? | todo |
| T023 | Decision | What H means: once per plan on receipt, checked against the true future | done |

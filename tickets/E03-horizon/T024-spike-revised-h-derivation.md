# T024 — A revised derivation of H

**Type:** Spike
**Epic:** E03
**Status:** review
**Depends on:** T009, T013, T014, T023
**Timebox:** one day

## Question
The baseline equations (PDF) give H = 0 for about half of the plans. Why, and what derivation of H
makes sense for our system? Write it out in LaTeX, with proofs, so Ralph can work through it.

**Design rule (Ralph, 2026-11-12):** do not tune on our numbers; five clips are far too few. Take the
lessons of the scenarios into the *structure* of the equations; take the *numbers* from physics,
regulation and published literature, without overfitting.

## Why it matters
H is the deadline the car sends to the scheduler (T023). If it is 0 most of the time it carries no
information; if it is not safe it is worse than useless. Every later ticket (T015 safety check, T016
statistics, T019 bounds) depends on which equations H uses.

## Method
- [x] Read the PDF derivation (pp. 1–9) and the current solver (T013).
- [x] Diagnosis on our data: why H = 0 (box gap against disk overlap for the 516 plans with H = 0,
      from the T014 run `results/t014-baseline`); how objects move along and across their heading,
      stopped against moving, turning; label noise. Used for structure and sanity checks only.
- [x] Literature for every parameter: physics (friction), regulation (UN R157, R79), safety models
      (RSS and its reference library, IEEE 2846), studies of driving and walking. Every reference was
      checked against the paper, Crossref or PubMed; claims that could not be confirmed are marked
      (e.g. no primary source for a 5 m turning radius: an assumption).
- [x] Worked examples, checked against closed forms (RSS car following: 0.48 s against 0.481 s).

## Hypotheses
1. H = 0 comes from the disks, not from the motion bounds. **Supported** (98.4% of H = 0 plans have
   a real box gap above 0.5 m; median 1.48 m).
2. The object bound has the wrong structure (isotropic, no nonholonomy, no turning distinction).
   **Supported**: along/across the heading differ by about 3× at p90; a stopped car's box drifts
   sideways about 5× less than a moving car's; the large sideways moves are turns.
3. H as the first possible contact is not a safe deadline. **Supported** by argument: at the deadline
   a contact may already be unavoidable; the time-to-react is the right quantity.

## Findings
The derivation is in `docs/derivation/horizon_derivation.tex`. Scripts:
- `uv run python scripts/spike_t024_object_motion.py`: the diagnosis numbers (Section 2).
- `uv run python scripts/t024_worked_examples.py`: the worked examples and sensitivity (Section 7),
  ~2 min. Part 5 of the first script reads `results/t014-baseline` (PR #17) and is skipped without it.

Worked examples, H_c / H_r in s (foreseeable set: lane-following vehicles with the RSS lateral
response, heading-estimate error 0.05 rad, fallback braking 4 m/s²):

| Scene | Baseline | Physical | Foreseeable |
|---|---|---|---|
| E1 next-lane car abreast, 10 m/s | 0.00 | 0.15 / 0.00 | 1.02 / 0.00 |
| E2 following at 4 m/s, 4 m gap | 1.36 | 0.73 / 0.03 | 0.97 / 0.46 |
| E3 hand-example pedestrian | 2.00 | 1.65 / 0.00 | 2.95 / 0.96 |
| E4 passing a parked car, 10 m/s | 0.76 | 0.84 / 0.00 | 1.05 / 0.00 (capped if parked cars yield) |

What generalises:
- Following reduces exactly to the RSS safe distance.
- A car beside the ego is limited by how well its heading is known (or by a lane map), not by its
  behaviour: with ε_ψ = 0 and RSS lateral behaviour it never limits H.
- A parked car is a responsibility question (does a car pulling out yield?), not a number.

## Answer
Replace the disks by oriented boxes (ego tracking error as ρ_p + 2·R_F·sin(ē_θ/2) = 0.39 m). Predict
each object as a box with separate front, back and side allowances, derived from explicit assumptions:
turning vehicles (Proposition 1), lane-following vehicles with the RSS lateral response
(Proposition 3), pedestrians (Proposition 2). Use two parameter sets: physical (relevance filter,
reference) and foreseeable (H). Use two horizons: the contact horizon H_c and the reaction horizon
H_r ≤ H_c, the latter as the scheduler deadline. The guarantee is exact and conditional on the stated
assumptions; the safety check measures how often each assumption is violated (validation, not
calibration). Confidence: high on the diagnosis and the derivation; the parameter values are sourced
but several are explicit assumptions (ε_ψ, κ_max, v_turn, ρ_p).

## Follow-ups
Proposed, not created until Ralph has read the derivation (Section 9 of the document):
- new solver `method: boxes` (geometry, Propositions 1–3, H_c);
- H_r in the solver, with the responsibility switches; amendment to T023 (deadline = H_r);
- T015: same boxes, fallback check, per-assumption violation record;
- T019 rewritten: "how often does each assumption hold?" instead of "which values cover the data?".
Decisions D1–D6 for Ralph are listed in the document.

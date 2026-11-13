# T024 — A revised derivation of H that fits our data

**Type:** Spike
**Epic:** E03
**Status:** review
**Depends on:** T009, T013, T014, T023
**Timebox:** one day

## Question
The baseline equations (PDF) give H = 0 for about half of the plans. Why, and what derivation of H
makes sense for our system and our data? Write it out in LaTeX, with proofs, so Ralph can work through
it.

## Why it matters
H is the deadline the car sends to the scheduler (T023). If it is 0 most of the time it carries no
information; if it is not safe it is worse than useless. Every later ticket (T015 safety check, T016
statistics, T019 bounds) depends on which equations H uses.

## Method
- [x] Read the PDF derivation (pp. 1–9) and the current solver (T013).
- [x] Data check: why H = 0 (box gap against disk overlap for the 516 plans with H = 0, from the T014
      run `results/t014-baseline`).
- [x] Data check: how objects move against constant velocity, along and across their heading, by
      class and motion state; motion limits from 1 s chords; label noise from parked tracks.
- [x] Literature: set-based prediction, time-to-react, RSS, fail-safe verification, conformal
      prediction, collision checking. Every reference was checked against the paper, Crossref or
      PubMed. Corrections that changed the text:
  - time-to-react is credited to Hillenbrand et al. 2006, not Tamke et al. 2011;
  - Lindemann et al. use a per-step union bound, not a max score;
  - ABS braking on dry roads is 8.5–10.6 m/s², so the physical layer uses 10 m/s²;
  - no primary source was found for a 5 m turning radius, so it is marked as an assumption.
- [x] Worked examples with both models, checked against closed forms.

## Hypotheses
1. H = 0 comes from the disks, not from the motion bounds. **Supported** (98.4% of H = 0 plans have
   a real box gap above 0.5 m; median 1.48 m).
2. The object bound β has the wrong shape. **Supported**: the error grows about linearly in τ
   (velocity noise), not quadratically; β is too small below 1 s and far too large after.
3. H as the first possible contact is not a safe deadline. **Supported** by argument: at the deadline
   a contact may already be unavoidable; the time-to-react is the right quantity.

## Findings
The derivation is in `docs/derivation/horizon_derivation.tex`. Evidence scripts:
- `uv run python scripts/spike_t024_object_motion.py`: data numbers (Sections 2 and 5).
- `uv run python scripts/t024_worked_examples.py`: the four worked examples (Section 8), including a
  check of the numerical reaction horizon against the RSS closed form (0.52 s against 0.522 s).
  Part 5 of the first script reads the T014 run `results/t014-baseline` (PR #17) and is skipped
  without it.

Key numbers (5 clips, offline, recorded stand-in):

| | Value |
|---|---|
| Plans with H = 0 (baseline) | 516 of 1 005 (52%) |
| Real box gap at τ = 0 for those plans, p10 / p50 / p90 | 0.95 / 1.48 / 3.67 m |
| Vehicle CV residual, p99 fit | 0.60 + 1.21τ + 0.26τ² m (baseline β: 0.3 + 0.5τ + τ²) |
| Vehicle cases outside β at 0.5 s / 1 s | 1.80% / 1.66% |
| Side overhang at 1 s, p99: stopped / moving vehicles | 0.43 / 2.02 m |
| Speed change over 1 s windows, min / p0.1 | −7.20 / −4.67 m/s² |
| Label noise, parked tracks: position RMS p95, yaw RMS p95 | 0.35 m, 2.8° |
| Person speed max, \|Δv\| per s p99.9 | 2.62 m/s, 1.32 m/s² |

Worked examples (H in s; baseline / revised physical / revised calibrated p99):
- E1 next-lane car abreast at 10 m/s: 0.00 / 0.09 / 0.60
- E2 following at 4 m/s, 4 m gap: 1.36 / 0.50 / 1.85; reaction horizon 0.03 / 1.66
- E3 hand-example pedestrian: 2.00 / 2.23 / 2.57
- E4 passing a parked car at 10 m/s: 0.76 / 0.75 / capped

## Answer
Replace the disks by oriented boxes (ego tracking error as ρ_p + 2·R_F·sin(ē_θ/2) = 0.39 m); predict
each object as a box with separate front, back and side allowances, from a physical layer intersected
with a data-calibrated (conformal) layer conditioned on class and motion state; and use two horizons:
the contact horizon H_c and the reaction horizon H_r ≤ H_c, the latter as the scheduler deadline. The
guarantee becomes explicit: P(contact with a known object) ≤ Σ δ_k over relevant objects.
Confidence: high on the diagnosis (data); medium on the calibrated numbers (in-sample, 5 clips,
per-τ rather than joint).

## Follow-ups
Proposed, not created until Ralph has read the derivation (Section 9 of the document):
- new solver `method: boxes` (geometry + calibrated layer + H_c);
- T019 rewritten as the conformal calibration experiment on held-out clips, plus more clips;
- H_r in the solver and an amendment to T023 (deadline = H_r);
- T015: same boxes, fallback check, responsibility labels;
- physical layer as relevance filter.
Decisions D1–D6 for Ralph are listed in the document.

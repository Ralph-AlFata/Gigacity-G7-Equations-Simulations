# T018 — What the truth check validates

**Type:** Decision
**Epic:** E03
**Status:** done
**Depends on:** none

## Context
The margin has two halves:
- **Ego tracking allowance:** ρ(τ) = ρ0 + q̄τ, with q̄ from ē_v, ē_θ, v̄_ref.
- **Object uncertainty:** βₖ(τ) = ε_p + ε_v·τ + ½·ā·τ².

The truth check (T015) compares H with the real future scene. That tests the object half, because
we know where the objects really went. It cannot test the ego half:
- **Alpamayo references:** no car actually drove the trajectory, so there is no real tracking error
  to compare with ρ(τ).
- **Recorded-path reference:** the car follows its own path exactly, so the tracking error is 0
  and ρ(τ) is never exercised.

Testing the ego half needs a controller following the plan, in a simulator or on a real car.

**Update 2026-11-12 (T023):** the safety check is now real contact, with the ego's box following
the plan exactly and not enlarged by ρ(τ). This makes the scope below explicit: ρ0 and q̄ shorten H,
but nothing on the dataset tests whether they are right. The decision stands unchanged.

## Options
### A. Claim H is validated
- Pros: simple headline.
- Cons: false. The tracking bounds are config values that nothing checks.

### B. State the scope explicitly
- Pros: honest results. Makes clear what a later phase must add.
- Cons: none.

## Decision
B. In phase 1, a "valid" H means: **assuming the ego tracks the reference within ρ(τ), the objects
stayed within their predicted regions until H.** ē_v, ē_θ, q̄ and ρ0 are assumptions, not
validated.

## Consequences
- **Reporting:** T015 and T016 label the truth check "object-side validation" in every output and
  summary.
- **Separate validation:** the object bounds are validated on their own in T019.
- **Revisit:** when a controller or simulator is available. That is the point where the ego half
  can be measured.

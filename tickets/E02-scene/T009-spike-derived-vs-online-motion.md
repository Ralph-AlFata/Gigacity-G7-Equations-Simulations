# T009 — Derived ego motion vs online egomotion

**Type:** Spike
**Epic:** E02
**Status:** todo
**Depends on:** T007
**Timebox:** half a day

## Question
Are velocities, accelerations and yaw rates derived from poses (T007 step 7) close to the motion
the car itself recorded, i.e. the velocity and acceleration columns of online `egomotion`?

## Why it matters
- **For objects**, derived motion is the only motion we have.
- **For the equations**, speed and acceleration feed the bounds directly.
- **If they differ a lot**, derived object motion is unreliable too, and the method must change
  before the equations use it (decided 2026-10-31).

## Method
- [ ] Data check: on both clips, compare derived ego speed, longitudinal and lateral acceleration,
      and yaw rate (from each ego source) with the online columns, resampled to the 10 Hz grid.
- [ ] Data check: report mean, 95th percentile and max error per quantity, and plot them over time.
- [ ] If errors are large, try smoothing alternatives and compare them the same way:
  - a larger difference window;
  - Savitzky–Golay;
  - a constant-acceleration Kalman smoother.

## Hypotheses
1. Speed agrees within ~0.1 m/s; acceleration is noisy without smoothing.

## Findings
<…>

## Answer
<…>

## Follow-ups
<Update T007 step 7 with the chosen method.>

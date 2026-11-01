# T006 — Is the rig origin the rear-axle midpoint?

**Type:** Spike
**Epic:** E02
**Status:** todo
**Depends on:** T002
**Timebox:** 2 hours

## Question
Where is the rig frame's origin on the car: the rear-axle midpoint on the ground, or somewhere else?
And what are its axes (x forward, y left, z up)?

## Why it matters
- **Equations:** the PDF's bicycle model and tracking-error bound use the rear-axle midpoint as the
  ego reference point.
- **Ego shape:** the ego footprint is placed relative to that point. The old code assumed
  rear axle → front bumper = 4.688/2 + 1.312 m.
- **Consequence:** if the origin is elsewhere, every margin shifts by that offset.

## Method
- [ ] Documentation:
  - NVIDIA wiki, coordinate-system page;
  - columns of `vehicle_dimensions` (e.g. rear-axle-to-bumper distances);
  - the Alpamayo 1.5 model card / loader (which point its trajectories describe).
- [ ] Data check: `sensor_extrinsics` gives sensor positions in the rig frame. Their x/z values
      should match a car whose origin is the rear axle (cameras ahead of it, lidar on the roof).
- [ ] Data check: during a constant-radius turn, the rear axle moves with no sideways velocity
      (bicycle model), and other points do. Measure the lateral velocity of the rig origin.

## Hypotheses
1. The origin is the rear-axle midpoint projected to the ground, with x forward, y left, z up.

## Findings
<…>

## Answer
<…>

## Follow-ups
<Set `ego.rear_axle_offset` in the scene meta (zero if the hypothesis holds).>

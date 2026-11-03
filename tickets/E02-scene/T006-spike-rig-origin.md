# T006 — Is the rig origin the rear-axle midpoint?

**Type:** Spike
**Epic:** E02
**Status:** review
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
- [x] Documentation:
  - NVIDIA wiki, coordinate-system page;
  - columns of `vehicle_dimensions` (e.g. rear-axle-to-bumper distances);
  - the Alpamayo 1.5 model card / loader (which point its trajectories describe).
- [x] Data check: `sensor_extrinsics` gives sensor positions in the rig frame. Their x/z values
      should match a car whose origin is the rear axle (cameras ahead of it, lidar on the roof).
- [x] Data check: during a constant-radius turn, the rear axle moves with no sideways velocity
      (bicycle model), and other points do. Measure the lateral velocity of the rig origin.

## Hypotheses
1. The origin is the rear-axle midpoint projected to the ground, with x forward, y left, z up.

## Findings
Script: `scripts/spike_t006_rig_origin.py` (`uv run python scripts/spike_t006_rig_origin.py`, ~10 s), on
the `scene` data from T002. Numbers are b789fab8 / b4d979af. Both clips have the same vehicle:
length 4.688 m, width 1.999 m, height 1.439 m, wheelbase 2.730 m, `rear_axle_to_bbox_center` 1.312 m.

### Documented (NVIDIA)
- Wiki, *Coordinate Systems*: the rig frame section is only a figure ("Rig Frame (Vehicle
  Coordinate System)") with no caption. The text names neither the origin nor the axes. The dataset card
  doesn't document `vehicle_dimensions` either.
- SDK 0.2.3 (`calibration.py`) exposes `rear_axle_to_bbox_center` and `wheelbase` without explaining them.
- **NVIDIA's NCore converter for this dataset** (`NVIDIA/ncore`,
  `tools/data_converter/pai/converter.py`) builds the ego box **in the rig frame** as
  `centroid = (rear_axle_to_bbox_center, 0.0, height * 0.5)`, `dim = (length, width, height)`,
  `rot = 0`. The same box is used to filter the ego's own lidar points. That box only makes sense if the rig origin
  is the rear axle (x), on the car's centerline (y), at ground level (z = box bottom).
- Alpamayo: the model cards say trajectories are in the "ego vehicle coordinate frame", but not
  which point on the car. Left to T012 (see Follow-ups).
  (https://github.com/NVlabs/physical_ai_av/wiki, https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py,
  https://github.com/NVlabs/alpamayo)

### From the data
**1. Sensors against the NCore ego box.** If the origin is the rear axle on the ground, the box spans x −1.032 … +3.656,
y ±0.999, z 0 … 1.439 m, and the front axle sits at x = 2.730.
| Sensor | x | y | z | |
|---|---|---|---|---|
| front wide / tele cameras | +1.62 … +1.71 | ≈ −0.1 | +1.21 … +1.24 | inside: windshield top |
| cross left / right cameras | +2.26 … +2.30 | ±0.9 | +0.77 … +0.79 | inside: front fenders, just behind the front axle |
| rear left / right cameras | +1.82 … +1.88 | ±0.97 | +0.89 … +0.90 | inside: mirrors, at the box's side edge |
| rear tele camera | +0.28 … +0.30 | +0.39 … +0.43 | +1.27 … +1.29 | inside: rear window |
| lidar | +1.00 | −0.01 / +0.02 | +1.89 | 0.45 m above the roof (roof mount) |
All sensors sit where they would on that car, and the lidar is on a mount above the roof. If the
origin were at axle height (~0.35 m), the fender cameras would be at ~1.1 m and the lidar ~0.8 m above the roof.

**2. Lateral velocity in turns.** A point a distance `a` ahead of the rear axle slides sideways at
`v_lat = a × yaw_rate` (no-slip bicycle model). Using online egomotion velocities rotated into the body
frame, at speed > 2 m/s and |yaw rate| > 0.05 rad/s (turns of −67° / +56°, up to 0.39 / 0.25 rad/s):
| | b789fab8 | b4d979af |
|---|---|---|
| samples (1 s blocks) | 463 (6) | 210 (4) |
| fitted `a` (95% block bootstrap) | **−0.09 m** (−0.18 … −0.01) | **+0.05 m** (−0.24 … +0.05) |
| median \|v_lat\| | 0.041 m/s | 0.012 m/s |
| what a box-center origin would give, \|yaw rate\| × 1.365 | 0.354 m/s | 0.298 m/s |
Candidate origins would give `a` = 0 (rear axle), 1.31 (box center), 1.37 (wheelbase middle) or
2.73 (front axle). The fit is within 0.1 m of the rear axle and excludes the others by more than 1 m.
The origin moves almost straight, as a rear axle does.

**3. Ground height under the origin.**
| | b789fab8 | b4d979af |
|---|---|---|
| lidar ground plane at x = y = 0, 20 spins | median **+0.075 m** (−0.07 … +0.14) | median **+0.015 m** (−0.03 … +0.05) |
| bottom of labeled cars within 15 m | median **−0.023 m** (IQR −0.09 … +0.04) | median **+0.049 m** (IQR +0.00 … +0.10) |
The ground is at z = 0 within ~0.1 m, by two independent measures. An axle-height origin would put it at about −0.35 m.

**Axes.** Body-frame forward velocity is positive (median +4.96 / +2.00 m/s), so **x is forward**.
The left cameras have y > 0, so **y is left**. The lidar and cameras have z > 0 above the ground, so **z is up**.
The frame is right-handed.

### Hypotheses
1. *The origin is the rear-axle midpoint projected to the ground, with x forward, y left, z up:*
   **supported.** NVIDIA's own converter assumes it, and sensor placement, turn kinematics (within
   0.1 m in x) and ground height (within 0.1 m in z) all agree, on both clips.

## Answer
**The rig origin is the rear-axle midpoint on the ground: x forward, y left, z up.**
`ego.rear_axle_offset = (0, 0, 0)`, so ego poses (`egomotion.offline`) already describe the rear-axle point.
Ego footprint in that frame, as in NCore: center `(rear_axle_to_bbox_center, 0, height/2)` =
(1.312, 0, 0.720) m, size 4.688 × 1.999 × 1.439 m. So x spans −1.032 (rear bumper) … +3.656 (front bumper).
These values come from `vehicle_dimensions` per clip, not hard-coded.
Confidence: high for x and the axes (documentation in code plus an independent kinematic test). High
for z within ~0.1 m. Finer than that is limited by suspension, tyre load and road camber, and doesn't matter for planar H.

## Follow-ups
- E02 README and T007: the "rear-axle point" is the rig origin itself; there is no offset to apply.
- T013: `r_veh` (disk around the rear axle covering the footprint) uses the box above. The farthest corner from
  the rear axle is (3.656, ±0.999), so the radius is 3.79 m.
- T012: whether Alpamayo's trajectory describes the rig origin is still to confirm from its loader.
  It is trained on this dataset's egomotion, which describes the rig origin, so the offset is expected to be 0.

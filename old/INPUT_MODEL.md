# Obstacle input contract and bounded-motion replay

This extension fills missing motion estimates with an explicit reachable-position set. It does not infer unobserved positions or use future labels. The active-scene default retires old tracks explicitly; the original strict-history and all-seen bounded-motion modes remain reproducible through their saved configurations.

## Active-scene default

`inventory_policy=active_tracks`, `track_timeout_s=0.25`: only tracks whose last available observation is less than 0.25 s old enter the calculation. Both observation time and availability time must be <= t0. The timeout was chosen to match the existing maximum-gap setting; measured median intervals in both clips are about 0.1 s. It was not chosen to maximize the horizon.

Newly observed objects enter immediately, even with one sample, using the bounded-speed fallback below. When observations have a gap >= timeout, the current episode starts after the gap; old motion and shape history are not mixed into the returning episode. This prevents velocity estimates across disappearances or apparent identity discontinuities. Retired tracks have status `retired_timeout` and remain in the diagnostics. n_observed_tracks = n_active_tracks + n_retired_tracks. n_obstacles = n_active_tracks when estimation succeeds.

Observation timeout is a scene-membership convention, not proof of departure. Occluded or missed objects can still exist. H is conditional on the inventory known at its anchor. A future arrival is included at the next evaluation that can observe it, never retroactively in the earlier anchor. The audit splits new IDs from returning retired IDs and reports the first unmodeled arrival time. A hindsight-only inventory-limited prefix min(H, first future unmodeled arrival) is reported separately; it is not a causal operational deadline.

Use `examples/active_scene_config.json` for this policy. `examples/bounded_motion_config.json` retains all seen tracks and reproduces the preceding experiment described below. The main batch-analysis script now defaults to the active-scene configuration.

## Original equations retained

For elapsed time s = observation_age + tau:

- nominal object center = last_position + velocity_estimate * s
- beta = eps_position + eps_velocity * s + 0.5 * acceleration_bound * s^2
- rho = rho0 + qbar * tau
- margin = distance(reference, nominal_center) - ego_radius - rho - object_radius - beta

The solver checks positive margins continuously over a polyline prefix, conservatively subdividing uncertain intervals. H=0 means no positive modeled clearance at the starting instant. It does not establish collision. A numeric result is not proof the assumptions hold.

## What must be supplied

| Quantity | Source / requirement |
|---|---|
| Reference positions and times | Runpod exports, transformed in 3D to the clip anchor before projecting to XY; tau=0 prepended |
| Ego physical radius | Rear-axle enclosing disk from vehicle dimensions and axle offset |
| Ego tracking allowance | rho0 and qbar; still illustrative, not measured controller performance |
| Object position and age | Last available label at or before t0; original age preserved |
| Object physical radius | Maximum past labeled footprint radius plus configured padding; shape bound remains assumed |
| Object nominal velocity | Bounded endpoint estimate if usable; otherwise zero as the center of an all-direction speed envelope |
| Object uncertainty | Derived velocity-error bound or explicit absolute-speed envelope, below |
| Scope | Default: recently observed labeled tracks with logged retirements. Optional legacy: every previously observed track. Unseen objects, road edges, and unidentified obstacles are not covered |

## Usable history

Use available observations in the configured recent history window ending at the track's last observation. Require at least two samples, distinct increasing timestamps, and acceptable gaps. No future rows or rows with future availability enter the estimate.

For first/last measured positions y0,y1 separated by dt, use vhat=(y1-y0)/dt. If each position has error norm <= ep and acceleration norm <= A on this interval, the velocity error at its endpoint obeys:

    ||v(last) - vhat|| <= 2 ep / dt + A dt / 2

This follows by bounding the two endpoint measurement errors and the difference between average velocity and terminal velocity. The adapter uses the maximum of this derived bound and the configured velocity-error allowance. Future motion then uses the original beta equation. Injected position/velocity noise, if configured, is included in the bounded-motion allowances.

Two observations 0.1 seconds apart with ep=0.3 m and A=2 m/s^2 imply an error bound of 6.1 m/s, not the former fixed 0.5 m/s. Thus relaxing the observation count without updating uncertainty was insufficient for a defensible input construction.

The existing constant-velocity bound validity clock remains measured from the last observation. Unexpired estimates still reduce the global cap by their remaining validity.

## Missing, gapped, or expired motion estimates

Assume an explicit bound ||v(s)|| <= V on the object's center speed throughout the full elapsed interval. Then:

    possible_position(s) is inside B(last_measured_position, ep + V*s)

Represent this using the original equation with nominal velocity=0, eps_velocity=V, acceleration term=0. Zero nominal velocity does NOT assert the object is stationary: motion in every direction up to speed V is covered. Acceleration can change direction without violating this set if the speed bound continues to hold.

This is a different assumption from the expired constant-velocity error model. It covers the time from the last observation through the queried future interval; rebasing the age to zero would be incorrect. These objects do not inherit the expired CV-model validity cap. Reference duration, log duration, configured horizon, and remaining validity of other CV estimates still cap H.

`examples/bounded_motion_config.json` selects V=40 m/s (144 km/h) uniformly as an **exploratory, uncalibrated bound**, not a dataset fact or operational guarantee. Larger/smaller values are scientific assumptions to report, not knobs to tune until H becomes positive. In particular, a smaller class bound requires justification. No future labels were used to select V.

## Why numerical completeness may produce many zero horizons

Previously seen objects stay in scope. Their possible-position regions can grow to include the ego vehicle before the query time. Then m(0)<=0 and the original equation genuinely returns H=0 under these assumptions. We do not replace invalid values with zeros. Full inventory counts and the limiting track's age, uncertainty, and model are saved for inspection.

Empty inventories and malformed/nonfinite input data remain errors or unavailable; a missing scene is not free space. This extension resolves the short-history and stale-estimate cases in the current 12-timestamp batch, not every conceivable data failure.

## Reproduce the complete batch

Run from the project root, with the existing environment and data:

```bash
.venv/bin/python scripts/analyze_runpod_results.py \
  --input artifacts/outputs_20261031_003733 \
  --out results/new_bounded_motion_run \
  --config examples/bounded_motion_config.json
.venv/bin/python scripts/report_complete_horizons.py --results results/new_bounded_motion_run
```

Output directory must be new. No network access or GPU inference is involved. The per-track CSV records the estimator, history failure reasons, actual bounds, ages, and fallback uses for every sensitivity configuration. All five original sensitivity settings remain; the zero-growth diagnostic retains fallback speed uncertainty and is therefore not literally uncertainty-free in this mode.

Future labels are checked only after computing H, both through the full cap and separately through H. Newly appearing track IDs before H are reported. Label checks are discrete proxy diagnostics, not proof of continuous coverage. Even a positive conditional H does not establish complete-scene safety when an unseen actor appears.

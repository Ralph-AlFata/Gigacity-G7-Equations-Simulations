# Clearance-horizon experiment pipeline

Research prototype prepared for the deadline-aware multi-vehicle serving project, 2026-10-29.

**Purpose:** evaluate the disk-tube equations on recorded PhysicalAI Autonomous Vehicles scenes, obtain distributions of conditional clearance horizons, and identify which assumptions make the horizons short.

**Current validation status:** 34 tests pass locally. The user ran Alpamayo 1.5 on an A40 and returned 36 single-front-camera trajectories for two train clips. Local frame conversion and displacement metrics were checked. The active-scene adapter returns all 240 requested horizons (36 generated + 12 recorded references, five settings), logging each inclusion, retirement, and history reset. Numerical completeness is not empirical validation of the bounds: some horizons are zero, retrospective tube violations occur, and future arrivals need reevaluation. Start with [the latest active-scene results](results/alpamayo_every3s_active_scene/START_HERE.md) and [the input equations and assumptions](INPUT_MODEL.md). Earlier strict-history and all-seen runs remain preserved; see [VALIDATION.md](VALIDATION.md).

The default now uses `active_tracks` with a 0.25 s observation timeout, two-observation endpoint velocity, and a bounded-speed fallback for insufficient motion history. Use `--config examples/active_scene_config.json` to make this explicit. `examples/bounded_motion_config.json` preserves the all-previously-seen inventory; the older sensitivity configurations preserve strict-history rejection. All these bounds and the timeout are experimental assumptions. Details below describing history rejection refer to the legacy strict mode.

## 1. What to test first

Start with an **offline label replay**, without video detection or Alpamayo inference. The dataset provides `egomotion.offline`, `obstacle.offline`, and `vehicle_dimensions`; query feature presence for the selected revision rather than assuming every clip has them [1–3]. The first run answers:

> Given a recorded driving path, labeled surrounding agents, and explicitly chosen uncertainty bounds, how long do our equations retain positive clearance?

It does **not** yet answer how well Alpamayo drives, how accurately an onboard detector works, or whether a vehicle can safely wait for a remote server. Those need additional experiments.

The pipeline is:

1. Select eligible clip UUIDs from a fixed dataset revision/split.
2. Fetch only the two motion-label features and vehicle dimensions.
3. Normalize timestamps and transform obstacle centers into the ego path's fixed frame.
4. At each replay time, estimate agent velocity using only observations at or before that time.
5. Use the recorded future ego path as an explicitly labeled offline reference, or supply an available planned trajectory.
6. Evaluate clearance continuously along a piecewise-linear reference.
7. Save horizon statistics, failure reasons, proxy uncertainty-coverage diagnostics, and plots.

These calculations use the CPU. Videos are useful for auditing selected cases; they are not necessary to reconstruct metric positions already present in the labels. A camera-only detector/tracker would introduce another research problem before we know whether the geometry is useful.

## 2. Installation and a local smoke test

Use Python 3.11 or newer. From the extracted directory:

```bash
uv sync --extra dataset
uv run --extra dataset python -m unittest discover -s tests -v
uv run --extra dataset python -m horizonlab demo --out my_demo
```

The demo contains six deterministic synthetic scenes: lead braking, adjacent-lane disk overlap, pedestrian crossing, pulling away, oncoming motion, and a stationary roadside obstacle. Some paths intentionally intersect to test the equations; these are not representative driving statistics.

Outputs are in `my_demo/results/`. The included `demo_run/results/` is a completed example of this workflow.

## 3. First run on NVIDIA data

NVIDIA's dataset requires acceptance of its access conditions and HF authentication [1,2]. Use the account/environment already authorized for your project. Keep tokens in the normal HF credential mechanism; do not put them in this package or experiment outputs.

```bash
uv run --extra dataset hf auth login
uv run --extra dataset python -m horizonlab fetch --out data/pai_one_clip --n 1 --seed 7 --split test --split-name alpamayo_aligned
uv run --extra dataset python -m horizonlab run --data data/pai_one_clip --out results/pai_one_clip --config examples/sensitivity_config.json
```

Complete login in your local terminal; never paste a token into chat. The CLI does not automatically load `.env`. Prefer the normal HF credential store. Do not use `uv add ".[dataset]"`, which attempts to add this project as its own dependency.

Audit the first clip's actual schemas, frame metadata, time alignment, vehicle dimensions, estimated velocities, and initial margins before interpreting its horizons. After that succeeds, expand to five clips at the same commit (replace `COMMIT` with `resolved_revision` from the first manifest):

```bash
uv run --extra dataset python -m horizonlab fetch --out data/pai_five_clips --n 5 --seed 7 --split test --split-name alpamayo_aligned --revision COMMIT
uv run --extra dataset python -m horizonlab run --data data/pai_five_clips --out results/pai_five_clips --config examples/sensitivity_config.json
```

`fetch` selects uniformly from eligible clips in the chosen split. It records the requested revision, resolved dataset commit, SDK version, seed, UUIDs, chunks, and feature list in `download_manifest.json`. The installed SDK 0.2.3 reads named splits from `splits/<split_name>.parquet`; dependency declarations require at least that version. Explicit branch/tag revisions are resolved before constructing the SDK interface. Save the resolved revision and use `--revision COMMIT` for subsequent runs. A mutable default branch is not a reproducible experiment identifier.

For manually curated scenes, put one eligible UUID per line in `clips.txt` and replace `--n 5` with `--clip-ids clips.txt`. This keeps selection explicit. Changing `--split` may be needed if your supplied UUIDs belong to another split; the code does not silently cross splits.

Only label/calibration features are requested, but downloads are still organized into chunks. Five randomly chosen clips may occupy five different chunks. Do not interpret clip count as a download-size guarantee. Keep downloads outside your measured horizon-computation time.

If normalization fails, inspect the clip's `schema.json`, the retained raw parquet tables, and the manifest error. Unknown frames, duplicate track/time rows, missing columns, or missing dimensions are errors, not permission to guess. If `reference_frame` is absent, inspect the chosen revision and explicitly provide `--obstacle-frame rig` or `--obstacle-frame anchor`. NVIDIA's radar reader treats the offline boxes as rig-frame poses [4]; its NCore converter preserves per-row frame metadata [3]. A known metadata/override contradiction is rejected.

The code does not fetch videos automatically. For a small audit subset, NVIDIA's SDK can retrieve a camera feature separately:

```python
from physical_ai_av import PhysicalAIAVDatasetInterface
ds = PhysicalAIAVDatasetInterface(revision="YOUR_RECORDED_DATASET_COMMIT")
camera = "camera_front_wide_120fov"
ds.download_clip_features(["YOUR_SELECTED_CLIP_UUID"], [camera])
reader = ds.get_clip_feature("YOUR_SELECTED_CLIP_UUID", camera)
```

Inspect surrounding views when checking side/rear agents. The chosen front camera is an audit view, not a claim of complete coverage.

## 4. Mapping the equations to data

| Quantity | Construction in this prototype | What remains assumed |
|---|---|---|
| Ego rear-axle position | `egomotion.offline`: `x,y,z`, pose quaternion | Label accuracy; locally planar kinematics |
| Reference path | Recorded future ego XY, marked `recorded_future_oracle` | This is an offline counterfactual reference, not a live predictor |
| Obstacle position | `center_x,y,z`, transformed using the row's frame and frame timestamp | Machine-label accuracy and completeness |
| Track identity | `track_id` | Identity stability; missing/reappearing tracks are not automatically repaired |
| Obstacle velocity | Linear least-squares slope over the most recent 0.8 s of available track history | Velocity-error bound is an experiment input |
| Object footprint radius | Maximum past half-diagonal `0.5*hypot(size_x,size_y)`, plus explicit padding | Upright, planar footprint; bounded size; no future size lookup |
| Ego footprint radius | Vehicle dimensions and rear-axle-to-center offset | Same planar rectangular footprint assumption |
| Initial tracking radius | Initial reference/ego displacement plus configured allowance | Localization/control uncertainty allowance |
| `qbar` | Configured directly, or derived from speed/heading error bounds | Bounds must hold across the interval |
| `eps_position`, `eps_velocity`, `a_bound` | Explicit settings, optionally overridden by class | Not supplied as certified bounds by this dataset |
| Validity time | Explicit configuration, reduced by stale observation age | Chosen assumptions hold over that duration |

The canonical per-clip files are:

- `ego.csv`: `t_s,x_m,y_m,yaw_rad`.
- `agents.csv`: `t_s,available_s,track_id,x_m,y_m,radius_m,class_name`.
- `metadata.json`: frame conventions, ego radius, provenance, data quality, dimensions, and scenario.

All times use the same integer microsecond origin before conversion to seconds. Ego/obstacle streams are not independently reset to zero. Quaternions use XYZW order. Obstacle positions are transformed in 3D, then projected to XY. The radius model is still planar: hilly scenes, rolled/pitched objects, trailers with changing articulation, overpasses, or inaccurate boxes need separate treatment. `ego_max_abs_roll_pitch_deg` is retained for audit. A zero planar margin in a vertically separated scene is not evidence of physical collision.

The normalization explicitly uses `reference_frame_timestamp_us` when it is present. A box timestamp and the time defining its coordinate frame need not be identical [3]. Velocities are estimated **after** conversion into the fixed frame. Differentiating ego-relative positions would confuse relative velocity with absolute obstacle velocity.

## 5. Equations and numerical interpretation

For an object last observed `age` seconds before evaluation time, use:

```text
elapsed = age + tau
predicted_center = last_estimated_position + estimated_velocity * elapsed
beta = eps_position + eps_velocity * elapsed + 0.5 * a_bound * elapsed^2
ego_radius_at_tau = r_veh + rho0 + qbar * tau
margin = norm(reference_position - predicted_center) - ego_radius_at_tau - r_object - beta
```

This propagates stale observations instead of treating them as fresh. The object's validity interval also begins at its last observation, so the global cap is reduced accordingly.

For an ego box of length `l`, width `w`, whose center is `d` metres from the rear axle:

```text
r_veh = sqrt((l/2 + abs(d))^2 + (w/2)^2)
```

Using half the box diagonal around the rear axle would fail to enclose the whole car. The SDK exposes the required dimension/offset fields [5].

The horizon is the first loss of the **prefix** clearance condition, capped by reference duration, chosen experiment horizon, validity, and available log duration. Mathematically:

```text
H = min(T_cap, inf{tau in [0,T_cap] : min_k margin_k(tau) <= 0})
inf(empty set) = infinity
```

If the first boundary is H, strict clearance holds before H, not at a zero-margin endpoint. Initial disk overlap gives H=0, meaning no certificate from this bound, not a confirmed collision.

The numerical solver does not only test a grid. Between consecutive reference waypoints, relative nominal position is affine. Its minimum distance on an interval can be computed exactly as distance to a line segment. The required separation is nondecreasing, so:

```text
minimum_center_distance_over_interval - required_separation_at_interval_end
```

is a lower bound on margin throughout that interval. If positive, the interval is clear under the model. Otherwise, recursively subdivide. At the time-resolution limit, return the left endpoint conservatively. This also catches crossings between clear endpoints and tangential contact. It is numerical research code, not formally verified floating-point software.

`H_s` is the certified prefix found. `interval_end_s` has the following interpretation:

| Status | Meaning |
|---|---|
| `cap_reached` | Clear through the cap; the eventual first boundary was not observed |
| `initial_overlap_or_tolerance` | No initial positive margin beyond numerical tolerance |
| `boundary_bracket` | A nonpositive endpoint was found in the first unresolved interval |
| `resolution_limit` | That interval could not be certified at the chosen resolution; an actual boundary is not established |
| `insufficient_track_history` / `track_history_gap` | Velocity estimate could not be formed under the history policy |
| `stale_track_bound_expired` | A previously seen track outlived the validity assumption |
| `no_observed_agents` | No usable inventory; real replay does not interpret an empty table as free space |

The checker uses a piecewise-linear reference by definition. If a controller tracks a different curve between waypoints, its interpolation/tracking mismatch must be included in the ego tube.

## 6. What the sample configurations mean

`examples/sensitivity_config.json` is a sensitivity experiment, **not a collection of literature-validated noise bounds**. Its numbers are illustrative engineering settings. The experiments are:

1. An illustrative baseline.
2. Zero uncertainty/growth as a diagnostic lower-separation case.
3. Remove ego growth to isolate its cost.
4. Remove obstacle acceleration growth to isolate its cost.
5. Larger uncertainty and motion bounds as a stress test.

Do not select a smaller acceleration bound simply to get a longer horizon. First check whether future recorded motion remains inside the resulting tube. A zero-growth setting is particularly unlikely to be a defensible guarantee.

For the derived `qbar`, remove `qbar_mps` from an experiment and instead set `speed_error_bound_mps` and `heading_error_bound_deg`. The runner obtains a maximum reference segment speed over the capped interval and applies your formula. Use an absolute speed bound if reverse motion is allowed.

`class_overrides` maps exact class strings from the labels to per-class obstacle bounds. Inspect the actual classes first. Optional bounded position/velocity noise is reproducible from the seed, clip, time, and track; identical draws are reused across configurations. Synthetic injected noise is an experiment assumption, not a measured detector error distribution.

## 7. Reporting and deciding what to improve

The runner writes:

| Output | Use |
|---|---|
| `horizons.csv` | One row per clip/time/configuration, including H, cap, limiting agent, initial margin, qbar, age, and invalid-window reason |
| `summary.json` | Mean capped horizon, percentiles, zero/cap fractions, deadline coverage, proxy tube violations, and compute-time diagnostics |
| `by_scenario.csv` | Per-scenario window counts, means, and medians |
| `horizon_summary.png` | Empirical CDF and scenario comparison |
| `config_used.json` | Exact parameters used |

The mean is the mean **capped certified horizon**. `cap_reached` values are censored observations of the unknown first boundary; they do not mean that safety ends at 6.4 s. Cap causes are saved. Compare configurations on matched clips/anchors and report cap rates. Later portions of a 20 s clip have less observable future; use a fixed full-duration subcohort as an additional analysis if needed.

Report the median, 5th/10th percentiles, fraction with H=0, and fractions with H above 0.1/0.25/0.5/1/2 s. The thresholds in the example are reporting choices, not automotive safety standards. The summary gives both a conditional valid-window rate and an all-window rate: missing/invalid windows are not silently counted as successful deadlines. It also bootstraps per-clip mean horizons, because overlapping windows are correlated. If drive/session grouping is available, split and resample at that coarser level too.

Future labels are used only AFTER computing H, to measure whether known agents' later labeled centers fit inside their predicted tubes over the capped interval. The outputs count shape-bound violations and newly appearing, unmodeled track IDs before H. This is a sample-time consistency check against machine labels. It is not independent perception ground truth, continuous-time coverage, or a validated joint safety probability.

Recommended staged experiment:

| Stage | Scope | Decision enabled |
|---|---|---|
| Smoke test | 5 clips; audit frames, timestamps, dimensions, velocities, initial margins | Is extraction correct? |
| Pilot | 30–50 clips spanning lead braking, side-by-side traffic, junctions/pedestrians, turns, sparse roads, and dense traffic | Which term or assumption dominates? |
| Calibration | Separate clips for choosing error/growth envelopes; simulator/controller rollouts for tracking errors | Are the assumptions supported at the claimed coverage level? |
| Held-out evaluation | Frozen parameters and a larger diverse clip set; paired configuration comparisons | Are horizon distributions useful and repeatable? |
| Planned-reference replay | Export Alpamayo trajectories available at each decision time | Does the model's planned geometry change the horizon? |
| Closed-loop simulation | Actually track those references under disturbances and latency | Do the tracking bounds hold and can the full system meet deadlines? |

Real clip scenarios initially have `scenario: unclassified`. Assign audited scenario labels in each clip's `metadata.json` before claiming a stratified comparison. Random clip sampling supports a simple pilot; deliberate scenario balancing supports stress tests but changes the population represented by its mean.

The first useful complexity increase should follow the observed bottleneck:

- Frequent H=0 in side-by-side traffic even with zero growth: improve footprint geometry (multiple disks or oriented boxes), not the motion predictor. Account for heading uncertainty when moving disk centers away from the rear axle.
- Reasonable initial clearance but fast collapse under `0.5*a*tau^2`: investigate class/road-constrained or directional reachable sets, after validating what maneuver constraints are defensible.
- Strong difference when `qbar` is removed: calibrate the controller's tracking envelope; a constant linear tube may be too loose.
- Many stale/insufficient-track windows: solve the observation/track availability problem before interpreting horizon averages.
- Long H but poor tube coverage: the assumptions are too optimistic; longer horizons are not a successful result.

## 8. Use actual Alpamayo trajectories later

The default recorded-future reference intentionally uses hindsight. It isolates geometry and sensitivity and must never be presented as a causal deployed planner. Also, the difference between Alpamayo predictions and the human-driven recorded future is **planning disagreement**, not the tracking error of a controller that executed Alpamayo's plan.

To evaluate an exported plan, provide one JSON object per line:

```json
{"clip_id":"YOUR_CLIP_UUID","t0_s":3.0,"observation_cutoff_s":2.8,"available_time_s":3.0,"coordinate_frame":"clip_anchor","reference_point":"rear_axle","source":"alpamayo_export","tau_s":[0.0,0.1,0.2],"xy_m":[[30.0,0.0],[31.0,0.0],[32.0,0.0]]}
```

The coordinates above illustrate the schema only. Supply the actual plan with its full duration. `tau_s=0` refers to the evaluation time `t0_s`, after trimming/resampling an older plan as necessary. Do not move a delayed plan's timestamps forward to pretend it is new. Every reference must be available by `t0_s` and derived only from observations available before that time. Local model outputs must be transformed from their documented reference frame into the same clip anchor frame; do not concatenate local coordinates with world coordinates.

```bash
python -m horizonlab run --data data/pai_pilot --references references.jsonl --out results/alpamayo_plans --config examples/sensitivity_config.json
```

This interface consumes plans but does not run or optimize the Alpamayo model. The horizon computation does not require reconstructing steering/acceleration inputs when a geometric reference is already available. The bicycle/controller model becomes important for validating the tracking envelope.

## 9. Scope before using H as a serving deadline

The default implementation covers **recently observed labeled agents**, with a logged 0.25 s timeout. New tracks enter as soon as an available label exists; a track returning after a timeout starts a fresh motion-history episode. Timed-out tracks are retired from the current inventory, not asserted physically absent. Retirements, active counts, future first-seen and returning IDs are reported. No future labels enter an earlier prediction. At each new evaluation time the inventory is rebuilt; a real deployment must recheck its reference when observations change. The legacy all-seen policy is preserved in explicit configurations.

This does not represent every curb, wall, lane boundary, unknown region, or currently unseen road user. `retrospective_inventory_limited_H_s` truncates H at the first later unmodeled label arrival in the recording; it is a hindsight audit, not an online horizon or guarantee. Removing a track after a timeout changes the scope and can omit an occluded object. Report that limitation alongside larger active-scene horizons.

Machine labels may be produced with offline smoothing or future information. History-only finite differences do not make the underlying labels an online perception system. Calibrate a causal estimator separately when moving toward deployment. A sample maximum or percentile is not automatically a hard deterministic bound; a probabilistic version needs an explicit failure-probability interpretation across time and agents.

For a later serving study, convert a horizon measured at an explicit time into a remaining budget, subtracting only latency still to be incurred and a justified reserve. `communication_and_actuation_s` and `reserve_s` are subtracted from H in `usable_budget_s`; both default to zero. Already modeled observation age must not be charged twice. These fields do not implement a network, controller, or fallback policy.

Clearance of a current trajectory does not by itself establish a viable fallback at the deadline. Full system safety also needs obstacle completeness, road constraints, validated tracking/dynamics, and fallback feasibility. This prototype is intended to tell us whether the basic bound is useful enough to justify those next steps.

## 10. Primary sources inspected

1. [NVIDIA PhysicalAI Autonomous Vehicles dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles): access, feature availability, chunking, machine-label status, and release/split history.
2. [NVIDIA PhysicalAI AV developer kit](https://github.com/NVlabs/physical_ai_av) and [dataset reader](https://github.com/NVlabs/physical_ai_av/blob/main/src/physical_ai_av/dataset.py): feature-based downloads, metadata, and returned objects. Inspected repository tree commit: `87f0208b2da4058d00e5ac0d0404d8ef7beb417d`.
3. [NVIDIA NCore PhysicalAI converter](https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py): obstacle fields, class IDs, `reference_frame`, `reference_frame_timestamp_us`, and rear-axle-relative ego box.
4. [NVIDIA radar-reprojection schema](https://github.com/NVIDIA/radar-reprojection/blob/main/docs/SCHEMA.md) and [PhysicalAI adapter](https://github.com/NVIDIA/radar-reprojection/blob/main/examples/python/physical_ai_av_adapter.py): concrete obstacle field names and rig-to-world pose composition. This prototype does not reuse that adapter's central finite-difference estimator for online-style inputs.
5. [NVIDIA calibration classes](https://github.com/NVlabs/physical_ai_av/blob/main/src/physical_ai_av/calibration.py): length, width, rear-axle-to-bounding-box-center offset, and wheelbase.
6. [NVIDIA coordinate-system documentation](https://github.com/NVlabs/physical_ai_av/wiki/2.--Coordinate-Systems) and [egomotion documentation](https://github.com/NVlabs/physical_ai_av/wiki/4.-Egomotion-and-Motion-Compensation): anchor frame, pose fields, timestamp units, and motion compensation.
7. [NVIDIA Toronto AI Lab BEV construction](https://github.com/nv-tlabs/sil-wheel/blob/main/scripts/build_bev_data.py): corroborating obstacle schema and rear-axle/body-center distinction. Its nearest-observation display matching is not used by the causal-history estimator here.

When the broad dataset description and a particular reader differ on frames or packaging, inspect the selected revision's actual schema. The adapter follows feature metadata for packing and per-row frame metadata for transforms rather than hard-coding the dataset card's illustrative file tree.

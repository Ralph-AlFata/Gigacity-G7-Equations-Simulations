# Validation record

## Selected-clip fetch repair — 2026-10-30

The user's fetch completed six label/calibration files and normalized two clips, despite repeated `_AggregatedTqdm.set_postfix_str` callback errors. The saved manifest has no clip errors. Installed SDK source inspection located the incomplete progress wrapper in the SDK batch downloader.

The project now obtains chunk filenames from SDK feature metadata, deduplicates them, and uses the SDK single-file downloader with native Hub progress reporting. Skipped/failed required files remain errors. Explicit 40-character commit revisions no longer require a Hub API lookup, permitting cache-only replay of extraction. Branches and tags are still resolved before loading data.

- `HF_HUB_OFFLINE=1 uv run --offline --no-sync --extra dataset python -m unittest discover -s tests -v`: **19 tests passed**.
- Re-ran `fetch_dataset` for both selected clips at commit `39414de6f0ab867238ad6a0299fcd18b26081b42`, split `train`, split name `alpamayo_aligned`, entirely offline into a temporary directory. The regenerated `ego.csv`, `agents.csv`, `metadata.json`, and `schema.json` matched existing outputs byte for byte. Canonical finite-value, timestamp-order, duplicate-track/time, and radius checks passed.
- `b789fab8-41f1-4cf3-b7f8-a8e94939f67d`: 202 ego rows, 7,140 obstacle rows, 110 distinct track IDs across the clip.
- `b4d979af-254e-4bc0-afdd-3a36531f4540`: 202 ego rows, 53,109 obstacle rows, 653 distinct track IDs across the clip.
- Both clips span 20 seconds, have ego radius approximately 3.790 m, and drop no obstacle rows for being outside ego support. Distinct track totals are not simultaneous vehicle counts and do not validate tracking identity quality.

No additional data or dependencies were downloaded, and original clip outputs were preserved. A fresh network transfer through the replacement downloader has not been tested. This establishes cached extraction and structural checks, not scientific validation of bounds or scenes. No new video or horizon run was performed during this repair. Earlier entries below describe the state at their respective stages and are retained as history.

## Local macOS audit — 2026-10-30

- No `.git` directory is present. Live files were inspected; historical modifications cannot be reconstructed. Existing `demo_run` and `my_demo` outputs were preserved.
- `uv sync --extra dataset`: succeeded.
- Python 3.11.13; NumPy 2.4.6; pandas 3.1.0; SciPy 1.17.1; Matplotlib 3.11.2; physical_ai_av 0.2.3; PyArrow 26.0.0; huggingface_hub 2.4.0.
- The explicit writable copies for obstacle centers and quaternions were already present. They were preserved. A new regression forces non-copy DataFrame arrays to be read-only, checks the transformed center and shared timestamps, and verifies that input tables remain unchanged. Removing the copies in an in-memory variant reproduced the read-only failure.
- Related array paths were inspected: replay's in-place velocity perturbation operates on a newly computed array, and the remaining pandas-derived views are read rather than mutated. No additional array-copy changes were needed.
- SDK inspection demonstrated that an explicitly supplied branch/tag remained unresolved in SDK 0.2.3. `fetch_dataset` now resolves it through the HF API before constructing the SDK, and records both requested and resolved revisions. The new test uses mocked SDK/HF responses; it is not evidence of a real download.
- `uv run --extra dataset python -m unittest discover -s tests -v`: **17 tests passed**.
- `uv run --extra dataset python -m horizonlab demo --out results/audit_2026_10_30_demo`: **540 experiment-windows**, all computationally valid (six synthetic clips, 18 anchors each, five configurations). Summary CSV/JSON and the rendered plot were inspected.

### Fresh synthetic results

All references are `recorded_future_oracle`. Parameters remain illustrative and unvalidated. Fractions below use the 108 valid windows per configuration.

| Configuration | Mean capped H (s) | Median (s) | H=0 | Cap reached | Proxy tube violations |
|---|---:|---:|---:|---:|---:|
| illustrative_base | 3.817 | 4.509 | 17.59% | 67.59% | 0.97% |
| larger_bounds_stress | 3.598 | 4.000 | 18.52% | 57.41% | 0.00% |
| no_acceleration_growth_ablation | 3.939 | 5.000 | 17.59% | 69.44% | 3.07% |
| no_ego_growth_ablation | 3.824 | 4.575 | 17.59% | 67.59% | 0.97% |
| zero_growth_diagnostic | 3.962 | 5.000 | 17.59% | 69.44% | 3.27% |

The 1st, 5th, and 10th percentiles are zero in every configuration. No windows were invalid. Baseline threshold coverage at 0.1/0.25/0.5/1/2 seconds is respectively 80.56%/79.63%/79.63%/78.70%/69.44%. Full statuses, threshold coverage, and clip-bootstrap intervals are in `results/audit_2026_10_30_demo/results/summary.json`; per-window limiting tracks and initial margins are in `horizons.csv`. Every fixture contains only `agent_1`, so it is the limiting track for every uncapped window.

The adjacent-lane fixture gives zero H at all 18 anchors in every configuration. Even with zero growth its initial margin is -2.623 m: center separation is 3.6 m, ego rear-axle disk radius is about 3.723 m, and obstacle radius is 2.5 m. This demonstrates the disk approximation's conservatism; shrinking uncertainty cannot remove this overlap. Baseline has one additional zero-H lead-braking window. Stronger stress bounds add a zero-H pedestrian window.

Positive computed H does not validate the assumptions: the baseline tube misses some held-out synthetic positions. Cap-reached results are clear through the cap, not estimates of collision time. Eight stress windows stop at numerical resolution limits. The six scenes are designed engineering fixtures, not a random driving sample; clip resampling acknowledges overlapping-window correlation but does not make their confidence intervals population estimates.

### Real-data audit remains pending

No usable HF credential was visible to `huggingface_hub.get_token()`, either with the ordinary environment or with `.env` loaded. Only credential-presence booleans were printed. No real-data fetch, normalization, horizon calculation, or five-clip expansion was performed. Local login and dataset access acceptance are required before proceeding.

The installed SDK's source was inspected for split loading, feature-presence handling, metadata-driven chunk/member names, decoding, and dimensions. Official [coordinate documentation](https://github.com/NVlabs/physical_ai_av/wiki/2.--Coordinate-Systems) and [egomotion documentation](https://github.com/NVlabs/physical_ai_av/wiki/4.-Egomotion-and-Motion-Compensation) describe an anchor frame and microsecond timestamps. The [NVIDIA converter](https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py) retains obstacle frame IDs and frame timestamps. These checks support the adapter design; actual selected-revision data still need inspection. The supplied radar schema URL could not be retrieved during this audit.

Before interpreting one-clip results, inspect:

1. Resolved commit, selected UUID/chunk, split, seed, SDK version, feature presence, and the feature packaging metadata. A missing feature-presence column is currently not evidence of presence; verify retrieval and actual schemas.
2. Raw ego/obstacle dtypes and units, timestamp ranges/cadence, duplicates, shared origin, dropped out-of-support observations, and obstacle frame timestamps versus observation times.
3. Per-row frame IDs, hand-checked 3D-to-anchor transforms, rig/rear-axle origin, and roll/pitch against the planar approximation.
4. Actual dimensions, rear-axle offset, resulting ego radius, obstacle box sizes/radii, and class identifiers.
5. Fixed-frame past-only velocity fits, availability cutoffs, stale inventory, history gaps, and finite speed ranges. Offline labels are not causal onboard measurements.
6. Initial margin decomposition for limiting obstacles, especially zero-H windows; distinguish physical geometry from added uncertainty and observation-age growth.
7. Invalid reasons, full capped distributions, deadline coverage with all windows in the denominator, tube/shape violations, and newly appearing unmodeled tracks. Missing labels do not imply free space.

Expand to five clips only after this audit succeeds, retaining the same resolved revision. A larger scenario-diverse study should follow that smoke test, separating sampling/calibration/evaluation clips and accounting for clip/drive correlation. No scheduler, inference, controller validation, fallback-feasibility result, or operational safety guarantee has been established.

## Earlier author-provided record

- Date: 2026-10-29.
- Python: 3.12.14; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; Matplotlib 3.10.8.
- `python -m unittest discover -s tests -v`: **15 tests passed**.
- Tests cover an analytic closing case, collision between clear endpoints, tangential contact, initial overlap, capped/no-obstacle behavior, observation-age growth, bound monotonicity, frame/timestamp conversion, future-data exclusion, stale tracks, and invalid-window reporting.
- `python -m horizonlab demo --out demo_run`: completed **540 experiment-windows** (six synthetic clips, 18 anchors per clip, five configurations).
- `python -m compileall -q horizonlab tests`: passed.
- Summary plot rendered and visually inspected.

The real Hugging Face download and NVIDIA SDK path has **not** been executed here. No real-world horizon statistics, calibrated error bounds, Alpamayo performance measurements, or closed-loop validation are included.


## Complete bounded-motion input replay

- Re-read original supplied equations; unchanged disk geometry, tracking tube, margin, and conservative interval solver.
- Added explicit bounded_motion adapter and configuration; original strict-history settings preserved.
- Usable past histories use an endpoint velocity and derived error >= 2 ep / dt + A dt / 2.
- Single-observation, gapped, and expired histories use an absolute-speed reachable disk, retain their original observation age, and remain in the inventory. V=40 m/s is exploratory and uncalibrated.
- Constant-velocity validity caps still apply to those estimates. Speed fallback has a separate all-elapsed-time absolute-speed assumption.
- 29 unit/regression tests pass, including analytic speed-envelope horizon, age propagation, endpoint measurement/acceleration error bound, future-availability rejection, gap fallback, speed monotonicity, mixed validity caps, and end-to-end replay of a stale singleton.
- Replayed all 36 generated and 12 recorded references under five settings: 240 numerical horizons, no unavailable rows, all observed tracks retained. Outputs: results/alpamayo_every3s_complete.
- Baseline generated references: 6 positive, 30 zero, mean 0.125109 s, median 0, maximum 1.376562 s. The original sparse-valid mean is not directly comparable.
- Baseline future-label checks through H: 8 center-tube violations / 881 checks across 36 model references, with three windows encountering newly appearing track IDs. Samples reuse labels, so counts are correlated. No complete-scene safety or calibrated coverage claim.
- Baseline sample 0 was also replayed at fallback speed 20 and 60 m/s; all 12 horizons matched the 40 m/s run. This narrow sensitivity check does not calibrate the speed bound or prove monotonicity across model switches. Configs and results: results/fallback_speed_sensitivity.
- All computation local; no downloads or new GPU jobs. START_HERE.md, the timestamp matrix, statistics, limiting-object table, per-object diagnostics, and implementation hashes make the result inspectable.


## Active-scene lifecycle and final Alpamayo batch

- Default policy now includes available labels newer than a 0.25 s timeout, explicitly retires older tracks, and resets velocity/shape history after a gap >= timeout. New single-observation tracks are included using the existing bounded-speed fallback. Legacy configurations remain unchanged.
- 34 tests pass, including arrival/retirement timing, delayed availability, exact timeout boundary, reset on reappearance, and separate retrospective new/returning-track diagnostics.
- Updated default synthetic demo completed 540/540 numerical experiment-windows in results/active_scene_default_demo; these are engineering checks, not dataset statistics.
- Final real-data run: results/alpamayo_every3s_active_scene. 36 generated + 12 recorded references, five settings, 240/240 numerical horizons. Active/retired counts balance and every active object is represented.
- Baseline generated horizons: mean 0.454167 s, median 0.166406 s, min 0, max 1.507031 s; 21 positive / 15 zero; no cap-reached values. Three results terminate at the numerical resolution limit rather than an observed boundary. Recorded references: mean 0.468676 s, median 0.165443 s.
- Baseline future-label checks through H: 122 violations / 17842 checks across the model references, correlated across trajectory samples. 18/36 windows encounter first-seen or returning unmodeled tracks before H. These arrivals are not fed into earlier predictions. The hindsight inventory-limited prefix averages 0.091049 s and is explicitly distinct from the causal conditional H.
- Timeout sensitivity on sample 0: 0.15 s gives mean 0.413607 s (one of 12 horizons differs); 0.25 and 0.5 s both give 0.450260 s. Settings were not selected by maximizing H. See results/active_scene_timeout_sensitivity.
- Inactive is not physically absent. This changes inventory scope from the all-seen experiment. Larger H is not evidence of improved real-world safety; bounds, missed detections, and future entry remain modeling limitations. No new downloads or GPU inference.

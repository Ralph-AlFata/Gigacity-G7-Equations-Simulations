# T001 — Data scope for phase 1

**Type:** Decision
**Epic:** E01
**Status:** done
**Depends on:** none

## Context
Phase 1 exists to run the horizon equations offline and get statistics on H: given a planned
trajectory, how long can it be followed safely. Research on the dataset (2026-10-31) raised four
choices: which ego poses to use, which cameras, which clips, and which split.

## Options and decisions

### 1. Ego poses: `egomotion` (online) vs `egomotion.offline`
- `egomotion` is what NVIDIA's Alpamayo 1.5 loader feeds the model, and exists for 100% of clips.
- `egomotion.offline` is post-processed and exists for 97.4% of clips. Our existing copy is 10 Hz
  starting at 0, while NVIDIA's docs describe egomotion as ~100 Hz absolute timestamps.

**Decision:** keep both, and build every downstream step so it can run on either.
**Revisit (T004):** we do not yet know how the two differ (rate, smoothness, offsets, and which
one the obstacle labels were placed with). Choosing blindly could put obstacles in a slightly
wrong place relative to the ego, which directly biases H. Decide after comparing them on real data.

### 2. Status of the offline data
**Decision:** treat the offline labels (ego poses, obstacles, calibration) as ground truth for
phase 1. The goal is to assess the equations, not perception quality.
**Consequence:** H measured here is conditional on perfect-but-offline knowledge of the scene;
label errors and online availability delays are out of scope until a later phase.

### 3. Clips
**Decision:** download data only for the current two clips (b789fab8, b4d979af) for now.
The download code itself is general: any clip, any feature, fetched on request (T002), and clips
are chosen from the index without downloading (T003). Scaling up is then a clip list, not new code.

### 4. Train/test splits
**Decision:** ignore splits. We are not training or evaluating the model's generalization; we
measure how long a given trajectory stays safe.

### 5. Cameras and video
**Decision:** record locations and frame timestamps for all 7 cameras (needed for a full scene
rebuild and later Alpamayo runs with its default 4 cameras), but do not store the MP4s. Fetch a
video on demand when we want to watch it or run inference.

### 6. Lidar (added 2026-10-31, from E02)
**Decision:** fetch `lidar_top_360fov` for our clips. Objects are in the scene only when they exist
in the labels **and** the lidar hits them (E02, T008), so lidar is required input, not optional.
Radar stays out (absent for our two clips, ~25% coverage overall).

## Consequences
- Downstream code must take the ego source (`online` / `offline`) as a parameter until T004 closes.
- Statistics from two clips describe these two scenes only, not driving in general.

# T004 — `egomotion` vs `egomotion.offline`

**Type:** Spike
**Epic:** E01
**Status:** done
**Depends on:** T002
**Timebox:** half a day

## Question
How do the online and offline ego poses differ for our two clips, and which one should the
scene geometry use (and which one should Alpamayo's input use)?

## Why it matters
Obstacle labels are given relative to the car at each label's timestamp. Placing them in the
ground frame uses an ego pose. If we use a pose source that differs from the one the labels were
produced with, obstacles land in slightly wrong places, which directly biases the margins and H.
Alpamayo was trained on the online `egomotion`, so its input may need a different choice.
Recorded in T001 as "keep both, revisit".

## Method
- [x] Documentation: NVIDIA wiki pages on egomotion; dataset card notes on `.offline` features.
- [x] Data check: sample rate, time range, start pose, and columns of each (the online one has
      velocity/acceleration/curvature columns according to the SDK).
- [x] Data check: position and heading difference between the two over each clip.
- [x] Data check: transform obstacles to the ground frame with each source and compare how still
      parked objects stay (smaller spread = consistent with the labels).

## Hypotheses
1. Offline is a smoothed, resampled (10 Hz) version of online with near-identical positions.
2. The obstacle labels were produced with the offline poses, so parked objects stay stiller
   with offline than with online.

## Findings
Script: `scripts/spike_t004_egomotion.py` (`uv run python scripts/spike_t004_egomotion.py`), on the
`scene` data from T002. Numbers are b789fab8 / b4d979af.

### Documented (NVIDIA)
- Wiki, *Egomotion and Motion Compensation*: egomotion is "~2,224 poses per clip, ~100Hz",
  `timestamp` is "Absolute timestamp in microseconds", in the anchor frame. It holds position,
  quaternion, velocity, acceleration and curvature.
- Wiki, *Coordinate Systems*: the anchor origin is the rig origin at t=0, and yaw is zero at t=0,
  but "any pitch and roll are not normalized, and estimated with respect to gravity".
- Dataset card: `.offline` features (added in 26.03, 97% of clips) are "offline-optimized" by
  "additional signal processing" for NuRec reconstruction. The method, rate and frame are not
  given. Neither source says which poses the obstacle labels were made with.
  (https://github.com/NVlabs/physical_ai_av/wiki, https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

### From the data
| | online `egomotion` | `egomotion.offline` |
|---|---|---|
| Columns | timestamp, x y z, qx qy qz qw, vx vy vz, ax ay az, curvature | timestamp, x y z, qx qy qz qw |
| Rate | 100 Hz from −0.19 s to 20.20 s, then a **sparse tail** (110–150 ms steps, up to 2 s) to 108.5 s / 140.1 s | 10 Hz, 202 rows, 0 … 20.000 s |
| Timestamps | own 100 Hz clock | rows 0 and 20 s, plus **the lidar spin starts** (169 / 154 of 200 exactly equal, the rest within a fraction of a ms) |
| Pose at t=0 | origin, yaw 0, **pitch −3.80° roll +3.40°** / pitch +0.60° roll +0.72° (gravity-referenced, as documented) | **exact identity**: the anchor is the full rig pose at t=0, so pitch/roll are *not* gravity-referenced, unlike the docs |
| Velocity columns | in the **anchor** frame (match d(x,y,z)/dt to 0.003 m/s; rig frame: 3.4 m/s off) | none (derive, T009) |

Online re-anchored to its own rig pose at t=0, against offline at the offline timestamps over 0–20 s:
- Horizontal position difference: median 0.26 / 0.08 m, max (at the end) **0.68 / 0.25 m**.
  Vertical: max 0.84 / 0.11 m.
- Heading difference: median 0.40 / 0.21°, max 0.65 / 0.81°. Path length: 106.5 vs 107.1 m and
  42.9 vs 43.1 m.
- **Clock offset:** online speed at t + lag best matches offline speed at t for a lag of
  **+50 / +40 ms**, i.e. online timestamps run about 40–50 ms late relative to offline.
- Offline is *not* smoother at the same sample times: the median 3rd difference of position is
  14.6 / 8.9 mm for offline against 2.9 / 1.0 mm for online.

Parked objects (≥ 20 sightings, horizontal span < 3 m under either source; 67 / 314 tracks).
Each sighting is placed in the ground frame with the pose interpolated at its `timestamp_us`.
The metric is the RMS of horizontal position around the track mean:
| Pose source | b789fab8 median RMS | b4d979af median RMS | Best time offset |
|---|---|---|---|
| offline | **0.164 m** | **0.212 m** | 0 ms / −10 ms |
| online | 0.238 m | 0.280 m | +20 ms / +40 ms |
| online resampled at the offline times (control) | 0.244 m | 0.282 m | +20 ms / +40 ms |
| online, at its best offset | 0.188 m | 0.228 m | |
- Offline gives the smaller spread for 40 / 67 and 249 / 314 tracks. Yaw spread is the same for
  both (0.6° / 1.3°).
- The control rules out sampling rate as the reason: online at 10 Hz is as spread as at 100 Hz.
- The labels are time-aligned with offline (best offset ≈ 0). Online needs its clock shifted by
  +20–40 ms, the same lag the speed match found. Even shifted, it stays ~10–15% worse than offline.
- The remaining ~0.16–0.21 m is label noise plus slow movers caught by the 3 m filter. It is the
  floor for T007's "parked objects stay still" check.

### Hypotheses
1. *Offline is a smoothed, 10 Hz resample of online with near-identical positions:* **refuted.**
   It is 10 Hz on the lidar clock, but it is a different solution, not a resample. It has a
   different anchor orientation, differs by up to 0.7 m and 1° over 20 s, has a 40–50 ms clock
   offset, and is not smoother.
2. *The labels were produced with the offline poses, so parked objects stay stiller with offline:*
   **supported** on both clips, by spread (−25–30%) and by timing (best offset 0 ms for offline).

## Answer
**Scene geometry uses `egomotion.offline`.** That covers placing obstacles in the ground frame, the
ego pose on the 10 Hz grid, and everything the equations read. The obstacle labels agree with it in
space (25–30% stiller parked objects) and in time (no clock offset), and it shares the lidar's
clock. Interpolating it at label times is fine: the 10 Hz control shows the rate costs nothing.
**Alpamayo's input stays online `egomotion`**, because that is what the model was trained on and
what NVIDIA's loader feeds it (T012). Online's 40–50 ms clock offset and its gravity-referenced
anchor mean the two sources must never be mixed inside one scene. An Alpamayo trajectory is
expressed in the rig frame at its decision time, and is placed in the scene with the *offline*
pose at that time.
Confidence: high that offline is the consistent source for these two clips (same direction on
every check, both clips). Medium that it generalizes, since this is two clips.

## Follow-ups
- T001 decision 1 updated with this choice.
- E02 README, T007 and T016: "both ego sources until T004" replaced with offline for geometry. The
  `<ego_source>` path level stays, so an online scene can still be built for comparison (T010 `--compare`).
- T007: the grid covers 0–20 s (offline covers it; online's dense part covers −0.19–20.2 s).
  The "parked objects stay still" threshold (0.5 m) is comfortably above the 0.16–0.21 m floor found here.
- T007 / T009: the offline anchor is **not gravity-levelled** (it is the full rig pose at t=0, with
  pitch up to 3.8° here). That doesn't matter for relative geometry, but "planar x, y, yaw" and
  vertical acceleration would carry the initial tilt. Option: level the offline frame with online's
  t=0 pitch/roll. To decide in T007.
- T009: online `vx, vy, vz` / `ax, ay, az` are in the anchor frame. Compare them after the +40–50 ms shift.
- T012: Alpamayo outputs are placed with the offline pose at the decision time (see Answer).

# Pipeline guide

What the pipeline does so far (tickets T001–T008 and T010), how to run it, and what each part produces.
The README has the command reference. This guide explains how the pieces fit together.

**Status (2026-11-04):**
- E01 (raw data) is done.
- E02 (scenes) is done through T008. T010 (the Rerun viewer) is in review.
- Still to come:
  - T009: derived ego motion vs online egomotion.
  - E03: references, the H solver and the replay.

---

## 1. The big picture

The goal is to measure the clearance horizon **H** on PhysicalAI-AV clips with Alpamayo
trajectories. The equations will consume a stream of **scenes**. A scene says, every 0.1 s, where the
ego car is and which objects are around it and actually visible. So far the pipeline builds that stream.

```
 Hugging Face dataset (pinned revision 39414de)
          │
          │  horizon-catalog  → pick clips (reads index files only)
          │  horizon-fetch    → download the features of those clips
          ▼
 data/raw/<clip_id>/          raw, unmodified files + manifest.json (SHA-256 of each)
          │
          │  horizon-scene build
          ▼
 data/scene/<clip_id>/<ego_source>/
     meta.json        how it was built (rules, inputs, code version, report)
     ego.parquet      one row per tick: ego pose + motion
     objects.parquet  one row per (tick, object): pose, motion, relative pose, lidar visibility
          │
          ▼
 horizon-scene view → Rerun: the scene in 3D next to the real camera videos, boxes drawn on them
          │
          ▼
 (next) H solver / replay (E03)
```

---

## 2. Setup (once)

```sh
uv sync                      # creates .venv with all dependencies
```

Put a Hugging Face token that has access to `nvidia/PhysicalAI-Autonomous-Vehicles` in `.env`:

```
HF_TOKEN=hf_...
```

The token is read and never printed. If neither `.env` nor the environment has one, it falls back
to `hf auth login`.

---

## 3. Running everything, end to end

```sh
# 1. (optional) look at the dataset
uv run horizon-catalog summary
uv run horizon-catalog show b789fab8-41f1-4cf3-b7f8-a8e94939f67d

# 2. see what would be downloaded for our two clips, then download it
uv run horizon-fetch --clips-file clips_selected_train.txt --features scene --dry-run
uv run horizon-fetch --clips-file clips_selected_train.txt --features scene

# 3. build the scenes (offline + online ego poses, ~20 s per clip and source)
uv run horizon-scene build --clips-file clips_selected_train.txt --ego-source both

# 4. look at it: the scene in 3D next to the camera videos (fetch the videos once, ~95 MB)
uv run horizon-fetch --clips-file clips_selected_train.txt \
    --features camera_front_wide_120fov:video camera_cross_left_120fov:video camera_cross_right_120fov:video
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d

# 5. run the tests (~1 min; the real-data tests run only if data/raw/ is filled)
uv run pytest
```

Our two clips are listed in `clips_selected_train.txt`:
- `b789fab8-41f1-4cf3-b7f8-a8e94939f67d`
- `b4d979af-254e-4bc0-afdd-3a36531f4540`

Each one is about 20 s long.

---

## 4. The commands

### `horizon-catalog`: which clips exist? (T003)

This command reads only the dataset's index files (`clip_index`, `feature_presence`, `data_collection`).
It never downloads clip data.

| Subcommand | What it does |
|---|---|
| `summary` | clip count, how many clips have each feature, every metadata column and its values |
| `show <clip>` | one clip: chunk, metadata, which features exist and which are already in `data/raw/` |
| `select` | filter clips and write a clip list to `data/clip_lists/<name>.txt` |

```sh
uv run horizon-catalog select --require scene \
    --where country=Germany,France "hour_of_day>=20" --limit 50 --seed 0 --name eu_night
```

- `--require` uses the same names as `--features` below.
- `--where` takes `column<op>value`. The operators are `= != < <= > >=`, and conditions combine with AND.
- The clip list records, at the top in `#` lines, the command, the revision and the date.

### `horizon-fetch`: download features (T002)

This command downloads only what you ask for, cut to each clip and otherwise unmodified. It writes
the files to `data/raw/<clip_id>/` and records each file in `data/raw/manifest.json` (source,
SHA-256, size, rows, columns).

- Re-running skips files whose hash still matches, so it only ever adds.
- It asks before downloading more than 2 GB (`--confirm-gb N`, or `--yes`).
- `--dry-run` lists the files and their sizes and downloads nothing.
- `--remove` deletes features again, for example a video you looked at once.

`--features` takes any mix of:

| Form | Example |
|---|---|
| feature | `egomotion`, `obstacle.offline` |
| feature:member | `camera_front_wide_120fov:video` |
| group | `labels`, `calibration`, `camera`, `lidar`, `radar` |
| group:member | `camera:frame_timestamps` |
| preset | `default`, `scene`, `alpamayo`, `videos` ([`presets.toml`](../horizon_pipeline/raw/presets.toml)) |

Presets:
- `default`: the labels, the calibration, and the camera frame timestamps. A few MB per clip.
- `scene`: `default` plus the lidar, ~350 MB per clip. **This is what scene building needs.**
- `alpamayo`: online egomotion and the 4 camera videos that Alpamayo's loader reads.
- `videos`: all 7 camera videos.

### `horizon-scene build`: raw files → scene (T007, T008)

```sh
uv run horizon-scene build --clips-file clips_selected_train.txt --ego-source both
uv run horizon-scene build --clips b789fab8-41f1-4cf3-b7f8-a8e94939f67d --ego-source offline --no-lidar
```

| Option | Meaning |
|---|---|
| `--clips` / `--clips-file` | which clips |
| `--ego-source offline\|online\|both` | which ego poses to build from. **Offline is the one used for H**. Online is for comparison. |
| `--raw`, `--out` | input and output roots (default `data/raw`, `data/scene`) |
| `--no-lidar` | skip the lidar step (fast). The visibility columns stay empty. |

For each clip and source it prints a summary:
- the ticks and the object rows;
- the duplicates removed;
- a parked-car stability check, which shows whether the transforms are consistent;
- the observed share by class and by distance band.

It refuses to build if two rows of the same track less than 20 ms apart disagree.

### `horizon-scene view`: watch a scene next to the real video (T010)

```sh
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d              # opens the viewer
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d --compare \
    --save artifacts/scene/b789fab8.rrd                                            # or save a recording
uv run rerun artifacts/scene/b789fab8.rrd                                          # ...and open it later
```

What you see, all on one timeline (`clip_time`, play, pause and scrub):

| Panel | Shows |
|---|---|
| left: `scene` (3D, follows the ego) | the ego box, heading and path; every existing object box; the lidar spin nearest the tick |
| right: the three cameras | the real video frames, with the scene's boxes drawn on them |

- **Box colors:** vehicles are blue, VRUs orange and static objects green when the object is
  `in_scene`. Gray means it exists but the lidar does not see it. Clicking a box shows its label:
  track ID, class, speed and lidar points. `--labels` draws all labels at once.
- **Drawing boxes on the video:**
  - Each box is moved to the video frame's own time (position linear, rotation slerp), then from
    world to rig to camera.
  - It is projected with the camera's f-theta lens model, following NVIDIA's SDK formula
    (`camera.py`).
- **What to look for:**
  - Boxes should sit on their cars and people, and move with them.
  - Gray boxes should be hidden behind something.
- **Options:**

  | Option | Effect |
  |---|---|
  | `--compare` | adds the online ego source's box and path in red |
  | `--all-frames` | 30 Hz video instead of one frame per tick |
  | `--cameras` | picks cameras (`--cameras` with no name gives 3D only) |
  | `--lidar-every N` | changes how much lidar is drawn |
  | `--no-lidar` | leaves the lidar out |
  | `--labels` | draws every box's label (by default, shown when a box is selected) |
  | `--fade-beyond M` | fades objects more than M metres from the ego (display only; the scene is unchanged) |

- **Cost:** a saved recording is ~150–250 MB and takes 25–55 s. Most of that is lidar.

---

## 5. What a scene is and how it is built

### Conventions

- **Time grid:** 10 Hz starting at clip time 0 (`t0_us`). Tick `k` is at `t_s = 0.1·k`.
- **Ego reference point:** the rear-axle midpoint on the ground, with x forward, y left and z up.
  This is the dataset's rig origin itself (T006). The ego footprint comes from `vehicle_dimensions`
  and is stored in `meta.json`.
- **World frame:** anchored at the rig at clip time 0. All positions are in metres, angles in radians.
- **Ego poses:** `egomotion.offline` by default. T004 showed that the labels match it in both space and
  time. Online egomotion has a 40–50 ms clock offset and a sparse tail.
- **Truth:** NVIDIA's offline auto-labels (`obstacle.offline`) are taken as ground truth for which
  objects exist and where. The lidar only decides whether we could see them.

### The transform chain

```
p_rig   = R_ext · p_sensor + t_ext         sensor → rig      (sensor_extrinsics.offline)
p_world = R(t)  · p_rig    + x(t)          rig at time t → world (ego pose, interpolated)
p_ego   = R(t_k)ᵀ · (p_world − x(t_k))     world → ego frame at tick t_k
```

- Poses are interpolated: position linearly, rotation by slerp.
- Quaternions are stored as `(qx, qy, qz, qw)`.
- The code is in `horizon_pipeline/scene/frames.py`.

### Build steps (`horizon_pipeline/scene/build.py`)

1. **Load:**
   - the ego poses (dense part only);
   - the object sightings, where each row is (time, track ID, box relative to the rig at that time);
   - the vehicle dimensions.
2. **De-duplicate:** drop rows of one track less than 20 ms apart. They must agree (size identical,
   position within 0.2 m), otherwise the build stops.
3. **Sightings to world:** use the ego pose at each sighting's own timestamp.
4. **Resample each track onto the grid:**
   - **Lifetime:** a track exists from its first to its last sighting. There is no extrapolation.
   - **Gaps:** gaps of up to 0.5 s are bridged. A longer gap splits the track into separate runs.
   - **Interpolation:** position is linear, rotation uses slerp, size is linear.
5. **Derived motion:** compute velocity, acceleration and yaw rate by central differences, separately
   for the ego and for each run of each object.
6. **Relative pose:** express every object in the ego frame at that tick (`rel_x`, `rel_y`, `rel_z`, `rel_yaw`,
   `distance`).
7. **Lidar observability (T008, `lidar.py`):**
   - **Choose the spin.** Out of the lidar's 100 ms spins, choose the one whose sweep crossed the
     object nearest the tick (within ±50 ms).
   - **Place the box.** Put the object's box where it was **at that crossing moment**, so a moving
     car is in the right place.
   - **Place the points.** Place every lidar point with the ego pose at its own timestamp. The points
     are raw, not motion-compensated (T005).
   - **Count.** Count the points inside the box, enlarged by 0.2 m on each side and ignoring the
     bottom 0.2 m, which is the road.
   - **Decide.** 5 or more points means `observed`, and `in_scene = observed`.
8. **Write** `meta.json`, `ego.parquet` and `objects.parquet`.

### The output files

The full column list is in [`tickets/E02-scene/README.md`](../tickets/E02-scene/README.md). These are
the ones you will use most:

| File | Key columns |
|---|---|
| `ego.parquet` (1 row per tick) | `tick, t_s, x, y, z, yaw, vx, vy, speed, a_long, a_lat, yaw_rate, has_history` |
| `objects.parquet` (1 row per tick × existing track) | `tick, t_s, track_id, class_group, x, y, yaw, length, width, height, vx, vy, speed, rel_x, rel_y, rel_yaw, distance, lidar_points, in_scene` |
| `meta.json` | `rate_hz, t0_us, n_ticks`, ego dimensions and footprint, class groups, `rules`, input SHA-256s, code version, `build_report` |

- `class_group` is one of `vehicle`, `vru` (people, riders, animals), `static` or `unknown`.
- Rows where an object exists but the lidar does not see it are **kept**, with `in_scene = false`, so
  you can inspect why it was excluded. **The equations should read only `in_scene` rows.**

### Reading a scene in Python

```python
import json
import pandas as pd

d = "data/scene/b789fab8-41f1-4cf3-b7f8-a8e94939f67d/offline"
meta = json.load(open(f"{d}/meta.json"))
ego = pd.read_parquet(f"{d}/ego.parquet")          # 201 ticks
objects = pd.read_parquet(f"{d}/objects.parquet")

# The scene at one tick: timestamp, track ID, position.
tick = 50
scene = objects[(objects.tick == tick) & objects.in_scene]
print(ego.loc[tick, ["t_s", "x", "y", "yaw", "speed"]])
print(scene[["t_s", "track_id", "class_group", "rel_x", "rel_y", "distance"]])

# Stream every tick in order (this is what the H equations will consume).
for tick, rows in objects[objects.in_scene].groupby("tick"):
    ...
```

### Current numbers (offline poses)

| | b789fab8 | b4d979af |
|---|---|---|
| Tracks | 110 | 653 |
| Object rows observed by the lidar | 90.6% | 89.5% |
| Object rows observed at 4–20 m | 99.9% | 97.7% |

- **Close range:** objects closer than ~3.2 m sit in the lidar's blind zone.
- **The near misses on b4d979af** are occlusion in a crowded parking lot.
- **Many unseen rows at long range** have 1–4 points, so the 5-point threshold matters there.

---

## 6. Repository map

```
horizon_pipeline/
  env.py              reads the HF token from the environment or .env
  raw/
    source.py         talks to the Hugging Face dataset at the pinned revision
    registry.py       features.csv: feature names, groups, members
    presets.toml      named feature selections (default, scene, alpamayo, videos)
    fetch.py          resolves --features and downloads per clip
    store.py          data/raw layout and manifest.json
    catalog.py        clip index + metadata queries (no downloads)
    cli.py            horizon-fetch
    catalog_cli.py    horizon-catalog
  scene/
    frames.py         Poses (interpolation), frame transforms, yaw, slerp
    build.py          grid, de-dup, tracks, resampling, derived motion, tables, write
    lidar.py          spin decoding, point placement, sweep timing, box counts, in_scene
    checks.py         parked-object stability check
    camera.py         f-theta projection, camera poses, video decoding
    view.py           the Rerun view (3D scene, camera frames, boxes on the video)
    cli.py            horizon-scene (build, view)
scripts/              one-off spike scripts that print the evidence quoted in tickets
  spike_t004_egomotion.py     online vs offline egomotion
  spike_t005_lidar.py         lidar format, frame, timing, motion compensation
  spike_t006_rig_origin.py    is the rig origin the rear axle?
tests/                pytest; synthetic tests always run, real-data tests need data/raw/
tickets/              the project tracker (epics, tickets, decisions, evidence)
old/                  the previous pipeline, reference only
data/                 not in git: raw/, scene/, clip_lists/
artifacts/            not in git: saved Rerun recordings
```

Run a spike script with `uv run python scripts/spike_t005_lidar.py [clip_id ...]`. Each one
prints the numbers recorded in its ticket.

---

## 7. Tests

```sh
uv run pytest                          # everything (~1 min)
uv run pytest tests/test_scene.py      # one file
uv run pytest -k lidar                 # by name
```

| File | What it checks |
|---|---|
| `test_registry.py`, `test_fetch.py`, `test_store.py`, `test_catalog.py` | feature names and presets, download planning, manifest and hashing, catalog filters |
| `test_raw_data_checks.py` | fetched raw files match the old copies in `data/pai_selected_train/` |
| `test_scene.py` | synthetic clips with known answers: exact frames, offset clocks, gaps, lifetimes, duplicates |
| `test_lidar.py` | box counting, per-point placement, sweep timing, and an end-to-end fake lidar with a moving car |
| `test_view.py` | f-theta projection equals NVIDIA's SDK; boxes land where they should in a camera; interpolation between ticks; a synthetic clip with a video recorded to `.rrd` and read back |
| `test_scene_real.py` | the real clips: grid and columns, parked cars stay still, no impossible jumps, lidar columns, near objects observed |

Real-data tests are **skipped** (not failed) when `data/raw/` is missing.

---

## 8. How work is organised

- Every change is a ticket in `tickets/`. The tracker table is in `tickets/README.md`.
- Ticket types:
  - Task: code.
  - Spike: a question answered with evidence.
  - Decision: a choice and its reasons.
  - Experiment: numbers.
- One ticket = one branch `feature/T-NNN-slug` off `main` = one PR. The status goes
  `todo → in-progress → review → done`.
- Claims about the dataset cite NVIDIA's docs or a data check, recorded in the ticket.

---

## 9. Known limits and open questions

- **Visibility uses the lidar, not the cameras.**
  - The lidar sees 360°. Alpamayo only uses front-facing and cross cameras.
  - The lidar sits on the roof at 1.9 m, so it also sees over parked cars that block a camera at
    0.8 m. The viewer shows such cases.
  - Whether H should count only objects a camera could see is still open.
- **`lidar_min_points = 5` is a parameter.** It is stored in `meta.json`. Tune it by looking at the
  viewer.
- **The camera overlay ignores the rolling shutter.** The sensor reads top to bottom over 30 ms.
- **Derived ego motion is not yet checked** against the vehicle's own measurements (T009).
- **Edge-tick timing:** in the first and last 0.1 s of a clip, the lidar spin can be up to ~97 ms from
  the tick, because there is no spin on one side.

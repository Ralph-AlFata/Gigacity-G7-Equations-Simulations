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
          │  horizon-reference recorded
          ▼
 data/reference/<clip_id>/<ego_source>/<source>/
     meta.json          source, frame, step size, inputs
     reference.parquet  one row per (plan tick, sample, step): the planned rear-axle path
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

# 5. the recorded-path references: at each tick, where the car actually drove afterwards (~1 s)
uv run horizon-reference recorded --clips-file clips_selected_train.txt --ego-source both

# 6. run the tests (~1 min; the real-data tests run only if data/raw/ is filled)
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
- the duplicates removed, and any that needed the range allowance (offset over 0.2 m);
- a parked-car stability check, which shows whether the transforms are consistent;
- the observed share by class and by distance band.

It refuses to build if two rows of the same track less than 20 ms apart disagree.

### `horizon-scene view`: watch a scene next to the real video (T010)

```sh
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d              # build (or reuse), then open
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d --compare \
    --save artifacts/b789fab8.rrd                                                  # or only write a file
uv run rerun artifacts/b789fab8.rrd                                                # ...and open it later
```

- **Build first, then open.** The command builds the whole recording first, printing each source
  as it finishes, then opens the finished file in Rerun. The full clip is there at once; nothing
  streams in.
- **Reuse.** Recordings are kept as `artifacts/scene/<clip_id>/<ego_source>-<options>.rrd`. The next
  `view` with the same options opens that file immediately, as long as it is newer than the scene.
  Rebuilding the scene makes it stale. `--rebuild` forces a new build, for example after changing
  the viewer code.

What you see, all on one timeline (`clip_time`, play, pause and scrub):

| Panel | Shows |
|---|---|
| left: `scene` (3D, follows the ego) | the ego box, heading and path; every existing object box; the lidar spin nearest the tick |
| left, below: three plots (T022) | the ego's speed (m/s), longitudinal and lateral acceleration (m/s²) and yaw rate (°/s) over the whole clip; the playhead marks the current tick. With `--compare`, the other ego source is overlaid |
| right: the three cameras | the real video frames, with the scene's boxes drawn on them |

- **Box colors:** vehicles are blue, VRUs orange and static objects green when the object is
  `in_scene`. Gray means it exists but the lidar does not see it. Clicking a box shows its label:
  track ID, class, speed and lidar points. `--labels` draws all labels at once.
- **Ego label:** the ego box always shows its speed and longitudinal acceleration at the current
  tick, e.g. `ego 7.1 m/s, a_long +0.7 m/s²` (T022). `?` at the first tick(s), where the causal motion
  doesn't exist yet (T009). The scene has one value per 0.1 s tick, so between ticks the viewer
  shows the latest tick at or before the playhead.
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
  | `--rebuild` | builds the recording again even if a current one exists |
  | `--save FILE.rrd` | writes the recording to FILE and does not open it |

- **Cost:** a recording is ~150–250 MB and takes 25–55 s to build, once. Most of that is lidar (~12 s for
  b789fab8); each camera takes ~3.5 s.

### `horizon-reference recorded`: the path the car actually drove (T011)

```sh
uv run horizon-reference recorded --clips-file clips_selected_train.txt --ego-source both
```

- **What it writes:** `data/reference/<clip>/<ego_source>/recorded/`, a `reference.parquet` and a
  `meta.json`. It needs the scene to be built first.
- **What a reference is:** the plan made at each tick t, as a list of steps 0.1 s apart. Every source
  (this one, Alpamayo later) uses the same format:

  | Column | Unit | Meaning |
  |---|---|---|
  | `tick` | int | scene tick the plan was made at (t) |
  | `sample` | int | sample index (0 when a source gives one trajectory) |
  | `step` | int | 0, 1, 2, ... |
  | `tau_s` | s | time after t, `step × 0.1` |
  | `x, y, z` | m | rear-axle position, in the scene's world frame |
  | `yaw` | rad | heading, in the scene's world frame |
  | `speed` | m/s | speed at this step |

- **The recorded source:** step k of the plan made at tick t is the scene's ego at tick t + k. So
  the plan at tick 0 runs to the end of the clip (201 steps, 20 s), and the plan at the last tick
  is a single step. A 201-tick clip gives 201·202/2 = 20 301 rows.
  - This is hindsight, so it counts as environment and may use the future.
  - Speed is the scene's speed at t + k. The scene has none at tick 0, so that one value is
    tick 1's speed (listed in `meta.json` under `speed_filled_ticks`).
- **Checks on reading:** `read_reference` (in `horizon_pipeline/reference/format.py`) refuses a file
  with missing columns, NaNs, non-integer ticks or steps, steps that are not 0, 1, 2, ... in order,
  a plan that appears twice, or `tau_s` ≠ `step × step_s`.

```python
from horizon_pipeline.reference.format import read_reference
ref, meta = read_reference("data/reference/b789fab8-41f1-4cf3-b7f8-a8e94939f67d/offline/recorded")
plan = ref[ref["tick"] == 50]          # the plan made at 5.0 s: where the car went from there
```

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
2. **De-duplicate:** drop rows of one track less than 20 ms apart. They must agree (same class, size identical,
   position within 0.2 m + 0.5% of the range, T020), otherwise the build stops.
3. **Sightings to world:** use the ego pose at each sighting's own timestamp.
4. **Resample each track onto the grid:**
   - **Lifetime:** a track exists from its first to its last sighting. There is no extrapolation.
   - **Gaps:** gaps of up to 0.5 s are bridged. A longer gap splits the track into separate runs.
   - **Interpolation:** position is linear, rotation uses slerp, size is linear.
5. **Derived motion:** compute velocity, acceleration and yaw rate from the current and earlier ticks
   only (T009), separately for the ego and for each run of each object:
   - velocity and yaw rate over the last 0.1 s;
   - acceleration as the change of velocity over the last 0.4 s;
   - NaN at a run's first tick (first two for acceleration), where nothing earlier exists.
6. **Relative pose:** express every object in the ego frame at that tick (`rel_x`, `rel_y`, `rel_z`, `rel_yaw`,
   `distance`).
7. **Lidar observability (T008, `lidar.py`):**
   - **Choose the spin.** Out of the lidar's 100 ms spins, choose the latest one whose sweep crossed
     the object at or before the tick, 0–100 ms earlier. The reading is causal, like a live sensor
     (T021). Before the first crossing an object has no reading and is not observed.
   - **Place the box.** Put the object's box where it was **at that crossing moment**, so a moving
     car is in the right place.
   - **Place the points.** Place every lidar point with the ego pose at its own timestamp. The points
     are raw, not motion-compensated (T005).
   - **Count.** Count the points inside the box, enlarged by 0.2 m on each side and ignoring the
     bottom 0.2 m, which is the road. Points recorded after the tick don't count: they haven't
     been measured yet.
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

## 6. How the viewer works (`horizon_pipeline/scene/view.py`, `camera.py`)

The viewer only **displays** the scene. It never changes it, and it decides nothing about which
objects count.

### Build, then open

1. **Read** the scene files (`meta.json`, `ego.parquet`, `objects.parquet`), plus the raw lidar,
   videos and calibration for display.
2. **Log** every source into one recording, one after another:
   - the ego and the boxes (under 1 s);
   - the lidar (~12 s);
   - each camera (~3.5 s each).
3. **Write** the recording to `artifacts/scene/<clip_id>/<ego_source>-<options>.rrd`. It goes to a
   `.partial` file first and is renamed only when complete.
4. **Open** the finished file in Rerun. The whole clip is loaded at once.
5. **Reuse** the file next time, as long as it is newer than the scene files. A change to what a
   recording contains bumps `RECORDING_VERSION` in `view.py`, which is part of the hash, so old
   recordings are not reused. The `<options>` part is
   a hash of the options, so different options get different files.

### What each piece is made from

| Entity | From | Notes |
|---|---|---|
| `world/ego` | `ego.parquet` | one transform per tick; the box is the footprint in `meta.json`, so it sits on the rear axle; its label is the speed and a_long at that tick |
| `plots/ego/<quantity>` | `ego.parquet` | `speed`, `a_long`, `a_lat`, `yaw_rate` (in °/s); ticks without a value yet are left out (T022) |
| `world/obstacles/<track_id>` | `objects.parquet` | one box per row; cleared at ticks where the track does not exist |
| `world/lidar` | raw spins | the spin whose **middle** is nearest the tick, each point placed with the ego pose at its own time, 1 point in 10 |
| `cameras/<cam>/image` | `video.mp4` + `frame_timestamps` | the frame nearest each tick, at that frame's own time, half resolution |
| `cameras/<cam>/boxes` | `objects.parquet` + calibration | the same boxes, projected into the camera |

### How a box gets onto the video

1. **Move to the frame's time.** Take the frame's timestamp, which can be up to ~11 ms off the tick.
   Interpolate the ego pose and every box between the two neighbouring ticks: position linear,
   rotation slerp, the same as the scene build.
2. **Make the box edges.** Each box edge is sampled at 16 points, so the fisheye curvature shows.
3. **Transform** each point from world to rig (inverse ego pose), then from rig to camera (inverse
   `sensor_extrinsics`). The camera frame has z forward, x right and y down.
4. **Project.** Use the f-theta lens model:
   - θ is the angle from the optical axis;
   - `angle_to_pixeldist_poly(θ)` gives the radius in pixels from the principal point;
   - the direction is that of (x, y).
   This is NVIDIA's SDK formula, and a test checks the two match.
5. **Cut at the field of view.** Points beyond `max_angle` (77° on the wide cameras) or behind the
   camera are dropped. Each edge is cut there, not wrapped around.

### Reading the labels

A label reads `<track_id> <class> <speed> m/s <N> pts`.
- **Speed** is the object's derived speed at that tick.
- **N** is T008's lidar count. It comes from the latest spin whose sweep crossed **that object** at
  or before the tick (T021), with the box placed at the crossing moment. `?` means no crossing yet.
- **The drawn dots won't match N exactly.** The 3D view draws the spin whose *middle* is nearest the
  tick, thinned to 1 point in 10.

### Limits

- **Rolling shutter is not modeled.** The sensor reads top to bottom over 30 ms, so a fast crossing
  object can look a little off near the top or bottom of the image.
- **Close objects look tall.** A box 1 m from a camera shows its top far above the visible roofline:
  the box includes the mirrors, and a real roof curves inward. That's geometry, not an error.

---

## 7. Trying new clips

The whole pipeline runs on any clip that has the inputs. To try new clips, from picking them to
checking them:

```sh
# 1. Pick clips at random among those that have everything the build and the viewer need.
uv run horizon-catalog select \
    --require scene camera_front_wide_120fov:video camera_cross_left_120fov:video camera_cross_right_120fov:video \
    --limit 3 --seed 20261104 --name random3_20261104

# 2. Download them (~1 GB for 3 clips, mostly lidar). Check the size first with --dry-run.
uv run horizon-fetch --clips-file data/clip_lists/random3_20261104.txt \
    --features scene camera_front_wide_120fov:video camera_cross_left_120fov:video camera_cross_right_120fov:video

# 3. Build the scenes (~20 s per clip and source with lidar).
uv run horizon-scene build --clips-file data/clip_lists/random3_20261104.txt --ego-source both

# 4. Look at each one.
uv run horizon-scene view --clip b6cb8b12-8b36-49c2-8609-63c4e2958e28
```

The clips picked on 2026-11-04 with seed 20261104:

| Clip | Where and when | Platform |
|---|---|---|
| `b6cb8b12-8b36-49c2-8609-63c4e2958e28` | Germany, January, 03:00 (night) | hyperion_8.1 |
| `ca823746-c767-48d9-bc0d-008ce50f6497` | Croatia, August, 19:00 | hyperion_8.1 |
| `23958e60-dbf9-41fa-b927-53960a114276` | United States, January, 00:00 (night) | **hyperion_8** (our first two clips are 8.1) |

**What to check on a new clip:**

- **The build summary.**
  - The parked-car position RMS should be about 0.2 m: the transforms agree with themselves.
  - The observed share at 4–20 m should be above 95%.
  - No unknown classes.
  - The build must not stop on a duplicate.
- **The viewer:**
  - boxes sit on their cars and people in all three cameras, and move with them;
  - parked cars stay still in 3D;
  - gray boxes are hidden behind something.
- **What would point to a problem:**
  - Boxes offset the same way in every frame: calibration, or a different camera model.
  - Boxes that trail behind moving objects: timing.
  - Many gray boxes in the open: the lidar step.


## 8. Repository map

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
  reference/
    format.py         the reference format: columns, validation, read/write
    recorded.py       the recorded-path reference (the ego's own future)
    cli.py            horizon-reference
scripts/              one-off spike scripts that print the evidence quoted in tickets
  spike_t004_egomotion.py     online vs offline egomotion
  spike_t005_lidar.py         lidar format, frame, timing, motion compensation
  spike_t006_rig_origin.py    is the rig origin the rear axle?
tests/                pytest; synthetic tests always run, real-data tests need data/raw/
tickets/              the project tracker (epics, tickets, decisions, evidence)
old/                  the previous pipeline, reference only
data/                 not in git: raw/, scene/, reference/, clip_lists/
artifacts/            not in git: saved Rerun recordings
```

Run a spike script with `uv run python scripts/spike_t005_lidar.py [clip_id ...]`. Each one
prints the numbers recorded in its ticket.

---

## 9. Tests

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
| `test_reference.py` | the reference format (round trip, each kind of bad file refused) and the recorded path on a synthetic ego |
| `test_reference_real.py` | the real clips: every step of every recorded plan lies exactly on the ego's track |

Real-data tests are **skipped** (not failed) when `data/raw/` is missing.

---

## 10. How work is organised

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

## 11. Known limits and open questions

- **Visibility uses the lidar, not the cameras.**
  - The lidar sees 360°. Alpamayo only uses front-facing and cross cameras.
  - The lidar sits on the roof at 1.9 m, so it also sees over parked cars that block a camera at
    0.8 m. The viewer shows such cases.
  - Whether H should count only objects a camera could see is still open.
- **`lidar_min_points = 5` is a parameter.** It is stored in `meta.json`. Tune it by looking at the
  viewer.
- **The camera overlay ignores the rolling shutter.** The sensor reads top to bottom over 30 ms.
- **Online egomotion can be much longer than the clip.** For ca823746 the online `egomotion` file
  spans −0.2 s to 140.2 s (offline: 0 to 20 s), so its online scene, and its recorded reference,
  have 1 402 ticks. The scene builder does not cut the grid to the clip's 20 s yet.

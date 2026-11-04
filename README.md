# Alpamayo horizon pipeline

Measures the clearance horizon H on PhysicalAI-AV clips with Alpamayo trajectories, to decide
the inference-time budget. Work is tracked in [`tickets/`](tickets/README.md). `old/` holds the
previous pipeline, for reference only.

## Setup

```sh
uv sync
```

Put a Hugging Face token with access to `nvidia/PhysicalAI-Autonomous-Vehicles` in `.env`
(`HF_TOKEN=…` or `HUGGING_FACE_KEY=…`). It is read, never printed.

## Fetching raw data (`horizon-fetch`)

Downloads features only when asked for, filtered to each clip and otherwise unmodified, into
`data/raw/<clip_id>/`, and records them in `data/raw/manifest.json`. The dataset is pinned to
commit `39414de6f0ab867238ad6a0299fcd18b26081b42`.

```sh
# What would be fetched, with sizes. Downloads nothing.
uv run horizon-fetch --clips-file clips_selected_train.txt --features scene --dry-run

# Fetch it. Asks for confirmation above 2 GB (--confirm-gb, or --yes to skip the prompt).
uv run horizon-fetch --clips-file clips_selected_train.txt --features scene

# One camera's video for one clip, then delete it again.
uv run horizon-fetch --clips b789fab8-41f1-4cf3-b7f8-a8e94939f67d --features camera_front_wide_120fov:video
uv run horizon-fetch --clips b789fab8-41f1-4cf3-b7f8-a8e94939f67d --remove camera_front_wide_120fov:video
```

`--features` (and `--remove`) take any mix of:

| Form | Example | Means |
|---|---|---|
| feature | `egomotion`, `camera_front_wide_120fov` | the feature, all its ZIP members |
| `feature:member` | `camera_front_wide_120fov:video` | one ZIP member |
| group | `labels`, `calibration`, `camera`, `lidar`, `radar` | every feature in that `features.csv` directory |
| `group:member` | `camera:frame_timestamps` | that member of every feature in the group |
| preset | `default`, `scene`, `alpamayo`, `videos` | a named list in [`presets.toml`](horizon_pipeline/raw/presets.toml) |

An unknown name prints the valid ones. Features a clip does not have (per `feature_presence`)
are skipped and listed as `absent` in the manifest. Re-running skips files whose SHA-256 still
matches the manifest, so it only adds.

Layout:

```
data/raw/
  manifest.json                     dataset revision, SDK version; per clip: chunk, feature_presence,
                                    files (source path, SHA-256, size, rows, columns), absent, not_fetched
  <clip_id>/
    clip_metadata.json              clip_index, feature_presence and data_collection rows
    <feature>.parquet               labels and calibration
    <feature>/<member>.<ext>        camera, lidar and radar ZIP members
```

ZIP members are streamed one at a time through the `physical_ai_av` SDK, so a chunk ZIP is never
downloaded whole. The SDK also keeps the extracted member in `~/.cache/huggingface/hub/subsetcache/`.

## Browsing clips (`horizon-catalog`)

Answers "which clips exist and what do they have?" from the dataset's index files only
(`clip_index`, `metadata/feature_presence`, `metadata/data_collection`). It never downloads clip data.

```sh
# Clip count, coverage per group and feature, and every metadata column with its values.
uv run horizon-catalog summary

# Chunk, metadata, and per feature: present in the dataset, and what is stored in data/raw/.
uv run horizon-catalog show b789fab8-41f1-4cf3-b7f8-a8e94939f67d

# 50 random clips (reproducible with --seed) that have everything `scene` needs, recorded in
# Germany or France at night, written to data/clip_lists/eu_night.txt.
uv run horizon-catalog select --require scene \
    --where country=Germany,France "hour_of_day>=20" \
    --limit 50 --seed 0 --name eu_night

# ...which the fetcher takes as is.
uv run horizon-fetch --clips-file data/clip_lists/eu_night.txt --features default --dry-run
```

- `--require` takes the same selections as `horizon-fetch --features`. A clip qualifies when it has
  every feature they name.
- `--where` takes `column<op>value` with `=`, `!=`, `<`, `<=`, `>`, `>=` on any metadata column
  that `summary` lists. With `=` or `!=` the value may be a comma list, and `null` matches missing
  values (e.g. `radar_config=null`). Conditions combine with AND.
- Clip lists start with `#` lines recording the command, the dataset revision and the date.

## Building scenes (`horizon-scene`)

Turns a clip's raw labels into a **scene**: every 0.1 s tick, the ego state and every object that
exists then, all in one world frame and also relative to the ego's rear axle. The schema is in
[`tickets/E02-scene/README.md`](tickets/E02-scene/README.md).

```sh
# Both clips, with offline and online ego poses (the H pipeline uses offline; online is for comparison).
uv run horizon-scene build --clips-file clips_selected_train.txt --ego-source both
```

Output, per clip and ego source:

```
data/scene/<clip_id>/<ego_source>/
  meta.json          grid (t0_us, n_ticks), ego dimensions and footprint, class groups, rules,
                     SHA-256 of the raw inputs, code version, build report
  ego.parquet        one row per tick: pose, yaw, derived velocity / acceleration / yaw rate
  objects.parquet    one row per (tick, track) where the track exists: world pose, size, derived
                     motion, pose relative to the ego, sighting age, lidar points in its box and
                     in_scene (= observed by the lidar)
```

It needs `egomotion[.offline]`, `obstacle.offline`, `vehicle_dimensions`, `sensor_extrinsics.offline`
and the lidar in `data/raw/<clip_id>/` (the `scene` preset). The lidar step decodes every spin
(~20 s per clip and ego source); `--no-lidar` skips it and leaves those columns empty. The build
stops if two rows of a track less than 20 ms apart disagree.

## Viewing scenes (`horizon-scene view`)

Shows a built scene in Rerun: the scene in 3D on the left, the camera videos on the right with
the scene's boxes drawn on them. One timeline, `clip_time`, drives all of it.

It first builds the whole recording into `artifacts/scene/<clip_id>/<ego_source>-<options>.rrd`
(~25 s for b789fab8, ~55 s for b4d979af), then opens it, so the full clip is there at once. The
next time it reuses that file as long as it is newer than the scene. `--rebuild` builds it again.

```sh
# The camera videos first (the three Alpamayo uses, ~95 MB for both clips).
uv run horizon-fetch --clips-file clips_selected_train.txt \
    --features camera_front_wide_120fov:video camera_cross_left_120fov:video camera_cross_right_120fov:video

# Build the recording (or reuse it) and open it.
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d

# Or write it to a file of your choice without opening it (~150-250 MB), and open it later.
uv run horizon-scene view --clip b789fab8-41f1-4cf3-b7f8-a8e94939f67d --compare --save artifacts/b789fab8.rrd
uv run rerun artifacts/b789fab8.rrd
```

- **Box colors:** boxes are colored by class group (vehicle blue, vru orange, static green) when
  `in_scene`, and gray when the object exists but the lidar does not see it.
- **Lidar:** the lidar shows the spin nearest each tick, thinned to every 10th point
  (`--lidar-every`, `--no-lidar`).
- **Cameras:** `--cameras` picks the cameras (none for 3D only). `--all-frames` shows all 30 Hz
  instead of the frame nearest each tick.
- **Ego sources:** `--compare` adds the other ego source's box and path in red.
- **Labels and distance:** labels show when you select a box. `--labels` draws them all.
  `--fade-beyond 30` fades objects more than 30 m from the ego.

## Tests

```sh
uv run pytest
```

`tests/test_raw_data_checks.py` compares `data/raw/` with the old copies in
`data/pai_selected_train/`. It is skipped until the data has been fetched.

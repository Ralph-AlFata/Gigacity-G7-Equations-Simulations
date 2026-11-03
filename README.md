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

## Tests

```sh
uv run pytest
```

`tests/test_raw_data_checks.py` compares `data/raw/` with the old copies in
`data/pai_selected_train/`. It is skipped until the data has been fetched.

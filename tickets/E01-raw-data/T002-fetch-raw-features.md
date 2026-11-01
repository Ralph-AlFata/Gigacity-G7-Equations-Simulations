# T002 — General on-request feature fetcher

**Type:** Task
**Epic:** E01
**Status:** todo
**Depends on:** T001

## Goal
One command that downloads any feature of the dataset for any list of clips, only when asked,
stores it under `data/raw/<clip_id>/` filtered to that clip and otherwise unmodified, and records
it in a manifest. Nothing is hard-coded to our two clips or to a fixed feature list.

## How the dataset is packed (from `features.csv` at the pinned commit)
36 features, each stored per chunk of ~100 clips:
| Group | Features | Chunk file | Per-clip content |
|---|---|---|---|
| labels | `egomotion`, `egomotion.offline`, `obstacle.offline` | ZIP | one parquet member per clip |
| calibration | `camera_intrinsics(.offline)`, `sensor_extrinsics(.offline)`, `lidar_intrinsics.offline`, `vehicle_dimensions` | parquet | rows of ~100 clips, filtered by clip ID |
| camera (7) | `camera_<name>` | ZIP | members `video` (mp4), `frame_timestamps`, `blurred_boxes` |
| lidar (1) | `lidar_top_360fov` | ZIP | member `pointclouds` |
| radar (16) | `radar_<name>` | ZIP | member `scans` |

So every feature except calibration can be fetched clip by clip (one ZIP member, streamed);
calibration chunk files are small and get filtered.

## Inputs → Outputs
- Inputs: clip IDs (arguments or a file such as `clips_selected_train.txt`); what to fetch
  (feature names, `feature:member`, or a named preset); HF token from `.env`.
- Outputs:
  ```
  data/raw/
    manifest.json
    <clip_id>/
      <feature>.parquet                       labels and calibration
      <feature>/<member>.<ext>                ZIP members (camera, lidar, radar)
      clip_metadata.json                      clip_index, feature_presence, data_collection rows
  ```

## Specification
- **Driven by the dataset's own metadata.** Feature names, chunk paths and ZIP members come from
  `features.csv` through the `physical_ai_av` SDK. A feature NVIDIA adds later works without a
  code change. Unknown names give an error listing the valid ones.
- **Selection:** `--features` takes feature names, `feature:member` (e.g.
  `camera_front_wide_120fov:video`), or presets, which are just named lists in one config file:
  - `default` = 3 labels + all 6 calibration tables (incl. offline variants and
    `lidar_intrinsics.offline`) + `frame_timestamps` of all 7 cameras. Small (a few MB per clip).
  - `scene` = `default` + `lidar_top_360fov`. Everything E02 (scene building) needs: lidar is
    the evidence for whether an object is observed (T008).
  - `alpamayo` = `egomotion` + `video` of the 4 cameras NVIDIA's loader uses.
  - `videos` = `video` of all 7 cameras. `lidar`, `radar` = everything in that group.
  - A bare camera/lidar/radar feature name means all its members.
- **Presence:** check the clip's `feature_presence` first. Absent → recorded as absent in the
  manifest and skipped; never a placeholder.
- **Size before download:** a `--dry-run` that lists what would be fetched with sizes (ZIP members
  from the remote central directory, parquet from file metadata) and fetches nothing. Ask for
  confirmation above a threshold (default 2 GB).
- **Streaming:** ZIP members are extracted with `maybe_stream=True`; a whole chunk ZIP is never
  downloaded. Calibration chunk parquets go through the HF cache and are filtered.
- **Filtering:** keep only the clip's rows, with the same columns, dtypes, values and row order.
- **Manifest:** per clip and stored file: feature, member, source path in the repo, SHA-256, size,
  row count, column names and dtypes (for parquet). Per clip: the list of everything available
  but not fetched (the "links" to fetch later). Plus SDK version, dataset commit, fetch date.
- **Idempotent:** a file already stored with a matching SHA-256 is skipped. Re-running only adds.
- **Removal:** `--remove <feature[:member]>` deletes stored files and updates the manifest (for
  large things like videos and lidar).
- Never touches `data/pai_selected_train/` (old pipeline output).

## Assumptions
- The SDK's streaming extraction works for every ZIP feature, not only cameras. To check on a lidar
  member.
- Remote ZIP member sizes can be read without downloading the member. To check, otherwise the
  dry run reports "unknown size".
- Measured so far: camera videos 10–16 MB per camera per clip, timestamp members ~9 KB.
- Lidar size per clip is unknown (dataset card: ~200 Draco-compressed spins per clip). If it is
  much larger than expected, stop after the dry run and decide with the user before fetching.
- Lidar is stored exactly as downloaded. Decoding it belongs to T005, not to the fetcher.

## Subtasks
- [ ] Project skeleton (package folder, `pyproject.toml` rewritten, test runner)
- [ ] Load HF token from `.env` without printing it
- [ ] Feature registry from `features.csv` + presets file
- [ ] Parquet fetch + filter (calibration)
- [ ] ZIP member streaming (labels, camera, lidar, radar)
- [ ] Presence check, dry run with sizes, confirmation threshold
- [ ] Manifest write/update, `--remove`
- [ ] CLI, documented in the repo README
- [ ] Dry run of `scene` for our two clips (report lidar size), then fetch it

## Acceptance criteria
- [ ] Test: filtering a chunk keeps exactly the rows of the clip, unchanged (synthetic chunk).
- [ ] Test: unknown feature, member or preset gives a clear error with the valid names.
- [ ] Test: a feature marked absent is skipped and recorded as absent.
- [ ] Test: re-running does not rewrite unchanged files.
- [ ] Check: for both clips, stored `egomotion.offline` and `obstacle.offline` are identical to
      `data/pai_selected_train/<clip>/raw_*.parquet`.
- [ ] Check: front-wide `frame_timestamps` matches the existing `front_camera_timestamps.parquet`.
- [ ] Check: fetching `camera_front_wide_120fov:video` for b789fab8 reproduces the existing mp4
      (SHA-256 `9ef5d8b3…090d` from `data/pai_selected_train/video_manifest.json`), then `--remove` it.
- [ ] Check: `--dry-run --features lidar` on one clip reports a size and downloads nothing.
- [ ] Check: after fetching `scene`, both clips have `lidar_top_360fov/pointclouds.parquet`, its
      SHA-256 and size in the manifest, and the manifest lists it as present in `feature_presence`.
- [ ] Rerun check: not applicable.

## Outcome
<Filled in when done.>

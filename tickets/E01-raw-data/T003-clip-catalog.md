# T003 — Clip catalog: browse and select clips without downloading

**Type:** Task
**Epic:** E01
**Status:** review
**Depends on:** T002

## Goal
Answer "which clips exist and what do they have?" from the dataset's small index files only,
and produce clip lists that T002 can fetch. This is how we pick clips when we scale beyond two.

## Inputs → Outputs
- Inputs: the index files at the pinned commit: `clip_index.parquet`, `feature_presence`,
  `data_collection` metadata (country, time of day, platform, …, whatever the files hold).
- Outputs: printed summaries and clip-list files (`data/clip_lists/<name>.txt`) usable as
  T002's input.

## Specification
- `catalog summary`: number of clips, coverage per feature (e.g. labels 97.4%, radar ~25%).
- `catalog show <clip_id>`: chunk, presence of every feature, collection metadata, and whether
  each feature is already stored locally (from `data/raw/manifest.json`).
- `catalog select`: filter by required features (e.g. must have `obstacle.offline` and all
  7 cameras) and by metadata columns, with `--limit` and `--seed` for a reproducible random
  sample. Writes a clip-list file.
- Reads only index files. Never downloads clip data.

## Assumptions
- The exact metadata columns available for filtering are read from the files, not assumed.

## Subtasks
- [x] Load index files through the SDK cache
- [x] `summary`, `show`, `select`
- [x] Document in the repo README

## Acceptance criteria
- [x] Test: `select` with a fixed seed returns the same list twice; every returned clip has the
      required features.
- [x] Check: `show` for our two clips gives chunks 262 and 329 and all features present except
      radar.
- [x] Rerun check: not applicable.

## Outcome
Implemented in `horizon_pipeline/raw/catalog.py` (logic) and `catalog_cli.py`, CLI
`horizon-catalog summary | show | select`, documented in the repo README. 13 new tests (57 in total).

- **Index only:** the catalog loads the same index files as the fetcher through the SDK cache
  (`features.csv`, `clip_index`, `feature_presence`, `data_collection`, plus `splits/default`,
  which the SDK attaches as a `split` column). Nothing else is read. `summary` takes about 2 s.
- **Metadata columns found** (306,152 clips, 3,146 chunks): `clip_is_valid` (all true), `chunk`,
  `split` (default split: train / val / test), `country` (25), `month`, `hour_of_day`,
  `platform_class` (hyperion_8, hyperion_8.1), `radar_config` (low / med / high, missing for 47.5%).
  Filters accept whichever columns the files have.
- **Coverage:** labels, offline calibration and lidar 97.4%; online calibration, egomotion and
  cameras 100%. Radar: 0% of clips have all 19 sensors and 52.5% have at least one.
  `radar_config` fully determines the set (checked by cross-tab on all clips): `null` → no radar
  (145,391 clips), `low` → the 9 `*_srr_0` sensors (87,197), `med` → 8 sensors (`*_srr_3`,
  `*_mrr_2`, `lrr_1`; 50,747), `high` → those 8 plus `side_left/right_srr_3` (22,817). So
  `--where radar_config=low` is the simple way to select radar clips.
- `--require` reuses the fetcher's selection syntax (features, groups, presets), so
  `--require scene` means "has everything `horizon-fetch --features scene` would fetch".
- `--limit` / `--seed`: a pandas sample with `random_state=seed`, returned in index order.
- Check: `show` for our two clips gives chunks 262 and 329, everything present except the 19
  radars, and every `scene` item stored.
- Under the SDK's default split our clips are `val` (b789fab8) and `test` (b4d979af), although
  `clips_selected_train.txt` was drawn from `alpamayo_aligned` / train. Splits are ignored (T001),
  so this only matters for the file name.

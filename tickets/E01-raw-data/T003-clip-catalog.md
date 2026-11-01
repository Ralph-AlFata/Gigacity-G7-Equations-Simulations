# T003 — Clip catalog: browse and select clips without downloading

**Type:** Task
**Epic:** E01
**Status:** todo
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
- [ ] Load index files through the SDK cache
- [ ] `summary`, `show`, `select`
- [ ] Document in the repo README

## Acceptance criteria
- [ ] Test: `select` with a fixed seed returns the same list twice; every returned clip has the
      required features.
- [ ] Check: `show` for our two clips gives chunks 262 and 329 and all features present except
      radar.
- [ ] Rerun check: not applicable.

## Outcome
<Filled in when done.>

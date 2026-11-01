# E01 — Raw data access layer

**Type:** Epic
**Status:** todo

## Goal
A general, on-request data layer for PhysicalAI-AV: any clip, any feature the dataset offers,
downloaded only when asked for, filtered to that clip, and recorded in a manifest. Phase 1 uses it
for our current two clips only, but nothing in the code is specific to them.

## Scope
- In: browsing the dataset index without downloading data (which clips exist, which
  features each one has); fetching any of the 36 features in `features.csv` (labels, calibration,
  7 cameras, lidar, 16 radars), and any ZIP member of them (video, frame timestamps, blurred boxes,
  point clouds, radar scans); size estimate before downloading; a manifest of what is stored.
- In: running it for b789fab8 and b4d979af, checked against the data we already have.
- Out: downloading more clips now (T001), decoding or processing the data (later epics).
- Dataset commit pinned to `39414de6f0ab867238ad6a0299fcd18b26081b42`.

## Tickets
| ID | Type | Title | Status |
|---|---|---|---|
| T001 | Decision | Data scope for phase 1 | done |
| T002 | Task | General on-request feature fetcher | todo |
| T003 | Task | Clip catalog: browse and select clips without downloading | todo |
| T004 | Spike | `egomotion` vs `egomotion.offline` | todo |

## Exit criteria
- One command fetches any feature (or member) for any clip list, e.g. our two clips' `scene` set,
  or one camera's video, or lidar, with no code change.
- `data/raw/<clip_id>/` holds the `scene` set (labels, calibration, camera timestamps, lidar) for
  our two clips, and `data/raw/manifest.json` says where every stored file came from.
- T004 answered and T001 updated with the egomotion choice.

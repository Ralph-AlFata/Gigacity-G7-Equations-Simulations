# T010 — Stream a scene to Rerun

**Type:** Task
**Epic:** E02
**Status:** todo
**Depends on:** T007

## Goal
One command that plays a built scene in Rerun as a 3D "video", so every scene ticket can be
checked by eye.

## Inputs → Outputs
- Inputs: `data/scene/<clip_id>/<ego_source>/`; optionally the raw lidar spins and a camera video.
- Outputs: a live Rerun viewer, or a saved `.rrd` file under `artifacts/`.

## Specification
- One timeline, `clip_time`, in seconds (repo convention).
- `world/ego`: ego box (from vehicle dimensions), heading arrow, rear-axle point.
- `world/obstacles/<track_id>`: 3D box for every existing object.
  - Colored by `class_group`.
  - In-scene boxes solid; existing but unobserved boxes faded.
  - Label shows track ID, class, speed, and lidar point count.
- `world/lidar`: the spin nearest each tick (after T008), downsampled if needed.
- Optional: `cameras/<camera>`, video frames on the same timeline (needs the video fetched).
- `--ego-source` picks the scene. `--compare` overlays both ego sources.

## Assumptions
- `rerun-sdk` is added as a dependency, with its version pinned.

## Subtasks
- [ ] Add rerun-sdk, write the logger
- [ ] CLI `scene view --clip … --ego-source …`
- [ ] Short section in the repo README

## Acceptance criteria
- [ ] Test: logging a tiny synthetic scene to an `.rrd` file runs without errors and contains the
      expected entity paths.
- [ ] Rerun check: for b789fab8, boxes and ego move smoothly over 20 s; parked cars stay still.

## Outcome
<Filled in when done.>

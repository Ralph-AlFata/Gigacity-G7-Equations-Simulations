# T010 — Stream a scene to Rerun

**Type:** Task
**Epic:** E02
**Status:** review
**Depends on:** T007

## Goal
One command that plays a built scene in Rerun as a 3D "video", next to the real camera videos with
the scene's boxes drawn on them. With it, every scene ticket can be checked by eye against what the
car actually saw.

## Inputs → Outputs
- Inputs:
  - `data/scene/<clip_id>/<ego_source>/`;
  - the raw lidar spins;
  - the camera videos with their frame timestamps;
  - `camera_intrinsics.offline` and `sensor_extrinsics.offline`.
- Outputs: a live Rerun viewer, or a saved `.rrd` file under `artifacts/`.

## Specification
- One timeline, `clip_time`, in seconds (repo convention).
- `world/ego`:
  - the ego transform per tick;
  - static children: the box (from the footprint), a heading arrow and the rear-axle point.
- `world/ego_path`: the whole ego path.
- `world/obstacles/<track_id>`: a 3D box for every existing object.
  - Colored by `class_group` when `in_scene`; gray and faded when it exists but is not observed.
  - The label shows track ID, class, speed and lidar point count. It appears when a box is
    selected; `--labels` draws all of them.
  - `--fade-beyond M` fades objects more than M m from the ego, for display only.
  - Cleared at ticks where the track does not exist.
- `world/lidar`: the spin whose middle is nearest each tick. Each point is placed with the ego pose
  at its own time, as T008 counts them, and thinned to every 10th point by default.
- `cameras/<camera>/image`: video frames on the same timeline, at each frame's own timestamp.
  - By default, the frame nearest each tick; `--all-frames` gives all 30 Hz.
- `cameras/<camera>/boxes`: the scene's boxes projected into that camera with its f-theta model,
  using the same colors.
  - The ego pose and boxes are interpolated to the frame's time, the same way the scene build
    interpolates: position linear, rotation slerp.
- Default cameras: the three Alpamayo uses that are fetched with the `alpamayo` preset.
- Layout: the 3D scene (tracking the ego) on the left; the front camera above the two cross
  cameras on the right.
- `--ego-source` picks the scene. `--compare` adds the other ego source's box and path in red.

## Assumptions
- `rerun-sdk` is added as a dependency, pinned to 0.40.0 (released 2026-10-21).
- Rerun's own camera model is pinhole only. The cameras are f-theta, so we project ourselves and
  draw 2D lines over the images.

## Subtasks
- [x] Add rerun-sdk, write the logger (`horizon_pipeline/scene/view.py`)
- [x] Camera module: f-theta projection, camera poses, video decoding (`horizon_pipeline/scene/camera.py`)
- [x] CLI `horizon-scene view --clip … --ego-source …`
- [x] Short section in the repo README and in `docs/GUIDE.md`

## Acceptance criteria
- [x] Test: logging a tiny synthetic scene, with a synthetic camera video, to an `.rrd` file runs
      without errors and contains the expected entity paths. A track is cleared when it ends.
- [x] Test: the f-theta projection matches NVIDIA's SDK (`FThetaCameraModel.ray2pixel`) to 1e-9 px.
      A box straight ahead lands on the principal point, including after the ego is moved and turned.
- [x] Check: on real frames of both clips, boxes sit on the cars and people in all three cameras, and
      lidar points fall on the facades and car bodies (see Outcome).
- [ ] Rerun check (by eye, in the viewer): for b789fab8, boxes and ego move smoothly over 20 s; parked
      cars stay still. This also covers the Rerun checks deferred by T007 and T008.

## Notes
- **Camera frame:** z forward, x right, y down, as the SDK documents. Checked on our calibration:
  - the front camera's z axis is the rig's +x;
  - its x axis is the rig's −y;
  - its y axis is the rig's −z.
- **Projection:** `angle_to_pixeldist_poly(θ)` gives the radius from the principal point, where θ is
  the angle from the optical axis. Points beyond `max_angle` (77° on the wide cameras) are cut.
  - `linear_cde` is (1, 0, 0) on all our cameras, so it changes nothing.
  - The SDK's loader reads an older column layout (`fw_poly_*`, `cx`). Our files hold the
    parameters as JSON in `model_parameters`, so `camera.py` reads that.
- **Frame timing:** the front camera's frames land within 26 µs of the ticks on b789fab8. On
  b4d979af the cross cameras are up to 10.7 ms off. That's why boxes are interpolated to each
  frame's time rather than drawn from the nearest tick.
- **Rolling shutter is not modeled.** The sensor reads top to bottom over 30.6 ms
  (`shutter_delay_us`). That is at most ~0.5 m of displacement for an object crossing at 15 m/s,
  and less near the image middle.
- **Size:**
  - b789fab8: 152 MB, 24 s.
  - b4d979af: 247 MB, 54 s (652 tracks).
  - The lidar is most of it: ~335 000 points per spin. `--lidar-every` and `--no-lidar` reduce it.

## Outcome
Done, pending the by-eye check in the viewer.

- **Projection on real frames:** I rendered the overlay code path to images (front and cross-right
  on b789fab8; front and cross-left on b4d979af).
  - Boxes sit on the van ahead, the parked cars and the pedestrians.
  - Projected lidar points fall on the building facades and car bodies.
  - With a car 1 m beside the cross-left camera, the box top shows above the roofline. This is
    expected: the box's near face is 1.1 m from the camera, its top is 1.0 m above it (43°), and the
    2.07 m box width includes the mirrors. The box bottom meets the wheels.
- **Lidar vs camera visibility:** in b4d979af's lot, people standing behind that car are `in_scene`.
  The roof lidar at 1.9 m sees over the car; the cross camera at 0.78 m cannot. This is the open
  question of whether H should count lidar-visible or camera-visible objects, and it can now be
  looked at clip by clip.
- **Occlusion confirmed:** the faded (not observed) boxes in b4d979af's crowded lot all sit behind
  other parked cars. This is the occlusion T008 predicted.

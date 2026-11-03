# E02 — Scene building

**Type:** Epic
**Status:** in-progress
**Depends on:** E01

## Goal
Turn the raw data of a clip into a normalized **scene**: at every tick of a 10 Hz grid, the ego
state and every object that is in the scene, all in the same frames and units. The equations only
read this scene; they never decide which objects count. That decision is made here, from data.

## Principles (decided 2026-10-31)
- **Separation:** scene building decides what is in the scene; the equations only consume numbers.
- **In the scene = exists AND observed.**
  - *Exists:* the tick lies inside the object's labeled lifetime (first to last sighting), with
    skips of at most 0.5 s bridged.
  - *Observed:* the lidar actually hits the object at that tick (points inside its box).
  - No timeouts.
- **Objects are their track IDs.** Different IDs are different objects. No ID-switch handling: H
  needs every object present at a tick, not who it was before.
- **Ground truth:** offline labels are treated as truth (T001). Positions between sightings are
  interpolated, even though that uses a sighting up to ~0.1 s in the future.
- **Ego source:** `egomotion.offline` for scene geometry (T004: the labels match it in space and
  time). Online `egomotion` scenes can still be built for comparison, but are never mixed with offline.
- **3D everywhere**, plus planar x, y, yaw for the equations.
- **Ego reference point:** rear-axle midpoint (to confirm in T006).

## Scope
- In scope:
  - the time grid;
  - frame transforms (rig at each sighting → world → ego frame at each tick);
  - de-duplication;
  - interpolation and gap bridging;
  - derived motion;
  - lidar observability;
  - class groups;
  - the scene files;
  - streaming the scene to Rerun.
- Out of scope:
  - reference trajectories (Alpamayo / recorded path), handled by a later epic;
  - margins and the H solver, handled by a later epic;
  - perception or timing delays, handled in a later phase.

## Scene schema
One folder per clip and ego source: `data/scene/<clip_id>/<ego_source>/`.

**`meta.json`**
- `clip_id`, `ego_source`, `rate_hz` (10), `t0_us` (clip time 0), `n_ticks`
- `ego`:
  - vehicle dimensions (length, width, height);
  - rear-axle offset from the rig origin (from T006);
  - footprint relative to the rear axle.
- `class_groups`:
  - vehicle: automobile, bus, heavy_truck, trailer;
  - vru: person, rider, stroller, animal;
  - static: protruding_object;
  - unknown: anything else.
- `rules`: `gap_bridge_s` (0.5), `dedup_s` (0.020), `lidar_min_points`, `lidar_box_margin_m`,
  `history_s` (1.6)
- `inputs`: SHA-256 of every raw file used (from `data/raw/manifest.json`), code version.

**`ego.parquet`**: one row per tick.

| Column | Unit | Meaning |
|---|---|---|
| `tick` | int | 0 … n_ticks−1 |
| `t_s` | s | tick × 0.1, clip time |
| `timestamp_us` | µs | t0_us + t_s |
| `x, y, z` | m | rear-axle midpoint, world frame (anchor = rig at clip time 0) |
| `qx, qy, qz, qw` | — | orientation, world frame |
| `yaw` | rad | heading in the world x-y plane, unwrapped |
| `vx, vy, vz, speed` | m/s | derived from poses, world frame |
| `ax, ay, az` | m/s² | derived, world frame |
| `a_long, a_lat` | m/s² | derived, in the ego frame |
| `yaw_rate` | rad/s | derived |
| `has_history` | bool | t_s ≥ history_s (Alpamayo needs 1.6 s of past input) |

**`objects.parquet`**: one row per (tick, track) where the object **exists**. Rows where it exists
but is not observed are kept, with `in_scene = false`, so we can see why it was excluded.

| Column | Unit | Meaning |
|---|---|---|
| `tick`, `t_s` | | as in ego |
| `track_id` | str | from the labels; one ID = one object |
| `label_class`, `class_group` | str | |
| `x, y, z` | m | box center, world frame |
| `qx, qy, qz, qw`, `yaw` | | box orientation, world frame |
| `length, width, height` | m | box size (label `size_x, size_y, size_z`) |
| `vx, vy, vz, speed` | m/s | derived, world frame |
| `ax, ay, az` | m/s² | derived, world frame |
| `yaw_rate` | rad/s | derived |
| `rel_x, rel_y, rel_z` | m | box center in the ego frame at this tick (origin at the rear axle) |
| `rel_yaw` | rad | box heading relative to ego heading |
| `distance` | m | planar center-to-center distance from the ego rear axle |
| `last_sighting_age_s` | s | time since the last sighting at or before this tick |
| `gap_s` | s | 0 if bracketed by normal sightings, else the length of the bridged skip |
| `lidar_points` | int | lidar points inside the box (with margin) in the spin nearest this tick |
| `observed` | bool | lidar_points ≥ lidar_min_points |
| `in_scene` | bool | exists (always true for a row) AND observed |

The equations read only rows with `in_scene = true`.

## Exit criteria
- [ ] For both clips and both ego sources, `data/scene/...` holds the three files above.
- [ ] Every column has a test or a recorded check.
- [ ] Derived ego motion is compared with online egomotion (T009). If the difference is large, the
      method is revised before E02 closes.
- [ ] Rerun: playing `clip_time` shows the ego, all existing boxes (in-scene ones highlighted), and
      the lidar points; objects appear and disappear exactly when `in_scene` flips.

## Tickets
| ID | Type | Title | Status |
|---|---|---|---|
| T005 | Spike | Lidar format: columns, frame, timing, size | review |
| T006 | Spike | Is the rig origin the rear-axle midpoint? | todo |
| T007 | Task | Scene builder: grid, ego, objects, frames, derived motion | todo |
| T008 | Task | Lidar observability and `in_scene` | todo |
| T009 | Spike | Derived ego motion vs online egomotion | todo |
| T010 | Task | Stream a scene to Rerun | todo |

## Notes
Measured on our two clips (old raw files):
- **Sighting format:** sightings are (time, track ID, box relative to the rig at that time).
- **Clocks:** each track runs on its own ~10 Hz clock.
- **Skips:** 0.2 s skips appear 245 times (b789fab8) and 100 times (b4d979af).
- **Duplicates:** rows of the same track 0–5 ms apart, 236 and 129 pairs respectively.
  - Within each pair: identical size, same class, positions within 7.4 cm.
  - About a third of them precede a 0.2 s skip.
- **Track counts:** 110 tracks in b789fab8, 653 in b4d979af.
- **Clip-boundary cut-offs:** in b4d979af, 207 tracks start and 302 end at the clip boundary,
  which means the clip cut them off, not that they entered or left.

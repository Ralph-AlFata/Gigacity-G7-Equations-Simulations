# T005 — Lidar format: columns, frame, timing, size

**Type:** Spike
**Epic:** E02
**Status:** review
**Depends on:** T002
**Timebox:** half a day

## Question
What exactly is in `lidar_top_360fov` for one clip? We need to know:
- the columns and their encoding;
- the coordinate frame of the points;
- the timestamps, per spin and per point;
- whether points are motion-compensated;
- the size per clip.

## Why it matters
T008 counts lidar points inside each object box. If we get the frame or timing wrong, points land
off the boxes. Then objects that are really seen get marked unobserved, or the reverse, and that
decides which objects enter H.

## Method
- [x] Documentation:
  - the dataset card says ~200 spins per clip (10 Hz), Draco-compressed (DracoPy);
  - check the `physical_ai_av` wiki for the parquet schema;
  - note that the SDK 0.2.3 has no lidar reader.
- [x] Data check: `fetch --dry-run --features lidar` for one clip to get its size, then fetch it.
- [x] Data check: decode one spin and record:
  - point count and fields (xyz, intensity, per-point time?);
  - its timestamp against the ego and label clocks.
- [x] Data check: frame test.
  - Transform points with each candidate frame (rig at spin time, sensor frame + extrinsics, …).
  - Check which one puts points on a parked car's labeled box across several spins.
- [x] Data check: motion compensation. Does a static wall stay still between spins while the ego
      moves, once the frame from the previous step is applied?

## Hypotheses
1. Points are in the rig frame at the spin timestamp, already motion-compensated.
2. One spin is about 0.1 s and its timestamp is the spin start or middle; it is not aligned to the
   label clocks.

## Findings
Script: `scripts/spike_t005_lidar.py` (`uv run python scripts/spike_t005_lidar.py`, ~1.5 min), on the
`scene` data from T002 (lidar was already stored, nothing new fetched). It decodes with `DracoPy`,
now a project dependency. Numbers are b789fab8 / b4d979af.

### Documented (NVIDIA)
- Dataset card: one parquet per clip, ~200 spins (10 Hz), Draco-compressed, "decoded, e.g., by
  using the DracoPy library". The column schema is deferred to the SDK wiki.
- Wiki, *Sensor Details*: 128-channel rotating lidar on the roof, "100ms per complete rotation".
  It lists columns `reference_timestamp` + `draco_encoded_pointcloud`, with intensity in a red channel and a
  16-bit relative time in green/blue, "[-105,000, +105,000] microseconds relative to
  `reference_timestamp`". **Our files don't match this** (see below).
- Wiki, *Coordinate Systems*: "Points represent 3D positions relative to the lidar sensor"; the
  extrinsic transforms sensor → rig ("`point_rig = T_rig_to_sensor @ point_sensor`").
- Wiki, *Egomotion and Motion Compensation*: "Points captured at different times are in different
  reference frames"; the example compensates each point with the pose at `point.timestamp`. It
  implies the points are raw but does not say so outright.
  (https://github.com/NVlabs/physical_ai_av/wiki, https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

### Format and size (from the data)
| | |
|---|---|
| File | `data/raw/<clip>/lidar_top_360fov/pointclouds.parquet`, **370 / 333 MB**, 199 rows, one row group per spin |
| Columns | `spin_index` (0…198, contiguous), `spin_start_timestamp`, `spin_end_timestamp` (µs), `draco_encoded_pointcloud` (binary, 1.7–1.9 MB) |
| Draco attributes | position `float32 ×3` (unnamed); `intensity` `uint8` 0–255; `timestamp` `uint32`; `model_element` `uint32 ×2` |
| `timestamp` | **absolute µs on the clip clock** (same clock as egomotion and labels), not relative. 100% of points lie in [spin start, spin end] |
| `model_element` | (laser channel 0–127, azimuth column 0–3599 at 0.1° steps) |
| Points per spin | 297k–339k (68–74% of 128 × 3600), ~5.6 bytes/point encoded |
| Range | horizontal 3.2–209 m. **No point closer than ~3.2–3.4 m** (blind zone, no ego-body returns) |
| Read + decode | ~40 ms per spin, ~8 s per clip |

### Timing
- Spins are back-to-back: duration 99.9–100.05 ms, next start = previous end (± 1 µs). Starts at
  13.4 ms / 99.5 ms of clip time, last start 19.81 / 19.90 s.
- `egomotion.offline` rows sit on the spin starts (169 / 154 of 202 exactly, as T004 found).
- Within a spin the column advances 36 per ms (360° in 100 ms). The first points face **backward**
  (azimuth ≈ +178° in the sensor frame) and the sweep runs **clockwise** seen from above.
- **Label timestamps are lidar sweep times.** No obstacle sighting sits on a spin start (0.0%). A
  label's time is when the sweep crossed the box center's azimuth: median −0.2 ms, within 5 ms for
  98.9% / 99.7% of 7 140 / 53 109 sightings. This explains the "own ~10 Hz clock per track" in the E02 notes:
  a track's clock is its bearing from the car.

### Frame
**Points are in the lidar sensor frame**, not the rig frame:
- The blind zone is centered on (0.0, 0.0) ± 0.1 m in every sampled spin. In the rig frame it would
  be at the lidar mount, (+1.0, 0.0).
- The fitted ground plane is at z = −1.76 … −1.93 m and level (slope ≤ 0.017). The lidar extrinsic
  puts the sensor 1.888 / 1.896 m above the rig origin.
- The frame is attached to the car: columns 0–19 always point to azimuth ≈ 171°, while the ego yaw
  goes from 0° to −68°.
- Online and offline lidar extrinsics are identical for both clips (0.0 mm, 0.000°). Lidar
  extrinsic: x 1.00 m, y −0.008 / +0.022 m, z 1.89 m; roll/pitch/yaw within 0.75°.
- Placing points with the extrinsic beats not placing them: without it, **0** points land in
  parked cars' boxes.

### Motion compensation
**The points are not motion-compensated.** Each point is in the sensor frame at its own
`timestamp`. Two independent tests agree. Both use only spins where the car drives faster than 3 m/s, where the
hypotheses differ by v × 100 ms = 0.3–0.9 m.

1. *Seam test.* Columns 3599 and 0 sit next to each other but are 100 ms apart in time. Taking the signed
   range jump per channel (last − first), median over spins:

   | | ground channels | structure channels |
   |---|---|---|
   | prediction if raw | ≈ 0 (a channel hits flat ground at a fixed range) | ≈ +v·0.1 s |
   | prediction if compensated | ≈ −v·0.1 s | ≈ 0 |
   | **as stored** | **−0.00 / −0.03 m** | **+0.34 / +0.28 m** |
   | after per-point compensation | −0.55 / −0.36 m | −0.21 / −0.09 m |

   (v·0.1 s median 0.58 / 0.34 m.) The stored points match "raw" on both kinds of surface.
   *Pitfall:* a first version that pooled all channels read ~0 as stored and concluded "compensated".
   Flat ground hides the raw seam.
2. *Parked-car boxes.* For 544 / 906 (spin, parked car) pairs, count points inside the tight labeled box
   (box placed with the offline pose at its label time, as T004). Then try each single pose time for the
   whole spin (spin start + offset), and per-point poses:

   | car swept at (ms into spin) | best single offset | per-point / best single | @start | @end |
   |---|---|---|---|---|
   | 0–25 | +10 / −40 ms | 0.995 / 0.990 | 1.00 / 0.99 | 0.67 / 0.91 |
   | 25–50 | +40 / +110 ms | 1.003 / 0.998 | 0.91 / 0.99 | 0.89 / 1.00 |
   | 50–75 | +80 / +70 ms | 0.998 / 1.001 | 0.67 / 0.91 | 0.99 / 1.00 |
   | 75–100 | +70 / +70 ms | 0.996 / 0.999 | 0.93 / 0.99 | 0.97 / 0.99 |
   | all | +60 / +60 ms | **1.015 / 1.002** | 0.85 / 0.97 | 0.97 / 0.99 |

   On the faster clip (b789, to 9 m/s) the best single pose time follows when the sweep passed the car.
   Per-point poses match the best single time in every bin, and beat any single time over all cars. The
   slower clip (b4d, ≤ 5 m/s) is flatter but agrees. A fixed spin-start pose costs up to 33% of the points on a car.

A cross-spin test (static world in spin k vs k+3) is **blind** to this question. Every azimuth is
swept at the same phase in every spin, so the error repeats and cancels. The first version of the script used it,
and it showed nothing (2.8–3.7 cm under every hypothesis).

### Hypotheses
1. *Points are in the rig frame at the spin timestamp, already motion-compensated:* **refuted on both
   counts.** They are in the **sensor** frame (apply the lidar extrinsic), and **raw** (each point at
   its own time).
2. *One spin is ~0.1 s, its timestamp is the spin start or middle, and it is not aligned to the label clocks:*
   **partly supported.** A spin is 100 ms with explicit start and end, and every point has its own time. But
   labels **are** aligned to the lidar: each label carries the time the sweep crossed the object.

## Answer
- **Reading:** decode `draco_encoded_pointcloud` with DracoPy. Positions are `cloud.points`. The
  named attributes are `intensity`, `timestamp` (absolute µs, clip clock) and `model_element`
  (channel, column). The wiki's `reference_timestamp` and RGB-packed relative time do not apply to
  this dataset revision.
- **Frame:** the lidar sensor frame. To the rig: `p_rig = R_ext · p + t_ext`, from
  `sensor_extrinsics.offline` row `lidar_top_360fov`.
- **Timing and compensation:** raw. To the world: `p_world = Pose_offline(point.timestamp) · p_rig`,
  with the pose interpolated per point. A single pose for the whole spin misplaces points by up to
  v × 100 ms (0.9 m at 9 m/s).
- **Size:** ~350 MB per 20 s clip, about 99% of a clip's `scene` data. ~40 ms to decode a spin.

Confidence: high for format, frame and "raw" (the seam and box tests are independent and agree on both clips).
High that labels are timed by the sweep (99% within 5 ms).

## Follow-ups
- **T008 specification** (updated): compensate points per point, into the world frame with the offline pose at
  each point's time, then into the ego frame at the tick. The "nearest spin" and the time difference are
  per object: the spin whose sweep crossed the object nearest the tick. Points within ~3.2 m of the
  sensor are never measured, so the "near objects are observed" check needs that exception.
- **T006:** the ground lies 1.76–1.93 m below the sensor, i.e. within ~0.1 m of the rig origin's
  height (sensor at z = 1.89 m in the rig). That supports a rig origin at ground level, which is to be checked there.
- **T007:** since each label is stamped at its sweep time, sightings of different tracks in one
  spin are up to 100 ms apart. The per-track interpolation onto the 10 Hz grid already handles that.
- **T010:** the Rerun spin view must compensate the same way, or static walls appear doubled at the seam.

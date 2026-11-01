# T005 — Lidar format: columns, frame, timing, size

**Type:** Spike
**Epic:** E02
**Status:** todo
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
- [ ] Documentation:
  - the dataset card says ~200 spins per clip (10 Hz), Draco-compressed (DracoPy);
  - check the `physical_ai_av` wiki for the parquet schema;
  - note that the SDK 0.2.3 has no lidar reader.
- [ ] Data check: `fetch --dry-run --features lidar` for one clip to get its size, then fetch it.
- [ ] Data check: decode one spin and record:
  - point count and fields (xyz, intensity, per-point time?);
  - its timestamp against the ego and label clocks.
- [ ] Data check: frame test.
  - Transform points with each candidate frame (rig at spin time, sensor frame + extrinsics, …).
  - Check which one puts points on a parked car's labeled box across several spins.
- [ ] Data check: motion compensation. Does a static wall stay still between spins while the ego
      moves, once the frame from the previous step is applied?

## Hypotheses
1. Points are in the rig frame at the spin timestamp, already motion-compensated.
2. One spin is about 0.1 s and its timestamp is the spin start or middle; it is not aligned to the
   label clocks.

## Findings
<…>

## Answer
<…>

## Follow-ups
<Update T008's specification with the frame and timing to use.>

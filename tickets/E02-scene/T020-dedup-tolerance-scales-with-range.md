# T020 — Duplicate check is too strict for distant objects

**Type:** Bug
**Epic:** E02
**Status:** todo
**Found in:** T007, `horizon_pipeline/scene/build.py:125` (`deduplicate`), on clip
`b6cb8b12-8b36-49c2-8609-63c4e2958e28` (random clip picked 2026-11-04)

## Observed
`horizon-scene build` stops on b6cb8b12:

```
FAILED b6cb8b12-8b36-49c2-8609-63c4e2958e28 offline: 1 duplicate pairs disagree:
  track 228 at 577673 / 578877 us: class heavy_truck / heavy_truck, size [10.763, 2.942, 3.785] / [10.763, 2.942, 3.785], offset 0.772 m
```

- **Track 228:** a heavy truck **197 m behind the ego**.
  - It has 3 sightings, all within 0.48–0.58 s, and then disappears.
  - The two rows are 1.2 ms apart. Class and size are identical.
  - The centers differ by 0.53 m along the truck's length and 0.11 m across it.
  - In 1.2 ms the truck and the ego (11.6 m/s) move ~1 cm, so this is two label estimates
    disagreeing, not motion.
  - 0.77 m at 197 m is ~0.2°.
- **The build stops at the first failure.** The other two random clips were therefore not built in
  the same run.

**Survey of every duplicate pair** (rows of one track < 20 ms apart) on all 5 clips:

| Clip | Pairs | Median offset | p99 | Max | > 0.2 m |
|---|---|---|---|---|---|
| b789fab8 | 236 | 1 mm | 3 cm | 4 cm | 0 |
| b4d979af | 129 | 1 mm | 1.5 cm | 2 cm | 0 |
| b6cb8b12 | 85 | 2 mm | 18 cm | 78 cm | 1 (at 197 m) |
| ca823746 | 5 | 7 cm | 11 cm | 11 cm | 0 |
| 23958e60 | 220 | 1 cm | 3.5 cm | 4.5 cm | 0 |

Class and size agree in every pair.

## Expected
The check exists to catch truly inconsistent labels: a different object, class or size behind one
track ID. It should not stop on ordinary label noise.

Label position error grows with range, because box placement is an angular estimate and a far
object's extent along its length is barely observed. So the allowed offset should **scale with
range**, not be a flat 0.2 m.

## Reproduce
```bash
uv run horizon-fetch --clips b6cb8b12-8b36-49c2-8609-63c4e2958e28 --features scene
uv run horizon-scene build --clips b6cb8b12-8b36-49c2-8609-63c4e2958e28 --ego-source offline
```

## Cause
`deduplicate` compares the rig-frame centers of a pair against a flat `dedup_max_offset_m = 0.2`,
whatever the object's range.

## Fix (decided 2026-11-04: scale with range)
- **Rule:** a pair agrees if class and size match (unchanged) and the center offset is
  ≤ `dedup_max_offset_m + dedup_offset_per_m × range`.
  - Defaults: 0.2 m and 0.005 (0.5% of range).
  - `range` is the pair's planar distance from the rig.
  - This allows 0.25 m at 10 m, 0.45 m at 50 m and 1.19 m at 197 m.
  - The current worst pair, 0.18 m in b6cb8b12, is below its limit at any range, so no other pair
    changes.
- **Report:** the build report records how many pairs needed the range allowance (offset above the
  flat 0.2 m) and the largest such offset with its range, so loosening the check stays visible.
- [ ] Code change: `build.py` (`Rules.dedup_offset_per_m`, `deduplicate`, build report); the rule
      appears in `meta.json` and in the E02 README.
- [ ] Regression test that fails before the fix and passes after: a far pair (0.8 m at 200 m) is
      merged; a near pair (0.8 m at 10 m) still stops the build; a class or size mismatch stops it
      at any range.
- [ ] Check: b6cb8b12 builds, for both ego sources.
- [ ] Rerun results affected? Our first two clips have no pair above 0.2 m, so their scenes do not
      change. Rebuild them anyway, so `meta.json` records the new rule.

## Workaround until fixed
Build the other clips without b6cb8b12:
```bash
uv run horizon-scene build --clips ca823746-c767-48d9-bc0d-008ce50f6497 23958e60-dbf9-41fa-b927-53960a114276 --ego-source both
```

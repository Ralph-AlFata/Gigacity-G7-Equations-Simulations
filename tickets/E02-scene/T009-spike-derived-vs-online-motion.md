# T009 — Derived ego motion vs online egomotion

**Type:** Spike
**Epic:** E02
**Status:** review
**Depends on:** T007
**Timebox:** half a day

## Question
Are velocities, accelerations and yaw rates derived from poses (T007 step 7) close to the motion
the car itself recorded, i.e. the velocity and acceleration columns of online `egomotion`?

## Why it matters
- **For objects**, derived motion is the only motion we have.
- **For the equations**, speed and acceleration feed the bounds directly.
- **If they differ a lot**, derived object motion is unreliable too, and the method must change
  before the equations use it (decided 2026-10-31).

## Method
- [x] Data check: on both clips, compare derived ego speed, longitudinal and lateral acceleration,
      and yaw rate (from each ego source) with the online columns, resampled to the 10 Hz grid.
- [x] Data check: report mean, 95th percentile and max error per quantity, and plot them over time.
- [x] If errors are large, try smoothing alternatives and compare them the same way:
  - a larger difference window;
  - Savitzky–Golay;
  - a constant-acceleration Kalman smoother.

## Hypotheses
1. Speed agrees within ~0.1 m/s; acceleration is noisy without smoothing.

## Findings
Script: `scripts/spike_t009_motion.py` (`uv run python scripts/spike_t009_motion.py`), on the scenes
built with T020's rule, for all 5 clips we have (our two plus the three random ones). Plots over time
are in `results/t009/<clip>.rrd`, one per clip (`rerun results/t009/<clip>.rrd`).
- **Layout:** `plots/<quantity>/recorded` and one series per method, with `plots/<quantity>/error/<method>`
  for each.
- **Methods plotted:** the current method, Savitzky–Golay 1.1 s and Kalman q=10.

### The reference
Online `egomotion` over its dense 100 Hz part, interpolated at the scene ticks:
- **Speed:** `|v|`.
- **Longitudinal and lateral acceleration:** `a` rotated into the rig frame with online's own
  orientation. Its `v` and `a` are in online's anchor frame (T004), and `a` holds no gravity
  (mean `az` ≈ 0).
- **Yaw rate:** `curvature × forward speed`. There is no yaw-rate column. This formula matches the
  100 Hz derivative of online's own yaw with correlation 0.999–0.9998 on 4 clips.
- **23958e60 is the exception:** correlation 0.76, with a constant offset of about +1.1°/s. Both pose
  sources give the same yaw rate, and both are 1.1°/s away from the curvature-based value, so the
  `curvature` column is wrong on that clip. Its yaw rate is left out of the pooled numbers.

**Clock:** the offline scene is compared at its best-matching shift of the online clock, found from
speed. The shift is +55, +45, −35, +15 and +20 ms on the five clips. T004 saw +40–50 ms on the first
two; the offset is not the same on every clip.

**The reference is not ground truth.** Even the *online* scene, differentiated from the same poses
that sit next to these columns, differs from them:
| online scene vs its own columns | speed p95 | a_long p95 | a_lat p95 | yaw rate p95 |
|---|---|---|---|---|
| range over the 5 clips | 0.01 m/s | 0.17–0.35 m/s² | 0.14–0.42 m/s² | 0.07–1.55°/s (23958e60: 1.25, the bias) |

So differencing 10 Hz poses and the car's recorded acceleration disagree by ~0.2–0.4 m/s² p95 even
with no source mismatch. The recorded value is presumably a filtered inertial estimate.

### Ego: offline scene (the one H uses) against the recorded motion
| Quantity | b789fab8 mean / p95 / max | b4d979af mean / p95 / max | 5 clips pooled mean / p95 / max |
|---|---|---|---|
| speed (m/s) | 0.022 / 0.06 / 0.09 | 0.020 / 0.07 / 0.11 | **0.024 / 0.07 / 0.11** |
| a_long (m/s²) | 0.15 / 0.43 / 0.70 | 0.11 / 0.29 / 0.77 | **0.15 / 0.39 / 0.77** |
| a_lat (m/s²) | 0.16 / 0.43 / 0.63 | 0.11 / 0.26 / 0.33 | **0.14 / 0.38 / 0.84** |
| yaw rate (°/s) | 0.41 / 1.57 / 2.40 | 0.25 / 0.87 / 1.02 | **0.21 / 0.76 / 2.40** |

- **Biases are small:**
  - speed +0.014 m/s;
  - accelerations below 0.01 m/s²;
  - yaw rate −0.04°/s pooled (b4d979af −0.22°/s).
- **The ranges covered:** speeds 0–35 m/s, accelerations up to 3.3 m/s², yaw rates up to 24°/s.
- **Where the errors peak:** the largest yaw-rate errors (2.4°/s) fall in b789fab8's sharpest turn,
  at 24°/s. That is about 10% there, and smaller everywhere else.

### Smoothing alternatives (offline ego, 5 clips pooled, p95 |error|)
| Method | speed (m/s) | a_long (m/s²) | a_lat (m/s²) | yaw rate (°/s) |
|---|---|---|---|---|
| central ±0.1 s (now) | 0.07 | 0.39 | 0.38 | 0.76 |
| central ±0.2 s | 0.06 | 0.30 | 0.35 | 0.76 |
| central ±0.3 s | 0.05 | 0.31 | 0.35 | 0.82 |
| Savitzky–Golay 0.7 s (cubic) | 0.06 | 0.30 | 0.34 | 0.78 |
| Savitzky–Golay 1.1 s (cubic) | 0.05 | 0.29 | 0.35 | 0.73 |
| Kalman CA + RTS, q = 1 / 10 / 100 m²/s⁵ | 0.05 / 0.05 / 0.06 | 0.31 / 0.29 / 0.32 | 0.35 / 0.34 / 0.35 | 0.80 / 0.73 / 0.77 |

- **Speed and yaw rate:** every method is within noise of the current one.
- **Acceleration:** any smoothing gives about −25% at p95 (0.39 → 0.29–0.31 m/s²). No method stands
  out, and that is the same size as the reference's own disagreement above.
- **Correctness:** all methods are exact (≤ 3e-3 m/s²) on a constant-acceleration path away from
  the ends.

### Objects: noise floor from parked objects
Objects have no recorded motion. Instead I use tracks that barely move: at least 50 ticks, with a
horizontal span under 1 m. They are selected by position only. Their true speed is about 0, so the
derived motion measures noise. That is 27 416 object-ticks over the 5 clips, offline.
| Method | speed p50 / p95 (m/s) | horizontal acceleration p50 / p95 (m/s²) |
|---|---|---|
| central ±0.1 s (now) | 0.16 / 0.68 | 0.49 / 2.57 |
| central ±0.3 s | 0.13 / 0.58 | 0.29 / 1.54 |
| Savitzky–Golay 1.1 s | 0.15 / 0.66 | 0.35 / 1.86 |
| Kalman CA, q = 1 | 0.13 / 0.59 | 0.37 / 1.82 |

**By distance (current method):**
| Distance | speed p50 / p95 (m/s) | acceleration p50 / p95 (m/s²) |
|---|---|---|
| 0–10 m | 0.13 / 0.42 | 0.43 / 1.47 |
| 10–20 m | 0.09 / 0.41 | 0.32 / 1.51 |
| 20–40 m | 0.13 / 0.67 | 0.42 / 2.45 |
| 40–80 m | 0.18 / 0.71 | 0.57 / 2.68 |
| > 80 m | 0.22 / 0.82 | 0.61 / 3.57 |

- **Range dependence:** the noise grows with range, as label error does (T020). Creeping cars would
  not depend on range.
- **Selection:** tightening the span filter (0.5 m, 0.3 m) lowers the numbers. That is partly
  circular, because a noisier track spreads wider.
- **Size against the ego:** object noise is about 10× the ego's. It comes from the labels, not from
  the poses.

### Side finding: acceleration at the ends of a run is biased
- **What:** T007 uses `np.gradient` with first-order ends. On a constant-acceleration path, the
  acceleration at the first and last tick of every run is off by **half the true acceleration**
  (0.75 m/s² for a true 1.5 m/s²). Velocity is fine there.
- **Where it matters:** for the ego this is just the clip's first and last tick. For objects it is
  every track's birth, death, and each gap longer than 0.5 s.
- **Fix:** `np.gradient(..., edge_order=2)` is exact on that path.

### Hypotheses
1. *Speed agrees within ~0.1 m/s; acceleration is noisy without smoothing:* **speed supported**:
   p95 0.07 m/s, max 0.11 m/s over 5 clips. **Acceleration only partly**: for the ego it is
   0.39 m/s² p95, with smoothing worth −25%, and half of the gap is the reference itself. For
   objects, acceleration *is* noisy (p95 2.6 m/s²), and smoothing does not rescue it (best 1.5 m/s²).

## Answer
**Speed and yaw rate from 10 Hz poses can be used as they are.**
- Ego speed is within 0.07 m/s at p95 (max 0.11 m/s) of what the car recorded.
- Ego yaw rate is within 0.8°/s at p95.
- No smoothing method improves them.
- **Ego acceleration** is within about 0.4 m/s² at p95. Smoothing brings that to about 0.3 m/s²,
  which is about how far the car's own recorded acceleration is from its own poses.

**Recommendation: keep T007 step 7 as it is (central differences, no smoothing), plus
`edge_order=2` at the ends of a run.** No smoother buys enough to justify its tuning and its lag.

**Objects:** speed noise is about 0.15 m/s typical and 0.7 m/s p95, so it is usable with a margin.
Per-tick object acceleration is **not** usable as an input to the equations, smoothed or not: the
noise floor (p95 1.5–2.6 m/s²) is as large as real braking. The bounds should take object
acceleration from class limits (T013, T019), not from the derived column.

**Confidence:**
- high for ego speed and yaw rate: 5 clips from 0 to 35 m/s, all consistent;
- medium for the acceleration numbers, since the reference is itself an estimate;
- medium for the object noise floor, which rests on parked objects.

## Follow-ups
- **T007 step 7:** unchanged method, plus `edge_order=2` (velocity and acceleration at the ends of a
  run). This is a small code change and will get its own bug ticket if you agree. Tests: exact on a
  constant-acceleration path at the ends.
- **T013 / T019:** object acceleration comes from class bounds. The derived `ax, ay` are for viewing
  and sanity checks only. Speed noise (0.7 m/s p95, more at range) belongs in the margin.
- **T012:** online's clock offset to offline is not constant (−35 to +55 ms over 5 clips), so it
  must not be hard-coded when Alpamayo output is placed in the scene. Also, online `curvature` is
  wrong on 23958e60 (+1.1°/s yaw-rate offset), so don't rely on it if the Alpamayo input path uses it.

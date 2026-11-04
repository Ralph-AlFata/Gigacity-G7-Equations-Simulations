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
- [x] Added 2026-11-04 (review): **causal methods only**. The replay simulates a live car, so motion at
      tick t may use only ticks ≤ t. Repeat the comparison with backward differences, trailing
      Savitzky–Golay and a forward-only Kalman filter. Then change step 7 to the best one.

## Hypotheses
1. Speed agrees within ~0.1 m/s; acceleration is noisy without smoothing.

## Findings
### Part 1: centred methods (superseded by part 2)
These methods use ticks after t, so they are not allowed in the replay. They are kept here because
they show what the data allows when the future is known, and they describe the reference.

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

### Side finding: acceleration at the ends of a run is biased (moot after part 2)
- **What:** T007 uses `np.gradient` with first-order ends. On a constant-acceleration path, the
  acceleration at the first and last tick of every run is off by **half the true acceleration**
  (0.75 m/s² for a true 1.5 m/s²). Velocity is fine there.
- **Where it matters:** for the ego this is just the clip's first and last tick. For objects it is
  every track's birth, death, and each gap longer than 0.5 s.
- **Fix:** `np.gradient(..., edge_order=2)` is exact on that path.

### Part 2: causal methods (decided 2026-11-04)
Script: `scripts/spike_t009_causal.py` (`uv run python scripts/spike_t009_causal.py`, ~35 s).

**Methods:** each uses only ticks ≤ t, with a warm-up at the start of a run.
- **Backward difference** over 0.1, 0.2, 0.3 and 0.5 s, for velocity, then the same again for
  acceleration.
- **Trailing Savitzky–Golay** over 0.3–1.5 s, poly 1–3, evaluated at the newest tick.
- **Forward-only Kalman** (constant acceleration, no smoother pass), q = 0.3–300 m²/s⁵.
- **Mixed:** velocity by backward difference over 0.1 s, and acceleration as the change of *that*
  velocity over 0.1–0.5 s.

**Checks:**
1. **Ego:** derived from the offline poses, against the recorded motion. These are horizontal vector
   errors in the rig frame, not just speed.
2. **Delay:** the shift of the recorded motion that best matches the derived one.
3. **Objects under real noise:** the ego path plus the real label noise of each parked track (its
   deviations from its median position; 223 sequences, 27 416 ticks), against the ego's recorded
   motion. This tests object-like data where the truth is known. Results are split by the parked
   object's distance.
4. **Parked objects:** derived motion, whose true value is about 0.

**Selected rows, p95:**
| Method | v delay | a delay | ego v | ego a | ego a, \|a\|>1 | ego yaw rate | object v near / mid / far | object a near / mid / far |
|---|---|---|---|---|---|---|---|---|
| **backward 0.1 s** (v) | 0.05 s | — | **0.125** | — | — | **1.25** | **0.47 / 0.61 / 0.72** | — |
| backward 0.3 s | 0.15 s | 0.27 s | 0.32 | 0.56 | 0.63 | 1.47 | 0.52 / 0.61 / 0.71 | 1.63 / 2.06 / 2.39 |
| trailing SG 0.5 s p2 | 0.00 s | 0.19 s | 0.094 | 0.59 | 0.66 | 1.23 | 0.55 / 0.71 / 0.83 | 1.89 / 2.36 / 2.77 |
| trailing SG 0.9 s p2 | 0.01 s | 0.37 s | 0.104 | 0.60 | 0.68 | 1.27 | 0.62 / 0.83 / 0.94 | 1.55 / 1.92 / 2.23 |
| trailing SG 1.1 s p1 | 0.49 s | 0.47 s | 1.02 | 0.65 | 0.77 | 3.75 | 0.97 / 1.11 / 1.11 | 1.37 / 1.71 / 1.96 |
| Kalman fwd q=1 | 0.01 s | 0.47 s | 0.14 | 0.64 | 0.75 | 1.56 | 0.58 / 0.80 / 0.90 | 1.31 / 1.66 / 1.86 |
| Kalman fwd q=30 | 0.00 s | 0.27 s | 0.093 | 0.52 | 0.64 | 1.25 | 0.60 / 0.79 / 0.91 | 1.88 / 2.34 / 2.76 |
| **a = Δv over 0.4 s** (v from backward 0.1 s) | — | 0.19 s | — | **0.54** | **0.59** | — | — | **1.61 / 2.01 / 2.35** |
| a = Δv over 0.5 s | — | 0.27 s | — | 0.55 | 0.62 | — | — | 1.43 / 1.84 / 2.12 |

Units are m/s, m/s² and °/s. "Near" is under 20 m, "mid" 20–40 m, "far" over 40 m. The full grid
(29 + 7 methods) is printed by the script.

**What the numbers say:**
- **Velocity under object noise: the plainest method wins.** The backward difference over 0.1 s is
  best or tied at every distance, and only 0.05 s late.
  - **The noise is slow drift, not jitter.** Averaging longer barely reduces it (parked p95
    0.67 → 0.64 m/s from 0.1 to 0.3 s) and adds delay.
  - **Methods that model acceleration amplify it.** Savitzky–Golay p2/p3 and Kalman remove the delay
    on the clean ego (0.09 m/s), but are worse under label noise.
- **Velocity on the ego:** Kalman / Savitzky–Golay p2 beat the backward difference by only
  0.03 m/s (0.09 against 0.125 m/s p95). That is not worth a second method for the ego.
- **Acceleration:**
  - **A causal acceleration is always late.** The delay is 0.2–0.5 s; it needs a span of past
    velocity.
  - **Ego:** the change of the 0.1 s velocity over the last 0.4 s is the best simple choice:
    0.54 m/s² p95, 0.59 during manoeuvres over 1 m/s², 0.19 s late. Kalman q=30 is 0.52, but is
    tuned and just as late.
  - **Objects:** label noise keeps acceleration at 1.3 m/s² p95 or more at every setting, even near
    and 0.5 s late. It is not usable as an input.
- **Real data check:** the rebuilt offline scenes, read from `ego.parquet` itself, give velocity
  0.125 m/s and acceleration 0.54 m/s² p95 against the recorded motion over the 5 clips, the same as
  the experiment.

### Hypotheses
1. *Speed agrees within ~0.1 m/s; acceleration is noisy without smoothing:*
   - **Speed is supported:** p95 0.07 m/s centred, 0.125 m/s (vector) causal.
   - **Acceleration only partly:** for the ego, 0.39 m/s² p95 centred and 0.54 m/s² causal, and
     smoothing only trims it.
   - **For objects, acceleration is noisy** (≥ 1.3 m/s² p95 causal), and no smoother rescues it.

## Answer
**Step 7 is now causal**, implemented in `build.py` (`derive_motion`, `Rules.accel_span_s`):
- **Velocity and yaw rate:** backward difference over the last 0.1 s, the latest two positions.
  - For the ego it is within 0.125 m/s and 1.25°/s p95 of what the car recorded, 0.05 s late.
  - Under real object label noise it is within 0.47 m/s near, 0.61 mid and 0.72 far, the best of
    all causal methods tried.
- **Acceleration:** (v(t) − v(t−0.4 s)) / 0.4 s, with a shorter span while a run is younger.
  - For the ego it is 0.54 m/s² p95, 0.19 s late.
- **NaN where nothing earlier exists:** velocity at a run's first tick, acceleration at its first two.
  A live system wouldn't know a new object's velocity either.
- **No smoothing:** no causal smoother is better for objects.

**For the equations:**
- **Object velocity** carries about 0.5–0.7 m/s p95 of noise. That belongs in the margin.
- **Object acceleration** must come from class bounds (T013, T019), not from `ax, ay`: its noise
  floor (1.6–2.4 m/s² p95) is as large as real braking.

**Confidence:**
- high for the choice of the backward difference: it is best or tied on every object check, and the
  simplest;
- medium for the absolute object numbers, which rest on parked-track noise added to the ego path.

## Follow-ups
- **T007 step 7:** updated in this PR. Tests:
  - changing a later tick changes no earlier motion, for ego and objects;
  - acceleration is exact for constant acceleration after warm-up;
  - first-tick NaN, on synthetic and real scenes.

  The run-end `np.gradient` bias found in part 1 is gone with the central differences.
- **T013 / T014:** velocity can be NaN at a track's first tick (and after a gap longer than 0.5 s).
  The solver must handle an in-scene object with unknown velocity. Object acceleration comes from
  class bounds.
- **Other places the scene uses later data** (sorted 2026-11-04 by the rule "the environment may use
  the future; calculations and sensor readings may not"):
  1. **Interpolation between sightings:** up to 0.1 s ahead normally, 0.5 s across a bridged gap.
     This builds the environment (the true world state), so it is **fine as is**.
  2. **Lidar observability (T008):** this is a sensor reading. It uses the spin crossing nearest the
     tick, up to about 50 ms after it, so it **must change**: use the latest crossing at or before the
     tick, which makes readings 0–100 ms old. The box the points are counted against can stay the
     true interpolated box, since that is environment. This needs a new ticket (T008 follow-up).
- **T012:**
  - online's clock offset to offline is not constant (−35 to +55 ms over 5 clips), so it must not be
    hard-coded;
  - online `curvature` is wrong on 23958e60.

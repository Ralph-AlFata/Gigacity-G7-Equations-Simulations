# T004 — `egomotion` vs `egomotion.offline`

**Type:** Spike
**Epic:** E01
**Status:** todo
**Depends on:** T002
**Timebox:** half a day

## Question
How do the online and offline ego poses differ for our two clips, and which one should the
scene geometry use (and which one should Alpamayo's input use)?

## Why it matters
Obstacle labels are given relative to the car at each label's timestamp. Placing them in the
ground frame uses an ego pose. If we use a pose source that differs from the one the labels were
produced with, obstacles land in slightly wrong places, which directly biases the margins and H.
Alpamayo was trained on the online `egomotion`, so its input may need a different choice.
Recorded in T001 as "keep both, revisit".

## Method
- [ ] Documentation: NVIDIA wiki pages on egomotion; dataset card notes on `.offline` features.
- [ ] Data check: sample rate, time range, start pose, and columns of each (the online one has
      velocity/acceleration/curvature columns according to the SDK).
- [ ] Data check: position and heading difference between the two over each clip.
- [ ] Data check: transform obstacles to the ground frame with each source and compare how still
      parked objects stay (smaller spread = consistent with the labels).

## Hypotheses
1. Offline is a smoothed, resampled (10 Hz) version of online with near-identical positions.
2. The obstacle labels were produced with the offline poses, so parked objects stay stiller
   with offline than with online.

## Findings
<…>

## Answer
<…>

## Follow-ups
<Update T001 with the final choice.>

# T023 — What H means: computed once when a plan arrives, checked against the true future

**Type:** Decision
**Epic:** E03
**Status:** done
**Depends on:** T011

## Context
E03 needs a precise meaning for H, one that matches how it will be used in a real system.

**The real system (Ralph, 2026-11-12):**
1. The car receives a plan (reference trajectory) from the Alpamayo server.
2. On receipt, it computes H once from what it knows at that moment: how long it is safe to keep
   following this plan.
3. Its next inference request carries H. The server's priority scheduler serves the most urgent
   request first: the shorter H, the more urgent.
4. The new plan must arrive before the current one runs out of safe time.

Because H decides how long a car may wait, it must be safe. So two questions are answered on the
dataset:
- **A. How long is H?** Its statistics.
- **B. Was H right?** If the car had followed the plan for H, would it have touched anything?

## Options
### A. H recomputed at every tick while following the plan
- Pros: reacts to objects that appear while following.
- Cons: not what the system does. It also needs a simulated car driving between plans, and
  decisions about where each new plan starts.

### B. H computed once per plan, on receipt
- Pros: exactly what the car sends to the scheduler. Every plan is evaluated on its own, so no
  driving loop is needed to compute H.
- Cons: a plan's H does not react to anything seen after it arrives.

## Decision
**B.** Settled with Ralph on 2026-11-12:
- **H is computed once, when the plan is received,** from what is known at that tick. It is never
  updated while the plan is followed.
- **Every sample is evaluated.** When a snapshot has several trajectories, each gets its own H.
- **Unsafe means contact.** It is unsafe if the ego's box, placed on the plan, touches an object's
  true box before H. Near misses are not failures.
- **Failures are split by cause and reported separately:**
  - **Objects known at t** (in the scene at t): the equations or their bounds were wrong.
  - **Objects not known at t** (not yet existing, or not observed): H could not have known about
    them. This is a perception question, not an equation one.
  The total is shown as well.
- **Where the plans come from:** a separate process, run by Ralph, generates the dataset of
  snapshots and their Alpamayo plans (for example every 0.3 s of a clip) and does the simulation
  for it. This pipeline takes those plans as given, in the T011 format, at the ticks they exist.
  The recorded path (T011) is a stand-in until then.

**For the scheduler (not built here):** a request should carry the deadline (receipt time + H)
rather than H alone, because requests wait in a queue, and the server serves the earliest
deadline first. H cannot exceed the plan's length (6.4 s for Alpamayo), so all plans safe for
their whole length look equally non-urgent.

## Consequences
- **T014:** one H per (plan tick, sample), computed once, from the objects in the scene at that
  tick.
- **T015:** the main safety truth is box contact along the plan, with the ego following the plan
  exactly. Failures are labelled by cause. H\* (the first real contact) gives the slack H\* − H.
- **T016** reports both questions:
  - **A (length):** median and the low tail (p5, p10), the share capped, and the share below the
    inference time (~1.2 s on an A40).
  - **B (safety):** the failure rate with a 95% confidence bound, not just a count. With 0 failures
    in n independent plans, the bound is about 3/n. Plans close in time are not independent.
- **Limits stated with every result:**
  - objects replay their recording and do not react to a plan that differs from what the real car
    did;
  - the ego follows the plan perfectly, so the tracking allowance ρ is not tested (T018);
  - the recorded path is safe by construction, because the real car drove it, so its box check
    only tests the code.
- **Later:** a sweep over the bounds (ε_p, ε_v, ā), giving the failure rate against the median H;
  then a scheduler simulation with many cars.

# Hand example for the H solver (T013)

One plan, two objects and fixed config values, worked out step by step from the equations (E03
README, `Deadline_Aware_Multi_Vehicle_Serving.pdf` pp. 4–9). `tests/test_horizon.py` asserts the
solver gives these numbers. **Check the arithmetic once; after that, any solver change is checked
against it.**

## Inputs

**Config**

| Symbol | Value |
|---|---|
| r_veh | 2.0 m (given, not auto) |
| ρ0 | 0.2 m |
| ē_v | 0.3 m/s |
| ē_θ | 0.1 rad |
| ε_p | 0.3 m |
| ε_v | 0.5 m/s |
| ā | 2.0 m/s² |
| t_valid | 6.4 s |

**Plan:** straight along x at 10 m/s from the origin, steps every 0.1 s up to 6.4 s:
p_ref(τ) = (10τ, 0), speed 10 m/s at every step.

**Objects at t**

| | Position ô (m) | Velocity ŵ (m/s) | Box L × W (m) |
|---|---|---|---|
| A, parked car | (40, 3) | (0, 0) | 4 × 2 |
| B, pedestrian walking toward the path | (30, −8) | (0, 1.5) | 0.6 × 0.6 |

## Step 1: constants

- T_cap = min(6.4, 6.4) = **6.4 s**.
- v̄_ref = max speed over [0, 6.4] = **10 m/s**.
- q̄ = ē_v + 2·v̄_ref·sin(ē_θ/2) = 0.3 + 2·10·sin(0.05) = 0.3 + 20 × 0.0499792 = **1.299583 m/s**.
- r_A = ½√(4² + 2²) = ½√20 = **2.236068 m**.
- r_B = ½√(0.6² + 0.6²) = ½√0.72 = **0.424264 m**.
- The part of Sₖ that grows with τ is the same for both objects:
  (q̄ + ε_v)·τ + ½·ā·τ² = 1.799583·τ + τ².
- The constant parts:
  - A: r_veh + r_A + ρ0 + ε_p = 2 + 2.236068 + 0.2 + 0.3 = **4.736068 m**;
  - B: 2 + 0.424264 + 0.2 + 0.3 = **2.924264 m**.

## Step 2: distances, required separations and margins

- D_A(τ) = ‖(10τ, 0) − (40, 3)‖ = √((40 − 10τ)² + 9)
- D_B(τ) = ‖(10τ, 0) − (30, −8 + 1.5τ)‖ = √((30 − 10τ)² + (8 − 1.5τ)²)

| τ (s) | D_A | S_A | m_A | D_B | S_B | m_B |
|---|---|---|---|---|---|---|
| 0 | √1609 = 40.112342 | 4.736068 | **35.376274** | √964 = 31.048349 | 2.924264 | **28.124085** |
| 0.5 | √1234 = 35.128336 | 5.885860 | **29.242476** | √(625 + 52.5625) = 26.030031 | 4.074056 | **21.955975** |
| 1.0 | √909 = 30.149627 | 7.535651 | **22.613976** | √(400 + 42.25) = 21.029741 | 5.723847 | **15.305893** |

Check of one S entry: S_A(0.5) = 4.736068 + 1.799583 × 0.5 + 0.25 = 5.885860.

## Step 3: H

H is the first τ where min(m_A, m_B) ≤ 0. Margins every 0.1 s around the crossing:

| τ (s) | m_A | m_B |
|---|---|---|
| 1.9 | 9.4479 | 2.1924 |
| 2.0 | 7.8885 | **0.6569** |
| 2.1 | 6.3102 | **−0.8898** |
| 2.5 | −0.1880 | −7.1110 |

B crosses first, between 2.0 and 2.1 s. Solving D_B(τ) = S_B(τ) on that interval by bisection:
**H = 2.04255 s, limiting object B.** A alone would only cross at 2.48867 s.

So: the pedestrian limits the plan. At 10 m/s, the car may keep following it for about 2.04 s.

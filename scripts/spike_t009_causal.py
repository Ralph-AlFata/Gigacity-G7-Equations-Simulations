"""T009, part 2: causal differentiators only. Each value at tick t uses positions at t and before,
never after, because the replay must only know what a car would know at t (decided 2026-11-04).

Three checks per method, on the same 5 clips as part 1 (`spike_t009_motion.py`):
1. *Ego:* derived from the offline ego poses, against online egomotion's own motion columns.
2. *Objects, noise:* derived motion of parked objects, whose true motion is ~0.
3. *Objects, fidelity:* the ego path plus real label noise (the deviations of parked objects from
   their median position), against the ego's recorded motion. This tests object-like data where
   the truth is known.
Also the delay of each method: the time shift of the recorded acceleration that best matches it.

    uv run python scripts/spike_t009_causal.py
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from spike_t009_motion import CLIPS, RAW, SCENE, Recorded, best_lag  # noqa: E402

from horizon_pipeline.scene.build import DT_S  # noqa: E402
from horizon_pipeline.scene.frames import QUAT, XYZ  # noqa: E402

PARKED_MIN_TICKS = 50
PARKED_MAX_SPAN_M = 1.0
DELAYS_S = np.arange(0.0, 0.6001, 0.01)
BANDS = {"near": (0, 20), "mid": (20, 40), "far": (40, np.inf)}


# ---------------------------------------------------------------- causal differentiators


def backward(x: np.ndarray, k: int) -> np.ndarray:
    """(x[i] - x[i-k]) / (k dt); a shorter span while fewer than k earlier ticks exist; NaN at i = 0."""
    x = np.asarray(x, float)
    d = np.full_like(x, np.nan)
    for i in range(1, len(x)):
        j = max(0, i - k)
        d[i] = (x[i] - x[j]) / ((i - j) * DT_S)
    return d


@lru_cache
def trailing_coeffs(window: int, poly: int, deriv: int) -> np.ndarray:
    return savgol_coeffs(window, poly, deriv=deriv, delta=DT_S, pos=window - 1, use="dot")


def trailing_sg(x: np.ndarray, deriv: int, window: int, poly: int) -> np.ndarray:
    """Least-squares polynomial over the last `window` ticks, differentiated at the newest one.
    While fewer ticks exist the window grows (poly capped so the fit is determined); NaN until at
    least deriv + 1 ticks exist."""
    x = np.asarray(x, float)
    d = np.full_like(x, np.nan)
    for i in range(deriv, len(x)):
        w = min(window, i + 1)
        p = min(poly, w - 1)
        if p < deriv:
            continue
        c = trailing_coeffs(w, p, deriv)
        d[i] = np.tensordot(c, x[i - w + 1 : i + 1], axes=(0, 0))
    return d


def kalman_forward(x: np.ndarray, q: float, r: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Constant-acceleration Kalman filter, forward pass only (no smoother). q: white-jerk density
    (m²/s⁵), r: position noise (m). Velocity NaN at the first tick, acceleration at the first two."""
    x = np.asarray(x, float).reshape(len(x), -1)
    dt = DT_S
    F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    Q = q * np.array([
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
        [dt**3 / 6, dt**2 / 2, dt],
    ])  # fmt: skip
    H = np.array([1.0, 0, 0])
    vel, acc = np.full_like(x, np.nan), np.full_like(x, np.nan)
    for c in range(x.shape[1]):
        m, P = np.array([x[0, c], 0.0, 0.0]), np.diag([r * r, 1e4, 1e4])
        for i in range(len(x)):
            if i:
                m, P = F @ m, F @ P @ F.T + Q
            S = H @ P @ H + r * r
            K = P @ H / S
            m = m + K * (x[i, c] - H @ m)
            P = P - np.outer(K, H @ P)
            if i >= 1:
                vel[i, c] = m[1]
            if i >= 2:
                acc[i, c] = m[2]
    return vel, acc


def methods() -> dict:
    out = {}
    for k in (1, 2, 3, 5):
        out[f"backward {k * DT_S:.1f} s"] = lambda x, k=k: (backward(x, k), backward(backward(x, k), k))
    for w in (3, 5, 7, 9, 11, 15):
        for p in (1, 2, 3):
            if p >= w - 1 and w > 3:
                continue
            pa = max(p, 2)  # acceleration needs at least a quadratic
            out[f"trailing SG {w * DT_S:.1f} s p{p}"] = lambda x, w=w, p=p, pa=pa: (
                trailing_sg(x, 1, w, p), trailing_sg(x, 2, w, pa)
            )
    for q in (0.3, 1, 3, 10, 30, 100, 300):
        out[f"Kalman fwd q={q:g}"] = lambda x, q=q: kalman_forward(x, q)
    return out


# ---------------------------------------------------------------- measurements


def body(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).inv().apply(vec)


def derive(method, ego: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vel, acc = method(ego[XYZ].to_numpy(float, copy=True))
    yaw_rate, _ = method(ego["yaw"].to_numpy(float, copy=True)[:, None])
    return vel, acc, yaw_rate


def ego_errors(derived, ego: pd.DataFrame, rec: Recorded, t_us: np.ndarray, delay_s: float = 0.0) -> dict:
    """Horizontal velocity and acceleration errors in the rig frame (vector norms), and yaw rate (°/s)."""
    vel, acc, yaw_rate = derived
    q = ego[QUAT].to_numpy(float, copy=True)
    ref_t = t_us - int(round(delay_s * 1e6))
    ok = rec.covers(ref_t)
    v_ref, a_ref, w_ref = rec.vectors(ref_t[ok])
    v = body(np.nan_to_num(vel[ok]), q[ok])[:, :2]
    a = body(np.nan_to_num(acc[ok]), q[ok])[:, :2]
    good_v, good_a = ~np.isnan(vel[ok, 0]), ~np.isnan(acc[ok, 0])
    return {
        "v": np.linalg.norm(v - v_ref[:, :2], axis=1)[good_v],
        "a": np.linalg.norm(a - a_ref[:, :2], axis=1)[good_a],
        "a_dyn": np.linalg.norm(a - a_ref[:, :2], axis=1)[good_a & (np.linalg.norm(a_ref[:, :2], axis=1) > 1.0)],
        "w": np.abs(np.degrees(yaw_rate[ok, 0]) - w_ref)[good_v],
    }


def parked_runs(objects: pd.DataFrame) -> list[tuple[np.ndarray, float]]:
    """Positions (n x 3) and median distance from the ego of each run of consecutive ticks of each
    parked track."""
    runs = []
    for _, g in objects.groupby("track_id"):
        xy = g[["x", "y"]].to_numpy()
        if len(g) < PARKED_MIN_TICKS or np.ptp(xy, axis=0).max() >= PARKED_MAX_SPAN_M:
            continue
        tick = g["tick"].to_numpy()
        run = np.concatenate([[0], np.cumsum(np.diff(tick) != 1)])
        runs += [
            (g.loc[run == r, XYZ].to_numpy(float, copy=True), float(g.loc[run == r, "distance"].median()))
            for r in np.unique(run)
            if (run == r).sum() >= 3
        ]
    return runs


def main() -> None:
    clips = []
    for clip in CLIPS:
        rec = Recorded(RAW / clip)
        ego = pd.read_parquet(SCENE / clip / "offline" / "ego.parquet")
        lag = best_lag(ego, rec)  # the clock offset, found with the centred method of part 1
        t_us = ego["timestamp_us"].to_numpy(np.int64) + int(round(lag * 1e6))
        runs = parked_runs(pd.read_parquet(SCENE / clip / "offline" / "objects.parquet"))
        clips.append((clip, rec, ego, t_us, runs))
    all_runs = [r for *_, runs in clips for r in runs]
    noise = [(r - np.median(r, axis=0), d) for r, d in all_runs]  # real label noise sequences, ~zero mean

    P = lambda parts: np.percentile(np.concatenate(parts), 95) if parts else np.nan  # noqa: E731
    rows = []
    for name, m in methods().items():
        ego_e = {"v": [], "a": [], "a_dyn": [], "w": []}
        noisy_e = {band: {"v": [], "a": []} for band in BANDS}
        derived = [derive(m, ego) for _, _, ego, _, _ in clips]
        for (clip, rec, ego, t_us, _), d in zip(clips, derived):
            e = ego_errors(d, ego, rec, t_us)
            for k in ego_e:
                ego_e[k].append(e[k])
        # Fidelity under object noise: the ego path plus each noise sequence in turn, at
        # deterministic offsets along the clips.
        for n, (z, dist) in enumerate(noise):
            clip, rec, ego, t_us, _ = clips[n % len(clips)]
            L = min(len(z), len(ego))
            start = (n * 37) % (len(ego) - L + 1)
            seg = ego.iloc[start : start + L].reset_index(drop=True).copy()
            seg[XYZ] = seg[XYZ].to_numpy(float, copy=True) + z[:L]
            e = ego_errors(derive(m, seg), seg, rec, t_us[start : start + L])
            band = next(b for b, (lo, hi) in BANDS.items() if lo <= dist < hi)
            noisy_e[band]["v"].append(e["v"])
            noisy_e[band]["a"].append(e["a"])
        # Parked objects: derived motion is noise.
        pv, pa = [], []
        for r, _ in all_runs:
            v, a = m(r)
            pv.append(np.linalg.norm(v[:, :2], axis=1)[~np.isnan(v[:, 0])])
            pa.append(np.linalg.norm(a[:, :2], axis=1)[~np.isnan(a[:, 0])])
        # Delay: the shift of the recorded motion that best matches the derived velocity / acceleration.
        delay = {
            k: DELAYS_S[np.argmin([
                np.mean(np.concatenate([ego_errors(dv, ego, rec, t_us, d)[k] for (_, rec, ego, t_us, _), dv in zip(clips, derived)]) ** 2)
                for d in DELAYS_S
            ])]
            for k in ("v", "a")
        }  # fmt: skip
        rows.append({
            "method": name, "delay_v": delay["v"], "delay_a": delay["a"],
            "ego_v": P(ego_e["v"]), "ego_a": P(ego_e["a"]), "ego_a_dyn": P(ego_e["a_dyn"]), "ego_w": P(ego_e["w"]),
            **{f"obj_v_{b}": P(noisy_e[b]["v"]) for b in BANDS},
            **{f"obj_a_{b}": P(noisy_e[b]["a"]) for b in BANDS},
            "parked_v": P(pv), "parked_a": P(pa),
        })  # fmt: skip
    df = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.width", 200)
    print("p95 errors. ego_*: clean ego vs recorded; obj_*: ego + real label noise vs recorded; parked_*: parked")
    print("objects' derived motion (noise). v = horizontal velocity vector (m/s), a = acceleration (m/s²),")
    print("a_dyn = ticks with |a_recorded| > 1 m/s², w = yaw rate (°/s), delay_v / delay_a = best-matching delay (s).")
    print(f"obj_* by the parked object's distance: {', '.join(f'{b} {lo}-{hi} m' for b, (lo, hi) in BANDS.items())}.")
    for b, (lo, hi) in BANDS.items():
        z = [len(z) for z, d in noise if lo <= d < hi]
        print(f"  {b}: {len(z)} noise sequences, {sum(z)} ticks")
    print()
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()

"""T009 spike: is the motion derived from 10 Hz poses (T007 step 7) close to the motion the car
recorded (online `egomotion` velocity, acceleration and curvature)? And if not, which smoothing fixes it?

Reads data/raw/<clip>/egomotion.parquet and the built scenes data/scene/<clip>/<source>/ (run
`horizon-scene build --ego-source both` first). Prints the numbers quoted in
tickets/E02-scene/T009-spike-derived-vs-online-motion.md and writes time plots to
results/t009/<clip>.rrd (open with `rerun results/t009/<clip>.rrd`).

    uv run python scripts/spike_t009_motion.py [clip_id ...]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from horizon_pipeline.scene.build import DT_S
from horizon_pipeline.scene.frames import QUAT, XYZ, Poses, dense_end
from horizon_pipeline.scene.view import TIMELINE

RAW = Path("data/raw")
SCENE = Path("data/scene")
OUT = Path("results/t009")
CLIPS = [
    "b789fab8-41f1-4cf3-b7f8-a8e94939f67d",
    "b4d979af-254e-4bc0-afdd-3a36531f4540",
    "b6cb8b12-8b36-49c2-8609-63c4e2958e28",
    "ca823746-c767-48d9-bc0d-008ce50f6497",
    "23958e60-dbf9-41fa-b927-53960a114276",
]
QUANTITIES = ["speed", "a_long", "a_lat", "yaw_rate"]
UNITS = {"speed": "m/s", "a_long": "m/s²", "a_lat": "m/s²", "yaw_rate": "°/s"}
LAGS_S = np.arange(-0.100, 0.1001, 0.005)
PARKED_MIN_TICKS = 50
PARKED_MAX_SPAN_M = 1.0  # selected by position only, so the speed being measured plays no part
YAW_REF_MIN_CORR = 0.99  # below this, the recorded yaw rate disagrees with online's own yaw: not a reference


# ---------------------------------------------------------------- differentiators (10 Hz, one run of ticks)


def central(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Central difference over +-k ticks; np.gradient (+-1 tick, one-sided at the ends) where that
    does not fit. k = 1 is exactly T007 step 7."""
    d = np.gradient(x, DT_S, axis=0)
    if k > 1 and len(x) > 2 * k:
        d[k:-k] = (x[2 * k :] - x[: -2 * k]) / (2 * k * DT_S)
    return d


def savgol(x: np.ndarray, deriv: int, window: int, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay derivative; falls back to repeated central differences on a short run."""
    if len(x) < window:
        out = x
        for _ in range(deriv):
            out = central(out)
        return out
    return savgol_filter(x, window, poly, deriv=deriv, delta=DT_S, axis=0, mode="interp")


def kalman_ca(x: np.ndarray, q: float, r: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Constant-acceleration Kalman filter + RTS smoother, independently per column.
    q: white-jerk spectral density (m²/s⁵), r: measurement standard deviation (m)."""
    x = np.asarray(x, float).reshape(len(x), -1)
    dt = DT_S
    F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    Q = q * np.array([
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
        [dt**3 / 6, dt**2 / 2, dt],
    ])  # fmt: skip
    H = np.array([[1.0, 0, 0]])
    n = len(x)
    vel, acc = np.empty_like(x), np.empty_like(x)
    for c in range(x.shape[1]):
        m_f, P_f, m_p, P_p = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3))
        m, P = np.array([x[0, c], 0, 0]), np.diag([r * r, 100.0, 100.0])
        for i in range(n):
            if i:
                m, P = F @ m, F @ P @ F.T + Q
            m_p[i], P_p[i] = m, P
            S = H @ P @ H.T + r * r
            K = P @ H.T / S
            m = m + (K * (x[i, c] - H @ m)).ravel()
            P = (np.eye(3) - K @ H) @ P
            m_f[i], P_f[i] = m, P
        m_s = m_f.copy()
        for i in range(n - 2, -1, -1):
            G = P_f[i] @ F.T @ np.linalg.inv(P_p[i + 1])
            m_s[i] = m_f[i] + G @ (m_s[i + 1] - m_p[i + 1])
        vel[:, c], acc[:, c] = m_s[:, 1], m_s[:, 2]
    return vel, acc


METHODS = {
    "central ±0.1 s (T007, now)": lambda x: (central(x), central(central(x))),
    "central ±0.2 s": lambda x: (central(x, 2), central(central(x, 2), 2)),
    "central ±0.3 s": lambda x: (central(x, 3), central(central(x, 3), 3)),
    "Savitzky-Golay 0.7 s": lambda x: (savgol(x, 1, 7), savgol(x, 2, 7)),
    "Savitzky-Golay 1.1 s": lambda x: (savgol(x, 1, 11), savgol(x, 2, 11)),
    "Kalman CA, q=1": lambda x: kalman_ca(x, 1.0),
    "Kalman CA, q=10": lambda x: kalman_ca(x, 10.0),
    "Kalman CA, q=100": lambda x: kalman_ca(x, 100.0),
}
CURRENT = next(iter(METHODS))


def ego_motion(ego: pd.DataFrame, method) -> pd.DataFrame:
    """The scene's ego states re-differentiated with `method`: speed, body-frame accelerations, yaw rate."""
    vel, acc = method(ego[XYZ].to_numpy(float, copy=True))
    yaw_rate, _ = method(ego["yaw"].to_numpy(float, copy=True))
    acc_body = Rotation.from_quat(ego[QUAT].to_numpy(float, copy=True)).inv().apply(acc)
    return pd.DataFrame({
        "speed": np.linalg.norm(vel, axis=1), "a_long": acc_body[:, 0], "a_lat": acc_body[:, 1],
        "yaw_rate": np.degrees(np.ravel(yaw_rate)),
    })  # fmt: skip


# ---------------------------------------------------------------- the recorded motion


class Recorded:
    """Online egomotion's own motion columns over its dense 100 Hz part. Velocity and acceleration
    are in the online anchor frame (T004); they are rotated into the rig frame with online's own
    orientation. Yaw rate = curvature x forward speed (checked against the derivative of online yaw)."""

    def __init__(self, clip_dir: Path):
        e = pd.read_parquet(clip_dir / "egomotion.parquet")
        e = e.iloc[: dense_end(e["timestamp"].to_numpy(np.int64)) + 1]
        self.t = e["timestamp"].to_numpy(np.int64)
        self.cols = e
        self.poses = Poses.from_frame(e)

    def covers(self, t_us: np.ndarray) -> np.ndarray:
        return (t_us >= self.t[0]) & (t_us <= self.t[-1])

    def at(self, t_us: np.ndarray) -> pd.DataFrame:
        col = lambda names: np.column_stack([np.interp(t_us, self.t, self.cols[n]) for n in names])  # noqa: E731
        _, rot = self.poses.at(t_us)
        v_body = rot.inv().apply(col(["vx", "vy", "vz"]))
        a_body = rot.inv().apply(col(["ax", "ay", "az"]))
        return pd.DataFrame({
            "speed": np.linalg.norm(v_body, axis=1), "a_long": a_body[:, 0], "a_lat": a_body[:, 1],
            "yaw_rate": np.degrees(np.interp(t_us, self.t, self.cols["curvature"]) * v_body[:, 0]),
        })  # fmt: skip

    def yaw_rate_check(self) -> tuple[float, float]:
        """curvature x forward speed against the 100 Hz derivative of online yaw: (RMS °/s, correlation)."""
        _, rot = self.poses.at(self.t)
        yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])
        derived = np.degrees(np.gradient(yaw, self.t / 1e6))
        rec = self.at(self.t)["yaw_rate"].to_numpy()
        return float(np.sqrt(np.mean((rec - derived) ** 2))), float(np.corrcoef(rec, derived)[0, 1])


def best_lag(ego: pd.DataFrame, rec: Recorded) -> float:
    """The shift of the recorded clock (s) that best matches derived speed (T004 found +40-50 ms for
    offline against online)."""
    derived = ego_motion(ego, METHODS[CURRENT])["speed"].to_numpy()
    t = ego["timestamp_us"].to_numpy(np.int64)
    rms = []
    for lag in LAGS_S:
        q = t + int(round(lag * 1e6))
        ok = rec.covers(q)
        rms.append(np.sqrt(np.mean((derived[ok] - rec.at(q[ok])["speed"].to_numpy()) ** 2)))
    return float(LAGS_S[int(np.argmin(rms))])


def errors(derived: pd.DataFrame, recorded: pd.DataFrame) -> dict:
    out = {}
    for q in QUANTITIES:
        e = (derived[q] - recorded[q]).to_numpy()
        e = e[~np.isnan(e)]
        out[q] = {"bias": e.mean(), "mean": np.abs(e).mean(), "p95": np.percentile(np.abs(e), 95), "max": np.abs(e).max()}
    return out


# ---------------------------------------------------------------- objects: the parked-car noise floor


def parked_noise(objects: pd.DataFrame, method) -> tuple[np.ndarray, np.ndarray]:
    """Derived speed and horizontal acceleration of tracks that barely move (position-selected), per
    run of consecutive ticks. True speed is ~0, so what is measured is noise."""
    speed, acc = [], []
    for _, g in objects.groupby("track_id"):
        xy = g[["x", "y"]].to_numpy()
        if len(g) < PARKED_MIN_TICKS or np.ptp(xy, axis=0).max() >= PARKED_MAX_SPAN_M:
            continue
        tick = g["tick"].to_numpy()
        run = np.concatenate([[0], np.cumsum(np.diff(tick) != 1)])
        for r in np.unique(run):
            x = g.loc[run == r, XYZ].to_numpy(float, copy=True)
            if len(x) < 3:
                continue
            v, a = method(x)
            speed.append(np.linalg.norm(v, axis=1))
            acc.append(np.linalg.norm(a[:, :2], axis=1))
    return (np.concatenate(speed), np.concatenate(acc)) if speed else (np.array([]), np.array([]))


# ---------------------------------------------------------------- report


def section(title: str) -> None:
    print(f"\n--- {title}")


def table(rows: dict[str, dict]) -> None:
    """rows: label -> errors(...). Prints mean / p95 / max |error| and the bias per quantity."""
    head = "".join(f"{q + ' (' + UNITS[q] + ')':>30}" for q in QUANTITIES)
    print(f"{'':<30}{head}")
    print(f"{'':<30}" + f"{'mean   p95   max   bias':>30}" * len(QUANTITIES))
    for label, err in rows.items():
        cells = "".join(
            f"{err[q]['mean']:>9.3f}{err[q]['p95']:>6.2f}{err[q]['max']:>6.2f}{err[q]['bias']:>+7.3f}" + " " * 2
            for q in QUANTITIES
        )
        print(f"{label:<30}{cells}")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def layout() -> rrb.Blueprint:
    """One row per quantity: recorded and derived values on the left, their errors on the right."""
    rows = [
        rrb.Horizontal(
            rrb.TimeSeriesView(origin=f"plots/{q}", contents=["$origin/**", "- $origin/error/**"], name=f"{q} ({UNITS[q]})"),
            rrb.TimeSeriesView(origin=f"plots/{q}/error", name=f"{q} error ({UNITS[q]})"),
        )
        for q in QUANTITIES
    ]
    return rrb.Blueprint(rrb.Vertical(*rows), collapse_panels=True)


def plot(clip: str, t_s: np.ndarray, recorded: pd.DataFrame, series: dict[str, pd.DataFrame]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{clip}.rrd"
    rec = rr.RecordingStream("horizon-t009", recording_id=clip)
    rec.save(path, default_blueprint=layout())
    for q in QUANTITIES:
        rec.send_columns(f"plots/{q}/recorded", indexes=[rr.TimeColumn(TIMELINE, duration=t_s)],
                         columns=rr.Scalars.columns(scalars=recorded[q].to_numpy()))  # fmt: skip
        for name, df in series.items():
            for kind, values in (("", df[q]), ("error/", df[q] - recorded[q])):
                rec.send_columns(f"plots/{q}/{kind}{slug(name)}", indexes=[rr.TimeColumn(TIMELINE, duration=t_s)],
                                 columns=rr.Scalars.columns(scalars=values.to_numpy()))  # fmt: skip
    rec.flush()
    return path


def main(clips: list[str]) -> None:
    pooled: dict[str, list] = {m: [] for m in METHODS}
    no_yaw_ref = []
    online_rows, parked = {}, {m: ([], []) for m in METHODS}
    for clip in clips:
        rec = Recorded(RAW / clip)
        section(clip)
        rms, corr = rec.yaw_rate_check()
        print(f"recorded yaw rate (curvature x forward speed) vs d(online yaw)/dt: RMS {rms:.3f} °/s, corr {corr:.4f}")
        if corr < YAW_REF_MIN_CORR:
            no_yaw_ref.append(clip[:8])
            print(f"  -> curvature disagrees with online's own yaw: yaw rate left out of the pooled table")
        # Online scene against online's own columns: same clock and the same poses.
        ego_on = pd.read_parquet(SCENE / clip / "online" / "ego.parquet")
        t_on = ego_on["timestamp_us"].to_numpy(np.int64)
        ok = rec.covers(t_on)
        online_rows[clip[:8]] = errors(ego_motion(ego_on[ok].reset_index(drop=True), METHODS[CURRENT]), rec.at(t_on[ok]))
        # Offline scene (the one H uses): its clock runs ahead of online's (T004), so shift.
        ego = pd.read_parquet(SCENE / clip / "offline" / "ego.parquet")
        lag = best_lag(ego, rec)
        t = ego["timestamp_us"].to_numpy(np.int64) + int(round(lag * 1e6))
        ok = rec.covers(t)
        ego, recorded = ego[ok].reset_index(drop=True), rec.at(t[ok])
        print(f"offline: best clock lag {lag * 1e3:+.0f} ms, {ok.sum()} ticks, speed {recorded['speed'].min():.1f}-"
              f"{recorded['speed'].max():.1f} m/s, |a_long| max {recorded['a_long'].abs().max():.2f}, "
              f"|a_lat| max {recorded['a_lat'].abs().max():.2f} m/s², |yaw rate| max {recorded['yaw_rate'].abs().max():.1f} °/s")  # fmt: skip
        series = {name: ego_motion(ego, m) for name, m in METHODS.items()}
        rows = {name: errors(df, recorded) for name, df in series.items()}
        table(rows)
        for name, df in series.items():
            if clip[:8] in no_yaw_ref:
                df, recorded_ = df.assign(yaw_rate=np.nan), recorded.assign(yaw_rate=np.nan)
            else:
                recorded_ = recorded
            pooled[name].append((df, recorded_))
        objects = pd.read_parquet(SCENE / clip / "offline" / "objects.parquet")
        for name, m in METHODS.items():
            s, a = parked_noise(objects, m)
            parked[name][0].append(s)
            parked[name][1].append(a)
        keep = [CURRENT, "Savitzky-Golay 1.1 s", "Kalman CA, q=10"]
        print(f"plots: {plot(clip, ego['t_s'].to_numpy(), recorded, {k: series[k] for k in keep})}")

    section("online scene vs online's own columns (same clock, same poses), current method")
    table(online_rows)
    section(f"offline scenes, all {len(clips)} clips pooled (yaw rate without {', '.join(no_yaw_ref) or 'none'})")
    table({name: errors(pd.concat([d for d, _ in v]), pd.concat([r for _, r in v])) for name, v in pooled.items()})
    section(f"parked objects (>= {PARKED_MIN_TICKS} ticks, span < {PARKED_MAX_SPAN_M} m), offline, all clips: noise of derived motion")
    print(f"{'':<30}{'speed (m/s) p50   p95':>24}{'|a_xy| (m/s²) p50   p95':>28}")
    for name, (s, a) in parked.items():
        s, a = np.concatenate(s), np.concatenate(a)
        print(f"{name:<30}{np.median(s):>18.3f}{np.percentile(s, 95):>6.2f}{np.median(a):>22.3f}{np.percentile(a, 95):>6.2f}")
    print(f"({len(s)} object-ticks)")


if __name__ == "__main__":
    main(sys.argv[1:] or CLIPS)

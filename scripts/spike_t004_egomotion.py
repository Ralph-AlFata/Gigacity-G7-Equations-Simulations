"""T004 spike: how do `egomotion` (online) and `egomotion.offline` differ, and which one are the
obstacle labels consistent with?

Reads data/raw/<clip>/ (fetch the `scene` preset first) and prints the numbers quoted in
tickets/E01-raw-data/T004-spike-egomotion-online-vs-offline.md.

    uv run python scripts/spike_t004_egomotion.py [clip_id ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp

RAW = Path("data/raw")
DENSE_GAP_US = 50_000  # online samples further apart than this are the sparse tail
STATIC_RANGE_M = 3.0  # a track is "parked" if its horizontal positions span less than this
MIN_SIGHTINGS = 20
OFFSETS_MS = np.arange(-150, 151, 10)


class Poses:
    """Rig poses in the anchor frame, interpolated (linear translation, slerp rotation)."""

    def __init__(self, df: pd.DataFrame):
        self.t = df["timestamp"].to_numpy(np.int64)
        self.xyz = df[["x", "y", "z"]].to_numpy()
        self.rot = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy())
        self._slerp = Slerp(self.t.astype(float), self.rot)

    def at(self, t_us) -> tuple[np.ndarray, Rotation]:
        t = np.clip(np.asarray(t_us, dtype=float), self.t[0], self.t[-1])
        xyz = np.column_stack([np.interp(t, self.t, self.xyz[:, i]) for i in range(3)])
        return xyz, self._slerp(t)

    def subset(self, keep: np.ndarray) -> "Poses":
        df = pd.DataFrame(self.xyz[keep], columns=["x", "y", "z"])
        df[["qx", "qy", "qz", "qw"]] = self.rot[keep].as_quat()
        df["timestamp"] = self.t[keep]
        return Poses(df)


def yaw(rot: Rotation) -> np.ndarray:
    return rot.as_euler("ZYX")[..., 0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def section(title: str) -> None:
    print(f"\n--- {title}")


def timing(on: pd.DataFrame, off: pd.DataFrame, lidar: pd.DataFrame, obstacles: pd.DataFrame) -> np.ndarray:
    section("timing")
    t_on = on["timestamp"].to_numpy(np.int64)
    gaps = np.diff(t_on)
    first_gap = int(np.argmax(gaps > DENSE_GAP_US)) if (gaps > DENSE_GAP_US).any() else len(gaps)
    dense = np.zeros(len(t_on), bool)
    dense[: first_gap + 1] = True
    print(f"online : {len(on)} rows, {t_on[0] / 1e6:.3f} .. {t_on[-1] / 1e6:.3f} s")
    print(f"         dense part {t_on[0] / 1e6:.3f} .. {t_on[first_gap] / 1e6:.3f} s, "
          f"{dense.sum()} rows, median dt {np.median(np.diff(t_on[dense])) / 1e3:.2f} ms")
    tail = t_on[~dense]
    if len(tail):
        print(f"         sparse tail {len(tail)} rows to {tail[-1] / 1e6:.1f} s, "
              f"median dt {np.median(np.diff(t_on[first_gap:])) / 1e3:.0f} ms")
    t_off = off["timestamp"].to_numpy(np.int64)
    spins = lidar["spin_start_timestamp"].to_numpy(np.int64)
    inner = t_off[1:-1]
    print(f"offline: {len(off)} rows, {t_off[0] / 1e6:.3f} .. {t_off[-1] / 1e6:.3f} s, median dt {np.median(np.diff(t_off)) / 1e3:.2f} ms")
    print(f"         rows equal to a lidar spin start: {np.isin(t_off, spins).sum()} / {len(t_off)} "
          f"(inner rows {np.isin(inner, spins).sum()} / {len(inner)}; lidar spins: {len(spins)}); "
          f"first/last rows: {t_off[0]} / {t_off[-1]} us")
    others = inner[~np.isin(inner, spins)]
    if len(others):
        to_spin = np.abs(others[:, None] - spins[None, :]).min(1) / 1e3
        print(f"         the other {len(others)} inner rows: nearest spin start {np.median(to_spin):.1f} ms "
              f"(median), {to_spin.min():.1f} .. {to_spin.max():.1f} ms; their median dt to the previous row "
              f"{np.median(others - t_off[np.searchsorted(t_off, others) - 1]) / 1e3:.1f} ms")
    t_obs = obstacles["timestamp_us"].to_numpy(np.int64)
    print(f"obstacles: {t_obs.min() / 1e6:.3f} .. {t_obs.max() / 1e6:.3f} s, "
          f"inside online dense range: {np.mean((t_obs >= t_on[0]) & (t_obs <= t_on[first_gap])) * 100:.1f}%, "
          f"on a lidar spin start: {np.isin(t_obs, spins).mean() * 100:.1f}%")
    return dense


def frames_at_zero(online: Poses, offline: Poses) -> None:
    section("anchor frame at t = 0")
    for name, poses in (("online", online), ("offline", offline)):
        xyz, rot = poses.at([0])
        z, y, x = rot.as_euler("ZYX", degrees=True)[0]
        print(f"{name:<8} position {np.round(xyz[0], 4)} m, yaw {z:+.3f} deg, pitch {y:+.3f} deg, roll {x:+.3f} deg")


def trajectory_difference(online: Poses, offline: Poses) -> None:
    section("offline vs online, online re-anchored to its own rig pose at t = 0")
    t = offline.t[(offline.t >= online.t[0]) & (offline.t <= online.t[-1])]
    p0, r0 = online.at([0])
    p_on, r_on = online.at(t)
    rel_r = r0.inv() * r_on  # online rig pose expressed in the online rig frame at t=0
    rel_p = r0.inv().apply(p_on - p0)
    p_off, r_off = offline.at(t)
    d = rel_p - p_off
    horiz = np.linalg.norm(d[:, :2], axis=1)
    dyaw = np.degrees(wrap(yaw(rel_r) - yaw(r_off)))
    dangle = np.degrees((rel_r.inv() * r_off).magnitude())
    print(f"over {len(t)} offline timestamps, {t[0] / 1e6:.2f} .. {t[-1] / 1e6:.2f} s")
    print(f"horizontal |dp|: median {np.median(horiz):.3f}, max {horiz.max():.3f}, at end {horiz[-1]:.3f} m")
    print(f"vertical    dz : median {np.median(np.abs(d[:, 2])):.3f}, max {np.abs(d[:, 2]).max():.3f} m")
    print(f"yaw difference : median {np.median(np.abs(dyaw)):.3f}, max {np.abs(dyaw).max():.3f} deg")
    print(f"full rotation  : median {np.median(dangle):.3f}, max {dangle.max():.3f} deg")
    step_on = np.linalg.norm(np.diff(rel_p, axis=0), axis=1)
    step_off = np.linalg.norm(np.diff(p_off, axis=0), axis=1)
    print(f"path length    : online {step_on.sum():.2f} m, offline {step_off.sum():.2f} m")
    jerk = lambda p: np.linalg.norm(np.diff(p, n=3, axis=0), axis=1)  # noqa: E731
    print(f"3rd differences of position at the offline times (smoothness): online median "
          f"{np.median(jerk(rel_p)) * 1e3:.2f} mm, offline {np.median(jerk(p_off)) * 1e3:.2f} mm")

    # Time lag: speed is frame-independent, so compare online speed at t + lag with offline at t.
    def speed(poses: Poses, times):
        h = 50_000.0
        a, _ = poses.at(times - h)
        b, _ = poses.at(times + h)
        return np.linalg.norm(b - a, axis=1) / (2 * h / 1e6)

    inner_t = t[(t >= online.t[0] + 200_000) & (t <= online.t[-1] - 200_000)]
    v_off = speed(offline, inner_t)
    errors = np.array([np.sqrt(np.mean((speed(online, inner_t + lag * 1000) - v_off) ** 2)) for lag in OFFSETS_MS])
    best = OFFSETS_MS[int(np.argmin(errors))]
    print(f"speed match    : online(t + lag) best matches offline(t) at lag {best:+d} ms "
          f"(speed RMS diff {errors.min():.3f} m/s; at 0 ms {errors[OFFSETS_MS == 0][0]:.3f} m/s)")


def velocity_frame(on: pd.DataFrame, dense: np.ndarray) -> None:
    section("online velocity columns: anchor frame or rig frame?")
    d = on[dense]
    t = d["timestamp"].to_numpy(float) / 1e6
    v_cols = d[["vx", "vy", "vz"]].to_numpy()
    v_anchor = np.gradient(d[["x", "y", "z"]].to_numpy(), t, axis=0)
    v_rig = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].to_numpy(copy=True)).inv().apply(v_anchor)
    for name, v in (("anchor", v_anchor), ("rig", v_rig)):
        print(f"columns vs d(x,y,z)/dt in the {name} frame: RMS diff {np.sqrt(np.mean((v_cols - v) ** 2)):.3f} m/s")


def world_positions(obstacles: pd.DataFrame, poses: Poses, offset_us: float = 0.0):
    p, r = poses.at(obstacles["timestamp_us"].to_numpy(float) + offset_us)
    centers = obstacles[["center_x", "center_y", "center_z"]].to_numpy(copy=True)
    box = Rotation.from_quat(obstacles[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy(copy=True))
    return p + r.apply(centers), yaw(r * box)


def spreads(obstacles: pd.DataFrame, world: np.ndarray, heading: np.ndarray, tracks) -> pd.DataFrame:
    rows = {}
    for track in tracks:
        m = (obstacles["track_id"] == track).to_numpy()
        xy = world[m, :2]
        h = heading[m]
        rows[track] = {
            "pos_rms_m": float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean())),
            "yaw_std_deg": float(np.degrees(np.std(wrap(h - h[0])))),
        }
    return pd.DataFrame(rows).T


def parked_objects(obstacles: pd.DataFrame, sources: dict[str, Poses]) -> None:
    section("parked objects: spread of their ground-frame positions under each source")
    counts = obstacles["track_id"].value_counts()
    candidates = counts.index[counts >= MIN_SIGHTINGS]
    worlds = {name: world_positions(obstacles, poses) for name, poses in sources.items()}
    parked = []
    for track in candidates:
        m = (obstacles["track_id"] == track).to_numpy()
        spans = [np.ptp(w[m, :2], axis=0).max() for w, _ in worlds.values()]
        if min(spans) < STATIC_RANGE_M:
            parked.append(track)
    print(f"{len(parked)} parked tracks (>= {MIN_SIGHTINGS} sightings, span < {STATIC_RANGE_M} m under some source) "
          f"of {len(candidates)} tracks with enough sightings")
    if not parked:
        return
    table = {name: spreads(obstacles, w, h, parked) for name, (w, h) in worlds.items()}
    for name, s in table.items():
        print(f"{name:<22} position RMS median {s.pos_rms_m.median():.3f} m, mean {s.pos_rms_m.mean():.3f} m; "
              f"yaw std median {s.yaw_std_deg.median():.2f} deg")
    base = table["offline"].pos_rms_m
    for name, s in table.items():
        if name != "offline":
            print(f"offline smaller than {name}: {(base < s.pos_rms_m).sum()} / {len(base)} tracks")

    section("time offset that keeps parked objects stillest (pose looked up at label time + offset)")
    for name, poses in sources.items():
        rms = []
        for offset in OFFSETS_MS:
            w, h = world_positions(obstacles, poses, offset * 1000)
            rms.append(spreads(obstacles, w, h, parked).pos_rms_m.median())
        rms = np.array(rms)
        best = OFFSETS_MS[int(np.argmin(rms))]
        print(f"{name:<22} best offset {best:+d} ms (median RMS {rms.min():.3f} m; at 0 ms {rms[OFFSETS_MS == 0][0]:.3f} m; "
              f"at +-100 ms {rms[OFFSETS_MS == -100][0]:.3f} / {rms[OFFSETS_MS == 100][0]:.3f} m)")


def speed_summary(online: pd.DataFrame) -> None:
    v = np.linalg.norm(online[["vx", "vy", "vz"]].to_numpy(), axis=1)
    dense = online["timestamp"].between(0, 20_000_000)
    print(f"ego speed (online vx,vy,vz) over 0..20 s: median {np.median(v[dense]):.1f}, max {v[dense].max():.1f} m/s")


def main(clips: list[str]) -> None:
    for clip in clips:
        d = RAW / clip
        print(f"\n================ {clip}")
        on = pd.read_parquet(d / "egomotion.parquet")
        off = pd.read_parquet(d / "egomotion.offline.parquet")
        obstacles = pd.read_parquet(d / "obstacle.offline.parquet")
        lidar = pq.read_table(
            d / "lidar_top_360fov/pointclouds.parquet", columns=["spin_start_timestamp", "spin_end_timestamp"]
        ).to_pandas()
        speed_summary(on)
        dense = timing(on, off, lidar, obstacles)
        online, offline = Poses(on), Poses(off)
        online_dense = online.subset(dense)
        frames_at_zero(online, offline)
        velocity_frame(on, dense)
        trajectory_difference(online_dense, offline)
        # Control: online resampled at the offline timestamps, to separate rate from source.
        p, r = online_dense.at(offline.t)
        online_10hz = pd.DataFrame(p, columns=["x", "y", "z"])
        online_10hz[["qx", "qy", "qz", "qw"]] = r.as_quat()
        online_10hz["timestamp"] = offline.t
        parked_objects(
            obstacles,
            {"offline": offline, "online": online_dense, "online@offline times": Poses(online_10hz)},
        )


if __name__ == "__main__":
    main(sys.argv[1:] or Path("clips_selected_train.txt").read_text().split())

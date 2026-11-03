"""T005 spike: what is in `lidar_top_360fov` (columns, frame, timing, motion compensation, size)?

Reads data/raw/<clip>/ (fetch the `scene` preset first) and prints the numbers quoted in
tickets/E02-scene/T005-spike-lidar-format.md.

    uv run python scripts/spike_t005_lidar.py [clip_id ...]
"""

import sys
import time
from pathlib import Path

import DracoPy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from spike_t004_egomotion import Poses, section

RAW = Path("data/raw")
SAMPLE_SPINS = [0, 50, 100, 150, 198]
SEAM_COLUMNS = 5  # columns (0.1 deg each, ~0.14 ms) on each side of the seam
GROUND_BAND_M = 0.2  # points this close above the fitted ground plane count as ground
FAST_MPS = 3.0  # motion tests use spins faster than this; slower ones barely tell hypotheses apart
OFFSETS_MS = np.arange(-50, 151, 10)
PARKED_RANGE_M = 1.0  # a track is "parked" if its ground-frame center spans less than this
PARKED_MAX_DIST_M = 25.0


class Spin:
    """One decoded spin: points as stored, plus the per-point attributes."""

    def __init__(self, row: dict):
        self.index = row["spin_index"]
        self.start, self.end = row["spin_start_timestamp"], row["spin_end_timestamp"]
        cloud = DracoPy.decode(row["draco_encoded_pointcloud"])
        self.xyz = np.asarray(cloud.points, dtype=float)
        self.t = np.asarray(cloud.get_attribute_by_name("timestamp")["data"]).ravel().astype(np.int64)
        self.intensity = np.asarray(cloud.get_attribute_by_name("intensity")["data"]).ravel()
        element = np.asarray(cloud.get_attribute_by_name("model_element")["data"])
        self.channel, self.column = element[:, 0], element[:, 1]
        self.attributes = [(a["unique_id"], a["name"], a["num_components"], np.asarray(a["data"]).dtype) for a in cloud.attributes]
        self.bytes = len(row["draco_encoded_pointcloud"])


class Lidar:
    def __init__(self, path: Path):
        self.path = path
        self.file = pq.ParquetFile(path)
        self.meta = pq.read_table(path, columns=["spin_index", "spin_start_timestamp", "spin_end_timestamp"]).to_pandas()

    def spin(self, i: int) -> Spin:
        return Spin(self.file.read_row_group(i).to_pylist()[0])


def extrinsic(clip_dir: Path, table: str) -> tuple[np.ndarray, Rotation]:
    ex = pd.read_parquet(clip_dir / f"{table}.parquet").xs("lidar_top_360fov", level="sensor_name").iloc[0]
    return ex[["x", "y", "z"]].to_numpy(float, copy=True), Rotation.from_quat(ex[["qx", "qy", "qz", "qw"]].to_numpy(float, copy=True))


def ground_fit(xyz: np.ndarray) -> np.ndarray:
    """Plane z = a x + b y + c through the lowest points 5-20 m away (horizontal)."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    g = xyz[(r > 5) & (r < 20)]
    g = g[g[:, 2] < np.percentile(g[:, 2], 15) + 0.3]
    return np.linalg.lstsq(np.c_[g[:, :2], np.ones(len(g))], g[:, 2], rcond=None)[0]


def format_and_size(lidar: Lidar, spins: dict[int, Spin]) -> None:
    section("format and size")
    m = lidar.meta
    print(f"file {lidar.path.stat().st_size / 1e6:.0f} MB, {len(m)} spins (row groups {lidar.file.num_row_groups}), "
          f"columns {lidar.file.schema_arrow.names}")
    s = spins[SAMPLE_SPINS[0]]
    print(f"draco attributes (id, name, components, dtype): {s.attributes}")
    n = np.array([sp.xyz.shape[0] for sp in spins.values()])
    b = np.array([sp.bytes for sp in spins.values()])
    print(f"sampled spins {list(spins)}: points {n.min()} .. {n.max()}, encoded {b.min() / 1e6:.2f} .. {b.max() / 1e6:.2f} MB, "
          f"{b.mean() / n.mean():.2f} bytes/point")
    t0 = time.perf_counter()
    for i in range(5):
        lidar.spin(i)
    print(f"read + decode: {(time.perf_counter() - t0) / 5 * 1e3:.0f} ms per spin")
    print(f"channel {s.channel.min()} .. {s.channel.max()} ({np.unique(s.channel).size} distinct), "
          f"column {s.column.min()} .. {s.column.max()} ({np.unique(s.column).size} distinct); "
          f"fill {len(s.xyz) / (128 * 3600) * 100:.0f}% of 128 x 3600; intensity {s.intensity.min()} .. {s.intensity.max()}")
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    print(f"horizontal range {r.min():.2f} .. {r.max():.1f} m, z {s.xyz[:, 2].min():.2f} .. {s.xyz[:, 2].max():.2f} m")


def timing(lidar: Lidar, spins: dict[int, Spin], off: pd.DataFrame, obstacles: pd.DataFrame) -> None:
    section("timing")
    m = lidar.meta
    start, end = m["spin_start_timestamp"].to_numpy(np.int64), m["spin_end_timestamp"].to_numpy(np.int64)
    dur = (end - start) / 1e3
    print(f"spin_index contiguous: {bool((np.diff(m['spin_index']) == 1).all())}; starts {start[0] / 1e6:.4f} .. "
          f"{start[-1] / 1e6:.4f} s; duration {dur.min():.2f} .. {dur.max():.2f} ms (median {np.median(dur):.2f}); "
          f"next start - end: {np.unique(start[1:] - end[:-1]).tolist()} us")
    t_off = off["timestamp"].to_numpy(np.int64)
    print(f"egomotion.offline rows equal to a spin start: {np.isin(t_off, start).sum()} / {len(t_off)}")
    for i, s in spins.items():
        inside = ((s.t >= s.start) & (s.t <= s.end)).mean()
        # How the sweep advances: encoder column against time, and where the spin starts.
        slope, icpt = np.polyfit((s.t - s.start) / 1e3, s.column, 1)
        first = s.t <= s.start + 1_000
        print(f"spin {i:>3}: point times {(s.t.min() - s.start) / 1e3:+.3f} .. {(s.t.max() - s.start) / 1e3:.3f} ms "
              f"after start, {inside * 100:.1f}% inside [start, end]; column = {icpt:.0f} + {slope:.1f}/ms x t; "
              f"first ms covers columns {np.percentile(s.column[first], 5):.0f} .. {np.percentile(s.column[first], 95):.0f}")
    t_obs = obstacles["timestamp_us"].to_numpy(np.int64)
    print(f"obstacle sightings on a spin start: {np.isin(t_obs, start).mean() * 100:.1f}%")


def frame(spins: dict[int, Spin], ext_xyz: np.ndarray, ext_rot: Rotation) -> None:
    section("frame: sensor or rig?")
    z, y, x = ext_rot.as_euler("ZYX", degrees=True)
    print(f"lidar extrinsic (sensor in rig): xyz {np.round(ext_xyz, 3)} m, yaw {z:+.2f} pitch {y:+.2f} roll {x:+.2f} deg")
    for i, s in spins.items():
        a, b, c = ground_fit(s.xyz)
        az = np.arctan2(s.xyz[:, 1], s.xyz[:, 0])
        bins = ((az + np.pi) / (2 * np.pi) * 360).astype(int) % 360
        nearest = pd.Series(np.hypot(s.xyz[:, 0], s.xyz[:, 1])).groupby(bins).idxmin().to_numpy()
        hole = s.xyz[nearest, :2].mean(0)
        col0 = np.degrees(np.median(az[s.column < 20]))
        print(f"spin {i:>3}: blind-zone center ({hole[0]:+.2f}, {hole[1]:+.2f}) m; ground z at origin {c:+.2f} m, "
              f"slope x {a:+.3f} y {b:+.3f}; azimuth of columns 0-19: {col0:+.1f} deg")


def sensor_to_world(s: Spin, times_us, poses: Poses, ext_xyz, ext_rot) -> np.ndarray:
    """Sensor-frame points to the anchor frame, with the rig pose looked up at `times_us`."""
    p, r = poses.at(times_us)
    return r.apply(ext_rot.apply(s.xyz) + ext_xyz) + p


def raw_to_end(s: Spin, poses: Poses, ext_xyz, ext_rot) -> np.ndarray:
    """Treats the points as raw (each in the sensor frame at its own time) and expresses them all
    in the sensor frame at the spin end, i.e. motion-compensates them."""
    world = sensor_to_world(s, s.t, poses, ext_xyz, ext_rot)
    p_e, r_e = poses.at([s.end])
    return ext_rot.inv().apply(r_e.inv().apply(world - p_e) - ext_xyz)


def seam_jumps(s: Spin, xyz: np.ndarray, ground: np.ndarray) -> dict[str, float]:
    """Signed range jump across the seam, last columns minus first ones, per channel; median over
    the channels that hit ground on both sides, and over those that hit structure on both sides.

    The seam (column 3599 next to column 0) joins points 100 ms apart. Driving forward with the
    seam behind the car, the two kinds of surface tell raw from compensated points apart:
      raw:         ground ~0 (a channel hits flat ground at a fixed range), structure ~ +v x 0.1 s
      compensated: ground ~ -v x 0.1 s (rings around two sensor positions), structure ~0
    """
    rng = np.linalg.norm(xyz, axis=1)
    last, first = s.column >= 3600 - SEAM_COLUMNS, s.column < SEAM_COLUMNS
    out = {"ground": [], "structure": []}
    for ch in np.unique(s.channel):
        a, b = (s.channel == ch) & last, (s.channel == ch) & first
        if not a.any() or not b.any():
            continue
        g_a, g_b = ground[a].mean(), ground[b].mean()
        if min(g_a, g_b) > 0.5 or max(g_a, g_b) < 0.5:
            out["ground" if g_a > 0.5 else "structure"].append(np.median(rng[a]) - np.median(rng[b]))
    return {k: float(np.median(v)) if v else np.nan for k, v in out.items()}


def spin_speeds(lidar: Lidar, poses: Poses) -> np.ndarray:
    starts = lidar.meta["spin_start_timestamp"].to_numpy()
    return np.linalg.norm(poses.at(starts + 50_000)[0] - poses.at(starts - 50_000)[0], axis=1) / 0.1


def motion_compensation(lidar: Lidar, poses: Poses, ext_xyz, ext_rot) -> None:
    section("motion compensation: seam test on ground and on structure")
    v = spin_speeds(lidar, poses)
    picks = np.flatnonzero(v > FAST_MPS)[::3]
    print(f"ego speed: median {np.median(v):.1f}, max {v.max():.1f} m/s; {len(picks)} spins above {FAST_MPS} m/s "
          f"(every 3rd), v x 100 ms median {np.median(v[picks]) * 0.1:.2f} m")
    rows = []
    for k in picks:
        s = lidar.spin(k)
        a, b, c = ground_fit(s.xyz)
        ground = s.xyz[:, 2] < a * s.xyz[:, 0] + b * s.xyz[:, 1] + c + GROUND_BAND_M
        rows.append({**{f"stored {k_}": j for k_, j in seam_jumps(s, s.xyz, ground).items()},
                     **{f"compensated {k_}": j for k_, j in seam_jumps(s, raw_to_end(s, poses, ext_xyz, ext_rot), ground).items()}})
    med = pd.DataFrame(rows).median()
    print("seam jump (last - first), median over spins:")
    print(f"  as stored                   : ground {med['stored ground']:+.2f} m, structure {med['stored structure']:+.2f} m")
    print(f"  after per-point compensation: ground {med['compensated ground']:+.2f} m, structure {med['compensated structure']:+.2f} m")


def sweep_time_ms(s: Spin, xyz_sensor: np.ndarray) -> np.ndarray:
    """When the sweep passes the azimuth of `xyz_sensor`, in ms after the spin start: the median
    point time in that 1-degree azimuth bin of the spin itself (lasers have their own azimuth
    offsets, so a column does not map to one exact angle)."""
    bins = lambda xyz: (np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) // 1).astype(int) % 360  # noqa: E731
    table = pd.Series((s.t - s.start) / 1e3).groupby(bins(s.xyz)).median().reindex(range(360))
    return table.to_numpy()[bins(xyz_sensor)]


def label_timing(lidar: Lidar, obstacles: pd.DataFrame, ext_xyz, ext_rot) -> None:
    section("label timestamps against the sweep")
    starts = lidar.meta["spin_start_timestamp"].to_numpy(np.int64)
    t = obstacles["timestamp_us"].to_numpy(np.int64)
    phase = (t - starts[np.clip(np.searchsorted(starts, t, "right") - 1, 0, None)]) / 1e3
    s = lidar.spin(len(starts) // 2)
    az = np.degrees(np.arctan2(s.xyz[:, 1], s.xyz[:, 0]))
    first, mid = s.t < s.start + 1_000, np.abs(s.t - s.start - 25_000) < 1_000
    turn = np.degrees(np.angle(np.exp(1j * np.radians(az[mid])).mean() / np.exp(1j * np.radians(az[first])).mean()))
    print(f"sweep: the first ms points to azimuth {np.degrees(np.angle(np.exp(1j * np.radians(az[first])).mean())):+.1f} deg, "
          f"25 ms later {turn:+.1f} deg further ({'clockwise' if turn < 0 else 'counter-clockwise'} from above)")
    centers = ext_rot.inv().apply(obstacles[["center_x", "center_y", "center_z"]].to_numpy(float, copy=True) - ext_xyz)
    diff = (phase - sweep_time_ms(s, centers) + 50) % 100 - 50
    print(f"label time - time the sweep crosses the box center: median {np.median(diff):+.1f} ms, "
          f"within 5 ms {np.mean(np.abs(diff) < 5) * 100:.1f}%, within 10 ms {np.mean(np.abs(diff) < 10) * 100:.1f}% "
          f"of {len(diff)} sightings")


def obstacle_world(obstacles: pd.DataFrame, poses: Poses) -> tuple[np.ndarray, Rotation]:
    p, r = poses.at(obstacles["timestamp_us"].to_numpy(float))
    centers = obstacles[["center_x", "center_y", "center_z"]].to_numpy(float, copy=True)
    box = Rotation.from_quat(obstacles[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy(float, copy=True))
    return p + r.apply(centers), r * box


def parked_boxes(lidar: Lidar, obstacles: pd.DataFrame, poses: Poses, ext_xyz, ext_rot) -> None:
    section("parked cars: lidar points inside their labeled boxes, by when the sweep passed the car")
    cars = obstacles[obstacles["label_class"] == "automobile"].reset_index(drop=True)
    centers, rots = obstacle_world(cars, poses)
    t_label = cars["timestamp_us"].to_numpy(np.int64)
    sizes = cars[["size_x", "size_y", "size_z"]].to_numpy(float, copy=True)
    parked = [np.asarray(i) for i in cars.groupby("track_id").groups.values()
              if len(i) >= 20 and np.ptp(centers[np.asarray(i), :2], axis=0).max() < PARKED_RANGE_M]
    v = spin_speeds(lidar, poses)
    fast = np.flatnonzero(v > FAST_MPS)
    print(f"{len(parked)} parked cars (>= 20 sightings, center span < {PARKED_RANGE_M} m); {len(fast)} spins above "
          f"{FAST_MPS} m/s; pose time for the whole spin = spin start + offset, or each point's own time (raw)")
    fixed = np.zeros((4, len(OFFSETS_MS)))
    raw = np.zeros(4)
    pairs = np.zeros(4, int)
    for k in fast:
        s = lidar.spin(k)
        ego, _ = poses.at([s.start])
        w_raw = sensor_to_world(s, s.t, poses, ext_xyz, ext_rot)
        w_fixed = [sensor_to_world(s, [s.start + o * 1000], poses, ext_xyz, ext_rot) for o in OFFSETS_MS]
        for idx in parked:
            j = idx[np.argmin(np.abs(t_label[idx] - s.start))]
            if np.linalg.norm(centers[j, :2] - ego[0, :2]) > PARKED_MAX_DIST_M:
                continue
            inside = lambda w: (np.abs(rots[j].inv().apply(w - centers[j])) <= sizes[j] / 2).all(1)  # noqa: E731
            hit = inside(w_raw)
            if hit.sum() < 20:
                continue
            q = min(int((np.median(s.t[hit]) - s.start) // 25_000), 3)
            pairs[q] += 1
            raw[q] += hit.sum()
            for oi, w in enumerate(w_fixed):
                fixed[q, oi] += inside(w).sum()
    at = lambda q, o: fixed[q, OFFSETS_MS == o][0] / fixed[q].max()  # noqa: E731
    print(f"  {'car swept at':<14} {'pairs':>6} {'best offset':>12} {'raw / best':>11} {'@start':>7} {'@end':>6}")
    for q in range(4):
        if pairs[q]:
            print(f"  {q * 25:>3}-{q * 25 + 25:<3} ms     {pairs[q]:>6} {OFFSETS_MS[fixed[q].argmax()]:>+9d} ms "
                  f"{raw[q] / fixed[q].max():>11.3f} {at(q, 0):>7.2f} {at(q, 100):>6.2f}")
    total = fixed.sum(0)
    print(f"  all           {pairs.sum():>6} {OFFSETS_MS[total.argmax()]:>+9d} ms {raw.sum() / total.max():>11.3f} "
          f"{total[OFFSETS_MS == 0][0] / total.max():>7.2f} {total[OFFSETS_MS == 100][0] / total.max():>6.2f}")
    print("  (points counted inside the tight labeled box; ratios relative to the best fixed offset)")


def main(clips: list[str]) -> None:
    for clip in clips:
        d = RAW / clip
        print(f"\n================ {clip}")
        lidar = Lidar(d / "lidar_top_360fov/pointclouds.parquet")
        spins = {i: lidar.spin(i) for i in SAMPLE_SPINS if i < len(lidar.meta)}
        off = pd.read_parquet(d / "egomotion.offline.parquet")
        obstacles = pd.read_parquet(d / "obstacle.offline.parquet")
        ext_xyz, ext_rot = extrinsic(d, "sensor_extrinsics.offline")
        online_xyz, online_rot = extrinsic(d, "sensor_extrinsics")
        format_and_size(lidar, spins)
        timing(lidar, spins, off, obstacles)
        frame(spins, ext_xyz, ext_rot)
        print(f"online vs offline lidar extrinsic: |dxyz| {np.linalg.norm(online_xyz - ext_xyz) * 1e3:.1f} mm, "
              f"angle {np.degrees((online_rot.inv() * ext_rot).magnitude()):.3f} deg")
        poses = Poses(off)
        label_timing(lidar, obstacles, ext_xyz, ext_rot)
        motion_compensation(lidar, poses, ext_xyz, ext_rot)
        parked_boxes(lidar, obstacles, poses, ext_xyz, ext_rot)


if __name__ == "__main__":
    main(sys.argv[1:] or Path("clips_selected_train.txt").read_text().split())

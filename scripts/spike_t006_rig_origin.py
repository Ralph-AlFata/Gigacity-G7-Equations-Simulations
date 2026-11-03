"""T006 spike: is the rig origin the rear-axle midpoint on the ground, with x forward, y left, z up?

Reads data/raw/<clip>/ (fetch the `scene` preset first) and prints the numbers quoted in
tickets/E02-scene/T006-spike-rig-origin.md.

    uv run python scripts/spike_t006_rig_origin.py [clip_id ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from spike_t004_egomotion import section
from spike_t005_lidar import Lidar, extrinsic, ground_fit

RAW = Path("data/raw")
DENSE_US = (0, 20_000_000)  # online egomotion is 100 Hz over the clip, sparse after it (T004)
MIN_SPEED_MPS = 2.0
MIN_YAW_RATE = 0.05  # rad/s; below this the lateral-velocity fit has no lever arm
NEAR_LABEL_M = 15.0
GROUND_SPINS = range(0, 199, 10)


def dimensions(d: Path) -> pd.Series:
    return pd.read_parquet(d / "vehicle_dimensions.parquet").iloc[0]


def ego_box(dim: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The ego box in the rig frame if the origin is the rear axle on the ground (as NVIDIA's NCore
    converter builds it: center (rear_axle_to_bbox_center, 0, height / 2))."""
    center = np.array([dim["rear_axle_to_bbox_center"], 0.0, dim["height"] / 2])
    half = np.array([dim["length"], dim["width"], dim["height"]]) / 2
    return center - half, center + half


def sensors(d: Path, dim: pd.Series) -> None:
    section("sensor positions against the ego box (origin = rear axle on the ground)")
    lo, hi = ego_box(dim)
    print(f"vehicle: length {dim['length']} m, width {dim['width']} m, height {dim['height']} m, "
          f"wheelbase {dim['wheelbase']} m, rear axle -> box center {dim['rear_axle_to_bbox_center']} m")
    print(f"box: x {lo[0]:+.3f} .. {hi[0]:+.3f}, y {lo[1]:+.3f} .. {hi[1]:+.3f}, z {lo[2]:+.3f} .. {hi[2]:+.3f} m; "
          f"front axle at x = {dim['wheelbase']:+.3f}")
    ext = pd.read_parquet(d / "sensor_extrinsics.offline.parquet").droplevel("clip_id")
    for name, row in ext.iterrows():
        p = row[["x", "y", "z"]].to_numpy(float)
        outside = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
        where = "inside" if not outside.any() else "outside by " + ", ".join(
            f"{a} {v:.2f} m" for a, v in zip("xyz", outside) if v > 0)
        print(f"  {name:<26} x {p[0]:+.3f} y {p[1]:+.3f} z {p[2]:+.3f}  {where}")


def lateral_velocity(d: Path, dim: pd.Series) -> None:
    section("lateral velocity of the rig origin in turns (no-slip: v_lat = offset_from_rear_axle x yaw_rate)")
    on = pd.read_parquet(d / "egomotion.parquet")
    on = on[on["timestamp"].between(*DENSE_US)]
    t = on["timestamp"].to_numpy(float) / 1e6
    rot = Rotation.from_quat(on[["qx", "qy", "qz", "qw"]].to_numpy(float, copy=True))
    v_body = rot.inv().apply(on[["vx", "vy", "vz"]].to_numpy(float, copy=True))  # columns are in the anchor frame (T004)
    yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])
    w = np.gradient(yaw, t)
    print(f"online egomotion, {len(on)} samples: body-frame vx median {np.median(v_body[:, 0]):+.2f} m/s "
          f"(x forward if > 0); yaw {np.degrees(yaw[0]):+.0f} -> {np.degrees(yaw[-1]):+.0f} deg")
    use = (v_body[:, 0] > MIN_SPEED_MPS) & (np.abs(w) > MIN_YAW_RATE)
    if use.sum() < 50:
        print(f"only {use.sum()} samples turning faster than {MIN_YAW_RATE} rad/s above {MIN_SPEED_MPS} m/s: no fit")
        return
    a, b = np.polyfit(w[use], v_body[use, 1], 1)
    resid = v_body[use, 1] - (a * w[use] + b)
    # Bootstrap over contiguous 1 s blocks, since neighbouring 100 Hz samples are not independent.
    blocks = (t[use] // 1).astype(int)
    ids = np.unique(blocks)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        pick = np.isin(blocks, rng.choice(ids, len(ids)))
        if pick.sum() > 10:
            boot.append(np.polyfit(w[use][pick], v_body[use][pick, 1], 1)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{use.sum()} samples ({len(ids)} one-second blocks) with speed > {MIN_SPEED_MPS} m/s and |yaw rate| > "
          f"{MIN_YAW_RATE} rad/s; yaw rate up to {np.abs(w[use]).max():.2f} rad/s")
    print(f"fit v_lat = a * yaw_rate + b: a = {a:+.3f} m (95% {lo:+.3f} .. {hi:+.3f}), b = {b:+.3f} m/s, "
          f"residual RMS {np.sqrt(np.mean(resid ** 2)):.3f} m/s")
    print(f"expected a: rear axle 0, wheelbase middle {dim['wheelbase'] / 2:+.3f}, box center "
          f"{dim['rear_axle_to_bbox_center']:+.3f}, front axle {dim['wheelbase']:+.3f}")
    print(f"=> origin about {a:+.2f} m ahead of the rear axle; lateral velocity |v_lat| median "
          f"{np.median(np.abs(v_body[use, 1])):.3f} m/s vs |yaw rate x wheelbase/2| median "
          f"{np.median(np.abs(w[use])) * dim['wheelbase'] / 2:.3f} m/s")


def ground_height(d: Path, ext_xyz: np.ndarray, ext_rot: Rotation) -> None:
    section("ground height in the rig frame (z = 0 at the ground if the origin is on the ground)")
    lidar = Lidar(d / "lidar_top_360fov/pointclouds.parquet")
    heights = []
    for i in GROUND_SPINS:
        s = lidar.spin(i)
        rig = ext_rot.apply(s.xyz) + ext_xyz
        a, b, c = ground_fit(rig)
        heights.append(c)  # plane height under the rig origin (x = y = 0)
    h = np.array(heights)
    print(f"lidar ground plane under the origin, {len(h)} spins: median {np.median(h):+.3f} m, "
          f"{h.min():+.3f} .. {h.max():+.3f} m")
    ob = pd.read_parquet(d / "obstacle.offline.parquet")
    near = ob[(np.hypot(ob["center_x"], ob["center_y"]) < NEAR_LABEL_M) & ob["label_class"].eq("automobile")]
    bottom = near["center_z"] - near["size_z"] / 2
    print(f"bottom of labeled cars within {NEAR_LABEL_M} m ({len(near)} sightings): median {bottom.median():+.3f} m, "
          f"25-75% {bottom.quantile(.25):+.3f} .. {bottom.quantile(.75):+.3f} m")


def main(clips: list[str]) -> None:
    for clip in clips:
        d = RAW / clip
        print(f"\n================ {clip}")
        dim = dimensions(d)
        ext_xyz, ext_rot = extrinsic(d, "sensor_extrinsics.offline")
        sensors(d, dim)
        lateral_velocity(d, dim)
        ground_height(d, ext_xyz, ext_rot)


if __name__ == "__main__":
    main(sys.argv[1:] or Path("clips_selected_train.txt").read_text().split())

"""Lidar observability (T008): does the lidar actually hit each object that exists at a tick?

Facts this relies on (T005):
- one spin is 100 ms; points are raw, each in the lidar sensor frame at its own `timestamp`
  (absolute us, clip clock), so every point is placed with the ego pose at that time;
- the sweep crosses an object once per spin, and the object's label is stamped with that moment.

For each (tick, object) row we take the spin whose sweep crossed the object nearest to the tick,
place the object's box at that crossing time, and count that spin's points inside the box.
"""

from dataclasses import dataclass
from pathlib import Path

import DracoPy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from horizon_pipeline.scene.frames import Poses, to_local

LIDAR = "lidar_top_360fov"
POINTCLOUDS = Path(LIDAR) / "pointclouds.parquet"
EXTRINSICS = "sensor_extrinsics.offline"
SPIN_US = 100_000


@dataclass(frozen=True)
class LidarRules:
    lidar_min_points: int = 5
    lidar_box_margin_m: float = 0.2  # enlarge the box by this on every side
    ground_clearance_m: float = 0.2  # ignore points this close above the box bottom (road)


@dataclass
class Spin:
    index: int
    start: int
    end: int
    xyz: np.ndarray  # sensor frame, m
    t: np.ndarray  # absolute us per point


class SpinReader:
    """Decodes `lidar_top_360fov` spins one at a time (format in T005)."""

    def __init__(self, path: Path):
        self.file = pq.ParquetFile(path)
        meta = pq.read_table(path, columns=["spin_index", "spin_start_timestamp", "spin_end_timestamp"]).to_pandas()
        self.starts = meta["spin_start_timestamp"].to_numpy(np.int64, copy=True)
        self.ends = meta["spin_end_timestamp"].to_numpy(np.int64, copy=True)

    def __len__(self) -> int:
        return len(self.starts)

    def spin(self, i: int) -> Spin:
        row = self.file.read_row_group(i).to_pylist()[0]
        cloud = DracoPy.decode(row["draco_encoded_pointcloud"])
        t = np.asarray(cloud.get_attribute_by_name("timestamp")["data"]).ravel().astype(np.int64)
        return Spin(row["spin_index"], row["spin_start_timestamp"], row["spin_end_timestamp"],
                    np.array(cloud.points, dtype=float), t)  # fmt: skip


def lidar_extrinsic(clip_dir: Path) -> tuple[np.ndarray, Rotation]:
    """Pose of the lidar in the rig frame: `p_rig = R p_sensor + t`."""
    ex = pd.read_parquet(clip_dir / f"{EXTRINSICS}.parquet").xs(LIDAR, level="sensor_name").iloc[0]
    return ex[["x", "y", "z"]].to_numpy(float, copy=True), Rotation.from_quat(ex[["qx", "qy", "qz", "qw"]].to_numpy(float, copy=True))


POSE_BUCKET_US = 100  # points this close in time share one interpolated pose (<= 0.5 mm at 10 m/s)


def points_to_world(spin: Spin, poses: Poses, ext_xyz: np.ndarray, ext_rot: Rotation) -> np.ndarray:
    """Each point with the ego pose at its own timestamp (the points are not motion-compensated).

    Poses are interpolated once per `POSE_BUCKET_US` bucket rather than per point (~1 000 instead
    of ~300 000 per spin)."""
    bucket = np.clip((spin.t + POSE_BUCKET_US // 2) // POSE_BUCKET_US * POSE_BUCKET_US, poses.start, poses.end)
    times, which = np.unique(bucket, return_inverse=True)
    p, r = poses.at(times)
    rig = ext_rot.apply(spin.xyz) + ext_xyz
    return np.einsum("nij,nj->ni", r.as_matrix()[which], rig) + p[which]


def sweep_profile(spin: Spin) -> np.ndarray:
    """When the sweep passes each 1-degree azimuth bin of the sensor frame, in us after the spin
    start (median point time per bin; empty bins are interpolated around the circle)."""
    bins = (np.degrees(np.arctan2(spin.xyz[:, 1], spin.xyz[:, 0])) // 1).astype(int) % 360
    table = pd.Series(spin.t - spin.start).groupby(bins).median().reindex(range(360))
    if table.isna().any():
        known = np.flatnonzero(table.notna())
        # Unwrap the known times around the circle before interpolating so the seam does not blur.
        table[:] = np.interp(np.arange(360), known, table.to_numpy()[known], period=360)
    return table.to_numpy(float)


def assign_spins(t_us: np.ndarray, sweep_us: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows at times `t_us` whose objects the sweep passes `sweep_us` after a spin start, the
    spin whose crossing is nearest each row's time, and that crossing time (us)."""
    k = np.clip(np.searchsorted(starts, t_us - sweep_us) - 1, 0, len(starts) - 1)
    candidates = np.clip(k[:, None] + np.array([0, 1, 2]), 0, len(starts) - 1)
    crossing = starts[candidates] + sweep_us[:, None]
    best = np.abs(crossing - t_us[:, None]).argmin(1)
    rows = np.arange(len(t_us))
    return candidates[rows, best], crossing[rows, best].round().astype(np.int64)


def count_in_box(points: np.ndarray, center: np.ndarray, rot: Rotation, size: np.ndarray, rules: LidarRules) -> int:
    """Points inside the box enlarged by the margin, minus those within the ground clearance of
    its bottom (world-frame points; box center, orientation and size as labeled)."""
    local = to_local(points, center, rot)
    half = size / 2
    m = rules.lidar_box_margin_m
    inside = (
        (np.abs(local[:, 0]) <= half[0] + m)
        & (np.abs(local[:, 1]) <= half[1] + m)
        & (local[:, 2] <= half[2] + m)
        & (local[:, 2] >= -half[2] + rules.ground_clearance_m)
    )
    return int(inside.sum())


def observe(
    objects: pd.DataFrame,
    tracks: dict,
    poses: Poses,
    clip_dir: Path,
    rules: LidarRules = LidarRules(),
) -> tuple[pd.DataFrame, dict]:
    """Fills `lidar_points`, `observed`, `in_scene`, `lidar_spin` and `lidar_dt_s` in `objects`.

    `tracks` are the world-frame sightings from the scene build (`build.world_tracks`); `poses` is
    the same ego source the scene was built with.
    """
    reader = SpinReader(clip_dir / POINTCLOUDS)
    ext_xyz, ext_rot = lidar_extrinsic(clip_dir)
    objects = objects.copy()
    usable = np.flatnonzero((reader.starts >= poses.start) & (reader.ends <= poses.end))
    starts = reader.starts[usable]
    profile = sweep_profile(reader.spin(int(usable[len(usable) // 2])))

    # Where each object is seen from the sensor at its tick -> when the sweep passes it.
    rel = objects[["rel_x", "rel_y", "rel_z"]].to_numpy(float, copy=True)
    in_sensor = ext_rot.inv().apply(rel - ext_xyz)
    az_bin = (np.degrees(np.arctan2(in_sensor[:, 1], in_sensor[:, 0])) // 1).astype(int) % 360
    tick_us = (objects["t_s"].to_numpy(float) * 1e6).round().astype(np.int64)
    k, crossing = assign_spins(tick_us, profile[az_bin], starts)
    spin_index = usable[k]

    # The box at the crossing time, interpolated from the track's own sightings.
    center = np.zeros((len(objects), 3))
    quat = np.zeros((len(objects), 4))
    size = np.zeros((len(objects), 3))
    for track_id, rows in objects.groupby("track_id").indices.items():
        xyz, rot, sz = tracks[track_id].state_at(crossing[rows])
        center[rows], quat[rows], size[rows] = xyz, rot.as_quat(), sz

    counts = np.zeros(len(objects), np.int64)
    reach = np.linalg.norm(size / 2 + rules.lidar_box_margin_m, axis=1)
    for s in np.unique(spin_index):
        rows = np.flatnonzero(spin_index == s)
        world = points_to_world(reader.spin(int(s)), poses, ext_xyz, ext_rot)
        tree = cKDTree(world)
        for row, near in zip(rows, tree.query_ball_point(center[rows], reach[rows])):
            if near:
                counts[row] = count_in_box(world[near], center[row], Rotation.from_quat(quat[row]), size[row], rules)

    objects["lidar_points"] = pd.array(counts, dtype="Int64")
    objects["observed"] = pd.array(counts >= rules.lidar_min_points, dtype="boolean")
    objects["in_scene"] = objects["observed"].copy()  # every row already exists (T007)
    objects["lidar_spin"] = spin_index.astype(np.int64)
    objects["lidar_dt_s"] = (crossing - tick_us) / 1e6
    return objects, observability_report(objects)


DISTANCE_BANDS_M = [0, 10, 20, 40, 80, np.inf]


def observability_report(objects: pd.DataFrame) -> dict:
    """Share of existing rows observed, by class group and by distance band."""
    band = pd.cut(objects["distance"], DISTANCE_BANDS_M, right=False,
                  labels=[f"{a:g}-{b:g} m" for a, b in zip(DISTANCE_BANDS_M[:-1], DISTANCE_BANDS_M[1:])])  # fmt: skip
    observed = objects["observed"].astype(float)
    table = observed.groupby([objects["class_group"], band], observed=True).agg(["mean", "size"])
    return {
        "observed_share": float(observed.mean()),
        "lidar_dt_s_max_abs": float(objects["lidar_dt_s"].abs().max()),
        "by_class_and_distance": {
            f"{g} {b}": {"share": round(float(r["mean"]), 4), "rows": int(r["size"])} for (g, b), r in table.iterrows()
        },
    }

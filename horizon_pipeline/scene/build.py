"""Scene builder (T007): raw clip files + an ego source -> `meta.json`, `ego.parquet`, `objects.parquet`.

Schema and principles in tickets/E02-scene/README.md; the steps below follow T007's specification.
The lidar columns (`lidar_points`, `observed`, `in_scene`) are written empty; T008 fills them.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from horizon_pipeline.raw.store import sha256
from horizon_pipeline.scene.frames import QUAT, XYZ, Poses, dense_end, slerp_pairs, to_local, wrap, yaw_of

RATE_HZ = 10
TICK_US = 1_000_000 // RATE_HZ
DT_S = 1 / RATE_HZ
DEFAULT_RAW = Path("data/raw")
DEFAULT_OUT = Path("data/scene")

EGO_SOURCES = {"offline": "egomotion.offline", "online": "egomotion"}
OBSTACLES = "obstacle.offline"
DIMENSIONS = "vehicle_dimensions"

CLASS_GROUPS = {
    "vehicle": ["automobile", "bus", "heavy_truck", "trailer"],
    "vru": ["person", "rider", "stroller", "animal"],
    "static": ["protruding_object"],
}
UNKNOWN_GROUP = "unknown"


@dataclass(frozen=True)
class Rules:
    gap_bridge_s: float = 0.5  # bridge skips between sightings up to this long
    dedup_s: float = 0.020  # rows of one track closer than this are duplicates
    dedup_max_offset_m: float = 0.2  # duplicates must agree to within this (rig-frame centers)
    dedup_size_tol_m: float = 1e-3
    gap_report_s: float = 0.15  # gap_s is reported only above this (normal spacing is 0.1 s)
    history_s: float = 1.6  # Alpamayo needs this much past input


class DuplicateError(ValueError):
    """Two rows of a track closer than `dedup_s` that do not describe the same box."""


@dataclass
class Scene:
    meta: dict
    ego: pd.DataFrame
    objects: pd.DataFrame
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- inputs


def load_ego(clip_dir: Path, ego_source: str) -> Poses:
    """Ego poses of the source, cut to their dense part (online has a sparse tail, T004)."""
    df = pd.read_parquet(clip_dir / f"{EGO_SOURCES[ego_source]}.parquet").sort_values("timestamp")
    df = df.iloc[: dense_end(df["timestamp"].to_numpy(np.int64, copy=True)) + 1]
    return Poses.from_frame(df)


def load_sightings(clip_dir: Path) -> pd.DataFrame:
    obs = pd.read_parquet(clip_dir / f"{OBSTACLES}.parquet")
    obs["timestamp_us"] = obs["timestamp_us"].astype(np.int64)  # stored as uint64
    obs["track_id"] = obs["track_id"].astype(str)
    obs["label_class"] = obs["label_class"].astype(str)
    return obs.sort_values(["track_id", "timestamp_us"], kind="stable").reset_index(drop=True)


def load_dimensions(clip_dir: Path) -> dict:
    row = pd.read_parquet(clip_dir / f"{DIMENSIONS}.parquet").iloc[0]
    return {k: float(row[k]) for k in ["length", "width", "height", "rear_axle_to_bbox_center", "wheelbase", "track_width"]}


def code_version() -> dict:
    try:
        pkg = version("alpamayo-horizon-pipeline")
    except PackageNotFoundError:
        pkg = None
    repo = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()  # noqa: E731
    try:
        commit, dirty = git("rev-parse", "HEAD") or None, bool(git("status", "--porcelain"))
    except OSError:
        commit, dirty = None, None
    return {"package": pkg, "git_commit": commit, "git_dirty": dirty}


# ---------------------------------------------------------------- steps


def make_grid(poses: Poses) -> np.ndarray:
    """Step 1: tick times (us), every 0.1 s from clip time 0 (or the first pose after it) to the
    last pose. Clip time 0 is where both ego sources' anchor frames are defined, so online and
    offline scenes share their ticks."""
    first = max(0, -(-poses.start // TICK_US) * TICK_US)
    return np.arange(first, poses.end + 1, TICK_US, dtype=np.int64)


def deduplicate(obs: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, int]:
    """Step 3: drop the earlier row of each pair of a track's rows less than `dedup_s` apart,
    after checking that both describe the same box. Expects `obs` sorted by track and time."""
    t = obs["timestamp_us"].to_numpy(np.int64, copy=True)
    track = obs["track_id"].to_numpy(copy=True)
    same = np.zeros(len(obs), bool)
    same[:-1] = (track[:-1] == track[1:]) & (t[1:] - t[:-1] < rules.dedup_s * 1e6)
    i = np.flatnonzero(same)
    j = i + 1
    center = obs[["center_x", "center_y", "center_z"]].to_numpy(float, copy=True)
    size = obs[["size_x", "size_y", "size_z"]].to_numpy(float, copy=True)
    cls = obs["label_class"].to_numpy(copy=True)
    bad = (
        (cls[i] != cls[j])
        | (np.abs(size[i] - size[j]) > rules.dedup_size_tol_m).any(1)
        | (np.linalg.norm(center[i] - center[j], axis=1) > rules.dedup_max_offset_m)
    )
    if bad.any():
        pairs = [
            f"track {track[a]} at {t[a]} / {t[b]} us: class {cls[a]} / {cls[b]}, "
            f"size {size[a].round(3).tolist()} / {size[b].round(3).tolist()}, "
            f"offset {np.linalg.norm(center[a] - center[b]):.3f} m"
            for a, b in zip(i[bad][:10], j[bad][:10])
        ]
        raise DuplicateError(f"{bad.sum()} duplicate pairs disagree:\n  " + "\n  ".join(pairs))
    return obs.loc[~same].reset_index(drop=True), int(same.sum())


def sightings_to_world(obs: pd.DataFrame, poses: Poses) -> tuple[np.ndarray, Rotation]:
    """Step 4: each box is in the rig frame at its own timestamp; compose with the ego pose then."""
    p, r = poses.at(obs["timestamp_us"].to_numpy(np.int64, copy=True))
    center = obs[["center_x", "center_y", "center_z"]].to_numpy(float, copy=True)
    box = Rotation.from_quat(obs[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy(float, copy=True))
    return r.apply(center) + p, r * box


def resample_track(
    t: np.ndarray, xyz: np.ndarray, rot: Rotation, size: np.ndarray, ticks: np.ndarray, rules: Rules
) -> dict | None:
    """Steps 5-6 for one track: its state at every tick where it exists.

    `t` must be strictly increasing (deduplicated). Returns column arrays, or None if the track
    exists at no tick.
    """
    tk = ticks[(ticks >= t[0]) & (ticks <= t[-1])]
    if len(tk) == 0:
        return None
    i = np.searchsorted(t, tk, side="right") - 1  # latest sighting at or before the tick
    exact = t[i] == tk
    j = np.where(exact, i, np.minimum(i + 1, len(t) - 1))
    spacing = t[j] - t[i]
    keep = exact | (spacing <= rules.gap_bridge_s * 1e6)
    tk, i, j, exact, spacing = tk[keep], i[keep], j[keep], exact[keep], spacing[keep]
    if len(tk) == 0:
        return None
    frac = np.divide(tk - t[i], spacing, out=np.zeros(len(tk)), where=spacing > 0)
    gap = np.where(exact, 0.0, spacing / 1e6)
    return {
        "timestamp_us": tk,
        "sighting": i,
        "xyz": xyz[i] + (xyz[j] - xyz[i]) * frac[:, None],
        "rot": slerp_pairs(rot[i], rot[j], frac),
        "size": size[i] + (size[j] - size[i]) * frac[:, None],
        "last_sighting_age_s": (tk - t[i]) / 1e6,
        "gap_s": np.where(gap > rules.gap_report_s, gap, 0.0),
    }


def segments(tick: np.ndarray) -> np.ndarray:
    """Run index of consecutive ticks (a new run starts wherever a tick is skipped)."""
    return np.concatenate([[0], np.cumsum(np.diff(tick) != 1)])


def derive_motion(xyz: np.ndarray, yaw: np.ndarray, run: np.ndarray) -> dict:
    """Step 7: central differences over +-0.1 s, one-sided at the ends of each run of ticks; NaN for
    a run of one tick. Returns velocity, acceleration (world frame) and yaw rate."""
    vel = np.full_like(xyz, np.nan)
    acc = np.full_like(xyz, np.nan)
    yaw_rate = np.full(len(yaw), np.nan)
    for r in np.unique(run):
        m = run == r
        if m.sum() < 2:
            continue
        vel[m] = np.gradient(xyz[m], DT_S, axis=0)
        acc[m] = np.gradient(vel[m], DT_S, axis=0)
        yaw_rate[m] = np.gradient(np.unwrap(yaw[m]), DT_S)
    return {"vel": vel, "acc": acc, "yaw_rate": yaw_rate}


def class_group(label_class: str) -> str:
    for group, classes in CLASS_GROUPS.items():
        if label_class in classes:
            return group
    return UNKNOWN_GROUP


def ego_table(poses: Poses, ticks: np.ndarray, t0_us: int, rules: Rules) -> tuple[pd.DataFrame, np.ndarray, Rotation]:
    xyz, rot = poses.at(ticks)
    yaw = np.unwrap(yaw_of(rot))
    motion = derive_motion(xyz, yaw, np.zeros(len(ticks), int))
    acc_body = rot.inv().apply(motion["acc"])
    t_s = (ticks - t0_us) / 1e6
    ego = pd.DataFrame({"tick": np.arange(len(ticks)), "t_s": t_s, "timestamp_us": ticks})
    ego[XYZ] = xyz
    ego[QUAT] = rot.as_quat()
    ego["yaw"] = yaw
    ego[["vx", "vy", "vz"]] = motion["vel"]
    ego["speed"] = np.linalg.norm(motion["vel"], axis=1)
    ego[["ax", "ay", "az"]] = motion["acc"]
    ego["a_long"], ego["a_lat"] = acc_body[:, 0], acc_body[:, 1]
    ego["yaw_rate"] = motion["yaw_rate"]
    ego["has_history"] = t_s >= rules.history_s - 1e-9
    return ego, xyz, rot


OBJECT_COLUMNS = [
    "tick", "t_s", "track_id", "label_class", "class_group",
    "x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "length", "width", "height",
    "vx", "vy", "vz", "speed", "ax", "ay", "az", "yaw_rate",
    "rel_x", "rel_y", "rel_z", "rel_yaw", "distance",
    "last_sighting_age_s", "gap_s", "lidar_points", "observed", "in_scene",
]  # fmt: skip


def objects_table(
    obs: pd.DataFrame, poses: Poses, ticks: np.ndarray, t0_us: int, ego_xyz: np.ndarray, ego_rot: Rotation, rules: Rules
) -> pd.DataFrame:
    world_xyz, world_rot = sightings_to_world(obs, poses)
    size = obs[["size_x", "size_y", "size_z"]].to_numpy(float, copy=True)
    parts = []
    for track, idx in obs.groupby("track_id", sort=True).indices.items():
        state = resample_track(
            obs["timestamp_us"].to_numpy(np.int64, copy=True)[idx], world_xyz[idx], world_rot[idx], size[idx], ticks, rules
        )
        if state is None:
            continue
        tick = (state["timestamp_us"] - ticks[0]) // TICK_US
        yaw = yaw_of(state["rot"])
        motion = derive_motion(state["xyz"], yaw, segments(tick))
        df = pd.DataFrame({"tick": tick, "t_s": (state["timestamp_us"] - t0_us) / 1e6, "track_id": track})
        df["label_class"] = obs["label_class"].to_numpy(copy=True)[idx][state["sighting"]]
        df[XYZ] = state["xyz"]
        df[QUAT] = state["rot"].as_quat()
        df["yaw"] = wrap(yaw)
        df[["length", "width", "height"]] = state["size"]
        df[["vx", "vy", "vz"]] = motion["vel"]
        df[["ax", "ay", "az"]] = motion["acc"]
        df["yaw_rate"] = motion["yaw_rate"]
        df["last_sighting_age_s"] = state["last_sighting_age_s"]
        df["gap_s"] = state["gap_s"]
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=OBJECT_COLUMNS)
    objects = pd.concat(parts, ignore_index=True)
    objects["class_group"] = objects["label_class"].map(class_group)
    objects["speed"] = np.linalg.norm(objects[["vx", "vy", "vz"]].to_numpy(copy=True), axis=1)
    # Step 8: the object in the ego frame at the same tick (origin at the rear axle, T006).
    k = objects["tick"].to_numpy(copy=True)
    rel = to_local(objects[XYZ].to_numpy(copy=True), ego_xyz[k], ego_rot[k])
    objects[["rel_x", "rel_y", "rel_z"]] = rel
    objects["rel_yaw"] = yaw_of(ego_rot[k].inv() * Rotation.from_quat(objects[QUAT].to_numpy(copy=True)))
    objects["distance"] = np.hypot(rel[:, 0], rel[:, 1])
    objects["lidar_points"] = pd.array([pd.NA] * len(objects), dtype="Int64")  # T008
    objects["observed"] = pd.array([pd.NA] * len(objects), dtype="boolean")
    objects["in_scene"] = pd.array([pd.NA] * len(objects), dtype="boolean")
    return objects.sort_values(["tick", "track_id"], kind="stable").reset_index(drop=True)[OBJECT_COLUMNS]


def footprint(dim: dict) -> dict:
    """Ego box relative to the rear axle on the ground (T006; as NVIDIA's NCore converter builds it)."""
    cx = dim["rear_axle_to_bbox_center"]
    return {
        "x_min": cx - dim["length"] / 2, "x_max": cx + dim["length"] / 2,
        "y_min": -dim["width"] / 2, "y_max": dim["width"] / 2,
        "z_min": 0.0, "z_max": dim["height"],
    }  # fmt: skip


# ---------------------------------------------------------------- build / write


def build_scene(clip_dir: Path, ego_source: str, rules: Rules = Rules()) -> Scene:
    clip_dir = Path(clip_dir)
    if ego_source not in EGO_SOURCES:
        raise ValueError(f"ego_source must be one of {list(EGO_SOURCES)}, not {ego_source!r}")
    poses = load_ego(clip_dir, ego_source)
    ticks = make_grid(poses)
    t0_us = int(ticks[0])
    obs = load_sightings(clip_dir)
    n_raw = len(obs)
    covered = obs["timestamp_us"].between(poses.start, poses.end)
    obs = obs.loc[covered].reset_index(drop=True)
    obs, n_merged = deduplicate(obs, rules)
    ego, ego_xyz, ego_rot = ego_table(poses, ticks, t0_us, rules)
    objects = objects_table(obs, poses, ticks, t0_us, ego_xyz, ego_rot, rules)
    unknown = sorted(set(objects.loc[objects["class_group"] == UNKNOWN_GROUP, "label_class"]))
    dim = load_dimensions(clip_dir)
    files = [f"{EGO_SOURCES[ego_source]}.parquet", f"{OBSTACLES}.parquet", f"{DIMENSIONS}.parquet"]
    report = {
        "sightings": n_raw,
        "sightings_outside_ego_range": int((~covered).sum()),
        "duplicates_merged": n_merged,
        "tracks": int(obs["track_id"].nunique()),
        "tracks_in_scene_file": int(objects["track_id"].nunique()),
        "object_rows": len(objects),
        "unknown_classes": unknown,
    }
    meta = {
        "clip_id": clip_dir.name,
        "ego_source": ego_source,
        "rate_hz": RATE_HZ,
        "t0_us": t0_us,
        "n_ticks": len(ticks),
        "ego": {**dim, "rear_axle_offset": [0.0, 0.0, 0.0], "footprint": footprint(dim)},
        "class_groups": {**CLASS_GROUPS, UNKNOWN_GROUP: "any other label_class"},
        "rules": asdict(rules),
        "inputs": {"files": {f: sha256(clip_dir / f) for f in files}, "code": code_version()},
        "build_report": report,
    }
    return Scene(meta, ego, objects, report)


def write_scene(scene: Scene, out_root: Path = DEFAULT_OUT) -> Path:
    out = Path(out_root) / scene.meta["clip_id"] / scene.meta["ego_source"]
    out.mkdir(parents=True, exist_ok=True)
    scene.ego.to_parquet(out / "ego.parquet", index=False)
    scene.objects.to_parquet(out / "objects.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(scene.meta, indent=2) + "\n")
    return out

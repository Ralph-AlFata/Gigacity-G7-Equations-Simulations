"""PhysicalAI AV adapter. Dataset schemas were inspected in NVIDIA source code.

Real-data integration requires the optional dataset dependencies and authorized HF
access. No frames, units, or missing features are silently guessed.
"""
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from .core import rear_axle_radius

FEATURES = ["egomotion.offline", "obstacle.offline", "vehicle_dimensions"]
SOURCE_URLS = [
    "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles",
    "https://github.com/NVlabs/physical_ai_av/blob/main/src/physical_ai_av/dataset.py",
    "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py",
    "https://github.com/NVIDIA/radar-reprojection/blob/main/docs/SCHEMA.md",
]


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def table(value, feature):
    if isinstance(value, dict):
        candidates = [v for v in value.values() if isinstance(v, pd.DataFrame)]
        if len(candidates) != 1:
            raise ValueError(f"{feature}: expected one table; got keys {list(value)}")
        value = candidates[0]
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{feature}: expected DataFrame; got {type(value).__name__}")
    df = value.copy()
    for name in df.index.names:
        if name is not None and name not in df.columns:
            df = df.reset_index(level=name)
    return df


def require(df, columns, label):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing {sorted(missing)}; actual columns={list(df.columns)}")
    if df[list(columns)].isna().any().any():
        raise ValueError(f"{label}: missing values in required columns")


def normalize(ego_df, obstacle_df, dimensions, obstacle_frame="auto"):
    """Returns canonical ego/agents tables and metadata; all times share one origin.

    Rig obstacle poses use reference_frame_timestamp_us when supplied. Full 3D
    poses are transformed before XY projection. Radii are PLANAR circumscribed
    box radii: upright objects and a locally planar road are experiment assumptions.
    """
    ego = table(ego_df, "egomotion.offline")
    obs = table(obstacle_df, "obstacle.offline")
    require(ego, ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"], "ego")
    require(obs, ["timestamp_us", "track_id", "center_x", "center_y", "center_z",
                  "size_x", "size_y"], "obstacles")
    ego = ego.sort_values("timestamp").reset_index(drop=True)
    if len(ego) < 2 or ego.timestamp.duplicated().any():
        raise ValueError("need at least two unique ego timestamps")
    if obs.duplicated(["track_id", "timestamp_us"]).any():
        raise ValueError("duplicate track/time rows; resolve them before analysis")
    if (obs[["size_x", "size_y"]] <= 0).any().any():
        raise ValueError("nonpositive obstacle box size")
    t_origin_us = int(ego.timestamp.iloc[0])
    t = (ego.timestamp.to_numpy(dtype=np.int64) - t_origin_us) * 1e-6
    t_obs = (obs.timestamp_us.to_numpy(dtype=np.int64) - t_origin_us) * 1e-6
    frame_values = obs["reference_frame"].astype(str).str.lower().to_numpy() if "reference_frame" in obs else None
    frame_map = {"rig": "rig", "ego": "rig", "vehicle": "rig", "anchor": "anchor", "world": "anchor"}
    if obstacle_frame == "auto":
        if frame_values is None:
            raise ValueError("reference_frame absent; inspect the schema and explicitly pass --obstacle-frame rig or anchor")
        unknown = sorted(set(frame_values) - set(frame_map))
        if unknown:
            raise ValueError(f"unknown reference_frame values {unknown}; inspect and explicitly choose --obstacle-frame")
        frames = np.array([frame_map[v] for v in frame_values])
    else:
        if obstacle_frame not in {"rig", "anchor"}:
            raise ValueError("obstacle_frame must be auto, rig, or anchor")
        frames = np.repeat(obstacle_frame, len(obs))
        if frame_values is not None:
            conflicts = [v for v in set(frame_values) if v in frame_map and frame_map[v] != obstacle_frame]
            if conflicts:
                raise ValueError(f"explicit frame conflicts with metadata: {conflicts}")
    t_frame = t_obs.copy()
    if "reference_frame_timestamp_us" in obs:
        require(obs, ["reference_frame_timestamp_us"], "obstacles")
        t_frame = (obs.reference_frame_timestamp_us.to_numpy(dtype=np.int64) - t_origin_us) * 1e-6
    is_rig = frames == "rig"
    inside = (t_obs >= t[0]) & (t_obs <= t[-1])
    inside &= ~is_rig | ((t_frame >= t[0]) & (t_frame <= t[-1]))
    dropped = int((~inside).sum())
    obs = obs.loc[inside].reset_index(drop=True)
    t_obs, t_frame, is_rig = t_obs[inside], t_frame[inside], is_rig[inside]
    xyz = ego[["x", "y", "z"]].to_numpy(dtype=float)
    rot = Rotation.from_quat(
        ego[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float, copy=True)
    )
    centers = obs[["center_x", "center_y", "center_z"]].to_numpy(
        dtype=float, copy=True
    )
    if is_rig.any():
        tq = t_frame[is_rig]
        position = np.column_stack([np.interp(tq, t, xyz[:, j]) for j in range(3)])
        centers[is_rig] = Slerp(t, rot)(tq).apply(centers[is_rig]) + position
    # Interpolation uses the right-hand ego pose too. Its timestamp is a
    # lower bound on availability, even when the obstacle timestamp is earlier.
    # Offline labels have unknown processing latency: this is NOT a claim of
    # actual online availability.
    pose_support = t_obs.copy()
    pose_support[is_rig] = t[np.searchsorted(t, t_frame[is_rig], side="left")]
    angles = rot.as_euler("xyz")
    if not np.isfinite(centers).all() or not np.isfinite(xyz).all():
        raise ValueError("nonfinite positions")
    canonical_ego = pd.DataFrame({"t_s": t, "x_m": xyz[:, 0], "y_m": xyz[:, 1],
                                 "yaw_rad": angles[:, 2]})
    canonical_agents = pd.DataFrame({
        "t_s": t_obs, "available_s": np.maximum(t_obs, pose_support),
        "track_id": obs.track_id.astype(str), "x_m": centers[:, 0], "y_m": centers[:, 1],
        "radius_m": 0.5 * np.hypot(obs.size_x.to_numpy(float), obs.size_y.to_numpy(float)),
        "class_name": obs.label_class.astype(str) if "label_class" in obs else "unknown",
    }).sort_values(["track_id", "t_s"]).reset_index(drop=True)
    dims = asdict(dimensions) if is_dataclass(dimensions) else dict(dimensions)
    required_dims = ["length", "width", "rear_axle_to_bbox_center"]
    if any(k not in dims for k in required_dims):
        raise ValueError(f"vehicle_dimensions must supply {required_dims}")
    dims = {k: float(v) for k, v in dims.items()}
    metadata = {
        "kind": "physicalai_offline_machine_labels", "coordinate_frame": "clip_anchor",
        "reference_point": "rear_axle", "time_origin_us": t_origin_us,
        "obstacle_frame_selection": obstacle_frame,
        "raw_reference_frames": sorted(set(frame_values)) if frame_values is not None else None,
        "rows_outside_ego_support": dropped, "vehicle_dimensions": dims,
        "ego_radius_m": rear_axle_radius(dims["length"], dims["width"], dims["rear_axle_to_bbox_center"]),
        "ego_max_abs_roll_pitch_deg": float(np.rad2deg(np.abs(angles[:, :2])).max()),
        "geometry_assumption": "planar road and upright rigid rectangular footprints; constant shape bound",
        "obstacle_inventory_complete": False,
        "availability_contract": "observation_and_interpolation_support_v1",
        "availability_is_lower_bound": True,
        "offline_label_processing_latency_known": False,
        "source_urls": SOURCE_URLS,
    }
    return canonical_ego, canonical_agents, metadata


def ensure_clip_features(ds, clip_ids, features):
    """Fetch metadata-defined files without the SDK's incompatible progress wrapper.

    SDK 0.2.3's batch downloader supplies a partial tqdm implementation that
    newer Hub Xet callbacks cannot use. The single-file SDK API uses Hub's
    native progress reporting and also supports cache-only offline operation.
    """
    filenames = dict.fromkeys(
        ds.features.get_chunk_feature_filename(ds.get_clip_chunk(clip_id), feature)
        for clip_id in clip_ids for feature in features
    )
    for filename in filenames:
        if ds.download_file(filename) is None:
            raise RuntimeError(f"Required feature file was not downloaded: {filename}")


def fetch_dataset(output, n=5, seed=7, split="test", split_name="alpamayo_aligned",
                  revision=None, cache_dir=None, clip_ids=None, obstacle_frame="auto"):
    # Import lazily: demo/core/test workflow does not require the NVIDIA SDK or HF.
    from physical_ai_av import PhysicalAIAVDatasetInterface
    requested_revision = revision
    if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        from huggingface_hub import HfApi
        # SDK 0.2.3 resolves the default branch only when revision=None. Pin
        # explicit branches/tags too, before metadata and feature downloads.
        revision = HfApi().dataset_info(
            "nvidia/PhysicalAI-Autonomous-Vehicles", revision=revision
        ).sha
        if not revision:
            raise ValueError("Hugging Face did not return a resolved dataset commit")
    ds = PhysicalAIAVDatasetInterface(revision=revision, cache_dir=cache_dir, split_name=split_name)
    missing = set(FEATURES) - ds.features.ALL
    if missing:
        raise ValueError(f"dataset revision lacks required features: {missing}")
    idx = ds.clip_index.copy()
    if "clip_is_valid" not in idx or "split" not in idx:
        raise ValueError("expected clip_is_valid and split; use physical_ai_av >= 0.2.3")
    eligible = idx.clip_is_valid.fillna(False).astype(bool) & idx.split.eq(split)
    presence = ds.feature_presence.reindex(idx.index)
    for feature in FEATURES:
        if feature in presence:
            eligible &= presence[feature].fillna(False).astype(bool)
    candidates = idx.loc[eligible]
    if clip_ids:
        absent = set(clip_ids) - set(candidates.index)
        if absent:
            raise ValueError(f"clips not eligible for selected split/features: {sorted(absent)}")
        selected = candidates.loc[list(dict.fromkeys(clip_ids))]
    else:
        if n < 1 or n > len(candidates):
            raise ValueError(f"n must be between 1 and {len(candidates)}")
        selected = candidates.sample(n=n, random_state=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": ds.repo_id, "resolved_revision": ds.revision,
        "requested_revision": requested_revision,
        "sdk_version": version("physical_ai_av"), "split_name": split_name, "split": split,
        "seed": seed, "sampling": "explicit IDs" if clip_ids else "uniform random eligible clips",
        "eligible_clips": len(candidates), "features": FEATURES,
        "selected": selected.reset_index().to_dict(orient="records"), "errors": [],
    }
    json_write(output / "download_manifest.json", manifest)
    ids = selected.index.tolist()
    # Only label/calibration chunks. No video/LiDAR/model downloads are needed here.
    ensure_clip_features(ds, ids, FEATURES)
    for clip_id in ids:
        try:
            ego_raw = ds.get_clip_feature(clip_id, "egomotion.offline")
            obs_raw = ds.get_clip_feature(clip_id, "obstacle.offline")
            dims = ds.get_clip_feature(clip_id, "vehicle_dimensions")
            clip_dir = output / str(clip_id)
            clip_dir.mkdir(exist_ok=True)
            # Retain small label tables so normalization can be revisited without refetch.
            raw_ego, raw_obs = table(ego_raw, "ego"), table(obs_raw, "obstacles")
            raw_ego.to_parquet(clip_dir / "raw_ego.parquet", index=False)
            raw_obs.to_parquet(clip_dir / "raw_obstacles.parquet", index=False)
            json_write(clip_dir / "schema.json", {
                "ego": {k: str(v) for k, v in raw_ego.dtypes.items()},
                "obstacles": {k: str(v) for k, v in raw_obs.dtypes.items()},
            })
            ego, agents, meta = normalize(raw_ego, raw_obs, dims, obstacle_frame)
            meta.update({"clip_id": str(clip_id), "dataset_revision": ds.revision, "split": split})
            if clip_id in ds.data_collection.index:
                collection = ds.data_collection.loc[clip_id]
                # Optional collection metadata can be missing; preserve that
                # absence as JSON null without relaxing numeric data checks.
                meta["collection"] = {
                    key: (None if pd.isna(value) else
                          value.item() if isinstance(value, np.generic) else value)
                    for key, value in collection.items()
                }
            ego.to_csv(clip_dir / "ego.csv", index=False)
            agents.to_csv(clip_dir / "agents.csv", index=False)
            json_write(clip_dir / "metadata.json", meta)
        except Exception as exc:
            manifest["errors"].append({"clip_id": str(clip_id), "error": str(exc)})
    json_write(output / "download_manifest.json", manifest)
    if manifest["errors"]:
        raise RuntimeError(f"{len(manifest['errors'])} clip(s) failed; see {output / 'download_manifest.json'}")
    return manifest

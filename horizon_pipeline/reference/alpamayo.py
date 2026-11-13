"""Alpamayo reference (T012): Alpamayo 1.5 plans, imported into the T011 reference format.

Input contract (one directory per clip, written by the separate snapshot process):

    <from>/<clip_id>/run.json
    <from>/<clip_id>/t_<t0_us:09d>.npz      one per snapshot

Each `.npz` holds what NVIDIA's loader and `sample_trajectories_from_data_with_vlm_rollout` give:
- `clip_id`: str;
- `t0_us`: int, the input time on the clip's clock (= the scene's `timestamp_us`);
- `pred_xyz`: (1, 1, S, 64, 3) positions, rig frame at t0, steps t0 + 0.1 ... t0 + 6.4 s;
- `pred_rot`: (1, 1, S, 64, 3, 3) rotation matrices in the same frame.

The loader puts poses in the rig frame at t0 as `xyz_local = R_t0^-1 (xyz - xyz_t0)` and
`rot_local = R_t0^-1 R`. Since the scene's world frame differs from the egomotion frame only by a
fixed rigid transform, the inverse with the scene's ego pose at t0 lands in the scene's world frame:
`xyz_world = R_t0 xyz_local + xyz_t0`, `R_world = R_t0 rot_local`.

The rig origin is the rear-axle midpoint (T006) for both the scene and Alpamayo, which is trained on
rig egomotion, so no offset is applied. There is no step 0 in the output; the importer adds it as
the ego pose at t0, so every plan starts exactly at the ego.

`run.json` holds at least the keys in `RUN_KEYS`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from horizon_pipeline.raw.store import sha256
from horizon_pipeline.reference.format import COLUMNS, FRAME
from horizon_pipeline.scene.build import DT_S, RATE_HZ, code_version
from horizon_pipeline.scene.frames import QUAT, XYZ, to_parent, yaw_of

SOURCE = "alpamayo"
N_STEPS = 64  # model output steps, t0 + 0.1 ... t0 + 6.4 s
MATCH_TOL_US = 1_000  # an output matches the scene tick within 1 ms
RUN_KEYS = ["model", "ego_source", "snapshot_interval_s", "n_cameras", "history_s"]
SNAPSHOT_GLOB = "t_*.npz"


class AlpamayoImportError(ValueError):
    pass


def snapshot_name(t0_us: int) -> str:
    return f"t_{t0_us:09d}.npz"


def read_run(clip_dir: Path) -> dict:
    path = Path(clip_dir) / "run.json"
    if not path.is_file():
        raise AlpamayoImportError(f"no run.json in {clip_dir}")
    run = json.loads(path.read_text())
    missing = [k for k in RUN_KEYS if k not in run]
    if missing:
        raise AlpamayoImportError(f"{path} is missing {missing}")
    return run


def read_snapshot(path: Path) -> dict:
    """One `.npz`, checked against the contract. Returns clip_id, t0_us, xyz (S, 64, 3), rot (S, 64, 3, 3)."""
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in ["clip_id", "t0_us", "pred_xyz", "pred_rot"] if k not in f.files]
        if missing:
            raise AlpamayoImportError(f"{path.name} is missing {missing}")
        xyz, rot = f["pred_xyz"], f["pred_rot"]
        snap = {"clip_id": str(f["clip_id"]), "t0_us": int(f["t0_us"])}
    if xyz.ndim != 5 or xyz.shape[:2] != (1, 1) or xyz.shape[3:] != (N_STEPS, 3):
        raise AlpamayoImportError(f"{path.name}: pred_xyz has shape {xyz.shape}, expected (1, 1, S, {N_STEPS}, 3)")
    if rot.shape != xyz.shape[:4] + (3, 3):
        raise AlpamayoImportError(f"{path.name}: pred_rot has shape {rot.shape}, expected {xyz.shape[:4] + (3, 3)}")
    if not (np.isfinite(xyz).all() and np.isfinite(rot).all()):
        raise AlpamayoImportError(f"{path.name}: non-finite values in the trajectory")
    if path.name != snapshot_name(snap["t0_us"]):
        raise AlpamayoImportError(f"{path.name} holds t0_us = {snap['t0_us']}; expected the name {snapshot_name(snap['t0_us'])}")
    snap["xyz"], snap["rot"] = xyz[0, 0].astype(float), rot[0, 0].astype(float)
    return snap


def match_tick(t0_us: int, ego: pd.DataFrame) -> int:
    """The scene tick at `t0_us`, within 1 ms. Never rounds silently to a far tick."""
    ts = ego["timestamp_us"].to_numpy(np.int64)
    k = int(np.argmin(np.abs(ts - t0_us)))
    if abs(int(ts[k]) - t0_us) > MATCH_TOL_US:
        raise AlpamayoImportError(
            f"t0_us = {t0_us} matches no scene tick within {MATCH_TOL_US} µs "
            f"(nearest: tick {k} at {int(ts[k])} µs; ticks are every {int(1e6 / RATE_HZ)} µs)"
        )
    return int(ego["tick"].iloc[k])


def plans_at_tick(snap: dict, pose: pd.Series) -> pd.DataFrame:
    """All samples of one snapshot in the world frame, with step 0 = the ego pose at t0."""
    n_samples = snap["xyz"].shape[0]
    p0 = pose[XYZ].to_numpy(float)
    r0 = Rotation.from_quat(pose[QUAT].to_numpy(float))
    xyz = to_parent(snap["xyz"].reshape(-1, 3), p0, r0).reshape(n_samples, N_STEPS, 3)
    xyz = np.concatenate([np.broadcast_to(p0, (n_samples, 1, 3)), xyz], axis=1)
    yaw = yaw_of(r0 * Rotation.from_matrix(snap["rot"].reshape(-1, 3, 3))).reshape(n_samples, N_STEPS)
    # Continue the scene's (unwrapped) heading at t0 without 2π jumps.
    yaw = np.unwrap(np.concatenate([np.full((n_samples, 1), float(pose["yaw"])), yaw], axis=1), axis=1)
    speed = np.zeros((n_samples, N_STEPS + 1))
    speed[:, 1:] = np.linalg.norm(np.diff(xyz, axis=1), axis=2) / DT_S  # backward differences
    speed[:, 0] = float(pose["speed"])  # known at t0 (the scene's causal speed)
    steps = N_STEPS + 1
    ref = pd.DataFrame({
        "tick": int(pose["tick"]),
        "sample": np.repeat(np.arange(n_samples), steps),
        "step": np.tile(np.arange(steps), n_samples),
    })
    ref["tau_s"] = ref["step"] * DT_S
    ref[XYZ] = xyz.reshape(-1, 3)
    ref["yaw"] = yaw.reshape(-1)
    ref["speed"] = speed.reshape(-1)
    return ref[COLUMNS]


def build_alpamayo(scene_dir: Path, clip_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Every snapshot of `clip_dir` imported against the scene in `scene_dir`."""
    scene_dir, clip_dir = Path(scene_dir), Path(clip_dir)
    scene = json.loads((scene_dir / "meta.json").read_text())
    if scene["rate_hz"] != RATE_HZ:
        raise AlpamayoImportError(f"scene rate {scene['rate_hz']} Hz, expected {RATE_HZ} Hz")
    run = read_run(clip_dir)
    if run["ego_source"] != scene["ego_source"]:
        raise AlpamayoImportError(f"the run used {run['ego_source']} egomotion; the scene is {scene['ego_source']}")
    paths = sorted(clip_dir.glob(SNAPSHOT_GLOB))
    if not paths:
        raise AlpamayoImportError(f"no {SNAPSHOT_GLOB} in {clip_dir}")
    ego = pd.read_parquet(scene_dir / "ego.parquet").set_index("tick", drop=False)

    plans, ticks, n_samples = [], [], set()
    for path in paths:
        snap = read_snapshot(path)
        if snap["clip_id"] != scene["clip_id"]:
            raise AlpamayoImportError(f"{path.name} is for clip {snap['clip_id']}, the scene is {scene['clip_id']}")
        tick = match_tick(snap["t0_us"], ego)
        pose = ego.loc[tick]
        if not pose["has_history"]:
            raise AlpamayoImportError(f"{path.name}: t0 = {pose['t_s']:.1f} s has less than the {run['history_s']} s of history Alpamayo needs")
        plans.append(plans_at_tick(snap, pose))
        ticks.append(tick)
        n_samples.add(snap["xyz"].shape[0])
    if len(set(ticks)) != len(ticks):
        raise AlpamayoImportError("two snapshots match the same scene tick")

    ref = pd.concat(plans, ignore_index=True).sort_values(["tick", "sample", "step"], kind="stable").reset_index(drop=True)
    first = ref.loc[ref["step"] == 1].set_index("tick")
    # Check value: the first step's length minus 0.1 s at the ego's speed at t0 (~0 if the plan
    # starts where and how fast the car is).
    gap = (first["speed"] - ego.loc[first.index, "speed"]) * DT_S
    meta = {
        "source": SOURCE,
        "clip_id": scene["clip_id"],
        "ego_source": scene["ego_source"],
        "frame": FRAME,
        "step_s": DT_S,
        "n_steps": {"min": N_STEPS + 1, "max": N_STEPS + 1},
        "n_ticks": len(ticks),
        "n_samples": sorted(n_samples)[0] if len(n_samples) == 1 else sorted(n_samples),
        "snapshot_interval_s": run["snapshot_interval_s"],
        "step0": "added: the scene ego pose at t0 (Alpamayo outputs start at t0 + 0.1 s)",
        "step1_gap_m": {"max_abs": float(np.abs(gap).max()), "median": float(gap.median())},
        "run": run,
        "inputs": {
            "files": {
                "scene/ego.parquet": sha256(scene_dir / "ego.parquet"),
                **{f"alpamayo/{p.name}": sha256(p) for p in [clip_dir / "run.json", *paths]},
            },
            "code": code_version(),
        },
    }
    return ref, meta


def write_snapshot(out_dir: Path, clip_id: str, t0_us: int, xyz: np.ndarray, rot: np.ndarray) -> Path:
    """Write one snapshot in the contract's layout; `xyz` (S, 64, 3) and `rot` (S, 64, 3, 3), rig frame at t0."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / snapshot_name(t0_us)
    np.savez(path, clip_id=np.str_(clip_id), t0_us=np.int64(t0_us), pred_xyz=xyz[None, None], pred_rot=rot[None, None])
    return path


def loader_frame(ego: pd.DataFrame, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """The ego's own next 64 poses after `tick`, put in the rig frame at `tick` with the loader's
    transform `R_t0^-1 (xyz - xyz_t0)`, `R_t0^-1 R`. Shapes (64, 3) and (64, 3, 3).

    This is how Alpamayo's ground-truth future looks; the tests import it and expect the ego track
    back."""
    ego = ego.set_index("tick", drop=False)
    future = ego.loc[tick + 1 : tick + N_STEPS]
    if len(future) != N_STEPS:
        raise ValueError(f"tick {tick} has only {len(future)} future ticks, need {N_STEPS}")
    r0 = Rotation.from_quat(ego.loc[tick, QUAT].to_numpy(float))
    xyz = r0.inv().apply(future[XYZ].to_numpy(float) - ego.loc[tick, XYZ].to_numpy(float))
    rot = (r0.inv() * Rotation.from_quat(future[QUAT].to_numpy(float))).as_matrix()
    return xyz, rot

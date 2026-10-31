"""Package existing front-wide frames and past ego poses; no network access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_history(ego, t0_us):
    # Never let interpolation consume a raw pose after the prediction anchor.
    ego = ego[ego.timestamp <= t0_us]
    times = ego.timestamp.to_numpy(dtype=np.int64)
    query = t0_us + np.arange(-15, 1, dtype=np.int64) * 100_000
    if len(times) < 2 or query[0] < times[0] or query[-1] > times[-1]:
        raise ValueError("History needs past-only support; anchor at an observed ego timestamp")
    positions = np.column_stack([
        np.interp(query, times, ego[k].to_numpy(float)) for k in ("x", "y", "z")
    ])
    rotations = Slerp(times.astype(float), Rotation.from_quat(
        ego[["qx", "qy", "qz", "qw"]].to_numpy(float, copy=True)
    ))(query.astype(float))
    origin, rotation = positions[-1], rotations[-1]
    xyz = rotation.inv().apply(positions - origin)
    rot = (rotation.inv() * rotations).as_matrix()
    # Round-trip check before using this transform on predicted positions.
    np.testing.assert_allclose(rotation.apply(xyz) + origin, positions, atol=1e-9)
    return xyz, rot, origin, rotation.as_matrix(), query


def main():
    import av
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/runpod_selected_frontwide")
    parser.add_argument("--stride-s", type=float, help="Sample each cached clip on this dataset-time grid instead of the three selected moments")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    moments = json.loads((root / "results/selected_moments/moments.json").read_text())
    if args.stride_s is not None:
        if not np.isfinite(args.stride_s) or args.stride_s <= 1.5:
            raise ValueError("stride must exceed the 1.5 s history duration")
        requests = []
        for clip in sorted({m["clip_id"] for m in moments["moments"]}):
            directory = root / "data/pai_selected_train" / clip
            ego_times = pd.read_csv(directory / "ego.csv").t_s
            for t0 in np.arange(args.stride_s, float(ego_times.max()), args.stride_s):
                requests.append(dict(clip_id=clip, label_t0_s=float(t0), requested_video_s=None))
        moments["moments"] = requests
    cases = []
    for moment in moments["moments"]:
        clip = moment["clip_id"]
        directory = root / "data/pai_selected_train" / clip
        meta = json.loads((directory / "metadata.json").read_text())
        requested_t0_s = moment["label_t0_s"]
        requested_t0_us = round(requested_t0_s * 1e6) + meta["time_origin_us"]
        ego = pd.read_parquet(directory / "raw_ego.parquet").sort_values("timestamp")
        # Preserve the requested sampling grid, but use the latest observed pose
        # as the actual inference anchor. No extrapolated current pose is invented.
        eligible = ego.timestamp[ego.timestamp <= requested_t0_us]
        if eligible.empty:
            raise ValueError("No past ego pose for the requested anchor")
        t0_us = int(eligible.iloc[-1])
        t0_s = (t0_us - meta["time_origin_us"]) / 1e6
        timestamps = pd.read_parquet(directory / "front_camera_timestamps.parquet").timestamp.to_numpy(np.int64)
        targets = t0_us + np.arange(-3, 1, dtype=np.int64) * 100_000
        indices = np.searchsorted(timestamps, targets, side="right") - 1
        if (indices < 0).any() or (np.diff(timestamps) <= 0).any():
            raise ValueError("Camera timestamps cannot support this request")
        actual = timestamps[indices]
        if (actual > targets).any() or ((targets - actual) > 50_000).any():
            raise ValueError("Camera frame is future-dated or more than 50 ms old")
        decoded = {}
        with av.open(str(directory / "front_camera.mp4")) as video:
            for i, frame in enumerate(video.decode(video=0)):
                if i in indices:
                    decoded[i] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                if i >= indices.max():
                    break
        frames = np.stack([decoded[int(i)] for i in indices])
        xyz, rotations, origin, rotation, history_us = local_history(ego, t0_us)
        case_id = f"{clip[:8]}_t{t0_s:.6f}s"
        filename = f"{case_id}.npz"
        np.savez_compressed(out / filename, image_frames=frames,
                            ego_history_xyz=xyz.astype(np.float32)[None, None],
                            ego_history_rot=rotations.astype(np.float32)[None, None])
        # Reference labels are separate from model inputs. Never extrapolate the clip tail.
        future_us = t0_us + np.arange(1, 65, dtype=np.int64) * 100_000
        future_us = future_us[future_us <= int(ego.timestamp.max())]
        truth_xyz = np.column_stack([
            np.interp(future_us, ego.timestamp.to_numpy(np.int64), ego[k].to_numpy(float))
            for k in ("x", "y", "z")
        ])
        cases.append(dict(case_id=case_id, clip_id=clip, input_file=filename,
                          input_sha256=sha256(out / filename),
                          requested_video_s=moment["requested_video_s"],
                          t0_s=t0_s, t0_us=t0_us, requested_t0_s=requested_t0_s,
                          history_latest_source_timestamp_us=t0_us,
                          history_resampling="past_support_only_at_observed_pose",
                          time_origin_us=meta["time_origin_us"],
                          camera_indices=[1], camera_frame_timestamps_us=actual.tolist(),
                          history_timestamps_us=history_us.tolist(),
                          origin_in_clip_anchor_m=origin.tolist(),
                          rotation_local_to_clip_anchor=rotation.tolist(),
                          ground_truth_tau_s=((future_us - t0_us) / 1e6).tolist(),
                          ground_truth_xyz_clip_anchor=truth_xyz.tolist(),
                          raw_ego_sha256=sha256(directory / "raw_ego.parquet")))
    manifest = dict(dataset_revision=moments["dataset_revision"],
                    camera_mode="front_wide_only", ego_history_source="egomotion.offline",
                    history_resampling="past_support_only_at_observed_pose",
                    offline_label_processing_latency_known=False,
                    reference_point="rear_axle", future_step_s=0.1, future_steps=64,
                    anchor_time_basis="dataset seconds relative to time_origin_us; not trimmed preview time",
                    cases=cases)
    (out / "inputs.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    shutil.copy2(root / "scripts/runpod_generate_references.py", out / "run_selected.py")
    archive = out.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for path in sorted(out.iterdir()):
            z.write(path, arcname=f"{out.name}/{path.name}")
    print(f"Prepared {len(cases)} moments: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

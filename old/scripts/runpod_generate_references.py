"""Run packaged front-wide-only inputs on Runpod; all downloads disabled.

In the already-loaded notebook:
%run -i /workspace/runpod_selected_frontwide/run_selected.py --reuse-loaded-model

Or from a terminal with the notebook kernel shut down:
python /workspace/runpod_selected_frontwide/run_selected.py
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import zipfile

# Set before library imports in standalone mode. Reuse mode does not load models.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np


def reference_record(case, xyz_local, manifest, sample, seed, latency_s):
    """Convert 3D local waypoints before XY projection; prepend current pose."""
    xyz = np.asarray(xyz_local, dtype=float)
    if xyz.shape != (manifest["future_steps"], 3) or not np.isfinite(xyz).all():
        raise ValueError(f"Unexpected/nonfinite trajectory: {xyz.shape}")
    rotation = np.asarray(case["rotation_local_to_clip_anchor"])
    origin = np.asarray(case["origin_in_clip_anchor_m"])
    anchor_xyz = xyz @ rotation.T + origin
    xy = np.vstack([origin[:2], anchor_xyz[:, :2]])
    return dict(clip_id=case["clip_id"], t0_s=case["t0_s"],
                observation_cutoff_s=case["t0_s"], available_time_s=case["t0_s"],
                availability_assumption="offline_zero_latency_reference_comparison",
                coordinate_frame="clip_anchor", reference_point=manifest["reference_point"],
                source="alpamayo_1_5_front_wide_only_offline_ego",
                sample_id=sample, seed=seed, inference_latency_s=latency_s,
                tau_s=(np.arange(len(xy)) * manifest["future_step_s"]).tolist(),
                xy_m=xy.tolist())


def main(existing_model=None, existing_processor=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-loaded-model", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "inputs.json").read_text())
    # Verify every input before loading or running a model.
    for case in manifest["cases"]:
        path = root / case["input_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["input_sha256"]:
            raise ValueError(f"Input checksum mismatch: {path}")
    import torch
    from alpamayo1_5 import helper

    if args.reuse_loaded_model:
        if existing_model is None or existing_processor is None:
            raise RuntimeError("Run in the existing notebook with %run -i; model and processor must exist.")
        model, processor = existing_model, existing_processor
    else:
        from huggingface_hub import snapshot_download
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        snapshot = snapshot_download("nvidia/Alpamayo-1.5-10B", local_files_only=True)
        model = Alpamayo1_5.from_pretrained(snapshot, dtype=torch.bfloat16,
                                          local_files_only=True).to("cuda")
        processor = helper.get_processor(model.tokenizer)
    model.eval()
    output = root / ("outputs_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    provenance = dict(manifest=manifest, model_name="nvidia/Alpamayo-1.5-10B",
                      model_class=type(model).__name__,
                      model_commit=getattr(model.config, "_commit_hash", None),
                      model_origin=getattr(model.config, "_name_or_path", None),
                      reused_loaded_model=args.reuse_loaded_model,
                      samples=args.samples, gpu=torch.cuda.get_device_name(),
                      torch_version=torch.__version__,
                      transformers_version=importlib.metadata.version("transformers"),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      settings=dict(top_p=0.98, temperature=0.6, num_traj_samples=1,
                                    max_generation_length=256),
                      timing_scope="model_execution_only_no_warmup_excluded",
                      notes="Single-front-camera experiment; offline labeled ego history. "
                            "Availability is t0 for offline comparison, NOT measured online availability.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    comparison_rows = []
    for case in manifest["cases"]:
        with np.load(root / case["input_file"], allow_pickle=False) as data:
            frames = torch.from_numpy(data["image_frames"])
            history = torch.from_numpy(data["ego_history_xyz"])
            rotations = torch.from_numpy(data["ego_history_rot"])
        messages = helper.create_message(frames=frames, camera_indices=torch.tensor(case["camera_indices"]))
        tokens = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        inputs = helper.to_device(dict(tokenized_data=tokens,
                                      ego_history_xyz=history, ego_history_rot=rotations), "cuda")
        for sample in range(args.samples):
            seed = args.seed + sample
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inputs, return_extra=True, **provenance["settings"])
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            xyz = pred_xyz[0, 0, 0].detach().float().cpu().numpy()
            rotation = pred_rot[0, 0, 0].detach().float().cpu().numpy()
            record = reference_record(case, xyz, manifest, sample, seed, latency)
            stem = f"{case['case_id']}_sample_{sample:03d}"
            np.savez_compressed(output / f"{stem}.npz", predicted_xyz_local=xyz,
                                predicted_rot_local=rotation)
            # Separate files avoid duplicate (clip_id, t0_s) keys in local replay.
            with (output / f"references_sample_{sample:03d}.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            (output / f"{stem}_reasoning.json").write_text(
                json.dumps(extra.get("cot"), default=str, indent=2))
            if "ground_truth_tau_s" in case:
                truth = np.asarray(case["ground_truth_xyz_clip_anchor"], dtype=float)[:, :2]
                predicted = np.asarray(record["xy_m"])[1:]
                errors = np.linalg.norm(predicted[:len(truth)] - truth, axis=1)
                comparison_rows.append(dict(
                    clip_id=case["clip_id"], t0_s=case["t0_s"], sample_id=sample,
                    compared_points=len(truth), compared_duration_s=case["ground_truth_tau_s"][-1],
                    full_6_4s_comparison=len(truth) == manifest["future_steps"],
                    ade_m=float(errors.mean()), last_compared_error_m=float(errors[-1]),
                    inference_latency_s=latency))
                np.savez_compressed(output / f"{stem}_comparison.npz",
                                    tau_s=record["tau_s"], predicted_xy_clip_anchor=record["xy_m"],
                                    ground_truth_tau_s=case["ground_truth_tau_s"],
                                    ground_truth_xy_clip_anchor=truth, displacement_errors_m=errors)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 6))
                origin = np.asarray(case["origin_in_clip_anchor_m"])[:2]
                ax.plot(*(predicted - origin).T, label="Alpamayo: 6.4 s")
                ax.plot(*(truth - origin).T, label=f"Recorded: {case['ground_truth_tau_s'][-1]:g} s")
                ax.scatter(0, 0, marker="*", label="Start")
                ax.set(title=f"{case['case_id']} / sample {sample}\nFront-wide only; ADE {errors.mean():.2f} m",
                       xlabel="Clip-frame X offset (m)", ylabel="Clip-frame Y offset (m)")
                ax.axis("equal")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output / f"{stem}.png", dpi=140)
                plt.close(fig)
                with (output / "comparison_metrics.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0]))
                    writer.writeheader()
                    writer.writerows(comparison_rows)
            print(f"Saved {stem}: {latency:.2f}s", flush=True)
            del pred_xyz, pred_rot, extra
        del inputs, tokens
    (output / "COMPLETE").write_text(f"{len(manifest['cases']) * args.samples} trajectories\n")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            z.write(path, arcname=f"{output.name}/{path.name}")
    print(f"\nDownload this results archive: {archive}", flush=True)


if __name__ == "__main__":
    main(globals().get("model"), globals().get("processor"))

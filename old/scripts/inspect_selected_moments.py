"""Reproduce the three requested video moments using local data only.

Run from the project root with:
uv run --offline --no-sync --extra dataset python scripts/inspect_selected_moments.py

Full-inventory validity is preserved. Per-agent curves are teaching diagnostics,
not replacements for an invalid full-scene result. No network APIs are used.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from horizonlab.core import clearance_horizon
from horizonlab.dataset import json_write
from horizonlab.replay import (
    check_canonical, estimate_obstacles, interp_xy, recorded_reference,
    retrospective_diagnostics, validate_config,
)

ROOT = Path("data/pai_selected_train")
OUT = Path("results/selected_moments")
REQUESTS = [
    ("b4d979af-254e-4bc0-afdd-3a36531f4540", [2.0, 4.0], "29"),
    ("b789fab8-41f1-4cf3-b7f8-a8e94939f67d", [6.0], "2"),
]


def video_frames(path, requested_times):
    best = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            pts = float(frame.time)
            for requested in requested_times:
                error = abs(pts - requested)
                if requested not in best or error < best[requested][0]:
                    best[requested] = (error, index, pts, frame.to_image())
    return best


def view_coordinates(points, origin, yaw):
    delta = np.atleast_2d(points) - origin
    forward = delta @ np.array([np.cos(yaw), np.sin(yaw)])
    left = delta @ np.array([-np.sin(yaw), np.cos(yaw)])
    return np.column_stack([-left, forward])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("examples/sensitivity_config.json").read_text())
    validate_config(config)
    base = config["experiments"][0]
    assert base["name"] == "illustrative_base"
    assert base["injection_position_radius_m"] == base["injection_velocity_radius_mps"] == 0
    json_write(OUT / "config_used.json", config)
    summaries, report = [], []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    for clip, requested_times, focus_id in REQUESTS:
        directory = ROOT / clip
        ego = pd.read_csv(directory / "ego.csv")
        agents = pd.read_csv(directory / "agents.csv", dtype={"track_id": str})
        meta = json.loads((directory / "metadata.json").read_text())
        check_canonical(ego, agents)
        times = pd.read_parquet(directory / "front_camera_timestamps.parquet").timestamp.to_numpy(np.int64)
        frames = video_frames(directory / "front_camera.mp4", requested_times)
        for requested in requested_times:
            _, index, pts, frame = frames[requested]
            t0 = float((int(times[index]) - meta["time_origin_us"]) * 1e-6)
            stem = f"{clip[:8]}_video_{requested:g}s"
            frame.save(OUT / f"{stem}_frame.png")
            tau, xy = recorded_reference(ego, t0, config["reference_horizon_s"])
            now = xy[0]
            yaw = float(np.interp(t0, ego.t_s, np.unwrap(ego.yaw_rad)))
            rho0 = float(np.linalg.norm(interp_xy(ego, [t0])[0] - now)) + base["rho0_extra_m"]
            past = agents[(agents.t_s <= t0 + 1e-10) & (agents.available_s <= t0 + 1e-10)]
            inventory, inventory_error = estimate_obstacles(agents, t0, base, config, clip)
            rows, models = [], {}
            for track_id, track in past.groupby("track_id", sort=True):
                track = track.sort_values("t_s")
                last = track.iloc[-1]
                history = track[track.t_s >= last.t_s - config["history_window_s"] - 1e-10]
                obs, error = estimate_obstacles(track, t0, base, config, clip)
                row = {
                    "track_id": str(track_id), "class_name": str(last.class_name),
                    "past_samples": len(track), "history_samples": len(history),
                    "last_observation_s": float(last.t_s), "age_s": float(t0-last.t_s),
                    "last_x_m": float(last.x_m), "last_y_m": float(last.y_m),
                    "last_label_distance_m": float(np.linalg.norm(last[["x_m", "y_m"]].to_numpy(float)-now)),
                    "estimate_status": error or "available", "H_single_agent_s": None,
                }
                if not error:
                    o = obs[0]
                    models[str(track_id)] = o
                    cap = min(float(tau[-1]), config["validity_s"] - o.age_s)
                    result = clearance_horizon(tau, xy, [o], meta["ego_radius_m"], rho0,
                                               base["qbar_mps"], cap, config["time_tolerance_s"])
                    distance = float(np.linalg.norm(o.center(0)-now))
                    required = meta["ego_radius_m"] + rho0 + o.radius_m + float(o.beta(0))
                    assert np.isclose(result.initial_margin_m, distance-required)
                    # World->plot rotation preserves the distances used in the equations.
                    assert np.isclose(np.linalg.norm(view_coordinates(o.center(0), now, yaw)), distance)
                    row.update({
                        "center_x_m": float(o.center(0)[0]), "center_y_m": float(o.center(0)[1]),
                        "vx_mps": float(o.velocity_mps[0]), "vy_mps": float(o.velocity_mps[1]),
                        "speed_mps": float(np.linalg.norm(o.velocity_mps)),
                        "center_distance_m": distance, "physical_radius_m": float(track.radius_m.max()),
                        "radius_with_padding_m": o.radius_m, "beta_at_anchor_m": float(o.beta(0)),
                        "required_separation_m": required, "initial_margin_m": result.initial_margin_m,
                        "H_single_agent_s": result.H_s, "single_agent_status": result.status,
                        "single_agent_cap_s": cap,
                    })
                rows.append(row)
            table = pd.DataFrame(rows)
            table.to_csv(OUT / f"{stem}_tracks.csv", index=False)
            failed = table[table.estimate_status != "available"]
            failed.to_csv(OUT / f"{stem}_invalid_tracks.csv", index=False)
            valid = table[table.estimate_status == "available"]
            focus = table.set_index("track_id").loc[focus_id].to_dict()
            assert focus["estimate_status"] == "available"
            chosen = [focus_id] + valid[valid.track_id != focus_id].nsmallest(2, "center_distance_m").track_id.tolist()
            full_result = None
            if inventory_error is None:
                full_cap = min(float(tau[-1]), config["validity_s"] - max(o.age_s for o in inventory))
                full_result = asdict(clearance_horizon(tau, xy, inventory, meta["ego_radius_m"],
                                                      rho0, base["qbar_mps"], full_cap,
                                                      config["time_tolerance_s"]))
            # All five settings use the same history requirements; retain invalid outcomes.
            experiments = []
            for experiment in config["experiments"]:
                _, error = estimate_obstacles(agents, t0, experiment, config, clip)
                one, one_error = estimate_obstacles(past[past.track_id == focus_id], t0, experiment, config, clip)
                assert one_error is None
                o = one[0]
                cap = min(float(tau[-1]), config["validity_s"] - o.age_s)
                result = clearance_horizon(tau, xy, [o], meta["ego_radius_m"], experiment["rho0_extra_m"],
                                           experiment["qbar_mps"], cap, config["time_tolerance_s"])
                experiments.append({"experiment": experiment["name"], "full_inventory_error": error,
                                    "focus_agent_only": asdict(result),
                                    "focus_future_label_diagnostic": retrospective_diagnostics(
                                        agents[agents.track_id == focus_id], t0, cap, result.H_s, [o])})
            entry = {
                "clip_id": clip, "requested_video_s": requested, "video_frame_index": index,
                "selected_video_pts_s": pts, "label_t0_s": t0,
                "reference_source": "recorded_future_oracle", "experiment": base["name"],
                "full_inventory_status": inventory_error or full_result["status"],
                "full_inventory_result": full_result, "observed_track_count": len(table),
                "invalid_track_count": len(failed), "invalid_reasons": failed.estimate_status.value_counts().to_dict(),
                "focus_track_id": focus_id, "focus_agent": focus,
                "ego_radius_m": meta["ego_radius_m"], "rho0_m": rho0,
                "sensitivity": experiments,
                "figure": f"{stem}.png", "tracks_file": f"{stem}_tracks.csv",
                "invalid_tracks_file": f"{stem}_invalid_tracks.csv",
                "input_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                 for name in ["ego.csv", "agents.csv", "metadata.json", "front_camera_timestamps.parquet"]},
            }
            summaries.append(entry)

            fig = plt.figure(figsize=(14, 9), layout="constrained")
            grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
            photo = fig.add_subplot(grid[0, 0])
            photo.imshow(frame); photo.axis("off")
            photo.set_title(f"Front camera | video {pts:.3f} s = label {t0:.6f} s", loc="left")
            plan = fig.add_subplot(grid[:, 1])
            colors = {"automobile": "#3b836e", "person": "#d47817"}
            for row in rows:
                tid = row["track_id"]
                if tid in models:
                    o = models[tid]; point = view_coordinates(o.center(0), now, yaw)[0]
                    color = colors.get(row["class_name"], "#906ab1")
                    if tid == focus_id: color = "#be3455"
                    if -24 <= point[0] <= 24 and -14 <= point[1] <= 46:
                        plan.add_patch(Circle(point, row["physical_radius_m"], facecolor=color,
                                              edgecolor=color, alpha=.22, lw=1))
                        plan.add_patch(Circle(point, o.radius_m+float(o.beta(0)), fill=False,
                                              edgecolor=color, alpha=.65, ls="--", lw=.8))
                        plan.plot(*point, ".", color=color, ms=4)
                        if tid in chosen:
                            plan.annotate(f"{tid}: {row['class_name']}", point, xytext=(5, 5),
                                          textcoords="offset points", fontsize=8, color=color)
                            step = view_coordinates(o.center(1), now, yaw)[0]-point
                            plan.annotate("", xy=point+step, xytext=point,
                                          arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
                else:
                    point = view_coordinates([row["last_x_m"], row["last_y_m"]], now, yaw)[0]
                    plan.plot(*point, "x", color="#555555", ms=6)
                    if -22 < point[0] < 22 and -12 < point[1] < 44:
                        plan.annotate(tid, point, xytext=(4, 4), textcoords="offset points", fontsize=7)
            path = view_coordinates(xy, now, yaw)
            plan.plot(path[:, 0], path[:, 1], color="#2475ba", lw=2, zorder=5)
            dims = meta["vehicle_dimensions"]
            plan.add_patch(Rectangle((-dims["width"]/2, dims["rear_axle_to_bbox_center"]-dims["length"]/2),
                                     dims["width"], dims["length"], facecolor="#2475ba", alpha=.7))
            plan.add_patch(Circle((0, 0), meta["ego_radius_m"], fill=False, edgecolor="#2475ba", lw=2))
            plan.add_patch(Circle((0, 0), meta["ego_radius_m"]+rho0, fill=False,
                                  edgecolor="#2475ba", ls="--"))
            plan.plot(0, 0, "+", color="black", ms=7, zorder=6)
            plan.set(xlim=(-22, 22), ylim=(-12, 44), aspect="equal", xlabel="Right of ego at anchor (m)",
                     ylabel="Ahead of ego at anchor (m)", title="Overhead labels and disks at the anchor")
            plan.grid(alpha=.15)
            plan.legend(handles=[
                Line2D([], [], color="#2475ba", label="Recorded future ego reference"),
                Line2D([], [], color="#3b836e", marker="o", ls="", label="Automobile label"),
                Line2D([], [], color="#d47817", marker="o", ls="", label="Person label"),
                Line2D([], [], color="#be3455", marker="o", ls="", label=f"Focus track {focus_id}"),
                Line2D([], [], color="#555555", marker="x", ls="", label="Missing history: last known position"),
            ], fontsize=8, loc="upper right")

            margins = fig.add_subplot(grid[1, 0])
            curve_colors = ["#be3455", "#3b836e", "#906ab1"]
            curve_rows = []
            for tid, color in zip(chosen, curve_colors):
                o = models[tid]
                cap = min(float(tau[-1]), config["validity_s"]-o.age_s)
                future = np.linspace(0, cap, 500)
                ref = np.column_stack([np.interp(future, tau, xy[:, j]) for j in range(2)])
                centers = o.position_m+(future+o.age_s)[:, None]*o.velocity_mps
                distance = np.linalg.norm(ref-centers, axis=1)
                required = meta["ego_radius_m"]+rho0+base["qbar_mps"]*future+o.radius_m+o.beta(future)
                margin = distance-required
                visible = future <= 3.0
                margins.plot(future[visible], margin[visible], color=color,
                             label=f"Track {tid} ({table.set_index('track_id').loc[tid, 'class_name']})")
                curve_rows.extend({"track_id": tid, "tau_s": float(t), "center_distance_m": float(d),
                                   "required_separation_m": float(s), "margin_m": float(m)}
                                  for t, d, s, m in zip(future, distance, required, margin))
            margins.axhline(0, color="#555555", ls="--", lw=1)
            h = focus["H_single_agent_s"]
            margins.axvline(h, color="#be3455", ls=":", lw=1)
            margins.annotate(f"Focus-agent prefix: {h:.3f} s", (h, 0), xytext=(8, 12), textcoords="offset points", color="#be3455")
            margins.set(xlabel=r"Look-ahead $\tau$ from this anchor (s)", ylabel="Clearance margin (m)",
                        title="First 3 s of selected-agent margins; full-scene H unavailable", xlim=(0, 3.0))
            margins.legend(fontsize=8, loc="lower left"); margins.grid(alpha=.15)
            fig.suptitle(f"{clip[:8]} | video {requested:g} s | full scene: {inventory_error or 'valid'}\n"
                         f"{len(failed)} of {len(table)} observed tracks lack usable estimates; missing tracks have NOT been removed",
                         fontsize=13, fontweight="bold")
            fig.supxlabel("Plan view is cropped; all past tracks remain in the audit. Dashed disks include uncertainty at the anchor. "
                          "Arrows show 1 s of nominal motion. Camera-to-track projection is not calibrated here.", fontsize=8)
            fig.savefig(OUT / f"{stem}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            pd.DataFrame(curve_rows).to_csv(OUT / f"{stem}_selected_margin_curves.csv", index=False)
            print(json.dumps({k: entry[k] for k in ["clip_id", "requested_video_s", "label_t0_s", "full_inventory_status", "invalid_track_count", "focus_agent"]}), flush=True)

    json_write(OUT / "moments.json", {
        "dataset_revision": json.loads((ROOT/"download_manifest.json").read_text())["resolved_revision"],
        "scope": "Full inventory evaluated without silently dropping tracks; selected-agent curves are conditional diagnostics only.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "moments": summaries,
    })
    report.extend([
        "# Three selected video moments\n",
        "All work uses the existing local videos and labels. No downloads or changes to the horizon model were made.\n",
        "**All three full-scene windows are invalid (`insufficient_track_history`). Their horizon is unavailable, not zero.** "
        "The original estimator requires three past observations per previously seen track. "
        "Single-agent figures below explain the equations but cannot replace the full inventory.\n",
        "The video PTS selects a frame; its paired timestamp selects the label anchor. "
        "The fixed overhead axes point forward/right at that anchor. Blue is the rear-axle ego disk, rectangular body, "
        "and recorded future reference (oracle). Filled object disks show physical radii; dashed disks add padding "
        "and current uncertainty. Gray crosses show last known positions for tracks with no usable estimate, "
        "which must not be interpreted as their current positions. No boxes are projected onto camera pixels.\n",
        "## Equations used\n",
        "Baseline ego tube: `rho(tau) = 0.2 + 0.5*tau` metres. Obstacle uncertainty: "
        "`beta(age+tau) = 0.3 + 0.5*(age+tau) + (age+tau)^2` metres. "
        "Obstacle radius is the largest past box half-diagonal plus 0.1 m padding. "
        "Ego rear-axle radius is `sqrt((length/2 + abs(offset))^2 + (width/2)^2)` = 3.790163 m.\n",
        "`margin = center distance - ego radius - rho - obstacle radius - beta`. "
        "The solver certifies the positive prefix on intervals; the plotted sampled curves are visual aids, "
        "not how the horizon solver establishes clearance. The figure zooms into the first 3 seconds; "
        "the curve CSVs extend through each selected agent's cap. Parameters are illustrative and unvalidated.\n",
        "The bicycle equations describe how a rear-axle vehicle moves under acceleration and steering. "
        "This label-replay experiment does not integrate those equations or simulate a controller: it substitutes "
        "the recorded future rear-axle path and assumes a tracking tube around it. Here qbar is set directly to "
        "0.5 m/s; it has not been derived from measured controller errors. The optional formula "
        "`qbar = speed_error_bound + 2*reference_speed_bound*sin(heading_error_bound/2)` is implemented in "
        "`horizonlab/core.py`, but is not used by this configuration.\n",
    ])
    for e in summaries:
        f = e["focus_agent"]
        report.extend([
            f"## {e['clip_id']} — video {e['requested_video_s']:g} s\n",
            f"![Frame, overhead view, and margin curves]({e['figure']})\n",
            f"Frame {e['video_frame_index']} maps to label **{e['label_t0_s']:.6f} s**. "
            f"Full inventory: {e['observed_track_count']} previously seen tracks; {e['invalid_track_count']} invalid.\n",
            f"Focus: **track {e['focus_track_id']} ({f['class_name']})**, estimated speed {f['speed_mps']:.3f} m/s, "
            f"last observation age {f['age_s']:.6f} s. Initial margin decomposition:\n",
            f"`{f['center_distance_m']:.6f} - {e['ego_radius_m']:.6f} - {e['rho0_m']:.6f} "
            f"- {f['radius_with_padding_m']:.6f} - {f['beta_at_anchor_m']:.6f} = {f['initial_margin_m']:.6f} m`\n",
            f"Its **single-agent** conservative prefix is {f['H_single_agent_s']:.6f} s "
            f"(`{f['single_agent_status']}`). The full-scene H remains unavailable.\n",
            "| Sensitivity setting | Focus-agent H (s) | Initial margin (m) | Status | Future-label tube misses / checks |\n"
            "|---|---:|---:|---|---:|",
        ])
        for s in e["sensitivity"]:
            r = s["focus_agent_only"]
            diagnostic = s["focus_future_label_diagnostic"]
            report.append(f"| {s['experiment']} | {r['H_s']:.6f} | {r['initial_margin_m']:.6f} | {r['status']} "
                          f"| {diagnostic['tube_violations']} / {diagnostic['tube_checked_observations']} |")
        report.append(f"\n[All track diagnostics]({e['tracks_file']}) · [Tracks preventing a full result]({e['invalid_tracks_file']})\n")
    report.extend([
        "## What follows from these results\n",
        "Do not delete short-history or stale tracks simply to produce a number. A later extension could keep them "
        "with an explicitly chosen velocity set and position reachable set, but those assumptions need justification. "
        "Range-based exclusion likewise needs a reachability argument; being far away in the last observation is not sufficient.\n",
        "A nonpositive margin is loss of the modeled clearance certificate, not evidence of physical collision. "
        "A cap-reached result is clear through that cap. The counts are track IDs, not guaranteed distinct physical objects. "
        "The train clips and hand-selected times are explanatory cases, not a representative benchmark.\n",
        "The zero-growth diagnostic removes position allowances, radius padding, and motion-error growth. "
        "Its longer horizon is a sensitivity comparison, not evidence that those allowances should be removed. "
        "Tube miss counts use future labels through each single-agent cap, including times after its computed horizon, "
        "and never feed back into the horizon inputs. A nonzero miss count indicates failure of those proxy assumptions "
        "on at least some checked future observations.\n",
        "Reproduce from the project root: `HF_HUB_OFFLINE=1 uv run --offline --no-sync --extra dataset python scripts/inspect_selected_moments.py`. "
        "See `moments.json` for exact timestamps, input hashes, settings, and retrospective per-focus-track label diagnostics.\n",
    ])
    (OUT / "README.md").write_text("\n".join(report)+"\n")


if __name__ == "__main__":
    main()

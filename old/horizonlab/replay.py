"""Offline replay, history-only estimates, coverage diagnostics, and horizon statistics."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .core import Obstacle, clearance_horizon, q_from_tracking_bounds, nonnegative
from .dataset import json_write


def validate_config(config):
    inventory = config.get("inventory_policy", "all_previously_observed")
    if inventory not in {"all_previously_observed", "active_tracks"}:
        raise ValueError("unknown inventory_policy")
    if inventory == "active_tracks":
        timeout = config.get("track_timeout_s")
        if timeout is None or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("active_tracks requires positive finite track_timeout_s")
    policy = config.get("obstacle_input_policy", "strict_history")
    if policy not in {"strict_history", "bounded_motion"}:
        raise ValueError("unknown obstacle_input_policy")
    if policy == "bounded_motion":
        speed = config.get("fallback_speed_bound_mps")
        if speed is None or not np.isfinite(speed) or speed <= 0:
            raise ValueError("bounded_motion requires an explicit positive fallback_speed_bound_mps")
    for key in ["reference_horizon_s", "validity_s", "anchor_stride_s", "warmup_s",
                "history_window_s", "max_history_gap_s", "time_tolerance_s"]:
        value = float(config[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive")
    if config["min_history_samples"] < 2:
        raise ValueError("need at least two past samples for velocity")
    if not config["experiments"] or len({e["name"] for e in config["experiments"]}) != len(config["experiments"]):
        raise ValueError("experiments need unique names")
    for e in config["experiments"]:
        for key in ["rho0_extra_m", "eps_position_m", "eps_velocity_mps", "acceleration_bound_mps2",
                    "object_radius_padding_m", "injection_position_radius_m", "injection_velocity_radius_mps"]:
            nonnegative(key, e.get(key, 0))
        if "qbar_mps" in e:
            nonnegative("qbar_mps", e["qbar_mps"])
        else:
            q_from_tracking_bounds(e["speed_error_bound_mps"], np.deg2rad(e["heading_error_bound_deg"]), 1)
    for key in ["communication_and_actuation_s", "reserve_s"]:
        nonnegative(key, config.get(key, 0))


def load_reference_records(path):
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["coordinate_frame"] != "clip_anchor" or r["reference_point"] != "rear_axle":
            raise ValueError("export references in the clip anchor frame, at the rear axle")
        if not (r["observation_cutoff_s"] <= r["available_time_s"] <= r["t0_s"]):
            raise ValueError("reference uses future inputs or is not yet available")
        key = (str(r["clip_id"]), float(r["t0_s"]))
        if key in records:
            raise ValueError(f"duplicate reference snapshot {key}")
        t, xy = np.asarray(r["tau_s"], float), np.asarray(r["xy_m"], float)
        if len(t) < 2 or xy.shape != (len(t), 2) or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError("reference needs increasing tau_s starting at zero and matching xy_m")
        if not np.isfinite(t).all() or not np.isfinite(xy).all():
            raise ValueError("nonfinite reference")
        records[key] = r
    return records


def interp_xy(df, ts):
    return np.column_stack([np.interp(ts, df.t_s.to_numpy(), df[c].to_numpy()) for c in ["x_m", "y_m"]])


def recorded_reference(ego, t0, duration):
    end = min(t0 + duration, float(ego.t_s.iloc[-1]))
    internal = ego.t_s.to_numpy()
    internal = internal[(internal > t0) & (internal < end)]
    times = np.r_[t0, internal, end]
    return times - t0, interp_xy(ego, times)


def bounded_noise(radius, rng):
    angle = rng.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(rng.uniform())
    return r * np.array([np.cos(angle), np.sin(angle)])


def endpoint_velocity_bound(history, eps_position, acceleration):
    """Bound endpoint velocity error from two position measurements.

    Each position error has norm <= eps_position; acceleration norm <= A over
    the measurement interval. The secant differs from terminal velocity by at
    most 2*eps_position/dt + A*dt/2. Requires the same physical tracked center.
    """
    dt = float(history.t_s.iloc[-1] - history.t_s.iloc[0])
    if dt <= 0:
        raise ValueError("velocity history must span positive time")
    xy = history[["x_m", "y_m"]].to_numpy(float)
    return (xy[-1] - xy[0]) / dt, 2 * eps_position / dt + 0.5 * acceleration * dt


def obstacle_bound_cap(obstacles, validity_s):
    """Finite CV bound validity vs all-elapsed-time absolute-speed assumption."""
    return min((validity_s - o.age_s for o in obstacles
                if o.motion_model == "constant_velocity"), default=float("inf"))


def estimate_obstacles(agents, t0, experiment, config, clip_id, diagnostics=None):
    """Past-only state estimation with an explicit, logged inventory policy.

    active_tracks retires timed-out tracks and resets history after gaps. This
    narrows scope to recently observed agents, not guaranteed physical absence.
    Future arrivals cannot enter an earlier snapshot; they are audited afterward.
    """
    past = agents[(agents.t_s <= t0 + 1e-10) & (agents.available_s <= t0 + 1e-10)]
    if past.empty:
        return [], "no_observed_agents"
    result = []
    first_error = None
    robust = config.get("obstacle_input_policy", "strict_history") == "bounded_motion"
    active_policy = config.get("inventory_policy", "all_previously_observed") == "active_tracks"
    for track_id, track in past.groupby("track_id", sort=True):
        track = track.sort_values("t_s")
        last = track.iloc[-1]
        age = max(0.0, float(t0 - last.t_s))
        episode = track
        reset = False
        if active_policy:
            # Use only the current contiguous track episode for motion and shape.
            gaps = np.flatnonzero(np.diff(track.t_s.to_numpy()) >= config["track_timeout_s"] - 1e-10)
            if len(gaps):
                episode = track.iloc[int(gaps[-1]) + 1:]
                reset = True
        history = episode[episode.t_s >= last.t_s - config["history_window_s"] - 1e-10]
        if active_policy and age >= config["track_timeout_s"] - 1e-10:
            if diagnostics is not None:
                diagnostics.append(dict(clip_id=clip_id, t0_s=float(t0), track_id=str(track_id),
                    class_name=str(last.class_name), past_samples=len(track), history_samples=len(history),
                    first_observation_s=float(track.t_s.iloc[0]), last_observation_s=float(last.t_s),
                    age_s=age, required_history_samples=config["min_history_samples"],
                    history_issues="", status="retired_timeout", estimator="not_in_active_inventory",
                    inventory_state="retired", history_reset=reset,
                    episode_start_s=float(episode.t_s.iloc[0])))
            continue
        failures = []
        if age >= config["validity_s"]:
            failures.append("stale_track_bound_expired")
        if len(history) < config["min_history_samples"]:
            failures.append("insufficient_track_history")
        dt = np.diff(history.t_s.to_numpy())
        if len(dt) and ((dt <= 0).any() or dt.max() > config["max_history_gap_s"]):
            failures.append("track_history_gap")
        diagnostic = dict(clip_id=clip_id, t0_s=float(t0), track_id=str(track_id),
                                    class_name=str(last.class_name), past_samples=len(track),
                                    history_samples=len(history), first_observation_s=float(track.t_s.iloc[0]),
                                    last_observation_s=float(last.t_s), age_s=age,
                                    required_history_samples=config["min_history_samples"],
                                    history_issues=";".join(failures),
                                    inventory_state="active", history_reset=reset,
                                    episode_start_s=float(episode.t_s.iloc[0]),
                                    status=";".join(failures) if failures else "usable",
                                    estimator="two_point" if len(history) == 2 else "least_squares")
        if failures and not robust:
            if diagnostics is not None:
                diagnostics.append(diagnostic)
            first_error = first_error or failures[0]
            continue
        params = dict(experiment)
        params.update(experiment.get("class_overrides", {}).get(str(last.class_name), {}))
        ep = params["eps_position_m"]
        ev = params["eps_velocity_mps"]
        acc = params["acceleration_bound_mps2"]
        motion_model = "constant_velocity"
        if robust and failures:
            # Zero nominal velocity is a center for a reachable set, NOT a
            # stationary-object assumption. Speed is bounded in every direction.
            velocity = np.zeros(2)
            ev = config["fallback_speed_bound_mps"]
            acc = 0.0
            motion_model = "bounded_speed"
            diagnostic.update(status="usable_fallback", estimator="absolute_speed_envelope")
        elif robust:
            velocity, derived_ev = endpoint_velocity_bound(history, ep, acc)
            ev = max(ev, derived_ev)
            diagnostic.update(estimator="bounded_endpoint_difference", derived_eps_velocity_mps=derived_ev)
        else:
            times = history.t_s.to_numpy(float)
            centered = times - times.mean()
            velocity = (centered[:, None] * history[["x_m", "y_m"]].to_numpy(float)).sum(axis=0) / (centered @ centered)
        entropy = f"{config['seed']}|{clip_id}|{t0:.9f}|{track_id}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(entropy).digest()[:8], "little"))
        position = last[["x_m", "y_m"]].to_numpy(dtype=float) + bounded_noise(params.get("injection_position_radius_m", 0), rng)
        velocity_noise = params.get("injection_velocity_radius_mps", 0)
        if motion_model != "bounded_speed":
            velocity += bounded_noise(velocity_noise, rng)
        if robust:
            ep += params.get("injection_position_radius_m", 0)
            if motion_model != "bounded_speed":
                ev += velocity_noise
        diagnostic.update(motion_model=motion_model, eps_position_m=ep,
                          eps_velocity_mps=ev, acceleration_bound_mps2=acc,
                          vx_mps=float(velocity[0]), vy_mps=float(velocity[1]))
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        result.append(Obstacle(str(track_id), position, velocity,
                               float(episode.radius_m.max()) + params.get("object_radius_padding_m", 0),
                               ep, ev, acc, age, motion_model))
    if not result and not first_error:
        return [], "no_active_agents"
    return ([], first_error) if first_error else (result, None)


def retrospective_diagnostics(agents, t0, cap, H, obstacles):
    """Held-out future LABEL comparison, never input to the horizon calculation.

    This measures proxy tube coverage, not real-world perception accuracy or safety.
    """
    future = agents[(agents.t_s > t0) & (agents.t_s <= t0 + cap)]
    known = {o.track_id: o for o in obstacles}
    n, violations, shape_violations = 0, 0, 0
    before_n, before_violations, before_shapes = 0, 0, 0
    for track_id, f in future.groupby("track_id"):
        if str(track_id) not in known:
            continue
        o = known[str(track_id)]
        tau = f.t_s.to_numpy() - t0
        elapsed = tau + o.age_s
        predicted = o.position_m + elapsed[:, None] * o.velocity_mps
        residual = np.linalg.norm(f[["x_m", "y_m"]].to_numpy() - predicted, axis=1)
        beta = o.eps_position_m + o.eps_velocity_mps * elapsed + 0.5 * o.acceleration_bound_mps2 * elapsed ** 2
        violations += int((residual > beta + 1e-8).sum())
        shape_violations += int((f.radius_m.to_numpy() > o.radius_m + 1e-8).sum())
        before = tau <= H + 1e-10
        before_n += int(before.sum())
        before_violations += int(((residual > beta + 1e-8) & before).sum())
        before_shapes += int(((f.radius_m.to_numpy() > o.radius_m + 1e-8) & before).sum())
        n += len(f)
    # Arrival is based on when an observation becomes available, not hindsight
    # about physical presence. Reappearing retired tracks are also unmodeled.
    arrival_time = np.maximum(agents.t_s, agents.available_s) if "available_s" in agents else agents.t_s
    arriving = agents[(arrival_time > t0) & (arrival_time <= t0 + cap)
                      & ~agents.track_id.astype(str).isin(known)]
    arriving_times = arrival_time.loc[arriving.index]
    new = arriving[arriving_times <= t0 + H]
    past_ids = set(agents[(agents.t_s <= t0) & (arrival_time <= t0)].track_id.astype(str))
    first_arrival = float(arriving_times.min() - t0) if len(arriving) else None
    return {"tube_checked_observations": n, "tube_violations": violations,
            "shape_bound_violations": shape_violations,
            "tube_checked_observations_before_H": before_n,
            "tube_violations_before_H": before_violations,
            "shape_bound_violations_before_H": before_shapes,
            "future_unmodeled_track_count_before_H": int(new.track_id.nunique()),
            "future_first_seen_track_count_before_H": int(new[~new.track_id.astype(str).isin(past_ids)].track_id.nunique()),
            "future_reappearing_track_count_before_H": int(new[new.track_id.astype(str).isin(past_ids)].track_id.nunique()),
            "first_unmodeled_arrival_after_t0_s": first_arrival,
            "retrospective_inventory_limited_H_s": min(H, first_arrival) if first_arrival is not None else H}


def check_canonical(ego, agents):
    for df, cols in [(ego, ["t_s", "x_m", "y_m"]),
                     (agents, ["t_s", "available_s", "x_m", "y_m", "radius_m"])]:
        if not np.isfinite(df[cols].to_numpy(float)).all():
            raise ValueError("nonfinite canonical data")
    if len(ego) < 2 or np.any(np.diff(ego.t_s) <= 0):
        raise ValueError("ego timestamps must be increasing")
    if agents.duplicated(["track_id", "t_s"]).any() or (agents.radius_m < 0).any():
        raise ValueError("duplicate track/time or negative radius")


def run_replay(data_root, config, out_dir, reference_path=None):
    validate_config(config)
    refs = load_reference_records(reference_path) if reference_path else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_write(out_dir / "config_used.json", config)
    rows = []
    history_diagnostics = []
    clip_dirs = sorted(p.parent for p in Path(data_root).glob("*/metadata.json"))
    if not clip_dirs:
        raise ValueError("no canonical clip folders found")
    for directory in clip_dirs:
        meta = json.loads((directory / "metadata.json").read_text())
        clip_id = str(meta["clip_id"])
        ego = pd.read_csv(directory / "ego.csv").sort_values("t_s")
        agents = pd.read_csv(directory / "agents.csv", dtype={"track_id": str})
        check_canonical(ego, agents)
        if refs is None:
            anchors = np.arange(float(ego.t_s.iloc[0]) + config["warmup_s"],
                                float(ego.t_s.iloc[-1]) - 1e-6, config["anchor_stride_s"])
        else:
            anchors = sorted(t0 for cid, t0 in refs if cid == clip_id)
        for t0 in anchors:
            if not ego.t_s.iloc[0] <= t0 < ego.t_s.iloc[-1]:
                raise ValueError(f"anchor {t0} outside ego support for {clip_id}")
            if refs is None:
                tau, xy = recorded_reference(ego, t0, config["reference_horizon_s"])
                reference_source = "recorded_future_oracle"
            else:
                ref = refs[(clip_id, float(t0))]
                tau, xy = np.asarray(ref["tau_s"], float), np.asarray(ref["xy_m"], float)
                reference_source = str(ref["source"])
            for experiment in config["experiments"]:
                row = {"clip_id": clip_id, "scenario": meta.get("scenario", "unclassified"),
                       "kind": meta["kind"], "reference_source": reference_source,
                       "experiment": experiment["name"], "t0_s": float(t0)}
                track_diagnostics = []
                obs, error = estimate_obstacles(agents, t0, experiment, config, clip_id, track_diagnostics)
                history_diagnostics.extend(dict(d, experiment=experiment["name"]) for d in track_diagnostics)
                row.update(n_observed_tracks=len(track_diagnostics),
                           n_active_tracks=sum(d["inventory_state"] == "active" for d in track_diagnostics),
                           n_retired_tracks=sum(d["inventory_state"] == "retired" for d in track_diagnostics),
                           n_unusable_tracks=sum(d["inventory_state"] == "active" and not d["status"].startswith("usable") for d in track_diagnostics),
                           n_fallback_tracks=sum(d["status"] == "usable_fallback" for d in track_diagnostics),
                           n_short_history_tracks=sum("insufficient_track_history" in d["history_issues"] for d in track_diagnostics),
                           n_stale_tracks=sum("stale_track_bound_expired" in d["history_issues"] for d in track_diagnostics),
                           obstacle_input_policy=config.get("obstacle_input_policy", "strict_history"),
                           inventory_policy=config.get("inventory_policy", "all_previously_observed"))
                if error:
                    rows.append(row | {"status": error, "H_s": np.nan})
                    continue
                caps = {"reference_duration": float(tau[-1]),
                        "clip_end": float(ego.t_s.iloc[-1] - t0),
                        "chosen_reference_horizon": config["reference_horizon_s"],
                        "bound_validity": obstacle_bound_cap(obs, config["validity_s"])}
                cap = min(caps.values())
                cap_causes = ";".join(k for k, v in caps.items() if abs(v - cap) <= 1e-7)
                # Bounds must cover the full supplied cap. Only segments up to cap matter.
                relevant = tau[:-1] < cap
                speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(tau)
                ref_speed_max = float(speed[relevant].max())
                qbar = (experiment["qbar_mps"] if "qbar_mps" in experiment else
                        q_from_tracking_bounds(experiment["speed_error_bound_mps"],
                                               np.deg2rad(experiment["heading_error_bound_deg"]), ref_speed_max))
                ego_now = interp_xy(ego, [t0])[0]
                rho0 = float(np.linalg.norm(ego_now - xy[0])) + experiment["rho0_extra_m"]
                started = time.perf_counter()
                result = clearance_horizon(tau, xy, obs, meta["ego_radius_m"], rho0,
                                            qbar, cap, config["time_tolerance_s"])
                calc_ms = (time.perf_counter() - started) * 1000
                diag = retrospective_diagnostics(agents, t0, cap, result.H_s, obs)
                reserve = config.get("communication_and_actuation_s", 0) + config.get("reserve_s", 0)
                limiting = next((o for o in obs if o.track_id == result.limiting_track_id), None)
                rows.append(row | asdict(result) | diag | {
                    "limiting_motion_model": limiting.motion_model if limiting else None,
                    "limiting_observation_age_s": limiting.age_s if limiting else None,
                    "limiting_beta_at_start_m": float(limiting.beta(0)) if limiting else None,
                    "limiting_center_distance_at_start_m": float(np.linalg.norm(xy[0] - limiting.center(0))) if limiting else None,
                    "limiting_radius_m": limiting.radius_m if limiting else None,
                    "T_cap_s": cap, "cap_causes": cap_causes, "n_obstacles": len(obs),
                    "max_observation_age_s": max(o.age_s for o in obs),
                    "rho0_m": rho0, "qbar_mps": qbar, "ref_speed_max_mps": ref_speed_max,
                    "ego_radius_m": meta["ego_radius_m"], "horizon_compute_ms": calc_ms,
                    "usable_budget_s": max(0, result.H_s - reserve),
                    "scope": ("recently_observed_labeled_agents_only" if config.get("inventory_policy") == "active_tracks"
                              else "previously_observed_labeled_agents_only"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no replay anchors; check clip IDs and reference times")
    df.to_csv(out_dir / "horizons.csv", index=False)
    pd.DataFrame(history_diagnostics).to_csv(out_dir / "track_history_diagnostics.csv", index=False)
    summary = summarize(df, config["seed"], config.get("deadline_thresholds_s", [0.1, 0.25, 0.5, 1, 2]))
    json_write(out_dir / "summary.json", summary)
    return df, summary


def summarize(df, seed=7, deadlines=(0.1, 0.25, 0.5, 1.0, 2.0)):
    summaries = []
    for (exp, source), g in df.groupby(["experiment", "reference_source"]):
        valid = g[g.H_s.notna()]
        summary = {"experiment": exp, "reference_source": source, "windows": len(g),
                   "clips": int(g.clip_id.nunique()), "valid_windows": len(valid),
                   "status_counts": {str(k): int(v) for k, v in g.status.value_counts().items()}}
        if len(valid):
            h = valid.H_s.to_numpy()
            by_clip = valid.groupby("clip_id").H_s.mean().to_numpy()
            rng = np.random.default_rng(seed)
            ci = None
            if len(by_clip) >= 2:
                boot = [rng.choice(by_clip, len(by_clip), replace=True).mean() for _ in range(1000)]
                ci = np.quantile(boot, [0.025, 0.975]).tolist()
            checks = int(valid.tube_checked_observations.sum())
            summary |= {"mean_capped_certified_horizon_s": float(h.mean()),
                        "percentiles_s": {str(q): float(np.quantile(h, q / 100)) for q in [1, 5, 10, 50, 90, 95]},
                        "mean_of_valid_clip_means_s": float(by_clip.mean()),
                        "clip_bootstrap_95pct_interval_s": ci,
                        "cap_reached_fraction": float(valid.status.eq("cap_reached").mean()),
                        "zero_horizon_fraction": float((h == 0).mean()),
                        "proxy_tube_violation_fraction": float(valid.tube_violations.sum() / checks) if checks else None,
                        "shape_bound_violations": int(valid.shape_bound_violations.sum()),
                        "windows_with_future_unmodeled_tracks_before_H": int((valid.future_unmodeled_track_count_before_H > 0).sum()),
                        "solver_ms_p50_p95": np.quantile(valid.horizon_compute_ms, [0.5, 0.95]).tolist(),
                        "deadline_coverage": {str(d): {
                            "fraction_all_windows_with_H_ge_deadline": float((g.H_s >= d).mean()),
                            "fraction_valid_windows_with_H_ge_deadline": float((valid.H_s >= d).mean()),
                            "fraction_all_windows_with_usable_budget_ge_deadline": float((g.usable_budget_s >= d).mean()),
                        } for d in deadlines}}
        summaries.append(summary)
    return {"interpretation": "Capped conditional clearance on labeled agents; not average collision time or a validated operational safety deadline.",
            "sampling_warning": "Overlapping windows are correlated; confidence interval resamples clips, not windows. Same-drive correlation may remain.",
            "results": summaries}

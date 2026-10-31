from copy import deepcopy


def default_config():
    # These are deliberately uncalibrated sensitivity settings, NOT benchmarks.
    base = {"name": "illustrative_base", "qbar_mps": 0.5, "rho0_extra_m": 0.2,
            "eps_position_m": 0.3, "eps_velocity_mps": 0.5,
            "acceleration_bound_mps2": 2.0, "object_radius_padding_m": 0.1,
            "injection_position_radius_m": 0.0, "injection_velocity_radius_mps": 0.0}
    experiments = [base]
    for name, updates in [
        ("zero_growth_diagnostic", {"qbar_mps": 0.0, "rho0_extra_m": 0.0, "eps_position_m": 0.0,
                                    "eps_velocity_mps": 0.0, "acceleration_bound_mps2": 0.0,
                                    "object_radius_padding_m": 0.0}),
        ("no_ego_growth_ablation", {"qbar_mps": 0.0}),
        ("no_acceleration_growth_ablation", {"acceleration_bound_mps2": 0.0}),
        ("larger_bounds_stress", {"qbar_mps": 1.0, "eps_position_m": 0.6,
                                  "eps_velocity_mps": 1.0, "acceleration_bound_mps2": 4.0}),
    ]:
        experiment = deepcopy(base)
        experiment.update(updates)
        experiment["name"] = name
        experiments.append(experiment)
    return {"parameter_status": "illustrative_unvalidated_bounds_and_active_inventory",
            "inventory_policy": "active_tracks", "track_timeout_s": 0.25,
            "obstacle_input_policy": "bounded_motion", "fallback_speed_bound_mps": 40.0,
            "inventory_assumption": "Recently observed labels only; timeout does not prove physical departure. Future arrivals require reevaluation.",
            "fallback_assumption": "Uncalibrated absolute center-speed bound of 40 m/s throughout elapsed time for fallback objects.",
            "seed": 7, "reference_horizon_s": 6.4, "validity_s": 6.4,
            "anchor_stride_s": 1.0, "warmup_s": 2.0, "history_window_s": 0.8,
            "min_history_samples": 2, "max_history_gap_s": 0.25,
            "time_tolerance_s": 0.001, "communication_and_actuation_s": 0.0,
            "reserve_s": 0.0, "deadline_thresholds_s": [0.1, 0.25, 0.5, 1.0, 2.0],
            "experiments": experiments}

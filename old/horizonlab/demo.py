"""Deterministic synthetic engineering fixtures, never dataset measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
from .core import rear_axle_radius
from .dataset import json_write


def make_demo(root):
    root = Path(root)
    t = np.round(np.arange(0, 20.0001, 0.1), 6)
    moving_until_stop = np.minimum(t, 4)
    cases = {
        "lead_braking": (25 + 10 * moving_until_stop - 1.25 * moving_until_stop ** 2, np.zeros_like(t), 2.5),
        "parallel_lane_disk_overlap": (12 * t, np.full_like(t, 3.6), 2.5),
        "crossing_pedestrian": (np.full_like(t, 80), 25 - 4 * t, 0.45),
        "pulling_away": (25 + 18 * t, np.zeros_like(t), 2.5),
        "oncoming": (200 - 12 * t, np.full_like(t, 0.5), 2.5),
        "roadside_stationary": (np.full_like(t, 60), np.full_like(t, 9), 1.0),
    }
    for scenario, (x, y, radius) in cases.items():
        clip_id = "synthetic_" + scenario
        d = root / clip_id
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"t_s": t, "x_m": 12 * t, "y_m": 0.0, "yaw_rad": 0.0}).to_csv(d / "ego.csv", index=False)
        pd.DataFrame({"t_s": t, "available_s": t, "track_id": "agent_1", "x_m": x,
                      "y_m": y, "radius_m": radius, "class_name": "synthetic"}).to_csv(d / "agents.csv", index=False)
        json_write(d / "metadata.json", {
            "clip_id": clip_id, "scenario": scenario, "kind": "synthetic_fixture",
            "coordinate_frame": "clip_anchor", "reference_point": "rear_axle",
            "obstacle_inventory_complete": True, "ego_radius_m": rear_axle_radius(4.8, 1.9, 1.2),
            "note": "Synthetic stress fixture; some paths intentionally intersect. Not human driving or NVIDIA data.",
        })

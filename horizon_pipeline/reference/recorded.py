"""Recorded-path reference (T011): at tick t, the plan is the path the car actually drove afterwards.

Step k of the plan made at t is the scene's ego pose at tick t + k, so the plan runs to the end of
the clip and gets shorter near it. This is hindsight by definition, so it belongs to the
environment: it may use the future (see the E02 principles).

Speed is the scene's derived speed at t + k. The scene has none at tick 0 (the causal difference
needs an earlier tick, T009); that one value is filled with tick 1's speed, the distance from tick
0 to tick 1 over 0.1 s.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from horizon_pipeline.raw.store import sha256
from horizon_pipeline.reference.format import COLUMNS, FRAME
from horizon_pipeline.scene.build import DT_S, RATE_HZ, code_version

SOURCE = "recorded"


def fill_speed(speed: pd.Series) -> tuple[np.ndarray, int]:
    """Fill leading NaN speeds with the first known one. Returns the speeds and how many were filled."""
    known = speed.notna().to_numpy()
    if not known.any():
        raise ValueError("the scene has no ego speed")
    first = int(np.argmax(known))
    if not known[first:].all():
        raise ValueError(f"ego speed is missing after tick {first}; only leading gaps are filled")
    out = speed.to_numpy(float, copy=True)
    out[:first] = out[first]
    return out, first


def recorded_reference(ego: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The recorded-path reference from a scene ego table (ticks 0..n-1, in order). Returns the
    reference and the number of speeds filled."""
    n = len(ego)
    if not np.array_equal(ego["tick"].to_numpy(), np.arange(n)):
        raise ValueError("ego ticks must be 0, 1, ..., n-1")
    speed, filled = fill_speed(ego["speed"])
    plan = np.concatenate([np.full(n - t, t) for t in range(n)])  # tick t has n - t steps
    step = np.concatenate([np.arange(n - t) for t in range(n)])
    row = plan + step  # the ego tick each step lands on
    ref = pd.DataFrame({"tick": plan, "sample": 0, "step": step, "tau_s": step * DT_S})
    for c in ["x", "y", "z", "yaw"]:
        ref[c] = ego[c].to_numpy(float)[row]
    ref["speed"] = speed[row]
    return ref[COLUMNS], filled


def build_recorded(scene_dir: Path) -> tuple[pd.DataFrame, dict]:
    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "meta.json").read_text())
    if scene["rate_hz"] != RATE_HZ:
        raise ValueError(f"scene rate {scene['rate_hz']} Hz, expected {RATE_HZ} Hz")
    ego = pd.read_parquet(scene_dir / "ego.parquet")
    ref, filled = recorded_reference(ego)
    steps = ref.groupby("tick").size()
    meta = {
        "source": SOURCE,
        "clip_id": scene["clip_id"],
        "ego_source": scene["ego_source"],
        "frame": FRAME,
        "step_s": DT_S,
        "n_steps": {"min": int(steps.min()), "max": int(steps.max())},
        "n_ticks": int(ref["tick"].nunique()),
        "n_samples": 1,
        "speed_filled_ticks": list(range(filled)),
        "inputs": {"files": {"scene/ego.parquet": sha256(scene_dir / "ego.parquet")}, "code": code_version()},
    }
    return ref, meta

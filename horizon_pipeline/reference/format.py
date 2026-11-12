"""Reference format (T011): one file layout for every source of reference trajectories.

`reference.parquet` has one row per (tick, sample, step), sorted that way:

| column | unit | meaning |
|---|---|---|
| `tick` | int | scene tick the plan was made at (t) |
| `sample` | int | sample index (0 for single-trajectory sources) |
| `step` | int | 0, 1, 2, ... within each (tick, sample) |
| `tau_s` | s | time after t, `step * step_s` |
| `x, y, z` | m | rear-axle position, world frame (the scene's frame) |
| `yaw` | rad | heading, world frame |
| `speed` | m/s | reference speed at this step |

`meta.json` next to it holds at least the keys in `META_KEYS`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["tick", "sample", "step", "tau_s", "x", "y", "z", "yaw", "speed"]
INT_COLUMNS = ["tick", "sample", "step"]
META_KEYS = ["source", "clip_id", "ego_source", "frame", "step_s", "n_steps", "inputs"]
FRAME = "world: the scene's frame (rig at clip time 0); positions are the rear-axle midpoint"
TAU_TOL_S = 1e-9
DEFAULT_ROOT = Path("data/reference")


class ReferenceError(ValueError):
    pass


def reference_dir(root: Path, clip_id: str, ego_source: str, source: str) -> Path:
    return Path(root) / clip_id / ego_source / source


def validate(df: pd.DataFrame, meta: dict) -> None:
    """Raise `ReferenceError` unless `df` and `meta` follow the format."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ReferenceError(f"meta.json is missing {missing}")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ReferenceError(f"missing columns {missing}")
    if df.empty:
        raise ReferenceError("no rows")
    nan = [c for c in COLUMNS if df[c].isna().any()]
    if nan:
        raise ReferenceError(f"NaN in {nan}")
    not_int = [c for c in INT_COLUMNS if not pd.api.types.is_integer_dtype(df[c])]
    if not_int:
        raise ReferenceError(f"{not_int} must be integers")
    tick, sample, step = (df[c].to_numpy() for c in INT_COLUMNS)
    if (tick < 0).any() or (sample < 0).any():
        raise ReferenceError("negative tick or sample")
    # Each (tick, sample) is one block of steps 0, 1, 2, ...; blocks come in increasing (tick, sample).
    same = np.concatenate([[False], (tick[1:] == tick[:-1]) & (sample[1:] == sample[:-1])])
    prev = np.concatenate([[-1], step[:-1]])
    if (step[~same] != 0).any() or (step[same] != prev[same] + 1).any():
        bad = int(np.flatnonzero(np.where(same, step != prev + 1, step != 0))[0])
        raise ReferenceError(f"steps must run 0, 1, 2, ... within each (tick, sample); row {bad} breaks it")
    starts = np.flatnonzero(~same)
    key = tick[starts] * (sample.max() + 1) + sample[starts]
    if (np.diff(key) <= 0).any():
        raise ReferenceError("rows must be sorted by (tick, sample, step), each (tick, sample) once")
    if not np.allclose(df["tau_s"], step * meta["step_s"], rtol=0, atol=TAU_TOL_S):
        raise ReferenceError(f"tau_s must equal step * {meta['step_s']}")


def write_reference(df: pd.DataFrame, meta: dict, out: Path) -> Path:
    """Validate, then write `reference.parquet` and `meta.json` into `out`."""
    validate(df, meta)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(out / "reference.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def read_reference(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read and validate a reference directory."""
    path = Path(path)
    df = pd.read_parquet(path / "reference.parquet")
    meta = json.loads((path / "meta.json").read_text())
    validate(df, meta)
    return df, meta

"""Horizon runner (T014): one H per plan, computed once, from what is known at the plan's tick.

A plan is one (tick, sample) of a reference source. For the plan received at tick t, the solver sees
only the rows of `objects.parquet` at tick t with `in_scene = true`. Nothing from a later tick is
read for it (T023).

An in-scene object without a velocity yet (a track's first tick, T009) is handled by the config's
`objects.no_velocity`:
- `exclude` (default): left out of H, but counted (`n_no_velocity`) and listed
  (`no_velocity_track_ids`), so the safety check (T015) can label contacts it causes;
- `static`: treated as standing still.

Output, per (clip, ego source, reference source):
`results/<run_name>/<clip_id>/<ego_source>/<reference_source>/` with `horizons.parquet`,
`config_used.yaml` and `meta.json`.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from horizon_pipeline.horizon.config import HorizonConfig
from horizon_pipeline.horizon.solver import Objects, Plan, ego_radius, solve
from horizon_pipeline.raw.store import sha256
from horizon_pipeline.reference.format import read_reference
from horizon_pipeline.scene.build import code_version

DEFAULT_RESULTS = Path("results")
SOLVERS = {"baseline": solve}  # config `solver.method` -> solve(plan, objects, r_veh, cfg) -> Horizon

COLUMNS = [
    "tick", "t_s", "sample", "reference_source",
    "H_s", "capped", "limiting_track_id", "margin_at_0_m", "T_cap_s", "q_bar_mps", "v_ref_bar_mps", "r_veh_m",
    "n_in_scene", "n_no_velocity", "no_velocity_track_ids", "ego_speed_mps",
]  # fmt: skip


def results_dir(root: Path, run_name: str, clip_id: str, ego_source: str, reference_source: str) -> Path:
    return Path(root) / run_name / clip_id / ego_source / reference_source


def known_at(objects: pd.DataFrame, tick: int, no_velocity: str) -> tuple[Objects, int, list[str]]:
    """What the car knows at `tick`: the in-scene objects there. Returns the solver input, how many
    are in the scene, and the track IDs that have no velocity yet."""
    seen = objects[(objects["tick"] == tick) & objects["in_scene"].fillna(False).astype(bool)]
    no_vel = seen[["vx", "vy"]].isna().any(axis=1)
    ids = sorted(seen.loc[no_vel, "track_id"].astype(str))
    if no_velocity == "exclude":
        used = seen[~no_vel]
    elif no_velocity == "static":
        used = seen.assign(vx=seen["vx"].where(~no_vel, 0.0), vy=seen["vy"].where(~no_vel, 0.0))
    else:
        raise ValueError(f"unknown no_velocity setting {no_velocity!r}")
    return Objects.from_rows(used), len(seen), ids


def run_horizons(ego: pd.DataFrame, objects: pd.DataFrame, ref: pd.DataFrame, reference_source: str,
                 r_veh: float, cfg: HorizonConfig) -> pd.DataFrame:  # fmt: skip
    """One row per (tick, sample) of `ref`, each solved once with the objects known at its tick."""
    solver = SOLVERS[cfg.method]
    ego = ego.set_index("tick")
    by_tick = {t: g for t, g in objects.groupby("tick")}
    empty = objects.iloc[:0]
    rows = []
    for tick, plans in ref.groupby("tick", sort=True):
        tick = int(tick)
        if tick not in ego.index:
            raise ValueError(f"the reference has a plan at tick {tick}, which the scene does not have")
        known, n_in_scene, no_vel = known_at(by_tick.get(tick, empty), tick, cfg.no_velocity)
        for sample, plan in plans.groupby("sample", sort=True):
            h = solver(Plan.from_rows(plan), known, r_veh, cfg)
            rows.append({
                "tick": tick, "t_s": float(ego.at[tick, "t_s"]), "sample": int(sample),
                "reference_source": reference_source, **asdict(h),
                "n_in_scene": n_in_scene, "n_no_velocity": len(no_vel), "no_velocity_track_ids": no_vel,
                "ego_speed_mps": float(ego.at[tick, "speed"]),
            })  # fmt: skip
    out = pd.DataFrame(rows)
    # None when capped. pandas turns None into NaN in a mixed column; keep it a real missing value.
    out["limiting_track_id"] = pd.Series([None if pd.isna(v) else str(v) for v in out["limiting_track_id"]], dtype=object)
    return out[COLUMNS]


def run_one(scene_dir: Path, ref_dir: Path, cfg: HorizonConfig, config_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    scene_dir, ref_dir = Path(scene_dir), Path(ref_dir)
    scene = json.loads((scene_dir / "meta.json").read_text())
    ref, ref_meta = read_reference(ref_dir)
    if (ref_meta["clip_id"], ref_meta["ego_source"]) != (scene["clip_id"], scene["ego_source"]):
        raise ValueError(f"reference is for {ref_meta['clip_id']}/{ref_meta['ego_source']}, "
                         f"scene is {scene['clip_id']}/{scene['ego_source']}")  # fmt: skip
    ego = pd.read_parquet(scene_dir / "ego.parquet")
    objects = pd.read_parquet(scene_dir / "objects.parquet")
    r_veh = cfg.r_veh_m if cfg.r_veh_m is not None else ego_radius(scene["ego"]["footprint"])
    out = run_horizons(ego, objects, ref, ref_meta["source"], r_veh, cfg)
    files = {
        "scene/ego.parquet": scene_dir / "ego.parquet",
        "scene/objects.parquet": scene_dir / "objects.parquet",
        "scene/meta.json": scene_dir / "meta.json",
        f"reference/{ref_meta['source']}/reference.parquet": ref_dir / "reference.parquet",
        f"reference/{ref_meta['source']}/meta.json": ref_dir / "meta.json",
    }
    if config_path is not None:
        files["config"] = Path(config_path)
    meta = {
        "clip_id": scene["clip_id"],
        "ego_source": scene["ego_source"],
        "reference_source": ref_meta["source"],
        "method": cfg.method,
        "no_velocity": cfg.no_velocity,
        "r_veh_m": float(r_veh),
        "n_plans": len(out),
        "n_ticks_with_plan": int(out["tick"].nunique()),
        "n_ticks_without_plan": int(len(ego) - out["tick"].nunique()),
        "n_capped": int(out["capped"].sum()),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"files": {k: sha256(p) for k, p in files.items()}, "code": code_version()},
    }
    return out, meta


def write_results(out: pd.DataFrame, meta: dict, cfg: HorizonConfig, path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path / "horizons.parquet", index=False)
    (path / "config_used.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def summary(out: pd.DataFrame) -> str:
    h = out["H_s"].to_numpy()
    return (f"{len(out)} plans; H median {np.median(h):.2f} s, p10 {np.percentile(h, 10):.2f} s, "
            f"H = 0 at {np.mean(h == 0):.0%}, capped {out['capped'].mean():.0%}")  # fmt: skip

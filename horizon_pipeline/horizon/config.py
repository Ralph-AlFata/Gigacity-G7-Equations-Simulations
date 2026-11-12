"""Horizon config (T013): `configs/horizon.yaml` -> a validated `HorizonConfig`."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/horizon.yaml")

SECTIONS = {
    "ego": ["r_veh_m", "rho0_m"],
    "tracking": ["e_v_bar_mps", "e_theta_bar_rad"],
    "objects": ["eps_p_m", "eps_v_mps", "a_bar_mps2"],
    "horizon": ["t_valid_s"],
    "solver": ["grid_s", "time_tolerance_s"],
}
POSITIVE = {"r_veh_m", "t_valid_s", "grid_s", "time_tolerance_s"}  # > 0; every other number is ≥ 0


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class HorizonConfig:
    r_veh_m: float | None  # None = auto, from the ego footprint
    rho0_m: float
    e_v_bar_mps: float
    e_theta_bar_rad: float
    eps_p_m: float
    eps_v_mps: float
    a_bar_mps2: float
    t_valid_s: float
    grid_s: float
    time_tolerance_s: float

    def to_dict(self) -> dict:
        """Back to the YAML layout, e.g. for `config_used.yaml`."""
        flat = asdict(self)
        flat["r_veh_m"] = "auto" if self.r_veh_m is None else self.r_veh_m
        return {section: {k: flat[k] for k in keys} for section, keys in SECTIONS.items()}


def parse_config(raw: dict) -> HorizonConfig:
    """Validate a config as read from YAML. Rejects unknown or missing keys and bad values."""
    if not isinstance(raw, dict):
        raise ConfigError("the config must be a mapping of sections")
    if unknown := sorted(set(raw) - set(SECTIONS)):
        raise ConfigError(f"unknown sections {unknown}")
    values = {}
    for section, keys in SECTIONS.items():
        got = raw.get(section) or {}
        if unknown := sorted(set(got) - set(keys)):
            raise ConfigError(f"[{section}] unknown keys {unknown}")
        if missing := sorted(set(keys) - set(got)):
            raise ConfigError(f"[{section}] missing keys {missing}")
        for key in keys:
            v = got[key]
            if key == "r_veh_m" and v == "auto":
                values[key] = None
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigError(f"{section}.{key} must be a finite number, not {v!r}")
            if v < 0 or (key in POSITIVE and v == 0):
                raise ConfigError(f"{section}.{key} must be {'> 0' if key in POSITIVE else '≥ 0'}, not {v}")
            values[key] = float(v)
    if values["e_theta_bar_rad"] > math.pi:
        raise ConfigError(f"tracking.e_theta_bar_rad must be ≤ π, not {values['e_theta_bar_rad']}")
    if values["time_tolerance_s"] > values["grid_s"]:
        raise ConfigError("solver.time_tolerance_s must not exceed solver.grid_s")
    return HorizonConfig(**values)


def load_config(path: Path = DEFAULT_CONFIG) -> HorizonConfig:
    return parse_config(yaml.safe_load(Path(path).read_text()))

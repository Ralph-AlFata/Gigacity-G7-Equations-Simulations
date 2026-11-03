"""Sanity checks on a built scene, shared by the CLI summary and the real-data tests."""

import numpy as np
import pandas as pd

PARKED_MIN_TICKS = 20
PARKED_MAX_SPAN_M = 3.0  # same span cut as T004
PARKED_MAX_SPEED_MPS = 0.3  # median derived speed; the span cut alone admits slow walkers and creeping cars


def parked_spread(objects: pd.DataFrame) -> pd.Series:
    """RMS horizontal distance from the track's mean position (m), for tracks that look parked:
    at least `PARKED_MIN_TICKS` ticks, a horizontal span under `PARKED_MAX_SPAN_M` and a median
    speed under `PARKED_MAX_SPEED_MPS`."""
    out = {}
    for track, g in objects.groupby("track_id"):
        xy = g[["x", "y"]].to_numpy()
        if (
            len(xy) >= PARKED_MIN_TICKS
            and np.ptp(xy, axis=0).max() < PARKED_MAX_SPAN_M
            and g["speed"].median() < PARKED_MAX_SPEED_MPS
        ):
            out[track] = float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean()))
    return pd.Series(out, name="pos_rms_m", dtype=float)

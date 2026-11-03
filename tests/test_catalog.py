import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import CAMERAS, CLIP_A, make_features_df
from horizon_pipeline.raw.catalog import Catalog, Condition, write_clip_list
from horizon_pipeline.raw.fetch import Fetcher, read_clip_list

N = 500


@pytest.fixture
def index_source():
    """Index files only, for N synthetic clips with random presence and metadata."""
    rng = np.random.default_rng(1)
    features_df = make_features_df()
    ids = pd.Index([f"clip-{i:04d}" for i in range(N)], name="clip_id")
    presence = pd.DataFrame(
        {f: rng.random(N) < (0.25 if f.startswith("radar") else 0.95) for f in features_df.index},
        index=ids,
    )
    return SimpleNamespace(
        revision="0" * 40,
        features_df=features_df,
        clip_index=pd.DataFrame({"clip_is_valid": rng.random(N) < 0.98, "chunk": np.arange(N) // 100}, index=ids),
        feature_presence=presence,
        data_collection=pd.DataFrame(
            {
                "country": pd.Series(rng.choice(["Germany", "France", "United States"], N), dtype="string").to_numpy(),
                "hour_of_day": rng.integers(0, 24, N),
                "radar_config": pd.Series(rng.choice(["low", "high", None], N), dtype="string").to_numpy(),
            },
            index=ids,
        ),
    )


@pytest.fixture
def catalog(index_source):
    return Catalog(index_source)


def test_select_with_a_fixed_seed_is_reproducible_and_has_the_required_features(catalog, index_source):
    require = ["obstacle.offline", "camera"]
    first = catalog.select(require, limit=20, seed=7)
    second = Catalog(index_source).select(require, limit=20, seed=7)
    assert first == second
    assert len(first) == 20
    assert first != catalog.select(require, limit=20, seed=8)
    presence = index_source.feature_presence.loc[first]
    assert presence[["obstacle.offline", *CAMERAS]].all().all()


def test_select_without_limit_returns_every_match_in_index_order(catalog, index_source):
    clips = catalog.select(["lidar"])
    expected = index_source.feature_presence.index[index_source.feature_presence["lidar_top_360fov"]]
    assert clips == list(expected)


def test_limit_larger_than_matches_returns_all(catalog):
    everything = catalog.select(["radar"])
    assert catalog.select(["radar"], limit=10_000) == everything


def test_where_conditions(catalog, index_source):
    meta = index_source.data_collection.join(index_source.clip_index)
    cases = {
        "country=Germany,France": meta.country.isin(["Germany", "France"]),
        "country!=Germany": meta.country != "Germany",
        "hour_of_day>=18": meta.hour_of_day >= 18,
        "hour_of_day<6": meta.hour_of_day < 6,
        "radar_config=null": meta.radar_config.isna(),
        "radar_config!=null": meta.radar_config.notna(),
        "radar_config=low,null": meta.radar_config.isna() | (meta.radar_config == "low"),
        "clip_is_valid=true": meta.clip_is_valid,
        "chunk=3": meta.chunk == 3,
    }
    for text, mask in cases.items():
        assert catalog.select(where=[text]) == list(meta.index[mask.fillna(False).astype(bool)]), text


def test_conditions_combine_with_and(catalog, index_source):
    meta = index_source.data_collection
    clips = catalog.select(["egomotion"], where=["country=France", "hour_of_day>=12"])
    assert clips and all(meta.at[c, "country"] == "France" and meta.at[c, "hour_of_day"] >= 12 for c in clips)
    assert index_source.feature_presence.loc[clips, "egomotion"].all()


@pytest.mark.parametrize(
    "where, message",
    [
        ("planet=Mars", "Unknown metadata column 'planet'. Columns: clip_is_valid, chunk, country"),
        ("country=USA", "Values: France, Germany, United States"),
        ("hour_of_day>=noon", "numeric"),
        ("clip_is_valid=yes", "boolean"),
        ("country", "Cannot parse condition"),
    ],
)
def test_bad_conditions_explain_themselves(catalog, where, message):
    with pytest.raises(ValueError) as error:
        catalog.select(where=[where])
    assert message in str(error.value)


def test_unknown_required_feature_lists_valid_names(catalog):
    with pytest.raises(ValueError, match="presets:"):
        catalog.select(["lidr"])


def test_condition_parse():
    assert Condition.parse("hour_of_day >= 18") == Condition("hour_of_day", ">=", "18")
    assert Condition.parse("country=United States") == Condition("country", "=", "United States")


def test_summary_tables(catalog, index_source):
    coverage = catalog.feature_coverage()
    assert coverage.loc["lidar_top_360fov", "percent"] == pytest.approx(
        index_source.feature_presence["lidar_top_360fov"].mean() * 100
    )
    assert coverage.loc["egomotion", "group"] == "labels"
    groups = catalog.group_coverage()
    radar = index_source.feature_presence[["radar_front_center_mrr_2", "radar_side_left_srr_0"]]
    assert groups.loc["radar", "all_percent"] == pytest.approx(radar.all(axis=1).mean() * 100)
    assert groups.loc["radar", "any_percent"] == pytest.approx(radar.any(axis=1).mean() * 100)
    overview = catalog.metadata_overview()
    assert list(overview.index) == ["clip_is_valid", "chunk", "country", "hour_of_day", "radar_config"]
    assert overview.loc["hour_of_day", "values"] == "0 … 23"
    assert "null" in overview.loc["radar_config", "values"]


def test_show_reports_presence_and_what_is_stored(source, root):
    fetcher = Fetcher(source, root)
    fetcher.fetch([CLIP_A], fetcher.registry.resolve(["egomotion", "camera_front_wide_120fov:video"]))
    info = Catalog(source).show(CLIP_A, json.loads((root / "manifest.json").read_text()))
    assert info["metadata"] == {"clip_is_valid": True, "chunk": 7, "country": "Germany", "radar_config": None}
    assert info["features"]["egomotion"] == {"present": True, "stored": ["egomotion"]}
    assert info["features"]["camera_front_wide_120fov"] == {
        "present": True,
        "stored": ["camera_front_wide_120fov:video"],
    }
    assert info["features"]["lidar_top_360fov"] == {"present": True, "stored": []}
    assert info["features"]["radar_side_left_srr_0"] == {"present": False, "stored": []}
    assert Catalog(source).show(CLIP_A)["features"]["egomotion"]["stored"] == []


def test_show_unknown_clip(catalog):
    with pytest.raises(ValueError, match="Unknown clip"):
        catalog.show("nope")


def test_clip_list_file_is_fetcher_input(tmp_path, catalog):
    clips = catalog.select(["scene"], limit=5, seed=3)
    path = tmp_path / "lists" / "sample.txt"
    write_clip_list(path, clips, ["horizon-catalog select --require scene --limit 5 --seed 3", "revision 000"])
    assert path.read_text().startswith("# horizon-catalog select")
    assert read_clip_list(path) == clips

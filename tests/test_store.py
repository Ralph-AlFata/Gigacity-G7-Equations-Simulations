import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from horizon_pipeline.raw.store import filter_parquet_to_clip, parquet_info


def test_filtering_a_chunk_keeps_exactly_the_clip_rows_unchanged(tmp_path):
    # Interleaved clips, mixed dtypes, a (clip_id, sensor) index like sensor_extrinsics.
    rng = np.random.default_rng(0)
    n = 30
    clips = [f"clip-{i % 3}" for i in range(n)]
    chunk = pd.DataFrame(
        {
            "clip_id": clips,
            "sensor": [f"sensor_{i}" for i in range(n)],
            "x": rng.normal(size=n),
            "count": rng.integers(0, 100, size=n).astype("int32"),
            "flag": rng.random(n) > 0.5,
            "label": pd.Series([f"l{i}" if i % 4 else None for i in range(n)], dtype="string"),
            "t_us": rng.integers(0, 10**12, size=n),
        }
    ).set_index(["clip_id", "sensor"])
    source = tmp_path / "chunk.parquet"
    chunk.to_parquet(source)

    target = tmp_path / "out" / "clip.parquet"
    rows = filter_parquet_to_clip(source, target, "clip-1")

    expected = chunk[chunk.index.get_level_values("clip_id") == "clip-1"]
    stored = pd.read_parquet(target)
    assert rows == len(expected) == 10
    pd.testing.assert_frame_equal(stored, expected, check_exact=True)
    assert pq.read_schema(target).remove_metadata() == pq.read_schema(source).remove_metadata()


def test_filtering_an_unknown_clip_gives_zero_rows(tmp_path):
    source = tmp_path / "chunk.parquet"
    pd.DataFrame({"v": [1, 2]}, index=pd.Index(["a", "b"], name="clip_id")).to_parquet(source)
    assert filter_parquet_to_clip(source, tmp_path / "c.parquet", "zzz") == 0


def test_parquet_info(tmp_path):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": pa.array([1.0, 2.0]), "b": pa.array(["x", "y"], pa.string())}), path)
    assert parquet_info(path) == {"rows": 2, "columns": {"a": "double", "b": "string"}}

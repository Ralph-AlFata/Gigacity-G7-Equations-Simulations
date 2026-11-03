import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import CLIP_A, CLIP_B, CLIP_C, member_tag
from horizon_pipeline.env import load_hf_token, read_dotenv
from horizon_pipeline.raw.fetch import Fetcher, read_clip_list
from horizon_pipeline.raw.store import sha256


def manifest(root):
    return json.loads((root / "manifest.json").read_text())


def test_fetch_stores_files_filtered_and_unmodified(source, root):
    fetcher = Fetcher(source, root)
    fetcher.fetch([CLIP_A], fetcher.registry.resolve(["egomotion.offline", "vehicle_dimensions",
                                                      "camera_front_wide_120fov:video"]))
    clip_dir = root / CLIP_A
    assert member_tag(clip_dir / "egomotion.offline.parquet") == f"{CLIP_A}|egomotion.offline|egomotion.offline"
    assert member_tag(clip_dir / "camera_front_wide_120fov" / "video.mp4") == f"{CLIP_A}|camera_front_wide_120fov|video"

    calibration = pd.read_parquet(clip_dir / "vehicle_dimensions.parquet")
    chunk_path = source.repo / "calibration/vehicle_dimensions/vehicle_dimensions.chunk_0007.parquet"
    chunk = pd.read_parquet(chunk_path)
    pd.testing.assert_frame_equal(calibration, chunk.loc[chunk.index == CLIP_A])

    files = manifest(root)["clips"][CLIP_A]["files"]
    assert set(files) == {"egomotion.offline", "vehicle_dimensions", "camera_front_wide_120fov:video"}
    record = files["vehicle_dimensions"]
    assert record["sha256"] == sha256(clip_dir / "vehicle_dimensions.parquet")
    assert record["rows"] == 3
    assert record["columns"] == {f.name: str(f.type) for f in pq.read_schema(chunk_path)}
    assert list(record["columns"]) == ["value", "name", "clip_id"]
    assert record["source"] == {
        "repo_path": "calibration/vehicle_dimensions/vehicle_dimensions.chunk_0007.parquet",
        "zip_member": None,
    }
    assert files["camera_front_wide_120fov:video"]["source"]["zip_member"] == f"{CLIP_A}.camera_front_wide_120fov.mp4"


def test_manifest_lists_presence_and_what_was_not_fetched(source, root):
    fetcher = Fetcher(source, root)
    fetcher.fetch([CLIP_A], fetcher.registry.resolve(["egomotion"]))
    data = manifest(root)
    assert data["dataset"] == {"repo_id": source.repo_id, "revision": source.revision, "sdk_version": "test"}
    clip = data["clips"][CLIP_A]
    assert clip["chunk"] == 7
    assert clip["feature_presence"]["lidar_top_360fov"] is True
    assert clip["feature_presence"]["radar_side_left_srr_0"] is False
    not_fetched = {link["key"]: link for link in clip["not_fetched"]}
    assert "egomotion" not in not_fetched
    assert not_fetched["lidar_top_360fov:pointclouds"] == {
        "key": "lidar_top_360fov:pointclouds",
        "repo_path": "lidar/lidar_top_360fov/lidar_top_360fov.chunk_0007.zip",
        "zip_member": f"{CLIP_A}.lidar_top_360fov.parquet",
    }
    assert not any(key.startswith("radar") for key in not_fetched)  # absent, so no link

    metadata = json.loads((root / CLIP_A / "clip_metadata.json").read_text())
    assert metadata["clip_index"] == {"clip_is_valid": True, "chunk": 7}
    assert metadata["data_collection"] == {"country": "Germany", "radar_config": None}
    assert metadata["feature_presence"]["radar_side_left_srr_0"] is False


def test_absent_feature_is_skipped_and_recorded(source, root):
    fetcher = Fetcher(source, root)
    entries = fetcher.fetch([CLIP_A, CLIP_B], fetcher.registry.resolve(["radar_side_left_srr_0"]))
    assert [(e.clip_id, e.status) for e in entries] == [(CLIP_A, "absent"), (CLIP_B, "fetch")]
    assert not (root / CLIP_A / "radar_side_left_srr_0").exists()
    assert (root / CLIP_B / "radar_side_left_srr_0" / "scans.parquet").is_file()
    assert source.calls == [("zip", CLIP_B, "radar_side_left_srr_0", "scans")]
    clips = manifest(root)["clips"]
    assert clips[CLIP_A]["absent"] == ["radar_side_left_srr_0:scans"]
    assert clips[CLIP_A]["files"] == {}
    assert clips[CLIP_B]["absent"] == []


def test_member_missing_from_zip_is_recorded_absent(source, root):
    fetcher = Fetcher(source, root)
    source.feature_presence.loc[CLIP_A, "radar_side_left_srr_0"] = True  # presence says yes, ZIP has no member
    fetcher.fetch([CLIP_A], fetcher.registry.resolve(["radar_side_left_srr_0"]), log=lambda _: None)
    assert manifest(root)["clips"][CLIP_A]["absent"] == ["radar_side_left_srr_0:scans"]


def test_rerun_does_not_rewrite_unchanged_files(source, root):
    items = Fetcher(source, root).registry.resolve(["scene"])
    Fetcher(source, root).fetch([CLIP_A, CLIP_C], items)
    stored = {p: p.stat().st_mtime_ns for p in root.rglob("*.parquet")}
    first_calls = len(source.calls)

    entries = Fetcher(source, root).fetch([CLIP_A, CLIP_C], items)
    assert {e.status for e in entries} == {"stored"}
    assert len(source.calls) == first_calls
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*.parquet")} == stored


def test_rerun_refetches_a_changed_file_and_only_adds(source, root):
    Fetcher(source, root).fetch([CLIP_A], Fetcher(source, root).registry.resolve(["egomotion"]))
    (root / CLIP_A / "egomotion.parquet").write_text("corrupted")
    fetcher = Fetcher(source, root)
    entries = fetcher.fetch([CLIP_A], fetcher.registry.resolve(["egomotion", "obstacle.offline"]))
    assert [e.status for e in entries] == ["fetch", "fetch"]
    assert member_tag(root / CLIP_A / "egomotion.parquet") == f"{CLIP_A}|egomotion|egomotion"
    assert set(manifest(root)["clips"][CLIP_A]["files"]) == {"egomotion", "obstacle.offline"}


def test_dry_run_reports_sizes_and_fetches_nothing(source, root):
    fetcher = Fetcher(source, root)
    entries = fetcher.fetch([CLIP_A], fetcher.registry.resolve(["lidar"]), dry_run=True)
    assert [e.status for e in entries] == ["fetch"]
    assert entries[0].size > 0
    assert source.calls == []
    assert not root.exists()


def test_confirmation_above_threshold(source, root):
    fetcher = Fetcher(source, root)
    items = fetcher.registry.resolve(["videos"])
    prompts = []
    with pytest.raises(SystemExit):
        fetcher.fetch([CLIP_A], items, confirm_bytes=10, confirm=lambda p: prompts.append(p) or False)
    assert len(prompts) == 1 and "7 files" in prompts[0]
    assert source.calls == []
    fetcher.fetch([CLIP_A], items, confirm_bytes=10, confirm=lambda p: True)
    assert len(source.calls) == 7


def test_remove_deletes_files_and_updates_manifest(source, root):
    fetcher = Fetcher(source, root)
    fetcher.fetch([CLIP_A], fetcher.registry.resolve(["egomotion", "camera_front_wide_120fov:video"]))
    assert fetcher.remove([CLIP_A], fetcher.registry.resolve(["camera_front_wide_120fov:video"])) == 1
    assert not (root / CLIP_A / "camera_front_wide_120fov").exists()
    clip = manifest(root)["clips"][CLIP_A]
    assert set(clip["files"]) == {"egomotion"}
    assert "camera_front_wide_120fov:video" in {link["key"] for link in clip["not_fetched"]}


def test_unknown_clip_is_an_error(source, root):
    fetcher = Fetcher(source, root)
    with pytest.raises(ValueError, match="Unknown clip IDs"):
        fetcher.fetch(["nope"], fetcher.registry.resolve(["egomotion"]))


def test_manifest_from_another_revision_is_refused(source, root):
    Fetcher(source, root).fetch([CLIP_A], Fetcher(source, root).registry.resolve(["egomotion"]))
    source.revision = "1" * 40
    with pytest.raises(ValueError, match="revision"):
        Fetcher(source, root)


def test_read_clip_list(tmp_path):
    path = tmp_path / "clips.txt"
    path.write_text(f"{CLIP_A}\n\n# a comment\n{CLIP_B}  # trailing\n")
    assert read_clip_list(path) == [CLIP_A, CLIP_B]


def test_hf_token_from_dotenv(tmp_path, monkeypatch):
    for key in ("HF_TOKEN", "HUGGING_FACE_KEY", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# secrets\nOTHER=1\nexport HUGGING_FACE_KEY='hf_abc'\n")
    assert read_dotenv(dotenv)["OTHER"] == "1"
    assert load_hf_token(dotenv) == "hf_abc"
    assert load_hf_token(tmp_path / "missing") is None
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert load_hf_token(dotenv) == "hf_env"

"""A fake dataset with the same packing as PhysicalAI-AV, built from synthetic files."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

CAMERAS = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]
CALIBRATION = [
    "camera_intrinsics",
    "camera_intrinsics.offline",
    "lidar_intrinsics.offline",
    "sensor_extrinsics",
    "sensor_extrinsics.offline",
    "vehicle_dimensions",
]
LABELS = ["egomotion", "egomotion.offline", "obstacle.offline"]
RADARS = ["radar_front_center_mrr_2", "radar_side_left_srr_0"]

CLIP_A = "aaaaaaaa-0000-0000-0000-000000000000"
CLIP_B = "bbbbbbbb-0000-0000-0000-000000000000"  # same chunk as A, has radar
CLIP_C = "cccccccc-0000-0000-0000-000000000000"  # another chunk


def make_features_df() -> pd.DataFrame:
    """Same columns and conventions as `features.csv` after the SDK parses it."""
    rows = []
    for f in LABELS:
        rows.append((f, "labels", f"labels/{f}/{f}.chunk_{{chunk_id:04d}}.zip", {f: f"{{clip_id}}.{f}.parquet"}))
    for f in CALIBRATION:
        rows.append((f, "calibration", f"calibration/{f}/{f}.chunk_{{chunk_id:04d}}.parquet", float("nan")))
    for f in CAMERAS:
        rows.append((
            f,
            "camera",
            f"camera/{f}/{f}.chunk_{{chunk_id:04d}}.zip",
            {
                "video": f"{{clip_id}}.{f}.mp4",
                "frame_timestamps": f"{{clip_id}}.{f}.timestamps.parquet",
                "blurred_boxes": f"{{clip_id}}.{f}.blurred_boxes.parquet",
            },
        ))
    rows.append(("lidar_top_360fov", "lidar", "lidar/lidar_top_360fov/lidar_top_360fov.chunk_{chunk_id:04d}.zip",
                 {"pointclouds": "{clip_id}.lidar_top_360fov.parquet"}))
    for f in RADARS:
        rows.append((f, "radar", f"radar/{f}/{f}.chunk_{{chunk_id:04d}}.zip", {"scans": f"{{clip_id}}.{f}.parquet"}))
    return pd.DataFrame(rows, columns=["feature", "directory", "chunk_path", "clip_files_in_zip"]).set_index("feature")


def member_bytes(name: str, tag: str) -> bytes:
    """A parquet with one `tag` row for `.parquet` members, the tag itself otherwise."""
    if not name.endswith(".parquet"):
        return tag.encode()
    buffer = io.BytesIO()
    pd.DataFrame({"tag": [tag]}).to_parquet(buffer, index=False)
    return buffer.getvalue()


def member_tag(path: Path) -> str:
    """Inverse of `member_bytes`: which clip, feature and member a stored file came from."""
    if path.suffix != ".parquet":
        return path.read_text()
    return pd.read_parquet(path)["tag"].item()


class FakeSource:
    repo_id = "fake/PhysicalAI-AV"
    revision = "0" * 40
    sdk_version = "test"

    def __init__(self, repo: Path):
        self.repo = repo
        self.features_df = make_features_df()
        self.clip_index = pd.DataFrame(
            {"clip_is_valid": True, "chunk": [7, 7, 9]},
            index=pd.Index([CLIP_A, CLIP_B, CLIP_C], name="clip_id"),
        )
        presence = {f: [True, True, True] for f in self.features_df.index}
        for f in RADARS:
            presence[f] = [False, True, False]
        self.feature_presence = pd.DataFrame(presence, index=self.clip_index.index)
        self.data_collection = pd.DataFrame(
            {"country": ["Germany", "United States", "France"], "radar_config": [None, "x", None]},
            index=self.clip_index.index,
        )
        self.calls: list[tuple] = []
        self._build_files()

    def _build_files(self):
        for chunk, clips in self.clip_index.groupby("chunk").groups.items():
            for feature, row in self.features_df.iterrows():
                path = self.repo / row["chunk_path"].format(chunk_id=chunk)
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(row["clip_files_in_zip"], dict):
                    with zipfile.ZipFile(path, "w") as archive:
                        for clip in clips:
                            if not self.feature_presence.at[clip, feature]:
                                continue
                            for alias, template in row["clip_files_in_zip"].items():
                                name = template.format(clip_id=clip)
                                archive.writestr(name, member_bytes(name, f"{clip}|{feature}|{alias}"))
                else:
                    # Interleaved clips, several rows each, like sensor_extrinsics.
                    ids = [c for _ in range(3) for c in clips]
                    pd.DataFrame(
                        {"value": range(len(ids)), "name": [f"s{i}" for i in range(len(ids))]},
                        index=pd.Index(ids, name="clip_id"),
                    ).to_parquet(path)

    def remote_size(self, repo_path, zip_member):
        path = self.repo / repo_path
        if zip_member is None:
            return path.stat().st_size
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return archive.getinfo(zip_member).file_size if zip_member in names else None

    def fetch_zip_member(self, clip_id, feature, member):
        self.calls.append(("zip", clip_id, feature, member))
        row = self.features_df.loc[feature]
        chunk = self.clip_index.at[clip_id, "chunk"]
        name = row["clip_files_in_zip"][member].format(clip_id=clip_id)
        out = self.repo / "extracted" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.repo / row["chunk_path"].format(chunk_id=chunk)) as archive:
            out.write_bytes(archive.read(name))  # KeyError if the member is missing, like the SDK
        return out

    def fetch_file(self, repo_path):
        self.calls.append(("file", repo_path))
        return self.repo / repo_path


@pytest.fixture
def source(tmp_path) -> FakeSource:
    return FakeSource(tmp_path / "repo")


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "raw"

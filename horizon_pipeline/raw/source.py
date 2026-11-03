"""Where raw files come from. `HfSource` wraps the `physical_ai_av` SDK; tests use a fake.

A source exposes the dataset's own metadata tables and three operations: the size of a remote
file or ZIP member, extracting one ZIP member, and downloading one whole file.
"""

import os
import zipfile
from pathlib import Path
from typing import Protocol

import pandas as pd

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATASET_REVISION = "39414de6f0ab867238ad6a0299fcd18b26081b42"


class Source(Protocol):
    repo_id: str
    revision: str
    sdk_version: str
    features_df: pd.DataFrame  # features.csv, indexed by feature
    clip_index: pd.DataFrame  # indexed by clip_id, has a `chunk` column
    feature_presence: pd.DataFrame  # indexed by clip_id, one bool column per feature
    data_collection: pd.DataFrame  # indexed by clip_id

    def remote_size(self, repo_path: str, zip_member: str | None) -> int | None:
        """Bytes of the file (or of the ZIP member, uncompressed); None if it cannot be read."""

    def fetch_zip_member(self, clip_id: str, feature: str, member: str) -> Path:
        """Extracts one member of the clip's chunk ZIP (streamed) and returns its local path."""

    def fetch_file(self, repo_path: str) -> Path:
        """Downloads a whole repo file (through the HF cache) and returns its local path."""


class HfSource:
    """The PhysicalAI-AV dataset on Hugging Face, pinned to one commit."""

    def __init__(self, revision: str = DATASET_REVISION, token: str | None = None):
        import physical_ai_av

        self.sdk = physical_ai_av.PhysicalAIAVDatasetInterface(
            revision=revision,
            repo_id=REPO_ID,
            token=token,
            # The fetcher asks for confirmation itself, on the whole plan.
            confirm_download_threshold_gb=float("inf"),
        )
        self.repo_id = REPO_ID
        self.revision = self.sdk.revision
        self.sdk_version = physical_ai_av.__version__
        self.features_df = self.sdk.features.features_df
        self.clip_index = self.sdk.clip_index
        self.feature_presence = self.sdk.feature_presence
        self.data_collection = self.sdk.data_collection

    def remote_size(self, repo_path: str, zip_member: str | None) -> int | None:
        try:
            if zip_member is None:
                cached = self.sdk._get_cached_path(repo_path)
                if cached is not None:
                    return os.path.getsize(cached)
                info = self.sdk.api.get_paths_info(paths=[repo_path], **self.sdk.repo_snapshot_info)
                return info[0].size if info else None
            # Reads only the ZIP's central directory (range requests), not the members.
            with self.sdk.open_file(repo_path, maybe_stream=True) as f:
                with zipfile.ZipFile(f) as archive:
                    return archive.getinfo(zip_member).file_size
        except Exception:
            return None

    def fetch_zip_member(self, clip_id: str, feature: str, member: str) -> Path:
        paths = self.sdk.get_clip_feature_paths(clip_id, feature, members=member, maybe_stream=True)
        return Path(paths[member])

    def fetch_file(self, repo_path: str) -> Path:
        return Path(self.sdk.download_file(repo_path))

"""Writing raw files to disk unchanged, and the manifest that records them."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

CLIP_ID_COLUMN = "clip_id"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_target(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    return Path(tmp)


def copy_file(source: Path, target: Path) -> None:
    """Byte-for-byte copy, replacing `target` atomically."""
    tmp = _atomic_target(target)
    try:
        shutil.copyfile(source, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def filter_parquet_to_clip(source: Path, target: Path, clip_id: str) -> int:
    """Writes the rows of `clip_id` from a chunk parquet; returns the row count.

    Done in Arrow so columns, types, row order and the pandas index metadata are kept as is.
    """
    table = pq.read_table(source)
    if CLIP_ID_COLUMN not in table.column_names:
        raise ValueError(f"{source} has no {CLIP_ID_COLUMN!r} column to filter on.")
    clip_rows = table.filter(pc.equal(table[CLIP_ID_COLUMN], clip_id))
    tmp = _atomic_target(target)
    try:
        pq.write_table(clip_rows, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return clip_rows.num_rows


def parquet_info(path: Path) -> dict:
    """Row count and column types, from the file footer only."""
    metadata = pq.read_metadata(path)
    schema = metadata.schema.to_arrow_schema()
    return {
        "rows": metadata.num_rows,
        "columns": {field.name: str(field.type) for field in schema},
    }


class Manifest:
    """`data/raw/manifest.json`: everything stored, where it came from, and what was not fetched.

    Layout:
        dataset:  repo_id, revision, sdk_version
        updated_at
        clips:    <clip_id> -> chunk, feature_presence, files {key -> record}, absent [keys],
                  not_fetched [{key, repo_path, zip_member}]
    """

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    @classmethod
    def load(cls, path: Path, dataset: dict) -> "Manifest":
        if path.is_file():
            data = json.loads(path.read_text())
            stored = data.get("dataset", {})
            for key in ("repo_id", "revision"):
                if stored.get(key) != dataset[key]:
                    raise ValueError(
                        f"{path} was written for {key}={stored.get(key)!r}, not {dataset[key]!r}. "
                        "Use another --root to keep dataset revisions apart."
                    )
            data["dataset"] = dataset
        else:
            data = {"dataset": dataset, "updated_at": None, "clips": {}}
        return cls(path, data)

    def clip(self, clip_id: str) -> dict:
        return self.data["clips"].setdefault(
            clip_id, {"chunk": None, "feature_presence": {}, "files": {}, "absent": [], "not_fetched": []}
        )

    def save(self, now: str) -> None:
        self.data["updated_at"] = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _atomic_target(self.path)
        try:
            tmp.write_text(json.dumps(self.data, indent=2) + "\n")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)

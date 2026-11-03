"""On-request fetching of raw features into `data/raw/<clip_id>/`, recorded in a manifest."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from horizon_pipeline.raw.registry import Item, Registry
from horizon_pipeline.raw.source import Source
from horizon_pipeline.raw.store import Manifest, copy_file, filter_parquet_to_clip, parquet_info, sha256

DEFAULT_ROOT = Path("data/raw")
DEFAULT_CONFIRM_GB = 2.0


@dataclass
class PlanEntry:
    clip_id: str
    item: Item
    status: str  # "fetch", "stored" (already there, unchanged) or "absent"
    repo_path: str
    zip_member: str | None
    size: int | None = None  # bytes to fetch; for parquet features, the whole chunk file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_clip_list(path: Path) -> list[str]:
    """One clip ID per line; blank lines and `#` comments ignored."""
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


class Fetcher:
    def __init__(self, source: Source, root: Path = DEFAULT_ROOT, registry: Registry | None = None):
        self.source = source
        self.root = Path(root)
        self.registry = registry or Registry(source.features_df)
        self.manifest = Manifest.load(
            self.root / "manifest.json",
            {"repo_id": source.repo_id, "revision": source.revision, "sdk_version": source.sdk_version},
        )

    # --- planning -------------------------------------------------------------------------

    def _check_clips(self, clip_ids: list[str]) -> None:
        unknown = [c for c in clip_ids if c not in self.source.clip_index.index]
        if unknown:
            raise ValueError(f"Unknown clip IDs (not in clip_index): {', '.join(unknown)}.")

    def _chunk(self, clip_id: str) -> int:
        return int(self.source.clip_index.at[clip_id, "chunk"])

    def _is_present(self, clip_id: str, feature: str) -> bool:
        presence = self.source.feature_presence
        if feature not in presence.columns:
            return True  # not tracked by the presence table; let the fetch decide
        return bool(presence.at[clip_id, feature])

    def _is_stored(self, clip_id: str, item: Item) -> bool:
        record = self.manifest.clip(clip_id)["files"].get(item.key)
        path = self.root / clip_id / self.registry.local_path(item)
        return record is not None and path.is_file() and sha256(path) == record["sha256"]

    def plan(self, clip_ids: list[str], items: list[Item], sizes: bool = True) -> list[PlanEntry]:
        self._check_clips(clip_ids)
        entries = []
        for clip_id in clip_ids:
            chunk = self._chunk(clip_id)
            for item in items:
                if not self._is_present(clip_id, item.feature):
                    status = "absent"
                elif self._is_stored(clip_id, item):
                    status = "stored"
                else:
                    status = "fetch"
                entry = PlanEntry(
                    clip_id,
                    item,
                    status,
                    self.registry.repo_path(item, chunk),
                    self.registry.zip_member(item, clip_id),
                )
                if sizes and status == "fetch":
                    entry.size = self.source.remote_size(entry.repo_path, entry.zip_member)
                entries.append(entry)
        return entries

    # --- fetching -------------------------------------------------------------------------

    def fetch(
        self,
        clip_ids: list[str],
        items: list[Item],
        *,
        dry_run: bool = False,
        confirm_bytes: float = DEFAULT_CONFIRM_GB * 1024**3,
        confirm: Callable[[str], bool] = lambda prompt: False,
        log: Callable[[str], None] = print,
    ) -> list[PlanEntry]:
        """Fetches what is not stored yet. Returns the plan; nothing is written on a dry run.

        Above `confirm_bytes` of known size, `confirm(prompt)` must return True to proceed.
        """
        entries = self.plan(clip_ids, items)
        if dry_run:
            return entries
        to_fetch = [e for e in entries if e.status == "fetch"]
        total = sum(e.size or 0 for e in to_fetch)
        if total > confirm_bytes and not confirm(
            f"Fetch {len(to_fetch)} files, {format_size(total)}? [y/N] "
        ):
            raise SystemExit("Aborted, nothing fetched.")

        for clip_id in clip_ids:
            clip = self.manifest.clip(clip_id)
            for entry in (e for e in entries if e.clip_id == clip_id):
                if entry.status == "absent":
                    self._mark_absent(clip, entry.item)
                elif entry.status == "fetch":
                    log(f"fetch  {clip_id}  {entry.item.key}")
                    if self._fetch_one(clip, entry):
                        clip["absent"] = [k for k in clip["absent"] if k != entry.item.key]
                    else:
                        log(f"absent {clip_id}  {entry.item.key}  (member not in the chunk ZIP)")
                        self._mark_absent(clip, entry.item)
                    self._finish_clip(clip_id)
            self._finish_clip(clip_id)
        return entries

    def _fetch_one(self, clip: dict, entry: PlanEntry) -> bool:
        item = entry.item
        target = self.root / entry.clip_id / self.registry.local_path(item)
        if item.member is None:
            chunk_file = self.source.fetch_file(entry.repo_path)
            filter_parquet_to_clip(chunk_file, target, entry.clip_id)
        else:
            try:
                member_file = self.source.fetch_zip_member(entry.clip_id, item.feature, item.member)
            except KeyError:
                return False
            copy_file(member_file, target)

        record = {
            "feature": item.feature,
            "member": item.member,
            "path": target.relative_to(self.root).as_posix(),
            "source": {"repo_path": entry.repo_path, "zip_member": entry.zip_member},
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "fetched_at": _now(),
        }
        if target.suffix == ".parquet":
            record.update(parquet_info(target))
        clip["files"][item.key] = record
        return True

    @staticmethod
    def _mark_absent(clip: dict, item: Item) -> None:
        if item.key not in clip["absent"]:
            clip["absent"] = sorted(clip["absent"] + [item.key])

    def _finish_clip(self, clip_id: str) -> None:
        """Refreshes the clip's presence, not-fetched links and metadata file, then saves."""
        clip = self.manifest.clip(clip_id)
        clip["chunk"] = self._chunk(clip_id)
        clip["feature_presence"] = {
            f: bool(v) for f, v in self.source.feature_presence.loc[clip_id].items()
        }
        chunk = clip["chunk"]
        clip["not_fetched"] = [
            {
                "key": item.key,
                "repo_path": self.registry.repo_path(item, chunk),
                "zip_member": self.registry.zip_member(item, clip_id),
            }
            for feature in self.registry.features
            if self._is_present(clip_id, feature)
            for item in self.registry.items(feature)
            if item.key not in clip["files"] and item.key not in clip["absent"]
        ]
        self._write_clip_metadata(clip_id)
        self.manifest.save(_now())

    def _write_clip_metadata(self, clip_id: str) -> None:
        def row(table):
            return json.loads(table.loc[[clip_id]].iloc[0].to_json())

        metadata = {
            "clip_id": clip_id,
            "clip_index": row(self.source.clip_index),
            "feature_presence": row(self.source.feature_presence),
            "data_collection": row(self.source.data_collection),
        }
        path = self.root / clip_id / "clip_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n")

    # --- removal --------------------------------------------------------------------------

    def remove(self, clip_ids: list[str], items: list[Item], log: Callable[[str], None] = print) -> int:
        """Deletes stored files and their manifest records. Returns how many were removed."""
        self._check_clips(clip_ids)
        removed = 0
        for clip_id in clip_ids:
            clip = self.manifest.clip(clip_id)
            for item in items:
                path = self.root / clip_id / self.registry.local_path(item)
                if clip["files"].pop(item.key, None) is not None or path.exists():
                    path.unlink(missing_ok=True)
                    parent = path.parent
                    if parent != self.root / clip_id and parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                    log(f"remove {clip_id}  {item.key}")
                    removed += 1
            self._finish_clip(clip_id)
        return removed


def format_size(size: float | None) -> str:
    if size is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024

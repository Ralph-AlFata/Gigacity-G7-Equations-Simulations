"""Feature registry built from the dataset's `features.csv`, and selection parsing.

Nothing here names a specific feature: feature names, groups, chunk paths and ZIP members all
come from the table, and presets come from `presets.toml`.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

PRESETS_FILE = Path(__file__).with_name("presets.toml")


@dataclass(frozen=True)
class Item:
    """One file to store for a clip: a whole parquet feature (member None) or one ZIP member."""

    feature: str
    member: str | None = None

    @property
    def key(self) -> str:
        """Manifest key: the feature alone when the member adds nothing (labels, calibration)."""
        if self.member is None or self.member == self.feature:
            return self.feature
        return f"{self.feature}:{self.member}"


def load_presets(path: Path = PRESETS_FILE) -> dict[str, list[str]]:
    with open(path, "rb") as f:
        return tomllib.load(f)


class Registry:
    """What the dataset offers, from a `features.csv` table as parsed by `physical_ai_av`.

    `features_df` is indexed by feature, with columns `directory`, `chunk_path` and
    `clip_files_in_zip` (a dict alias -> member template for ZIP features, NaN otherwise).
    """

    def __init__(self, features_df: pd.DataFrame, presets: dict[str, list[str]] | None = None):
        self.features_df = features_df
        self.presets = load_presets() if presets is None else presets

    @property
    def features(self) -> list[str]:
        return list(self.features_df.index)

    @property
    def groups(self) -> dict[str, list[str]]:
        return {
            group: list(rows.index)
            for group, rows in self.features_df.groupby("directory", sort=False)
        }

    def members(self, feature: str) -> list[str] | None:
        """ZIP member aliases of a feature, or None for a direct (parquet) feature."""
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        return list(templates) if isinstance(templates, dict) else None

    def items(self, feature: str) -> list[Item]:
        """Every storable item of a feature."""
        members = self.members(feature)
        return [Item(feature)] if members is None else [Item(feature, m) for m in members]

    def repo_path(self, item: Item, chunk: int) -> str:
        return self.features_df.at[item.feature, "chunk_path"].format(chunk_id=chunk)

    def zip_member(self, item: Item, clip_id: str) -> str | None:
        if item.member is None:
            return None
        return self.features_df.at[item.feature, "clip_files_in_zip"][item.member].format(
            clip_id=clip_id
        )

    def local_path(self, item: Item) -> str:
        """Path under `data/raw/<clip_id>/`: `<feature>.<ext>` or `<feature>/<member>.<ext>`."""
        if item.member is None:
            return item.feature + PurePosixPath(self.features_df.at[item.feature, "chunk_path"]).suffix
        ext = PurePosixPath(self.features_df.at[item.feature, "clip_files_in_zip"][item.member]).suffix
        if self.members(item.feature) == [item.feature]:
            return item.feature + ext
        return f"{item.feature}/{item.member}{ext}"

    def resolve(self, selection: list[str]) -> list[Item]:
        """Expands feature names, `feature:member`, groups, `group:member` and presets.

        Entries may also be comma-separated. Order is kept, duplicates dropped.
        Raises ValueError naming the valid choices for anything unknown.
        """
        items: list[Item] = []
        for entry in selection:
            for token in filter(None, (t.strip() for t in entry.split(","))):
                for item in self._resolve_token(token, seen_presets=()):
                    if item not in items:
                        items.append(item)
        return items

    def _resolve_token(self, token: str, seen_presets: tuple[str, ...]) -> list[Item]:
        if token in self.presets:
            if token in seen_presets:
                raise ValueError(f"Preset cycle: {' -> '.join(seen_presets + (token,))}.")
            items = []
            for sub in self.presets[token]:
                items += self._resolve_token(sub, seen_presets + (token,))
            return items

        name, _, member = token.partition(":")
        if name in self.features_df.index:
            if not member:
                return self.items(name)
            valid = self.members(name)
            if valid is None:
                raise ValueError(f"Feature {name!r} is a single parquet file and has no members.")
            if member not in valid:
                raise ValueError(
                    f"Unknown member {member!r} of feature {name!r}. Valid members: {', '.join(valid)}."
                )
            return [Item(name, member)]

        groups = self.groups
        if name in groups:
            if not member:
                return [item for feature in groups[name] for item in self.items(feature)]
            items = [
                Item(feature, member)
                for feature in groups[name]
                if member in (self.members(feature) or [])
            ]
            if not items:
                valid = sorted({m for f in groups[name] for m in (self.members(f) or [])})
                raise ValueError(
                    f"No feature in group {name!r} has member {member!r}. "
                    f"Valid members: {', '.join(valid) or 'none'}."
                )
            return items

        raise ValueError(
            f"Unknown feature, group or preset {name!r}.\n"
            f"  presets:  {', '.join(self.presets)}\n"
            f"  groups:   {', '.join(groups)}\n"
            f"  features: {', '.join(self.features)}"
        )

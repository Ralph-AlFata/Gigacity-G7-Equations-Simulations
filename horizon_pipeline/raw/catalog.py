"""Clip catalog: which clips exist and what they have, from the dataset's index files only.

The metadata table is `clip_index` joined with `data_collection`, so every column those files
hold can be filtered on; nothing here names a specific column.
"""

import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from horizon_pipeline.raw.registry import Registry
from horizon_pipeline.raw.source import Source

DEFAULT_LISTS_DIR = Path("data/clip_lists")

_OPERATORS = {
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    "=": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}
_CONDITION = re.compile(r"^\s*([^!<>=\s]+)\s*(!=|>=|<=|=|>|<)\s*(.*?)\s*$")
_NULL = "null"


@dataclass(frozen=True)
class Condition:
    """`column op value`; with `=` / `!=`, value may be a comma-separated list; `null` matches missing."""

    column: str
    op: str
    value: str

    @classmethod
    def parse(cls, text: str) -> "Condition":
        match = _CONDITION.match(text)
        if not match:
            raise ValueError(
                f"Cannot parse condition {text!r}. Use column<op>value with op one of "
                f"{', '.join(_OPERATORS)}, e.g. country=Germany or hour_of_day>=18."
            )
        return cls(*match.groups())


class Catalog:
    def __init__(self, source: Source, registry: Registry | None = None):
        self.registry = registry or Registry(source.features_df)
        self.presence = source.feature_presence.astype(bool)
        self.metadata = source.clip_index.join(source.data_collection, how="left")
        self.revision = source.revision

    # --- summary --------------------------------------------------------------------------

    def feature_coverage(self) -> pd.DataFrame:
        """Per feature: its group and the share of clips that have it."""
        directory = self.registry.features_df["directory"]
        return pd.DataFrame(
            {
                "group": [directory.get(f, "?") for f in self.presence.columns],
                "clips": self.presence.sum().to_numpy(),
                "percent": (self.presence.mean() * 100).to_numpy(),
            },
            index=self.presence.columns,
        )

    def group_coverage(self) -> pd.DataFrame:
        """Per group: share of clips with all of its features, and with at least one."""
        rows = {}
        for group, features in self.registry.groups.items():
            present = self.presence[[f for f in features if f in self.presence.columns]]
            rows[group] = {
                "features": present.shape[1],
                "all_percent": present.all(axis=1).mean() * 100,
                "any_percent": present.any(axis=1).mean() * 100,
            }
        return pd.DataFrame(rows).T

    def metadata_overview(self, top: int = 6) -> pd.DataFrame:
        """Per metadata column: type, distinct values, missing share, and a short value summary."""
        rows = {}
        for column in self.metadata.columns:
            values = self.metadata[column]
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and values.nunique() > top:
                summary = f"{values.min()} … {values.max()}"
            else:
                counts = values.value_counts(dropna=False).head(top)
                summary = ", ".join(f"{'null' if pd.isna(k) else k} ({v})" for k, v in counts.items())
            rows[column] = {
                "dtype": str(values.dtype),
                "distinct": values.nunique(),
                "missing_percent": values.isna().mean() * 100,
                "values": summary,
            }
        return pd.DataFrame(rows).T

    # --- show -----------------------------------------------------------------------------

    def show(self, clip_id: str, manifest: dict | None = None) -> dict:
        """Chunk, collection metadata, and per feature: present, and which items are stored."""
        if clip_id not in self.metadata.index:
            raise ValueError(f"Unknown clip ID {clip_id!r} (not in clip_index).")
        stored = (manifest or {}).get("clips", {}).get(clip_id, {}).get("files", {})
        features = {}
        for feature in self.presence.columns:
            keys = [item.key for item in self.registry.items(feature)] if feature in self.registry.features_df.index else []
            features[feature] = {
                "present": bool(self.presence.at[clip_id, feature]),
                "stored": [key for key in keys if key in stored],
            }
        metadata = json.loads(self.metadata.loc[[clip_id]].iloc[0].to_json())
        return {"clip_id": clip_id, "metadata": metadata, "features": features}

    # --- select ---------------------------------------------------------------------------

    def select(
        self,
        require: list[str] | None = None,
        where: list[str] | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[str]:
        """Clips having every required feature and matching every condition, in index order.

        `require` takes the same selections as the fetcher (features, groups, presets, …); a
        clip qualifies when it has every feature they name. With `limit`, a random sample of
        that size drawn with `seed` (same inputs, same list).
        """
        mask = pd.Series(True, index=self.metadata.index)
        for feature in self.required_features(require or []):
            mask &= self.presence[feature]
        for text in where or []:
            mask &= self._condition_mask(Condition.parse(text))
        clips = self.metadata.index[mask]
        if limit is not None and limit < len(clips):
            sample = pd.Series(clips).sample(n=limit, random_state=seed)
            clips = clips[clips.isin(sample)]
        return list(clips)

    def required_features(self, require: list[str]) -> list[str]:
        features = list(dict.fromkeys(item.feature for item in self.registry.resolve(require)))
        missing = [f for f in features if f not in self.presence.columns]
        if missing:
            raise ValueError(f"No presence information for: {', '.join(missing)}.")
        return features

    def _condition_mask(self, condition: Condition) -> pd.Series:
        if condition.column not in self.metadata.columns:
            raise ValueError(
                f"Unknown metadata column {condition.column!r}. "
                f"Columns: {', '.join(self.metadata.columns)}."
            )
        values = self.metadata[condition.column]
        op = _OPERATORS[condition.op]
        if condition.op in ("=", "!="):
            wanted = [v.strip() for v in condition.value.split(",")]
            match_null = _NULL in wanted
            targets = [self._cast(values, v, condition) for v in wanted if v != _NULL]
            matched = values.isin(targets) | (values.isna() if match_null else False)
            return matched if condition.op == "=" else ~matched
        return op(values, self._cast(values, condition.value, condition)).fillna(False).astype(bool)

    @staticmethod
    def _cast(values: pd.Series, text: str, condition: Condition):
        if pd.api.types.is_bool_dtype(values):
            if text.lower() not in ("true", "false"):
                raise ValueError(f"{condition.column} is boolean; use true or false, not {text!r}.")
            return text.lower() == "true"
        if pd.api.types.is_numeric_dtype(values):
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"{condition.column} is numeric; {text!r} is not a number.") from None
            return int(number) if pd.api.types.is_integer_dtype(values) and number.is_integer() else number
        if condition.op in ("=", "!=") and text not in set(values.dropna()):
            valid = sorted(values.dropna().unique())
            raise ValueError(
                f"No clip has {condition.column}={text!r}. Values: {', '.join(map(str, valid))}."
            )
        return text


def write_clip_list(path: Path, clips: list[str], header: list[str]) -> None:
    """One clip ID per line, preceded by `#` comments saying how the list was made."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {line}" for line in header] + clips
    path.write_text("\n".join(lines) + "\n")

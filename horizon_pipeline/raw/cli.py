"""`horizon-fetch`: fetch (or remove) raw PhysicalAI-AV features for a list of clips."""

import argparse
import sys
from pathlib import Path

from horizon_pipeline.env import load_hf_token
from horizon_pipeline.raw.fetch import (
    DEFAULT_CONFIRM_GB,
    DEFAULT_ROOT,
    Fetcher,
    PlanEntry,
    format_size,
    read_clip_list,
)
from horizon_pipeline.raw.source import DATASET_REVISION, HfSource


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="horizon-fetch",
        description="Fetch raw PhysicalAI-AV features on request into data/raw/<clip_id>/.",
    )
    parser.add_argument("--clips", nargs="+", default=[], metavar="CLIP_ID")
    parser.add_argument("--clips-file", type=Path, help="one clip ID per line")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--features",
        nargs="+",
        metavar="SEL",
        help="features, feature:member, groups, group:member or presets (see presets.toml)",
    )
    what.add_argument("--remove", nargs="+", metavar="SEL", help="delete these stored files")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched, with sizes")
    parser.add_argument("--yes", action="store_true", help="do not ask above the size threshold")
    parser.add_argument("--confirm-gb", type=float, default=DEFAULT_CONFIRM_GB)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    if args.clips_file:
        args.clips += read_clip_list(args.clips_file)
    if not args.clips:
        parser.error("give --clips and/or --clips-file")
    args.clips = list(dict.fromkeys(args.clips))
    return args


def print_plan(entries: list[PlanEntry]) -> None:
    for e in entries:
        size = format_size(e.size) if e.status == "fetch" else ""
        print(f"{e.status:<7} {e.clip_id}  {e.item.key:<45} {size:>10}")
    to_fetch = [e for e in entries if e.status == "fetch"]
    unknown = sum(e.size is None for e in to_fetch)
    total = format_size(sum(e.size or 0 for e in to_fetch))
    print(
        f"\n{len(to_fetch)} to fetch ({total}" + (f", {unknown} of unknown size" if unknown else "") + "), "
        f"{sum(e.status == 'stored' for e in entries)} already stored, "
        f"{sum(e.status == 'absent' for e in entries)} absent."
    )
    if any(e.item.member is None for e in to_fetch):
        print("Parquet sizes are whole chunk files (~100 clips); only the clip's rows are stored.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fetcher = Fetcher(HfSource(args.revision, token=load_hf_token(args.env_file)), args.root)
    try:
        items = fetcher.registry.resolve(args.remove or args.features)
        if args.remove:
            removed = fetcher.remove(args.clips, items)
            print(f"\nremoved {removed} files")
            return 0
        entries = fetcher.fetch(
            args.clips,
            items,
            dry_run=args.dry_run,
            confirm_bytes=float("inf") if args.yes else args.confirm_gb * 1024**3,
            confirm=lambda prompt: input(prompt).strip().lower() in ("y", "yes"),
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print()
    print_plan(entries)
    if args.dry_run:
        print("Dry run: nothing fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

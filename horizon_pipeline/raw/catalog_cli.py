"""`horizon-catalog`: browse PhysicalAI-AV clips and write clip lists, from index files only."""

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from horizon_pipeline.env import load_hf_token
from horizon_pipeline.raw.catalog import DEFAULT_LISTS_DIR, Catalog, write_clip_list
from horizon_pipeline.raw.fetch import DEFAULT_ROOT
from horizon_pipeline.raw.source import DATASET_REVISION, HfSource


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="horizon-catalog",
        description="Browse PhysicalAI-AV clips and write clip lists. Reads index files only.",
    )
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("summary", help="clip count, feature coverage, metadata columns")

    show = commands.add_parser("show", help="one clip: chunk, metadata, features present and stored")
    show.add_argument("clip_ids", nargs="+", metavar="CLIP_ID")
    show.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="where the fetch manifest is")

    select = commands.add_parser("select", help="filter clips and write a clip list")
    select.add_argument(
        "--require",
        nargs="+",
        default=[],
        metavar="SEL",
        help="features the clip must have (features, groups, presets, as in horizon-fetch)",
    )
    select.add_argument(
        "--where",
        nargs="+",
        default=[],
        metavar="COND",
        help="metadata conditions, e.g. country=Germany,France hour_of_day>=18 radar_config=null",
    )
    select.add_argument("--limit", type=int, help="random sample of this many clips")
    select.add_argument("--seed", type=int, default=0)
    select.add_argument("--name", help="write data/clip_lists/<name>.txt (otherwise only print)")
    select.add_argument("--lists-dir", type=Path, default=DEFAULT_LISTS_DIR)
    return parser.parse_args(argv)


def print_summary(catalog: Catalog) -> None:
    n = len(catalog.metadata)
    print(f"{n} clips in {catalog.metadata['chunk'].nunique()} chunks (revision {catalog.revision[:12]})\n")
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 90):
        print("Groups (share of clips with all / any of the group's features):")
        print(catalog.group_coverage().to_string(float_format="{:.1f}".format), "\n")
        print("Features:")
        print(catalog.feature_coverage().to_string(float_format="{:.1f}".format), "\n")
        print("Metadata columns (usable in `select --where`):")
        print(catalog.metadata_overview().to_string(float_format="{:.1f}".format))


def print_show(info: dict) -> None:
    print(f"clip {info['clip_id']}")
    for key, value in info["metadata"].items():
        print(f"  {key:<16} {value}")
    print()
    for feature, state in info["features"].items():
        present = "present" if state["present"] else "absent"
        stored = ", ".join(state["stored"]) if state["stored"] else "-"
        print(f"  {feature:<34} {present:<8} stored: {stored}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = Catalog(HfSource(args.revision, token=load_hf_token(args.env_file)))
    try:
        if args.command == "summary":
            print_summary(catalog)
        elif args.command == "show":
            manifest_path = args.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
            for clip_id in args.clip_ids:
                print_show(catalog.show(clip_id, manifest))
        else:
            clips = catalog.select(args.require, args.where, args.limit, args.seed)
            for clip_id in clips:
                print(clip_id)
            print(f"\n{len(clips)} clips", file=sys.stderr)
            if args.name:
                path = args.lists_dir / f"{args.name}.txt"
                write_clip_list(
                    path,
                    clips,
                    [
                        f"horizon-catalog {shlex.join(sys.argv[1:] if argv is None else argv)}",
                        f"revision {catalog.revision}",
                        f"require {' '.join(args.require) or '-'}; where {' '.join(args.where) or '-'}; "
                        f"limit {args.limit}; seed {args.seed}",
                        f"{len(clips)} clips, written {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                    ],
                )
                print(f"wrote {path}", file=sys.stderr)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

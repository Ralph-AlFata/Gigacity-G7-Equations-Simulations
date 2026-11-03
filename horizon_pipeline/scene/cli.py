"""`horizon-scene`: build scenes from raw clip files into data/scene/<clip_id>/<ego_source>/."""

import argparse
import sys
from pathlib import Path

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.scene.build import DEFAULT_OUT, DEFAULT_RAW, EGO_SOURCES, DuplicateError, build_scene, write_scene
from horizon_pipeline.scene.checks import parked_spread


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="horizon-scene", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build meta.json, ego.parquet and objects.parquet")
    build.add_argument("--clips", nargs="+", default=[], metavar="CLIP_ID")
    build.add_argument("--clips-file", type=Path, help="one clip ID per line")
    build.add_argument("--ego-source", choices=[*EGO_SOURCES, "both"], default="offline")
    build.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.clips_file:
        args.clips += read_clip_list(args.clips_file)
    if not args.clips:
        parser.error("give --clips and/or --clips-file")
    args.clips = list(dict.fromkeys(args.clips))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = list(EGO_SOURCES) if args.ego_source == "both" else [args.ego_source]
    for clip in args.clips:
        for source in sources:
            try:
                scene = build_scene(args.raw / clip, source)
            except (DuplicateError, FileNotFoundError, ValueError) as e:
                print(f"FAILED {clip} {source}: {e}", file=sys.stderr)
                return 1
            out = write_scene(scene, args.out)
            r, m = scene.report, scene.meta
            spread = parked_spread(scene.objects)
            print(
                f"{clip} {source:<7} -> {out}\n"
                f"  {m['n_ticks']} ticks from {m['t0_us'] / 1e6:.3f} s; {r['sightings']} sightings "
                f"({r['sightings_outside_ego_range']} outside the ego poses, {r['duplicates_merged']} duplicates merged); "
                f"{r['tracks_in_scene_file']} tracks, {r['object_rows']} object rows"
                + (f"; unknown classes: {', '.join(r['unknown_classes'])}" if r["unknown_classes"] else "")
                + f"\n  parked tracks: {len(spread)}, position RMS median {spread.median():.3f} m, "
                f"max {spread.max():.3f} m"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

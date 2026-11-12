"""`horizon-reference`: build reference trajectories into data/reference/<clip_id>/<ego_source>/<source>/."""

import argparse
import sys
from pathlib import Path

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.reference.format import DEFAULT_ROOT, reference_dir, write_reference
from horizon_pipeline.reference.recorded import SOURCE, build_recorded
from horizon_pipeline.scene.build import DEFAULT_OUT, EGO_SOURCES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="horizon-reference", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recorded", help="the path the car actually drove after each tick")
    rec.add_argument("--clips", nargs="+", default=[], metavar="CLIP_ID")
    rec.add_argument("--clips-file", type=Path, help="one clip ID per line")
    rec.add_argument("--ego-source", choices=[*EGO_SOURCES, "both"], default="offline")
    rec.add_argument("--scenes", type=Path, default=DEFAULT_OUT)
    rec.add_argument("--out", type=Path, default=DEFAULT_ROOT)
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
        for ego_source in sources:
            scene_dir = args.scenes / clip / ego_source
            if not (scene_dir / "meta.json").is_file():
                print(f"no scene in {scene_dir}: run `horizon-scene build --clips {clip} --ego-source {ego_source}`", file=sys.stderr)
                return 1
            ref, meta = build_recorded(scene_dir)
            out = write_reference(ref, meta, reference_dir(args.out, clip, ego_source, SOURCE))
            n = meta["n_steps"]
            print(
                f"{clip} {ego_source:<7} -> {out}\n"
                f"  {meta['n_ticks']} plans, {len(ref)} rows, {n['max']} to {n['min']} steps "
                f"({(n['max'] - 1) * meta['step_s']:.1f} s to {(n['min'] - 1) * meta['step_s']:.1f} s ahead); "
                f"speed filled at ticks {meta['speed_filled_ticks']}; max speed {ref['speed'].max():.1f} m/s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

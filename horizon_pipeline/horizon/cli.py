"""`horizon-run`: one H per plan, into results/<name>/<clip_id>/<ego_source>/<reference_source>/ (T014)."""

import argparse
import sys
from pathlib import Path

from horizon_pipeline.horizon.config import DEFAULT_CONFIG, load_config
from horizon_pipeline.horizon.run import DEFAULT_RESULTS, results_dir, run_one, summary, write_results
from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.reference.format import DEFAULT_ROOT, reference_dir
from horizon_pipeline.scene.build import DEFAULT_OUT, EGO_SOURCES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="horizon-run", description=__doc__)
    parser.add_argument("--clips", nargs="+", default=[], metavar="CLIP_ID")
    parser.add_argument("--clips-file", type=Path, help="one clip ID per line")
    parser.add_argument("--ego-source", choices=[*EGO_SOURCES, "both"], default="offline")
    parser.add_argument("--reference", required=True, help="reference source: recorded, alpamayo, ...")
    parser.add_argument("--name", required=True, help="run name: results/<name>/...")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--references", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    if args.clips_file:
        args.clips += read_clip_list(args.clips_file)
    if not args.clips:
        parser.error("give --clips and/or --clips-file")
    args.clips = list(dict.fromkeys(args.clips))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    sources = list(EGO_SOURCES) if args.ego_source == "both" else [args.ego_source]
    for clip in args.clips:
        for ego_source in sources:
            scene_dir = args.scenes / clip / ego_source
            ref_dir = reference_dir(args.references, clip, ego_source, args.reference)
            for d, how in [(scene_dir, "horizon-scene build"), (ref_dir, f"horizon-reference {args.reference}")]:
                if not (d / "meta.json").is_file():
                    print(f"nothing in {d}: run `{how}` for {clip} ({ego_source}) first", file=sys.stderr)
                    return 1
            out, meta = run_one(scene_dir, ref_dir, cfg, args.config)
            path = write_results(out, meta, cfg, results_dir(args.out, args.name, clip, ego_source, args.reference))
            print(f"{clip} {ego_source:<7} -> {path}\n  {summary(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

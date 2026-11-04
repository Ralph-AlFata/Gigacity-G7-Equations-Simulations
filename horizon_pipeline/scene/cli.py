"""`horizon-scene`: build scenes from raw clip files into data/scene/<clip_id>/<ego_source>/, and
view them in Rerun next to the camera videos."""

import argparse
import subprocess
import sys
from pathlib import Path

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.scene.build import DEFAULT_OUT, DEFAULT_RAW, EGO_SOURCES, DuplicateError, build_scene, write_scene
from horizon_pipeline.scene.camera import ALPAMAYO_CAMERAS
from horizon_pipeline.scene.checks import parked_spread
from horizon_pipeline.scene.view import DEFAULT_RECORDINGS, ViewOptions, is_current, other_source, recording_path, write_recording


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="horizon-scene", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build meta.json, ego.parquet and objects.parquet")
    build.add_argument("--clips", nargs="+", default=[], metavar="CLIP_ID")
    build.add_argument("--clips-file", type=Path, help="one clip ID per line")
    build.add_argument("--ego-source", choices=[*EGO_SOURCES, "both"], default="offline")
    build.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--no-lidar", action="store_true", help="leave the lidar columns empty (T008)")
    view = sub.add_parser(
        "view", help="build a Rerun recording of a scene next to the camera videos, then open it",
        description="Builds the whole recording (or reuses one newer than the scene), then opens it in Rerun.",
    )  # fmt: skip
    view.add_argument("--clip", required=True, metavar="CLIP_ID")
    view.add_argument("--ego-source", choices=list(EGO_SOURCES), default="offline")
    view.add_argument("--compare", action="store_true", help="also draw the other ego source's box and path")
    view.add_argument("--cameras", nargs="*", default=ALPAMAYO_CAMERAS, metavar="CAMERA",
                      help="cameras to show (default: the three Alpamayo uses); none for 3D only")
    view.add_argument("--all-frames", action="store_true", help="every video frame (30 Hz), not only the one nearest each tick")
    view.add_argument("--no-lidar", action="store_true", help="leave out the lidar points")
    view.add_argument("--lidar-every", type=int, default=ViewOptions.lidar_every, help="keep every n-th lidar point (default 10)")
    view.add_argument("--labels", action="store_true", help="draw every box's label (default: shown when a box is selected)")
    view.add_argument("--fade-beyond", type=float, metavar="M", help="fade objects farther than M metres from the ego")
    view.add_argument("--save", type=Path, metavar="FILE.rrd", help="write the recording here and do not open it")
    view.add_argument("--rebuild", action="store_true", help="build the recording even if a current one exists")
    view.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS, help="where recordings are kept")
    view.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    view.add_argument("--scenes", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.command == "view":
        return args
    if args.clips_file:
        args.clips += read_clip_list(args.clips_file)
    if not args.clips:
        parser.error("give --clips and/or --clips-file")
    args.clips = list(dict.fromkeys(args.clips))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return view(args) if args.command == "view" else build(args)


def view(args: argparse.Namespace) -> int:
    scene_dir = args.scenes / args.clip / args.ego_source
    if not (scene_dir / "meta.json").is_file():
        print(f"no scene in {scene_dir}: run `horizon-scene build --clips {args.clip} --ego-source {args.ego_source}`", file=sys.stderr)
        return 1
    clip_dir = args.raw / args.clip
    missing = [c for c in args.cameras if not (clip_dir / c / "video.mp4").is_file()]
    if missing:
        print(f"no video for {', '.join(missing)}: fetch with `horizon-fetch --clips {args.clip} --features "
              + " ".join(f"{c}:video" for c in missing) + "`, or pick --cameras", file=sys.stderr)
        return 1
    opts = ViewOptions(lidar=not args.no_lidar, lidar_every=args.lidar_every, cameras=args.cameras,
                       all_frames=args.all_frames, compare=args.compare, labels=args.labels,
                       fade_beyond_m=args.fade_beyond)  # fmt: skip
    if opts.compare and not (scene_dir.parent / other_source(args.ego_source) / "meta.json").is_file():
        print(f"--compare needs the {other_source(args.ego_source)} scene too: run `horizon-scene build --clips {args.clip} --ego-source both`", file=sys.stderr)
        return 1
    path = args.save or recording_path(args.recordings, args.clip, args.ego_source, opts)
    if args.rebuild or args.save or not is_current(path, scene_dir, opts):
        print(f"building {path} ...")
        write_recording(path, scene_dir, clip_dir, opts, progress=print)
        print(f"  {path.stat().st_size / 1e6:.0f} MB")
    else:
        print(f"using {path} (newer than the scene; --rebuild to build it again)")
    if args.save:
        print(f"open it with `uv run rerun {path}`")
        return 0
    subprocess.Popen([sys.executable, "-m", "rerun", str(path)], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    print(f"opened {path} in Rerun")
    return 0


def build(args: argparse.Namespace) -> int:
    sources = list(EGO_SOURCES) if args.ego_source == "both" else [args.ego_source]
    for clip in args.clips:
        for source in sources:
            try:
                scene = build_scene(args.raw / clip, source, lidar=not args.no_lidar)
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
                + (f"\n  duplicates needing the range allowance: {d['pairs_over_flat_limit']}, largest "
                   f"{d['largest_offset_m']} m at {d['largest_offset_range_m']} m"
                   if (d := r["duplicate_check"])["pairs_over_flat_limit"] else "")
                + f"\n  parked tracks: {len(spread)}, position RMS median {spread.median():.3f} m, "
                f"max {spread.max():.3f} m"
            )
            if r["lidar"]:
                print(f"  lidar: {r['lidar']['observed_share'] * 100:.1f}% of object rows observed; "
                      f"crossing within {r['lidar']['lidar_dt_s_max_abs'] * 1e3:.0f} ms of the tick")
                for key, v in r["lidar"]["by_class_and_distance"].items():
                    print(f"    {key:<18} {v['share'] * 100:5.1f}% of {v['rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

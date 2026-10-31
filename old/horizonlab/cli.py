import argparse
import json
from pathlib import Path
from .dataset import fetch_dataset, json_write
from .defaults import default_config
from .demo import make_demo
from .replay import run_replay
from .report import make_plots


def main():
    parser = argparse.ArgumentParser(description="Offline conditional clearance-horizon experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("write-config", help="Write clearly labeled, uncalibrated sensitivity settings")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("demo", help="Run synthetic engineering fixtures; no HF credentials or GPU")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p = sub.add_parser("fetch", help="Select clips and fetch only offline labels and dimensions")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--split", default="test")
    p.add_argument("--split-name", default="alpamayo_aligned")
    p.add_argument("--revision", help="HF revision; resolved commit is recorded in the manifest")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--clip-ids", type=Path, help="Text file of UUIDs, one per line; overrides random sampling")
    p.add_argument("--obstacle-frame", choices=["auto", "rig", "anchor"], default="auto")
    p = sub.add_parser("run", help="Compute horizons, diagnostics, summary statistics, and plots")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--references", type=Path, help="JSONL of available planned trajectories; otherwise recorded oracle")
    args = parser.parse_args()
    if args.command == "write-config":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json_write(args.out, default_config())
        print(args.out)
        return
    if args.command == "fetch":
        ids = [x.strip() for x in args.clip_ids.read_text().splitlines() if x.strip()] if args.clip_ids else None
        manifest = fetch_dataset(args.out, args.n, args.seed, args.split, args.split_name,
                                 args.revision, args.cache_dir, ids, args.obstacle_frame)
        print(f"Normalized {len(manifest['selected'])} clips; revision={manifest['resolved_revision']}")
        return
    config = json.loads(args.config.read_text()) if args.config else default_config()
    if args.command == "demo":
        data, output = args.out / "data", args.out / "results"
        make_demo(data)
        print("SYNTHETIC DEMO ONLY. No NVIDIA clips or Alpamayo inference are used.")
        df, summary = run_replay(data, config, output)
    else:
        output = args.out
        df, summary = run_replay(args.data, config, output, args.references)
    make_plots(df, output)
    print(f"Saved {len(df)} experiment-windows to {output}")
    for item in summary["results"]:
        mean = item.get("mean_capped_certified_horizon_s")
        value = "n/a" if mean is None else f"{mean:.3f} s"
        print(f"{item['experiment']}: valid={item['valid_windows']}/{item['windows']}, capped mean={value}")

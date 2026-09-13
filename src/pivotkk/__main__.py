"""CLI: python -m pivotkk <command>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pivotkk")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("prepare", help="download and build all datasets (CPU, run locally)")
    d.add_argument("--pivots", nargs="+", default=["lin", "lua", "swc", "swh", "sag", "fra"])
    d.add_argument("--sizes", nargs="+", type=int, default=[10_000, 50_000, 100_000])
    d.add_argument("--sampling", choices=["stratified", "top", "random"], default="stratified")

    a = sub.add_parser("audit", help="is kon_Latn actually Kikongo?")
    a.add_argument("--sample", type=int, default=50_000)
    a.add_argument("--corpora", nargs="+", default=None)
    a.add_argument("--lexicons", type=Path, default=None,
                   help="directory of reference lexicons (default: data/lexicons)")

    t = sub.add_parser("train", help="run one experiment (two-stage)")
    t.add_argument("config", type=Path)
    t.add_argument("--smoke", action="store_true", help="tiny data, verifies the loop end-to-end")

    e = sub.add_parser("eval", help="score a checkpoint on both test sets")
    e.add_argument("checkpoint")
    e.add_argument("--limit", type=int, default=None)

    sub.add_parser("table", help="collect every run's metrics into one table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "prepare":
        from .prepare import main as run
        run(args.pivots, args.sizes, how=args.sampling)
    elif args.cmd == "audit":
        from .audit import main as run
        run(sample=args.sample, corpora=args.corpora, lexicon_dir=args.lexicons)
    elif args.cmd == "train":
        from .train import main as run
        run(args.config, smoke=args.smoke)
    elif args.cmd == "eval":
        from .evaluate import main as run
        run(args.checkpoint, limit=args.limit)
    elif args.cmd == "table":
        from .evaluate import collect
        df = collect()
        if df.empty:
            print("no runs with metrics.json yet")
        else:
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

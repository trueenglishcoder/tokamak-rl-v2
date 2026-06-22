#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokamak_rl_v2.sweeps.tcvdelta_single_segment_0p1s import summarize_root


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a TCV-delta 0.1 s single-segment reward sweep")
    parser.add_argument("root")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    summarize_root(Path(args.root), out_dir=None if args.out_dir is None else Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

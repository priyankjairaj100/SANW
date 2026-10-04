#!/usr/bin/env python3
"""Fit one bounded-pair scorer using training and development data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcr.practical_training import PracticalTrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--composition-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--composition-temperature", type=float, default=0.02)
    parser.add_argument("--composition-margin", type=float, default=0.005)
    parser.add_argument("--retention-margin-cap", type=float, default=0.01)
    parser.add_argument("--hard-negative-count", type=int, default=4)
    parser.add_argument("--pair-chunk", type=int, default=8192)
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    args = parser.parse_args()
    config = PracticalTrainingConfig(**{key: value for key, value in vars(args).items() if key not in ("repository", "output")})
    print(json.dumps(run_training(args.repository, args.output, config), indent=2), flush=True)


if __name__ == "__main__":
    main()

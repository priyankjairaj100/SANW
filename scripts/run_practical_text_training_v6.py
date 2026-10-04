#!/usr/bin/env python3
"""Fit the new token-level backup using training/development inputs only."""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcr.practical_text_training_v6 import TextTrainingConfig, run_text_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--token-protocol", type=Path)
    for field in dataclasses.fields(TextTrainingConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(parser.parse_args())
    repository = args.pop("repository")
    output = args.pop("output")
    cache = args.pop("cache_directory")
    token_protocol = args.pop("token_protocol")
    result = run_text_training(repository, output, cache, TextTrainingConfig(**args), token_protocol=token_protocol)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

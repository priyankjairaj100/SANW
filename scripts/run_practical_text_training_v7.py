#!/usr/bin/env python3
"""Prepared source-pair token runner. Launch only after root authorizes v7 fitting."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcr.practical_text_training_v6 import TextTrainingConfig
from gcr.practical_text_training_v7 import run_source_pair_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    for field in dataclasses.fields(TextTrainingConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(parser.parse_args())
    repository, output = args.pop("repository"), args.pop("output")
    cache, protocol = args.pop("cache_directory"), args.pop("protocol")
    print(json.dumps(run_source_pair_training(repository, output, cache, TextTrainingConfig(**args), protocol), indent=2))


if __name__ == "__main__":
    main()

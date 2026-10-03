#!/usr/bin/env python3
"""Synthetic-only CPU timing and forward/backward preflight; no study data."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.adapters import ResidualAdapter
from gcr.losses import contrastive_loss
from gcr.training import METHODS, atomic_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--captions-per-image", type=int, default=35)
    parser.add_argument("--output", default="results/training_preflight.json")
    args = parser.parse_args()
    if args.captions_per_image < 8:
        parser.error("At least eight captions per image are needed for the fixture.")
    seed_everything(5, args.threads)
    batch, dim, captions = 32, 512, args.captions_per_image
    images = F.normalize(torch.randn(batch, dim), dim=1)
    texts = F.normalize(torch.randn(batch * captions, dim), dim=1)
    relations = torch.zeros(batch, batch * captions, dtype=torch.int64)
    for row in range(batch):
        start = row * captions
        relations[row, start:start+5] = 1
        for offset in range(5, captions):
            relations[row, start+offset] = 2 + (offset - 5) % 3
    results = []
    for method in METHODS:
        model = ResidualAdapter(dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        generator = torch.Generator().manual_seed(101)
        durations, losses = [], []
        for step in range(args.steps + 1):
            started = time.perf_counter()
            im, tx = model(images, texts)
            loss = contrastive_loss(im, tx, relations, method, 100., generator=generator, semantic_text_features=texts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step:
                durations.append(time.perf_counter() - started)
                losses.append(float(loss))
        results.append({"method": method, "seconds_per_batch": sum(durations) / len(durations), "finite": all(math.isfinite(value) for value in losses)})
    estimate = sum(row["seconds_per_batch"] for row in results) * 3 * 3 * 10 * math.ceil(1200 / batch)
    output = {
        "synthetic_only": True, "natural_image_study_runs": 0,
        "threads": args.threads, "images_per_batch": batch, "captions_per_image": captions,
        "dimension": dim, "timed_steps_per_method": args.steps, "methods": results,
        "sequential_108_candidate_training_seconds_estimate": estimate,
        "estimate_excludes": ["validation", "checkpoint and history writes", "feature loading", "contention from concurrent workers"],
        "actual_candidate_caption_counts_may_differ": True,
    }
    atomic_json(ROOT / args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

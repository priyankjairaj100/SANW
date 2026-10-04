#!/usr/bin/env python3
"""Sequential authorized token replications for one encoder, three threads total."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {"vit_b32": [(29, 3), (43, 4)], "rn50": [(29, 5), (43, 6)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=PROTOCOLS, required=True)
    args = parser.parse_args()
    for seed, version in PROTOCOLS[args.encoder]:
        protocol = f"results/practical_v6/token_protocol_v{version}_{args.encoder}_seed{seed}.json"
        output = f"results/practical_v6/text_pilot/{args.encoder}_eps0.01_ret1_lr1e-5_seed{seed}_4epochs"
        command = [sys.executable, "-u", "scripts/run_practical_text_training_v6.py",
                   "--encoder", args.encoder, "--epsilon", ".01", "--retention-weight", "1",
                   "--learning-rate", "1e-5", "--seed", str(seed), "--epochs", "4",
                   "--batch-size", "32", "--text-batch-size", "64", "--threads", "3",
                   "--token-protocol", protocol,
                   "--cache-directory", f"results/practical_v6/text_prefix_cache/{args.encoder}",
                   "--output", output]
        print(f"START encoder={args.encoder} seed={seed}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"COMPLETE encoder={args.encoder} seed={seed}", flush=True)


if __name__ == "__main__":
    main()

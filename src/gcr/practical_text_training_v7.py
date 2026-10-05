"""Prepared v7 source-pair token fitting; v6 scientific files remain untouched.

Training eligibility uses only frozen training margins. Development composition
uses every valid source/source/contradiction triplet, without that margin filter.
Batched epoch metrics are screening records. Canonical single-caption inference
and development-only global amplitude calibration select the final model.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .practical_text_v6 import WEIGHTS, digest
from .practical_text_training_v6 import TextTrainingConfig, load_context, training_caption_indices
from .practical_source_pair_v7 import SourcePairTrainingExamples
from .practical_development_v7 import source_pair_validation_metrics
from .practical_training import select_development_epoch
from .review_training import _atomic_npz
from .training import atomic_json, atomic_torch_save, canonical_json, seed_everything

SOURCE_FACTORY = "gcr.practical_source_pair_v7.SourcePairTrainingExamples"


@torch.inference_mode()
def evaluate_source_pair_development(scorer, data, pool, output, epoch, config):
    """All validation pairs are scored before valid source triplets are formed."""
    scorer.eval()
    image_indices, text_indices, scores = [], [], []
    for image_index in data.split_indices["validation"]:
        columns = sorted(data.pairs[image_index])
        values = scorer.score_pairs(data.images[image_index:image_index + 1].expand(len(columns), -1),
                                     data.texts[columns])
        image_indices.extend([image_index] * len(columns))
        text_indices.extend(columns)
        scores.extend(values.tolist())
    composition, raw = source_pair_validation_metrics(data, np.asarray(image_indices),
                                                      np.asarray(text_indices), np.asarray(scores))
    composition_path = output / "development" / f"epoch_{epoch:02d}_source_pair_composition.npz"
    _atomic_npz(composition_path, **raw)
    composition["predictions"] = {"path": str(composition_path.relative_to(output)),
                                   "sha256": digest(composition_path)}

    image_scores, image_top = scorer.exact_topk(pool.images, pool.texts, k=1, direction="i2t")
    text_scores, text_top = scorer.exact_topk(pool.images, pool.texts, k=1, direction="t2i")
    image_correct = pool.owner[image_top[:, 0]] == torch.arange(len(pool.images))
    text_correct = text_top[:, 0] == pool.owner
    retrieval = {"i2t_r1": float(image_correct.double().mean()), "t2i_r1": float(text_correct.double().mean()),
                 "image_correct_count": int(image_correct.sum()), "text_correct_count": int(text_correct.sum()),
                 "image_count": len(pool.images), "text_count": len(pool.texts),
                 "tie_rule": "descending_float64_score_then_ascending_gallery_manifest_index",
                 "relevance": "source_caption_ownership"}
    retrieval_path = output / "development" / f"epoch_{epoch:02d}_retrieval.npz"
    _atomic_npz(retrieval_path, image_ids=np.asarray(pool.image_ids), text_ids=np.asarray(pool.text_ids),
                owner=pool.owner.numpy(), image_correct=image_correct.numpy(), text_correct=text_correct.numpy(),
                image_top_scores=image_scores.numpy(), image_top_indices=image_top.numpy(),
                text_top_scores=text_scores.numpy(), text_top_indices=text_top.numpy())
    retrieval["predictions"] = {"path": str(retrieval_path.relative_to(output)), "sha256": digest(retrieval_path)}
    return composition, retrieval


def validate_v7_protocol(repository, protocol_path, config):
    protocol = json.loads(protocol_path.read_text())
    for record in protocol["parent_protocols"]:
        if digest(repository / record["path"]) != record["sha256"]:
            raise ValueError("A frozen parent protocol changed")
    for name, expected in protocol["source_hashes"].items():
        if digest(repository / name) != expected:
            raise ValueError(f"Prepared v7 source changed: {name}")
    parent = json.loads((repository / protocol["parent_protocols"][0]["path"]).read_text())
    if protocol["practical_gate"] != parent["practical_gate"]:
        raise ValueError("The inherited practical gate changed")
    if config.encoder not in protocol["encoders"] or config.seed not in protocol["seeds"]:
        raise ValueError("Encoder/seed is outside the frozen plan")
    for name, expected in protocol["fit_config"].items():
        if getattr(config, name) != expected:
            raise ValueError(f"Configuration differs from v7 protocol: {name}")
    return protocol


def run_source_pair_training(repository, output, cache_directory, config, protocol_path):
    """No caller may launch this prepared experiment before root authorization."""
    config.validate()
    repository, output, protocol_path = Path(repository).resolve(), Path(output).resolve(), Path(protocol_path).resolve()
    protocol = validate_v7_protocol(repository, protocol_path, config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; earlier records are immutable")
    seed_everything(config.seed, config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    paths, data, pool, cache, _, model = load_context(repository, cache_directory, config)
    train = SourcePairTrainingExamples(data, config.hard_negative_count, config.epsilon)
    named_parameters = model.learned_named_parameters()
    parameters = [parameter for _, parameter in named_parameters]
    initial = {name: parameter.detach().clone() for name, parameter in named_parameters}
    tokenizer_distribution = importlib.metadata.distribution("open_clip_torch")
    identity = {
        "schema": "sanw_practical_source_pair_token_v7", "config": dataclasses.asdict(config),
        "source_factory": SOURCE_FACTORY,
        "protocol": {"path": str(protocol_path.relative_to(repository)), "sha256": digest(protocol_path)},
        "source_sha256": protocol["source_hashes"], "weight_identity": WEIGHTS[config.encoder],
        "inputs": {name: {"path": str(path.relative_to(repository)), "sha256": digest(path)} for name, path in paths.items()},
        "prefix_cache_receipt": {"path": str(Path(cache_directory) / "receipt.json"),
                                 "sha256": digest(Path(cache_directory) / "receipt.json")},
        "tokenizer": {"package": "open_clip_torch", "version": tokenizer_distribution.version,
                      "files": {name: digest(tokenizer_distribution.locate_file("open_clip/" + name))
                                for name in ("tokenizer.py", "bpe_simple_vocab_16e6.txt.gz")}},
        "mining": train.mining_summary,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "training_score_amplitude": 1.0,
        "deployment_score": "s0 + alpha*epsilon*tanh(dot(frozen_image, normalized_learned_text-normalized_reference_text)/epsilon)",
        "composition_development": "all_valid_distinct_source_source_contradiction_triplets_without_margin_eligibility_filter",
        "development_evaluation_role": "batched_screening_only; canonical_global_amplitude_selection_is_separate",
        "held_out_evaluation": "prohibited_by_runner",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    ledger_hash = hashlib.sha256(canonical_json(identity)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_hash})
    atomic_json(output / "training_mining.json", train.mining_summary)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                                  foreach=False, fused=False)
    development_texts = sorted({j for i in data.split_indices["validation"] for j in data.pairs[i]})
    development_indices = cache.indices_for_vectors(data.texts[development_texts]) + cache.indices_for_vectors(pool.texts)
    permutation_generator = torch.Generator().manual_seed(config.seed)
    history, steps = [], 0
    started = time.monotonic()
    for epoch in range(config.epochs + 1):
        epoch_started = time.monotonic()
        training = {}
        if epoch:
            model.train()
            permutation = torch.randperm(len(train.images), generator=permutation_generator).tolist()
            totals, count = {}, 0
            for offset in range(0, len(permutation), config.batch_size):
                batch = permutation[offset:offset + config.batch_size]
                captions = training_caption_indices(train, batch)
                indices = cache.indices_for_vectors(data.texts[captions])
                optimizer.zero_grad(set_to_none=True)
                scorer = cache.encode(model, indices, config.text_batch_size)
                scorer.epsilon = config.epsilon
                loss, values = train.batch_loss(scorer, batch, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source-pair training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                count += len(batch)
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value * len(batch)
            training = {name: value / count for name, value in totals.items()}
        checkpoint = output / "checkpoints" / f"epoch_{epoch:02d}.pt"
        update_norm = float(torch.sqrt(sum((parameter.detach() - initial[name]).double().square().sum()
                                          for name, parameter in named_parameters)))
        atomic_torch_save(checkpoint, {
            "schema": "sanw_practical_text_last_block_v1", "study_schema": "sanw_practical_source_pair_token_v7",
            "source_factory": SOURCE_FACTORY,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "config": dataclasses.asdict(config),
            "weight_identity": WEIGHTS[config.encoder], "epoch": epoch, "optimizer_steps": steps,
            "update_norm": update_norm, "ledger_sha256": ledger_hash,
            "torch_rng_state": torch.get_rng_state(), "permutation_rng_state": permutation_generator.get_state(),
        })
        model.eval()
        with torch.inference_mode():
            scorer = cache.encode(model, development_indices, config.text_batch_size)
            scorer.epsilon = config.epsilon
            if epoch == 0 and scorer.delta.count_nonzero():
                raise RuntimeError("Initial correction must be exactly zero")
            composition, retrieval = evaluate_source_pair_development(scorer, data, pool, output, epoch, config)
            maximum_delta = float(scorer.delta.abs().max())
        row = {"epoch": epoch, "optimizer_steps": steps, "update_norm": update_norm,
               "training": training, "composition": composition, "retrieval": retrieval,
               "maximum_development_embedding_component_change": maximum_delta,
               "checkpoint": {"path": str(checkpoint.relative_to(output)), "sha256": digest(checkpoint)},
               "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        atomic_json(output / "history.json", {"ledger_sha256": ledger_hash, "epochs": history})
        screening = select_development_epoch(history)
        screening["status"] = "screening_only_canonical_amplitude_calibration_required"
        atomic_json(output / "screening_selection.json", screening)
        print(json.dumps({"event": "source_pair_epoch_complete", "encoder": config.encoder, "seed": config.seed,
                          "epoch": epoch, "optimizer_steps": steps,
                          "source_pair_joint_accuracy": composition["paired_joint_accuracy"],
                          "i2t_r1": retrieval["i2t_r1"], "t2i_r1": retrieval["t2i_r1"],
                          "epoch_seconds": row["epoch_seconds"]}), flush=True)
    completion = {"status": "completed_development_only", "canonical_calibration_pending": True, "ledger_sha256": ledger_hash,
                  "epochs": config.epochs, "optimizer_steps": steps, "screening_selection": screening,
                  "history_sha256": digest(output / "history.json"), "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "completion.json", completion)
    return completion

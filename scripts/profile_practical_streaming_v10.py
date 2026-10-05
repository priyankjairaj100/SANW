#!/usr/bin/env python3
"""Original1200 training-only implementation equivalence and resource validation.

No fitting or outcome-dependent configuration choice occurs. A fixed random
coefficient and the frozen state exercise the exact full-gallery implementation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np

from run_practical_joint_v9 import training_view
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_joint_v9 import JointFitConfig, JointCompositionExamples, FullGallerySourceLoss
from gcr.practical_streaming_v10 import FrozenScoreCache, StreamingFullGalleryConstraints, StreamingFullGallerySourceLoss


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(function, repeats=1):
    seconds = []
    for _ in range(repeats):
        start = time.monotonic()
        result = function()
        seconds.append(time.monotonic() - start)
    return result, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--encoder', choices=('vit_b32', 'rn50'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an implementation-validation receipt')
    repo = args.repository.resolve()
    overall_start = time.monotonic()
    loaded, load_seconds = measure(lambda: training_view(repo, args.encoder))
    images, texts, source_rows, owner, sources, supported, contra, provenance = loaded
    if len(images) != 1200:
        raise ValueError('This profiler is restricted to the original1200 training owners')
    source_texts = texts[source_rows]
    config = JointFitConfig(rank=128, radius=1, composition_weight=.25)
    model, pca_seconds = measure(lambda: ConstrainedBilinearScorer.from_training(images, texts, config.rank))
    metadata_path = repo / provenance['inputs']['metadata']['path']
    scale = float(json.loads(metadata_path.read_text())['logit_scale'])
    cache, cache_seconds = measure(lambda: FrozenScoreCache.create(args.cache, images, source_texts, owner))
    constraints, constraints_seconds = measure(lambda: StreamingFullGalleryConstraints(images, source_texts, owner, model, cache))
    retrieval = StreamingFullGallerySourceLoss(constraints, scale)
    composition = JointCompositionExamples(images, texts, sources, supported, contra, model)
    point = np.random.default_rng(20261005).normal(size=model.coefficient.shape)
    point *= config.radius / np.linalg.norm(point)
    measurements, results = {}, {}
    for label, coefficient in (('zero', model.coefficient), ('fixed_random_radius1', point)):
        source_result, source_seconds = measure(lambda: retrieval.loss_gradient(coefficient), 2)
        joint_result, joint_seconds = measure(lambda: composition.loss_gradient(coefficient, composition.eligible, config))
        certificate, scan_seconds = measure(lambda: constraints.scan(coefficient, add=False, canonical=True))
        _, radial_seconds = measure(lambda: constraints.radial_feasibility_factor(coefficient))
        sample = np.random.default_rng(20261006).permutation(len(images))[:64]
        _, minibatch_seconds = measure(lambda: retrieval.loss_gradient(coefficient, sample), 3)
        measurements[label] = {'full_retrieval_seconds': source_seconds, 'full_joint_seconds': joint_seconds,
                               'canonical_full_scan_seconds': scan_seconds, 'radial_full_scan_seconds': radial_seconds,
                               'retrieval_minibatch64_seconds': minibatch_seconds,
                               'retrieval_value': source_result[0], 'retrieval_gradient_norm': float(np.linalg.norm(source_result[1])),
                               'joint_value': joint_result[0], 'joint_gradient_norm': float(np.linalg.norm(joint_result[1])),
                               'certificate': certificate}
        results[label] = source_result
    streamed_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dense, dense_init_seconds = measure(lambda: FullGalleryConstraints(images, source_texts, owner, model))
    dense_retrieval = FullGallerySourceLoss(dense, scale)
    equivalence = {}
    for label, coefficient in (('zero', model.coefficient), ('fixed_random_radius1', point)):
        (value, gradient), seconds = measure(lambda: dense_retrieval.loss_gradient(coefficient))
        old_value, old_gradient = results[label]
        dense_certificate, dense_scan_seconds = measure(lambda: dense.scan(coefficient, add=False, canonical=True))
        stream_certificate = measurements[label]['certificate']
        checks = {}
        for name, val in dense_certificate.items():
            equal = (abs(val-stream_certificate[name]) <= 1e-12 if name == 'min_constraint_slack' and val is not None
                     else val == stream_certificate[name])
            if not equal:
                raise AssertionError(f'Constraint mismatch: {name}')
            checks[name] = True
        np.testing.assert_allclose(old_gradient, gradient, atol=2e-12, rtol=2e-11)
        if abs(value-old_value) > 2e-12:
            raise AssertionError('Full retrieval scalar mismatch')
        equivalence[label] = {'absolute_loss_difference': abs(value-old_value),
                              'max_absolute_gradient_difference': float(np.abs(gradient-old_gradient).max()),
                              'all_constraint_summary_fields_equal': all(checks.values()),
                              'dense_full_retrieval_seconds': seconds, 'dense_canonical_scan_seconds': dense_scan_seconds}
    np.testing.assert_allclose(cache.i2t, dense.frozen, rtol=0, atol=2e-15)
    sources_to_bind = ('src/gcr/practical_streaming_v10.py', 'scripts/profile_practical_streaming_v10.py',
                       'src/gcr/practical_joint_v9.py', 'src/gcr/practical_constrained_v8.py',
                       'src/gcr/practical_constrained_evaluation_v8.py', 'scripts/run_practical_joint_v9.py',
                       'scripts/run_practical_constrained_v8.py')
    initial = measurements['zero']
    result = {'study': 'sanw_practical_v10_streaming_implementation_validation', 'encoder': args.encoder,
              'scope': 'original1200_training_only_no_fit_no_development_no_test', 'config': asdict(config),
              'fixed_nonzero_probe_seed': 20261005, 'config_selected_from_profiling': False,
              'gallery_images': len(images), 'gallery_source_texts': len(source_rows),
              'all_training_text_rows': len(texts), 'eligible_joint_images': len(composition.eligible),
              'normalization': 'unchanged_v8_float32_torch_F_normalize_once_then_float64',
              'training_provenance': provenance, 'source_sha256': {name: sha(repo/name) for name in sources_to_bind},
              'environment': {'numpy': np.__version__, 'thread_env': {key: os.environ.get(key) for key in
                              ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
              'load_seconds': load_seconds, 'pca_seconds': pca_seconds, 'cache_create_open_seconds': cache_seconds,
              'streamed_constraint_init_seconds': constraints_seconds, 'dense_constraint_init_seconds': dense_init_seconds,
              'cache_metadata': cache.metadata, 'timings': measurements, 'equivalence': equivalence,
              'fixed_initial_gradient_multiplier': config.composition_weight * initial['retrieval_gradient_norm'] / max(initial['joint_gradient_norm'], 1e-12),
              'process_peak_rss_kib_after_streaming': streamed_peak_kib,
              'process_peak_rss_kib_after_dense_comparison': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'memory_note': 'Process peaks include frozen feature loading and PCA; dense comparison occurs after streaming timing.',
              'optimizer_steps_for6000_owners32epochs': 32*((6000+config.batch_size-1)//config.batch_size),
              'elapsed_seconds': time.monotonic()-overall_start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('encoder', 'gallery_images', 'gallery_source_texts', 'elapsed_seconds', 'equivalence')}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare or encode fresh1500 only after the immutable twelve-state source lock.

This new pipeline never mutates or reads the training6000 feature arrays. Raw
JPEGs are streamed from the pinned archive; exact text reuse keeps distinct IDs.
Numeric/model imports are deferred until after the authoritative access gate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

from fresh_confirmation_support import (
    ROOT, HERE, CONTRACT, OWNER_LOCK, INPUT_RECORDS, EXPECTED_RUNTIME, PREPROCESS,
    PINNED_ENCODING_SOURCES, EXPORT_SOURCE, SUPPORT_SOURCE, build_manifest, canonical_json,
    decode_verified_jpeg, load_owner_metadata, record, require_access, sha,
    validate_manifest, verify_record, verify_runtime,
)

MANIFEST_PATH = 'data/fresh_confirmation1500/manifest.json'
PREPARATION_PATH = 'data/fresh_confirmation1500/preparation_receipt.json'
ENCODING_STUDY = 'sanw_practical_v10_fresh_confirmation_encoding'


def load_source(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def lock_record(args):
    return record(ROOT, Path(args.state_source_lock).resolve())


def access(args):
    return require_access(args.state_source_lock, args.lock_sha256, root=ROOT)


def prepare(args):
    lock = access(args)
    owners, audit_rows = load_owner_metadata(ROOT, lock)
    output = ROOT / MANIFEST_PATH
    receipt_path = ROOT / PREPARATION_PATH
    if output.exists() or receipt_path.exists():
        raise FileExistsError('Fresh preparation already exists; never silently rebuild confirmation inputs')
    sources = {name: verify_record(ROOT, INPUT_RECORDS[name])
               for name in ('train_annotations', 'source_captions', 'image_archive')}
    parser_module = load_source('recovery/training_expansion_private/build_additional_train_manifest.py',
                                'sanw_fresh_pinned_caption_parser')
    manifest = build_manifest(owners, sources['train_annotations'], sources['source_captions'],
                              audit_rows, parser_module.clean_caption)
    counts = dict(sorted(Counter(row['relation'] for row in manifest['pairs']).items()))
    if len(manifest['images']) != 1500 or counts.get('source') != 7500:
        raise ValueError('Fresh population differs from locked contract')
    # Recheck the entire state/source qualification before committing a new input.
    if access(args) != lock:
        raise ValueError('State/source lock changed during preparation')
    canonical_json(output, manifest)
    receipt = {
        'study': 'sanw_practical_v10_fresh_confirmation_preparation', 'complete': True,
        'state_source_lock': lock_record(args), 'contract': CONTRACT, 'owner_lock': OWNER_LOCK,
        'raw_inputs': INPUT_RECORDS, 'manifest': record(ROOT, output),
        'source_sha256': {name: sha(ROOT / name) for name in
                          (EXPORT_SOURCE, SUPPORT_SOURCE, *PINNED_ENCODING_SOURCES)},
        'image_count': 1500, 'text_count': len(manifest['texts']), 'source_caption_count': 7500,
        'relation_counts': counts, 'owner_order': "owner_lock['image_ids']",
        'all_raw_labels_preserved': True, 'training_eligibility_filter_applied': False,
        'caption_rows_deduplicated': False, 'fit_or_selection_allowed': False,
        'jpeg_files_extracted': 0, 'features_encoded': False, 'outcomes_scored': False,
    }
    canonical_json(receipt_path, receipt)
    print(json.dumps({'preparation_receipt': record(ROOT, receipt_path),
                      'image_count': 1500, 'text_count': len(manifest['texts']), 'relation_counts': counts}), flush=True)


def validate_preparation(args, lock, owners):
    path = ROOT / PREPARATION_PATH
    receipt = json.loads(path.read_text())
    expected_source = {name: sha(ROOT / name) for name in (EXPORT_SOURCE, SUPPORT_SOURCE, *PINNED_ENCODING_SOURCES)}
    if (receipt.get('study') != 'sanw_practical_v10_fresh_confirmation_preparation'
            or receipt.get('complete') is not True or receipt.get('state_source_lock') != lock_record(args)
            or receipt.get('contract') != CONTRACT or receipt.get('owner_lock') != OWNER_LOCK
            or receipt.get('raw_inputs') != INPUT_RECORDS or receipt.get('source_sha256') != expected_source
            or receipt.get('all_raw_labels_preserved') is not True
            or receipt.get('training_eligibility_filter_applied') is not False
            or receipt.get('caption_rows_deduplicated') is not False
            or receipt.get('fit_or_selection_allowed') is not False
            or receipt.get('manifest', {}).get('path') != MANIFEST_PATH):
        raise ValueError('Fresh preparation identity differs')
    manifest = json.loads(verify_record(ROOT, receipt['manifest']).read_text())
    validate_manifest(manifest, owners)
    counts = dict(sorted(Counter(row['relation'] for row in manifest['pairs']).items()))
    if (receipt['image_count'] != len(manifest['images']) or receipt['text_count'] != len(manifest['texts'])
            or counts != receipt['relation_counts'] or receipt['source_caption_count'] != counts.get('source')
            or len(manifest['images']) != 1500 or counts.get('source') != 7500):
        raise ValueError('Fresh manifest structural counts differ')
    return manifest, receipt


class FeatureBank:
    """Separate resumable bank; exact strings share vectors but never row IDs."""
    def __init__(self, path, identity, dimension, np):
        self.np, self.dimension = np, dimension
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS features(kind TEXT,key TEXT,value BLOB,PRIMARY KEY(kind,key))')
        value = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if previous and previous[0] != value:
            self.db.close()
            raise ValueError('Fresh resumable-bank identity changed')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (value,))
        self.db.commit()

    def get(self, kind, key):
        row = self.db.execute('SELECT value FROM features WHERE kind=? AND key=?', (kind, key)).fetchone()
        return None if row is None else self.np.frombuffer(row[0], dtype=self.np.float32)

    def put(self, kind, key, value):
        np = self.np
        value = np.asarray(value, dtype=np.float32)
        if (value.shape != (self.dimension,) or not np.isfinite(value).all()
                or abs(float(np.linalg.norm(value)) - 1.) > 2e-4):
            raise ValueError('Nonfinite or unnormalized fresh vector')
        old = self.get(kind, key)
        if old is not None and not np.array_equal(old, value):
            raise ValueError('A previously committed fresh vector changed')
        self.db.execute('INSERT OR IGNORE INTO features VALUES (?,?,?)', (kind, key, value.tobytes()))

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()


def text_key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_export(images, texts, manifest, np, dimension):
    if (images.shape != (1500, dimension) or texts.shape != (len(manifest['texts']), dimension)
            or images.dtype != np.float32 or texts.dtype != np.float32):
        raise ValueError('Fresh feature shapes or float32 dtype differ')
    for values in (images, texts):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-4, rtol=2e-4):
            raise ValueError('Final fresh vectors are nonfinite or unnormalized')


def encode(args):
    lock = access(args)
    owners, audit_rows = load_owner_metadata(ROOT, lock)
    manifest, preparation = validate_preparation(args, lock, owners)
    out = ROOT / 'results/fresh_confirmation1500' / args.encoder
    if (out / 'completion.json').exists():
        raise FileExistsError('Completed fresh feature export exists; refuse overwrite')
    if args.threads != 4:
        raise ValueError('Use the pinned four-thread encoding recipe')
    # All numeric/model packages are loaded only after the final twelve-state gate.
    import numpy as np
    import torch
    import torchvision
    import open_clip
    import PIL
    runtime = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
               'open_clip': open_clip.__version__, 'numpy': np.__version__, 'Pillow': PIL.__version__}
    verify_runtime(runtime)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    started = time.monotonic()
    image_archive = verify_record(ROOT, INPUT_RECORDS['image_archive'])
    weight_entry = INPUT_RECORDS['vit_weights' if args.encoder == 'vit_b32' else 'rn50_weights']
    weight = verify_record(ROOT, weight_entry)
    helper = load_source('recovery/training_expansion_private/encode_expansion_pilot.py', 'sanw_fresh_pinned_text_helper')
    name, dimension = ('ViT-B-32', 512) if args.encoder == 'vit_b32' else ('RN50', 1024)
    identity = {
        'encoder': args.encoder, 'state_source_lock': lock_record(args), 'contract': CONTRACT,
        'owner_lock': OWNER_LOCK, 'preparation_receipt': record(ROOT, ROOT / PREPARATION_PATH),
        'manifest': record(ROOT, ROOT / MANIFEST_PATH), 'weights': weight_entry,
        'runtime': runtime, 'preprocess_config': PREPROCESS, 'dtype': 'float32', 'normalization': 'L2',
        'threads': 4, 'image_batch': 32, 'text_batch': 128,
        'text_encoding': 'batch_causally_trimmed_after_last_eot',
        'source_sha256': {path: sha(ROOT / path) for path in (EXPORT_SOURCE, SUPPORT_SOURCE, *PINNED_ENCODING_SOURCES)},
    }
    out.mkdir(parents=True, exist_ok=True)
    bank = FeatureBank(out / 'resumable_bank.sqlite', identity, dimension, np)
    pending_images = [row for row in manifest['images'] if bank.get('image', row['sha256']) is None]
    strings = {row['text'] for row in manifest['texts'] if bank.get('text', text_key(row['text'])) is None}
    pending_text = sorted(strings, key=lambda text: (len(text), text))
    canonical_json(out / 'encoding_lock.json', {'identity': identity, 'pending_images': len(pending_images),
                                               'pending_unique_texts': len(pending_text)}, overwrite=True)
    if args.encoder == 'rn50':
        from open_clip.openai import load_openai_model
        from open_clip.transform import image_transform
        model = load_openai_model(str(weight), precision='fp32', device='cpu')
        preprocess = image_transform(224, is_train=False, mean=tuple(PREPROCESS['mean']), std=tuple(PREPROCESS['std']),
                                     interpolation='bicubic', resize_mode='shortest')
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=str(weight), device='cpu', image_mean=tuple(PREPROCESS['mean']), image_std=tuple(PREPROCESS['std']))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(name)
    with torch.inference_mode(), zipfile.ZipFile(image_archive) as archive:
        for kind, rows, batch in [('image', pending_images, 32), ('text', pending_text, 128)]:
            for start in range(0, len(rows), batch):
                chosen = rows[start:start + batch]
                if kind == 'image':
                    tensors = []
                    for row in chosen:
                        audited = audit_rows[row['id']]
                        if any(row[key] != audited[key] for key in ('sha256', 'bytes', 'rgb_size_prefixed_sha256')):
                            raise ValueError('Manifest image differs from frozen content audit')
                        with decode_verified_jpeg(archive, row) as rgb:
                            tensors.append(preprocess(rgb))
                    values = model.encode_image(torch.stack(tensors), normalize=True).cpu().numpy()
                    keys = [row['sha256'] for row in chosen]
                else:
                    values = helper.short_text(model, tokenizer(chosen)).cpu().numpy()
                    keys = [text_key(text) for text in chosen]
                for key, value in zip(keys, values):
                    bank.put(kind, key, value)
                bank.commit()
                if start % (batch * 16) == 0 or start + batch >= len(rows):
                    progress = {'encoder': args.encoder, 'kind': kind, 'done': min(start + batch, len(rows)),
                                'total': len(rows), 'elapsed_seconds': time.monotonic() - started}
                    canonical_json(out / 'progress.json', progress, overwrite=True)
                    print(json.dumps(progress), flush=True)
    images = np.stack([bank.get('image', row['sha256']) for row in manifest['images']])
    texts = np.stack([bank.get('text', text_key(row['text'])) for row in manifest['texts']])
    validate_export(images, texts, manifest, np, dimension)
    if access(args) != lock or record(ROOT, ROOT / MANIFEST_PATH) != identity['manifest']:
        raise ValueError('Decision or manifest identity changed during encoding')
    tmp = out / 'features.tmp.npz'
    np.savez_compressed(tmp, image_features=images, text_features=texts,
                        image_ids=np.array([row['id'] for row in manifest['images']]),
                        text_ids=np.array([row['id'] for row in manifest['texts']]))
    tmp.replace(out / 'features.npz')
    model_repository = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K' if args.encoder == 'vit_b32' else 'openai/CLIP'
    model_revision = '1a25a446712ba5ee05982a381eed697ef9b435cf' if args.encoder == 'vit_b32' else weight_entry['sha256']
    metadata = {
        'schema_version': 1, 'dataset': manifest['dataset'], 'encoder_id': args.encoder,
        'manifest_sha256': identity['manifest']['sha256'], 'features_sha256': sha(out / 'features.npz'),
        'model_repository': model_repository, 'model_revision': model_revision, 'weights_sha256': weight_entry['sha256'],
        'image_count': 1500, 'text_count': len(texts), 'source_caption_count': 7500, 'dimension': dimension,
        'dtype': 'float32', 'normalization': 'L2', 'preprocess_config': PREPROCESS,
        'open_clip_version': open_clip.__version__, 'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'logit_scale': float(model.logit_scale.exp()), 'text_encoding': 'causally_trimmed_after_last_eot',
        'state_source_lock': identity['state_source_lock'], 'encoding_identity': identity,
        'owner_lock': OWNER_LOCK, 'relation_counts': preparation['relation_counts'],
        'all_rows_confirmation_only': True, 'fit_or_selection_allowed': False,
        'training_eligibility_filter_applied': False, 'caption_rows_deduplicated': False,
        'raw_jpeg_files_extracted': 0, 'training_feature_arrays_read': False, 'outcomes_scored': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    # A crash after metadata but before completion may be resumed from the bound
    # bank. Completion is the immutable commit marker and was rejected above.
    canonical_json(out / 'metadata.json', metadata, overwrite=True)
    bank.close()
    completion = {
        'study': ENCODING_STUDY, 'complete': True, 'encoder': args.encoder,
        'state_source_lock': identity['state_source_lock'],
        'preparation_receipt': record(ROOT, ROOT / PREPARATION_PATH), 'manifest': identity['manifest'],
        'features': record(ROOT, out / 'features.npz'), 'metadata': record(ROOT, out / 'metadata.json'),
        'owner_lock': OWNER_LOCK, 'image_count': 1500, 'text_count': len(texts), 'source_caption_count': 7500,
        'elapsed_seconds': metadata['elapsed_seconds'], 'outcomes_scored': False,
    }
    canonical_json(out / 'completion.json', completion)
    print(json.dumps(completion), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'encode'])
    parser.add_argument('--state-source-lock', required=True)
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--encoder', choices=['vit_b32', 'rn50'])
    parser.add_argument('--threads', type=int, choices=[4], default=4)
    args = parser.parse_args(argv)
    if args.operation == 'encode' and args.encoder is None:
        parser.error('--encoder is required for encode')
    if args.operation == 'prepare' and args.encoder is not None:
        parser.error('--encoder is only valid for encode')
    (prepare if args.operation == 'prepare' else encode)(args)


if __name__ == '__main__':
    main()

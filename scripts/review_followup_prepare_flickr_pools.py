#!/usr/bin/env python3
"""Acquire and encode new, source-only e-ViL dev900 and test1000 pools.

Original data, model, feature NPZs, and SQLite bank are read-only inputs. New
images/manifests/receipts/features stay within review_followup namespaces.
No adapters are applied and no retrieval metrics are computed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

import numpy as np

from prepare_visual_entailment import (
    acquire_image, read_annotations, strip_entity_markup, ANNOTATION_REVISION,
    CAPTION_REVISION, FLICKR_REVISION, EXPECTED_ZIP_ETAG, SOURCE_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/review_followup'
FEATURES = ROOT / 'results/review_followup/features'
AUDIT = ROOT / 'results/review_followup/data_audit'
OLD_DATA = ROOT / 'data/visual_entailment'
OLD_FEATURES = ROOT / 'results/features/visual_entailment'
POOLS = {'e_vil_dev900': ('validation', 900), 'e_vil_test1000': ('test', 1000)}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing follow-up artifact differs; preserved: {path}')
        return
    with path.open('xb') as stream:
        stream.write(content)


def source_inputs():
    names = ['scripts/review_followup_prepare_flickr_pools.py', 'scripts/prepare_visual_entailment.py',
             'scripts/extract_features.py', 'data/visual_entailment/selection.json',
             'data/visual_entailment/manifest.json', 'data/visual_entailment/provenance.json',
             'data/visual_entailment/raw/esnlive_dev.csv', 'data/visual_entailment/raw/esnlive_test.csv',
             'data/visual_entailment/raw/flickr30k_entities_annotations.zip',
             'data/visual_entailment/raw/flickr30k_zip_selected_index.json',
             'results/features/visual_entailment/features.npz', 'results/features/visual_entailment/metadata.json',
             'data/model/open_clip_config.json', 'data/model/open_clip_model.safetensors']
    return {name: {'bytes': (ROOT / name).stat().st_size, 'sha256': digest(ROOT / name)} for name in names}


def prepare(workers):
    selection = json.loads((OLD_DATA / 'selection.json').read_text())
    old_provenance = json.loads((OLD_DATA / 'provenance.json').read_text())
    old_manifest = json.loads((OLD_DATA / 'manifest.json').read_text())
    sources = source_inputs()
    for name in ('esnlive_dev.csv', 'esnlive_test.csv', 'flickr30k_entities_annotations.zip'):
        if digest(OLD_DATA / 'raw' / name) != SOURCE_SHA256[name]:
            raise ValueError(f'Pinned raw source hash mismatch: {name}')
    dev = set(read_annotations(OLD_DATA / 'raw/esnlive_dev.csv'))
    test = set(read_annotations(OLD_DATA / 'raw/esnlive_test.csv'))
    calibration = set(selection['images']['calibration'])
    names_by_pool = {'e_vil_dev900': sorted(dev - calibration), 'e_vil_test1000': sorted(test)}
    if len(dev) != 1000 or len(test) != 1000 or len(calibration) != 100 or not calibration <= dev:
        raise ValueError('Official dev/test/calibration split counts differ')
    if dev & test or set(selection['images']['train']) & (dev | test):
        raise ValueError('Official image-ID leakage')
    for name, (_, count) in POOLS.items():
        if len(names_by_pool[name]) != count:
            raise ValueError(f'Wrong candidate count: {name}')
    wanted = set.union(*(set(v) for v in names_by_pool.values()))
    captions = {}
    with zipfile.ZipFile(OLD_DATA / 'raw/flickr30k_entities_annotations.zip') as archive:
        for filename in sorted(wanted):
            member = 'Sentences/' + Path(filename).stem + '.txt'
            values = [strip_entity_markup(line) for line in archive.read(member).decode('utf-8').splitlines() if line.strip()]
            if len(values) != 5:
                raise ValueError(f'Wrong source-caption count: {filename}')
            captions[filename] = values
    index = json.loads((OLD_DATA / 'raw/flickr30k_zip_selected_index.json').read_text())
    if index['archive_etag'] != EXPECTED_ZIP_ETAG or not wanted <= set(index['images']):
        raise ValueError('Pinned image archive index mismatch')
    image_dir = DATA / 'flickr_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    receipts = {}
    missing = []
    # Submit development additions first; both pools still require the complete
    # cross-pool byte-disjointness audit before feature publication.
    ordered_wanted = list(dict.fromkeys(filename for names in names_by_pool.values() for filename in names))
    for filename in ordered_wanted:
        old = old_provenance['images'].get(filename)
        if old:
            path = ROOT / old['path']
            if not path.is_file() or path.stat().st_size != old['size_bytes'] or digest(path) != old['sha256']:
                raise ValueError(f'Original image bytes differ: {filename}')
            receipts[filename] = {**old, 'acquisition': 'reused_original_verified_image'}
        else:
            missing.append(filename)
    print(json.dumps({'phase': 'acquisition', 'wanted': len(wanted), 'reuse_original': len(receipts),
                      'new_namespace_images': len(missing), 'workers': workers}), flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(acquire_image, filename, index['images'][filename], image_dir, ROOT): filename for filename in missing}
        for done, future in enumerate(as_completed(pending), 1):
            filename = pending[future]
            try:
                receipts[filename] = {**future.result(), 'acquisition': 'verified_pinned_archive_in_new_namespace'}
            except Exception as error:
                errors.append({'filename': filename, 'exception': type(error).__name__, 'error': str(error)})
            if done % 100 == 0 or done == len(missing):
                print(json.dumps({'phase': 'acquisition', 'finished': done, 'total': len(missing), 'errors': len(errors)}), flush=True)
    if errors:
        path = AUDIT / f'acquisition_failures_{time.time_ns()}.json'
        dump(path, errors)
        raise RuntimeError(f'{len(errors)} image acquisitions failed; see {path}')
    groups = {f'original_{split}': set(names) for split, names in selection['images'].items()}
    groups.update({name: set(names) for name, names in names_by_pool.items()})
    all_receipts = {**old_provenance['images'], **receipts}
    hashes = {name: {all_receipts[filename]['sha256'] for filename in members} for name, members in groups.items()}
    intersections = {}
    for a in groups:
        for b in groups:
            if a >= b:
                continue
            intersections[a + '__' + b] = {'image_ids': sorted(groups[a] & groups[b]),
                                            'image_id_count': len(groups[a] & groups[b]),
                                            'image_byte_sha256_count': len(hashes[a] & hashes[b]),
                                            'image_byte_sha256': sorted(hashes[a] & hashes[b])}
    forbidden = [('e_vil_dev900', 'original_train'), ('e_vil_dev900', 'original_calibration'),
                 ('e_vil_dev900', 'original_test'), ('e_vil_test1000', 'original_train'),
                 ('e_vil_test1000', 'original_calibration'), ('e_vil_test1000', 'original_validation'),
                 ('e_vil_dev900', 'e_vil_test1000')]
    violations = [{'left': a, 'right': b, 'ids': sorted(groups[a] & groups[b]),
                   'byte_hashes': sorted(hashes[a] & hashes[b])} for a, b in forbidden
                  if groups[a] & groups[b] or hashes[a] & hashes[b]]
    duplicates = {}
    for name, members in groups.items():
        by_hash = defaultdict(list)
        for filename in members:
            by_hash[all_receipts[filename]['sha256']].append(filename)
        duplicates[name] = {key: sorted(value) for key, value in by_hash.items() if len(value) > 1}
    split_audit = {'schema_version': 1, 'status': 'passed' if not violations else 'failed',
                   'definition': 'Official pinned e-ViL dev/test image pools; source captions from pinned Flickr30k Entities.',
                   'not_standard_karpathy_flickr1000': True,
                   'groups': {name: {'image_count': len(value), 'unique_image_byte_hashes': len(hashes[name])} for name, value in groups.items()},
                   'intersections': intersections, 'within_group_byte_duplicates': duplicates,
                   'forbidden_overlap_violations': violations,
                   'intentional_overlap': 'dev900 includes original validation100; test1000 includes original test400. Calibration100 is excluded from both.',
                   'inputs': sources}
    dump(AUDIT / 'split_disjointness.json', split_audit)
    if violations:
        raise ValueError('Forbidden ID or exact-image-byte overlap; no feature encoding permitted')
    old_text = {row['id']: row['text'] for row in old_manifest['texts']}
    manifests = {}
    for name, (split, _) in POOLS.items():
        images, texts, pairs = [], [], []
        for filename in names_by_pool[name]:
            stem = Path(filename).stem
            iid = 'flickr:' + stem
            receipt = receipts[filename]
            original_split = 'validation' if split == 'validation' else 'test'
            was_original = filename in set(selection['images'][original_split])
            subgroup = ('original_validation100' if was_original else 'additional_validation800') if split == 'validation' else ('original_test400' if was_original else 'additional_test600')
            images.append({'id': iid, 'path': receipt['path'], 'split': split, 'review_subgroup': subgroup,
                           'sha256': receipt['sha256'], 'bytes': receipt['size_bytes']})
            for i, caption in enumerate(captions[filename]):
                tid = f'source:{stem}:{i}'
                if tid in old_text and old_text[tid] != caption:
                    raise ValueError(f'Caption differs from original frozen text: {tid}')
                texts.append({'id': tid, 'text': caption})
                pairs.append({'image_id': iid, 'text_id': tid, 'relation': 'source'})
        manifest = {'schema_version': 1, 'dataset': name, 'images': images, 'texts': texts, 'pairs': pairs, 'triplets': []}
        path = DATA / name / 'manifest.json'
        dump(path, manifest)
        dump(path.with_name('provenance.json'), {'schema_version': 1, 'scope': name,
             'annotation_revision': ANNOTATION_REVISION, 'caption_revision': CAPTION_REVISION,
             'image_revision': FLICKR_REVISION, 'archive_etag': EXPECTED_ZIP_ETAG,
             'selection': 'All official e-ViL dev images except the original calibration100' if split == 'validation' else 'All official e-ViL test images',
             'source_caption_order': 'Five original Entities sentence lines; exact existing strip_entity_markup parser.',
             'images': {filename: receipts[filename] for filename in names_by_pool[name]},
             'manifest_sha256': digest(path), 'inputs': sources,
             'split_audit': {'path': str((AUDIT / 'split_disjointness.json').relative_to(ROOT)), 'sha256': digest(AUDIT / 'split_disjointness.json')}})
        manifests[name] = manifest
        print(json.dumps({'phase': 'manifest_complete', 'pool': name, 'images': len(images), 'texts': len(texts), 'sha256': digest(path)}), flush=True)
    return manifests


def encode(manifests, threads):
    import torch
    import open_clip
    from PIL import Image
    from extract_features import short_text, WEIGHT_SHA
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if json.loads((AUDIT / 'split_disjointness.json').read_text())['status'] != 'passed':
        raise ValueError('Passed split audit required')
    oldmeta = json.loads((OLD_FEATURES / 'metadata.json').read_text())
    weight = ROOT / 'data/model/open_clip_model.safetensors'
    config_path = weight.with_name('open_clip_config.json')
    if digest(weight) != WEIGHT_SHA or digest(OLD_FEATURES / 'features.npz') != oldmeta['features_sha256']:
        raise ValueError('Frozen weight or feature bytes differ')
    config = json.loads(config_path.read_text())
    prep = config.get('preprocess_cfg', {})
    if prep != oldmeta['preprocess_config'] or open_clip.__version__ != oldmeta['open_clip_version']:
        raise ValueError('Encoder or preprocessing configuration differs from original')
    db = sqlite3.connect('file:' + str(ROOT / 'results/features/bank.sqlite') + '?mode=ro', uri=True)
    expected_signature = json.dumps({'weight': WEIGHT_SHA, 'config': digest(config_path), 'open_clip': open_clip.__version__}, sort_keys=True)
    if db.execute("SELECT value FROM metadata WHERE key='encoder'").fetchone()[0] != expected_signature:
        raise ValueError('Read-only feature-bank encoder signature differs')
    with np.load(OLD_FEATURES / 'features.npz', allow_pickle=False) as data:
        old_images = {str(key): value.copy() for key, value in zip(data['image_ids'], data['image_features'])}
        old_texts = {str(key): value.copy() for key, value in zip(data['text_ids'], data['text_features'])}
    cache = {'image': {}, 'text': {}}
    status = {}
    pending = {'image': {}, 'text': {}}
    for name, manifest in manifests.items():
        rows = {'image': manifest['images'], 'text': manifest['texts']}
        for kind, items in rows.items():
            for item in items:
                key = item['sha256'] if kind == 'image' else hashlib.sha256(item['text'].encode()).hexdigest()
                token = (kind, item['id'])
                originals = old_images if kind == 'image' else old_texts
                if item['id'] in originals:
                    value = originals[item['id']]
                    if key in cache[kind] and not np.array_equal(cache[kind][key], value):
                        raise ValueError('Original duplicate-content vectors differ')
                    cache[kind][key] = value
                    status[token] = 'original_npz_exact'
                elif key in cache[kind]:
                    status[token] = 'reused_identical_content'
                else:
                    found = db.execute('SELECT value FROM features WHERE kind=? AND key=?', (kind, key)).fetchone()
                    if found:
                        value = np.frombuffer(found[0], dtype=np.float32).copy()
                        if value.shape != (512,):
                            raise ValueError('Malformed feature bank row')
                        cache[kind][key] = value
                        status[token] = 'readonly_bank_exact'
                    else:
                        pending[kind].setdefault(key, item)
                        status[token] = 'new_encoding'
    db.close()
    # A later existing ID can supply a key queued earlier under a different ID.
    for kind in pending:
        pending[kind] = {key: row for key, row in pending[kind].items() if key not in cache[kind]}
    print(json.dumps({'phase': 'encoding', 'new_unique_images': len(pending['image']), 'new_unique_texts': len(pending['text']), 'threads': threads}), flush=True)
    kwargs = {'image_' + key: prep[key] for key in ('mean', 'std', 'interpolation', 'resize_mode') if key in prep}
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=str(weight), **kwargs)
    model.eval()
    model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    parity = []
    with torch.inference_mode():
        for name, manifest in manifests.items():
            candidates = manifest['texts']
            indices = sorted(set([0, len(candidates) // 2, len(candidates) - 1] + list(range(min(20, len(candidates))))))
            tokens = tokenizer([candidates[i]['text'] for i in indices])
            error = float((model.encode_text(tokens, normalize=True) - short_text(model, tokens)).abs().max())
            if error > 2e-6:
                raise ValueError('Causal-padding parity failed')
            parity.append({'pool': name, 'max_abs_error': error, 'tolerance': 2e-6, 'examples': len(indices)})
        for kind, batch_size in (('image', 32), ('text', 128)):
            items = list(pending[kind].items())
            if kind == 'text':
                items.sort(key=lambda pair: len(pair[1]['text']))
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                if kind == 'image':
                    tensors = []
                    for _, item in batch:
                        with Image.open(ROOT / item['path']) as image:
                            tensors.append(preprocess(image.convert('RGB')))
                    values = model.encode_image(torch.stack(tensors), normalize=True)
                else:
                    values = short_text(model, tokenizer([item['text'] for _, item in batch]))
                values = values.cpu().numpy().astype(np.float32)
                if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5):
                    raise ValueError('New feature vectors invalid')
                for (key, _), value in zip(batch, values):
                    cache[kind][key] = value.copy()
                if start // batch_size % 10 == 0 or start + batch_size >= len(items):
                    print(json.dumps({'phase': 'encoding', 'kind': kind, 'complete': min(start + batch_size, len(items)), 'total': len(items)}), flush=True)
    for name, manifest in manifests.items():
        output = FEATURES / name
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'features.npz'
        if path.exists():
            raise FileExistsError(f'Feature output already exists, preserved: {path}')
        vectors = {}
        reuse = {}
        exact_checks = {}
        for kind in ('image', 'text'):
            items = manifest['images' if kind == 'image' else 'texts']
            keys = [row['sha256'] if kind == 'image' else hashlib.sha256(row['text'].encode()).hexdigest() for row in items]
            vectors[kind + '_features'] = np.stack([cache[kind][key] for key in keys]).astype(np.float32)
            vectors[kind + '_ids'] = np.asarray([row['id'] for row in items])
            originals = old_images if kind == 'image' else old_texts
            reused = [(i, row['id']) for i, row in enumerate(items) if row['id'] in originals]
            if not all(np.array_equal(vectors[kind + '_features'][i], originals[identifier]) for i, identifier in reused):
                raise ValueError('Original feature row changed on reuse')
            exact_checks[kind] = len(reused)
            reuse[kind] = dict((label, sum(status[(kind, row['id'])] == label for row in items))
                              for label in ('original_npz_exact', 'readonly_bank_exact', 'reused_identical_content', 'new_encoding'))
        temporary = output / 'features.partial.npz'
        np.savez_compressed(temporary, **vectors)
        temporary.replace(path)
        meta = {'schema_version': 1, 'dataset': name, 'manifest_sha256': digest(DATA / name / 'manifest.json'),
                'features_sha256': digest(path), 'image_count': len(manifest['images']), 'text_count': len(manifest['texts']),
                'dimension': 512, 'dtype': 'float32', 'normalization': 'L2', 'model_repository': oldmeta['model_repository'],
                'model_revision': oldmeta['model_revision'], 'weights_sha256': WEIGHT_SHA, 'logit_scale': oldmeta['logit_scale'],
                'preprocess_config': prep, 'open_clip_version': open_clip.__version__, 'torch_version': torch.__version__,
                'text_encoding': 'causally_trimmed_after_last_eot', 'causal_padding_parity': parity,
                'source_sha256': digest(Path(__file__)), 'source_feature_metadata_sha256': digest(OLD_FEATURES / 'metadata.json'),
                'reuse_counts': reuse, 'original_npz_rows_byte_exact_verified': exact_checks,
                'input_order': 'Exact manifest image/text order', 'no_adapter_applied': True, 'no_retrieval_scoring_performed': True,
                'split_audit': {'path': str((AUDIT / 'split_disjointness.json').relative_to(ROOT)), 'sha256': digest(AUDIT / 'split_disjointness.json')}}
        dump(output / 'metadata.json', meta)
        print(json.dumps({'phase': 'features_complete', 'pool': name, 'bytes': path.stat().st_size,
                          'sha256': meta['features_sha256'], 'reuse': reuse}), flush=True)
    originals = json.loads((AUDIT / 'split_disjointness.json').read_text())['inputs']
    checks = {name: digest(ROOT / name) == pin['sha256'] for name, pin in originals.items()}
    if not all(checks.values()):
        raise ValueError('Frozen source input changed')
    dump(AUDIT / 'construction_audit.json', {'status': 'passed', 'original_inputs_unchanged': checks,
         'feature_pools': {name: {'manifest_sha256': digest(DATA / name / 'manifest.json'),
                                'features_sha256': digest(FEATURES / name / 'features.npz'),
                                'metadata_sha256': digest(FEATURES / name / 'metadata.json')} for name in POOLS},
         'new_unique_image_encodings': len(pending['image']), 'new_unique_text_encodings': len(pending['text']),
         'source_sha256': digest(Path(__file__)), 'no_adapter_applied': True, 'no_retrieval_scoring_performed': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'encode', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    if args.phase in ('prepare', 'all'):
        manifests = prepare(args.workers)
    else:
        manifests = {name: json.loads((DATA / name / 'manifest.json').read_text()) for name in POOLS}
    if args.phase in ('encode', 'all'):
        encode(manifests, args.threads)
    print(json.dumps({'status': 'complete', 'phase': args.phase, 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()

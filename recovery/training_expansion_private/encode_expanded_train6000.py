#!/usr/bin/env python3
"""Resumable fixed-backbone encoding; preserve original train vectors verbatim."""
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import time
import numpy as np
import torch
import open_clip
from PIL import Image
from encode_expansion_pilot import short_text

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['vit_b32', 'rn50'], required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    started = time.monotonic()
    root = ROOT / 'results/official_train_expansion' / args.encoder
    out = root / 'train_6000'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion.json').exists():
        raise FileExistsError('Completed expanded cache already exists; do not overwrite.')
    manifest_path = ROOT / 'data/official_train_expansion/train_6000/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sample_path = HERE / 'FIXED_6000_TRAIN_SAMPLE.json'
    sample = json.loads(sample_path.read_text())
    old_manifest_path = ROOT / 'data/visual_entailment/manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    old_ii = [k for k, row in enumerate(old_manifest['images']) if row['split'] == 'train']
    old_ids = {old_manifest['images'][k]['id'] for k in old_ii}
    new_ids = set(sample['additional_image_ids'])
    expected_ids = old_ids | new_ids
    if len(old_ids) != 1200 or len(new_ids) != 4800 or old_ids & new_ids:
        raise ValueError('Training owner lock differs.')
    if len(manifest['images']) != 6000 or {x['id'] for x in manifest['images']} != expected_ids:
        raise ValueError('Train-only manifest differs from locked owner sample.')
    if any(row['split'] != 'train' for row in manifest['images']):
        raise ValueError('Nontraining owner in expanded fitting manifest.')
    fresh_ids = set(json.loads((HERE / 'FRESH_CONFIRMATION_1500_OWNER_LOCK.json').read_text())['image_ids'])
    if expected_ids & fresh_ids:
        raise ValueError('Fresh confirmation owner in training cache.')
    old_tid_set = {x['text_id'] for x in old_manifest['pairs'] if x['image_id'] in old_ids}
    old_ti = [k for k, row in enumerate(old_manifest['texts']) if row['id'] in old_tid_set]
    pilot_path = root / 'pilot100/receipt.json'
    pilot = json.loads(pilot_path.read_text())
    if not pilot['anchor_parity_passed'] or pilot['sample_sha256'] != sha(sample_path):
        raise ValueError('Successful exact-sample pilot with anchors is required.')
    assets = ROOT / 'data/official_train_expansion/assets'
    image_audit_path = HERE / 'IMAGE_DECODING_AND_CONTENT_AUDIT.json'
    audit = json.loads(image_audit_path.read_text())
    if not audit['passed_no_reserved_content_overlap']:
        raise ValueError('Image integrity audit failed.')
    audit_rows = {x['image_id']: x for x in audit['rows']}
    if args.encoder == 'vit_b32':
        name, dim = 'ViT-B-32', 512
        old_root = ROOT / 'results/resume_features/visual_entailment'
        weight = assets / 'open_clip_model.safetensors'
        weight_sha = 'ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6'
        model_repository = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
        revision = '1a25a446712ba5ee05982a381eed697ef9b435cf'
    else:
        name, dim = 'RN50', 1024
        old_root = ROOT / 'results/strengthen_second_encoder/features/visual_entailment'
        weight = assets / 'RN50.pt'
        weight_sha = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
        model_repository, revision = 'openai/CLIP', weight_sha
    if sha(weight) != weight_sha or pilot['weights_sha256'] != weight_sha:
        raise ValueError('Backbone identity mismatch.')
    old_metadata_path = old_root / 'metadata.json'
    old_features_path = old_root / 'features.npz'
    old_metadata = json.loads(old_metadata_path.read_text())
    old_feature_sha = sha(old_features_path)
    if old_metadata['features_sha256'] != old_feature_sha or old_metadata['weights_sha256'] != weight_sha:
        raise ValueError('Original feature-cache provenance mismatch.')
    with np.load(old_features_path, allow_pickle=False) as z:
        old_images = np.array(z['image_features'][old_ii], dtype=np.float32, copy=True)
        old_texts = np.array(z['text_features'][old_ti], dtype=np.float32, copy=True)
    original_images = {old_manifest['images'][j]['id']: old_images[k] for k, j in enumerate(old_ii)}
    original_texts = {old_manifest['texts'][j]['id']: old_texts[k] for k, j in enumerate(old_ti)}
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    preprocess_cfg = {'size': 224, 'mode': 'RGB', 'interpolation': 'bicubic',
                      'resize_mode': 'shortest', 'crop': 'center', 'mean': list(mean), 'std': list(std)}
    identity = {'encoder': args.encoder, 'weights_sha256': weight_sha,
                'torch': torch.__version__, 'open_clip': open_clip.__version__, 'numpy': np.__version__,
                'preprocess_config': preprocess_cfg, 'text_encoding': 'batch_causally_trimmed_after_last_eot',
                'image_batch': 32, 'text_batch': 128, 'threads': args.threads,
                'source_sha256': sha(Path(__file__)), 'suffix_source_sha256': sha(HERE / 'encode_expansion_pilot.py'),
                'manifest_sha256': sha(manifest_path), 'sample_sha256': sha(sample_path),
                'pilot_receipt_sha256': sha(pilot_path), 'original_features_sha256': old_feature_sha}
    identity_text = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    db = sqlite3.connect(out / 'resumable_bank.sqlite', timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS features(kind TEXT,key TEXT,value BLOB,PRIMARY KEY(kind,key))')
    previous = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if previous and previous[0] != identity_text:
        raise ValueError('Resumable-bank identity mismatch.')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (identity_text,))
    db.commit()
    def get(kind, key):
        row = db.execute('SELECT value FROM features WHERE kind=? AND key=?', (kind, key)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)
    def put(kind, key, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (dim,) or not np.isfinite(value).all() or abs(np.linalg.norm(value) - 1) > 2e-4:
            raise ValueError('Invalid feature vector.')
        old = get(kind, key)
        if old is not None and not np.array_equal(old, value):
            if float(np.max(np.abs(old - value))) > 2e-6:
                raise ValueError('Inconsistent feature for shared content key.')
            return
        db.execute('INSERT OR IGNORE INTO features VALUES (?,?,?)', (kind, key, value.tobytes()))
    def textkey(text):
        return hashlib.sha256(text.encode()).hexdigest()
    # Preserve every old row verbatim at export. Seed content reuse with its first
    # old occurrence, validating duplicate-content consistency independently.
    for k, j in enumerate(old_ii):
        put('image', audit_rows[old_manifest['images'][j]['id']]['sha256'], old_images[k])
    for k, j in enumerate(old_ti):
        put('text', textkey(old_manifest['texts'][j]['text']), old_texts[k])
    pilot_manifest = json.loads((ROOT / 'data/official_train_expansion/pilot100_manifest.json').read_text())
    with np.load(root / 'pilot100/features.npz', allow_pickle=False) as z:
        if sha(root / 'pilot100/features.npz') != pilot['features_sha256']:
            raise ValueError('Pilot feature archive changed.')
        for k, row in enumerate(pilot_manifest['images']):
            put('image', audit_rows[row['id']]['sha256'], z['image_features'][k])
        for k, row in enumerate(pilot_manifest['texts']):
            put('text', textkey(row['text']), z['text_features'][k])
    db.commit()
    pending_images = [(audit_rows[row['id']]['sha256'], row) for row in manifest['images']
                      if row['id'] in new_ids and get('image', audit_rows[row['id']]['sha256']) is None]
    unique_text = {}
    for row in manifest['texts']:
        key = textkey(row['text'])
        if get('text', key) is None:
            unique_text.setdefault(key, row)
    pending_text = sorted(unique_text.items(), key=lambda pair: (len(pair[1]['text']), pair[1]['text']))
    atomic_json(out / 'encoding_lock.json', {'identity': identity, 'pending_images': len(pending_images),
                                          'pending_unique_texts': len(pending_text), 'fresh_confirmation_rows': 0})
    print(json.dumps({'encoder': args.encoder, 'pending_images': len(pending_images),
                      'pending_unique_texts': len(pending_text)}), flush=True)
    if args.encoder == 'rn50':
        from open_clip.openai import load_openai_model
        from open_clip.transform import image_transform
        model = load_openai_model(str(weight), precision='fp32', device='cpu')
        preprocess = image_transform(224, is_train=False, mean=mean, std=std, interpolation='bicubic', resize_mode='shortest')
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=str(weight), device='cpu', image_mean=mean, image_std=std)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(name)
    with torch.inference_mode():
        for kind, rows, batch in [('image', pending_images, 32), ('text', pending_text, 128)]:
            stage = time.monotonic()
            for pos in range(0, len(rows), batch):
                chosen = rows[pos:pos+batch]
                if kind == 'image':
                    tensors = []
                    for key, row in chosen:
                        path = ROOT / row['path']
                        if sha(path) != key:
                            raise ValueError('Training image changed after integrity audit.')
                        with Image.open(path) as im:
                            tensors.append(preprocess(im.convert('RGB')))
                    values = model.encode_image(torch.stack(tensors), normalize=True).cpu().numpy()
                else:
                    values = short_text(model, tokenizer([row['text'] for _, row in chosen])).cpu().numpy()
                for (key, _), value in zip(chosen, values):
                    put(kind, key, value)
                db.commit()
                if pos % (batch * 16) == 0 or pos + batch >= len(rows):
                    progress = {'encoder': args.encoder, 'kind': kind, 'done': min(pos+batch, len(rows)),
                                'total': len(rows), 'stage_seconds': time.monotonic()-stage,
                                'elapsed_seconds': time.monotonic()-started}
                    atomic_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
    images = np.stack([original_images[row['id']] if row['id'] in original_images
                       else get('image', audit_rows[row['id']]['sha256']) for row in manifest['images']])
    texts = np.stack([original_texts[row['id']] if row['id'] in original_texts
                      else get('text', textkey(row['text'])) for row in manifest['texts']])
    image_ids = np.array([row['id'] for row in manifest['images']])
    text_ids = np.array([row['id'] for row in manifest['texts']])
    i_lookup, t_lookup = {v: i for i, v in enumerate(image_ids)}, {v: i for i, v in enumerate(text_ids)}
    if not np.array_equal(images[[i_lookup[old_manifest['images'][j]['id']] for j in old_ii]], old_images):
        raise ValueError('Old image rows changed.')
    if not np.array_equal(texts[[t_lookup[old_manifest['texts'][j]['id']] for j in old_ti]], old_texts):
        raise ValueError('Old text rows changed.')
    if images.shape != (6000, dim) or texts.shape != (len(manifest['texts']), dim):
        raise ValueError('Export shape mismatch.')
    for value in (images, texts):
        if not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=2e-4, rtol=2e-4):
            raise ValueError('Export nonfinite or unnormalized.')
    temp = out / 'features.tmp.npz'
    np.savez_compressed(temp, image_features=images, text_features=texts, image_ids=image_ids, text_ids=text_ids)
    temp.replace(out / 'features.npz')
    metadata = {'schema_version': 1, 'dataset': manifest['dataset'], 'encoder_id': args.encoder,
                'manifest_sha256': sha(manifest_path), 'features_sha256': sha(out / 'features.npz'),
                'model_repository': model_repository, 'model_revision': revision, 'weights_sha256': weight_sha,
                'image_count': len(images), 'text_count': len(texts), 'dimension': dim, 'dtype': 'float32',
                'normalization': 'L2', 'preprocess_config': preprocess_cfg,
                'open_clip_version': open_clip.__version__, 'torch_version': torch.__version__,
                'numpy_version': np.__version__, 'logit_scale': float(model.logit_scale.exp()),
                'text_encoding': 'causally_trimmed_after_last_eot', 'original_features_sha256': old_feature_sha,
                'original_manifest_sha256': sha(old_manifest_path), 'original_metadata_sha256': sha(old_metadata_path),
                'old_image_rows_bitwise_preserved': len(old_images), 'old_text_rows_bitwise_preserved': len(old_texts),
                'sample_lock_sha256': sha(sample_path), 'fresh_confirmation_lock_sha256': sha(HERE / 'FRESH_CONFIRMATION_1500_OWNER_LOCK.json'),
                'image_integrity_audit_sha256': sha(image_audit_path), 'pilot_receipt_sha256': sha(pilot_path),
                'source_sha256': sha(Path(__file__)), 'encoding_identity': identity,
                'all_rows_training_only': True, 'fresh_confirmation_rows': 0,
                'elapsed_seconds': time.monotonic()-started}
    atomic_json(out / 'metadata.json', metadata)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    atomic_json(out / 'completion.json', {'complete': True, 'metadata_sha256': sha(out / 'metadata.json'),
                                        'features_sha256': metadata['features_sha256'], 'elapsed_seconds': metadata['elapsed_seconds']})
    print(json.dumps({'encoder': args.encoder, 'complete': True, 'elapsed_seconds': metadata['elapsed_seconds'],
                      'features_sha256': metadata['features_sha256']}), flush=True)

if __name__ == '__main__':
    main()

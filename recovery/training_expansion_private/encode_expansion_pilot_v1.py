#!/usr/bin/env python3
"""Measure frozen encoding for fixed 100 training owners and old-cache anchors."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def short_text(model, tokens):
    end = tokens.argmax(dim=-1)
    length = int(end.max()) + 1
    tokens = tokens[:, :length]
    dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(tokens).to(dtype)
    x = x + model.positional_embedding[:length].to(dtype)
    x = model.transformer(x, attn_mask=model.attn_mask[:length, :length])
    x = model.ln_final(x)[torch.arange(len(tokens)), end]
    if model.text_projection is not None:
        x = model.text_projection(x) if isinstance(model.text_projection, torch.nn.Linear) else x @ model.text_projection
    return F.normalize(x, dim=-1)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder', choices=['vit_b32', 'rn50'], required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    out = ROOT / 'results/official_train_expansion' / args.encoder / 'pilot100'
    if out.exists():
        raise FileExistsError('Pilot output already exists; do not overwrite measured evidence.')
    out.mkdir(parents=True)
    assets = ROOT / 'data/official_train_expansion/assets'
    if args.encoder == 'vit_b32':
        name, dim = 'ViT-B-32', 512
        weight = assets / 'open_clip_model.safetensors'
        weight_sha = 'ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6'
        feature_path = ROOT / 'results/resume_features/visual_entailment/features.npz'
        kwargs = {'image_mean': (0.48145466, 0.4578275, 0.40821073),
                  'image_std': (0.26862954, 0.26130258, 0.27577711)}
    else:
        name, dim = 'RN50', 1024
        weight = assets / 'RN50.pt'
        weight_sha = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
        feature_path = ROOT / 'results/strengthen_second_encoder/features/visual_entailment/features.npz'
        kwargs = {}
    if sha(weight) != weight_sha:
        raise ValueError('Pinned model weights mismatch')
    image_audit = json.loads((HERE / 'IMAGE_DECODING_AND_CONTENT_AUDIT.json').read_text())
    if not image_audit['passed_no_reserved_content_overlap']:
        raise ValueError('Image audit failed')
    started = time.monotonic()
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=str(weight), device='cpu', **kwargs)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(name)
    setup_seconds = time.monotonic() - started
    manifest_path = ROOT / 'data/official_train_expansion/pilot100_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    locked = set(json.loads((HERE / 'FIXED_6000_TRAIN_SAMPLE.json').read_text())['pilot_image_ids'])
    if {x['id'] for x in manifest['images']} != locked or len(locked) != 100:
        raise ValueError('Pilot owners differ from predeclared sample')
    confirmation = set(json.loads((HERE / 'FRESH_CONFIRMATION_1500_OWNER_LOCK.json').read_text())['image_ids'])
    if confirmation & locked:
        raise ValueError('Pilot overlaps fresh confirmation')
    def encode_images(rows, archive=None):
        values = []
        for start in range(0, len(rows), 32):
            tensors = []
            for row in rows[start:start+32]:
                if archive is None:
                    image = Image.open(ROOT / row['path'])
                else:
                    member = 'flickr30k-images/' + row['id'].split(':', 1)[1] + '.jpg'
                    image = Image.open(io.BytesIO(archive.read(member)))
                with image:
                    tensors.append(preprocess(image.convert('RGB')))
            values.append(model.encode_image(torch.stack(tensors), normalize=True).cpu().numpy().astype(np.float32))
        return np.concatenate(values)
    def encode_texts(rows):
        unique = sorted({x['text'] for x in rows}, key=lambda x: (len(x), x))
        values = []
        for start in range(0, len(unique), 128):
            values.append(short_text(model, tokenizer(unique[start:start+128])).cpu().numpy().astype(np.float32))
        values = np.concatenate(values)
        lookup = {text: k for k, text in enumerate(unique)}
        return values[[lookup[x['text']] for x in rows]], len(unique)
    with torch.inference_mode():
        tokens = tokenizer([x['text'] for x in manifest['texts'][:22]])
        padding_error = float((model.encode_text(tokens, normalize=True) - short_text(model, tokens)).abs().max())
        if padding_error > 2e-6:
            raise ValueError('Causal text truncation parity failed')
        started = time.monotonic()
        images = encode_images(manifest['images'])
        image_seconds = time.monotonic() - started
        started = time.monotonic()
        texts, unique_texts = encode_texts(manifest['texts'])
        text_seconds = time.monotonic() - started
        old = json.loads((ROOT / 'data/visual_entailment/manifest.json').read_text())
        train_rows = [(i, x) for i, x in enumerate(old['images']) if x['split'] == 'train']
        image_anchors = train_rows[:20]
        anchor_iids = {x['id'] for _, x in train_rows}
        train_tids = {x['text_id'] for x in old['pairs'] if x['image_id'] in anchor_iids}
        text_anchors = [(i, x) for i, x in enumerate(old['texts']) if x['id'] in train_tids][:128]
        with zipfile.ZipFile(assets / 'flickr30k-images.zip') as z:
            ai = encode_images([x for _, x in image_anchors], z)
        at, _ = encode_texts([x for _, x in text_anchors])
        with np.load(feature_path, allow_pickle=False) as z:
            image_anchor_error = float(np.max(np.abs(ai - z['image_features'][[i for i, _ in image_anchors]])))
            text_anchor_error = float(np.max(np.abs(at - z['text_features'][[i for i, _ in text_anchors]])))
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError('Nonfinite pilot encoding')
    np.savez_compressed(out / 'features.npz', image_features=images, text_features=texts,
                        image_ids=np.array([x['id'] for x in manifest['images']]),
                        text_ids=np.array([x['id'] for x in manifest['texts']]))
    receipt = {'schema': 'sanw-expansion-cold-encoding-pilot-v1', 'encoder': args.encoder,
               'weights_sha256': weight_sha, 'manifest_sha256': sha(manifest_path),
               'sample_sha256': sha(HERE / 'FIXED_6000_TRAIN_SAMPLE.json'),
               'source_sha256': sha(Path(__file__)), 'torch': torch.__version__,
               'open_clip': open_clip.__version__, 'numpy': np.__version__, 'threads': args.threads,
               'image_batch': 32, 'text_batch': 128, 'setup_seconds': setup_seconds,
               'images': len(images), 'text_rows': len(texts), 'unique_text_strings': unique_texts,
               'image_seconds': image_seconds, 'text_seconds': text_seconds,
               'preprocess_repr': str(preprocess), 'logit_scale': float(model.logit_scale.exp()),
               'text_encoding': 'batch_causally_trimmed_after_last_eot',
               'padding_parity_max_error': padding_error,
               'anchor_images': len(image_anchors), 'anchor_texts': len(text_anchors),
               'old_image_anchor_max_abs_error': image_anchor_error,
               'old_text_anchor_max_abs_error': text_anchor_error,
               'anchor_tolerance': 2e-6,
               'anchor_parity_passed': max(image_anchor_error, text_anchor_error) <= 2e-6,
               'features_sha256': sha(out / 'features.npz'),
               'confirmation_images_encoded': 0, 'benchmark_outcomes_read': False}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    if not receipt['anchor_parity_passed']:
        raise ValueError('Old-cache anchor parity failed: do not concatenate new and old encodings.')

if __name__ == '__main__':
    main()

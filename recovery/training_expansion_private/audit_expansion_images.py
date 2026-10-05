#!/usr/bin/env python3
"""Decode selected training and reserved-confirmation images without features."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import struct
import time
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def main():
    start = time.monotonic()
    sample = json.loads((HERE / 'FIXED_6000_TRAIN_SAMPLE.json').read_text())
    confirmation = json.loads((HERE / 'FRESH_CONFIRMATION_1500_OWNER_LOCK.json').read_text())
    original = json.loads((ROOT / 'data/visual_entailment/manifest.json').read_text())
    old_train = {x['id'] for x in original['images'] if x['split'] == 'train'}
    old_nontrain = {x['id'] for x in original['images'] if x['split'] != 'train'}
    provenance = json.loads((ROOT / 'data/visual_entailment/provenance.json').read_text())
    old_hashes = {'flickr:' + Path(name).stem: row['sha256'] for name, row in provenance['images'].items()}
    reserved_hashes = {old_hashes[i] for i in old_nontrain}
    with zipfile.ZipFile(ROOT.parent / 'v8_input_recovery/SANW_SCIENTIFIC_INPUTS_20261004_part_05.zip') as z:
        for name in ['data/review_followup/e_vil_dev900/manifest.json',
                     'data/review_followup/e_vil_test1000/manifest.json',
                     'data/coco_karpathy/manifest.json', 'data/sugarcrepe/manifest.json',
                     'data/sugarcrepe_pp/manifest.json']:
            reserved_hashes.update(x['sha256'] for x in json.loads(z.read(name))['images'])
    new_train = set(sample['additional_image_ids'])
    fresh = set(confirmation['image_ids'])
    assert len(old_train) == 1200 and len(new_train) == 4800 and len(fresh) == 1500
    assert not old_train & new_train and not fresh & (old_train | new_train)
    targets = [(i, 'old_train') for i in sorted(old_train)]
    targets += [(i, 'new_train') for i in sorted(new_train)]
    targets += [(i, 'fresh_confirmation') for i in sorted(fresh)]
    archive_path = ROOT / 'data/official_train_expansion/assets/flickr30k-images.zip'
    acquisition = json.loads((HERE / 'flickr30k-images_zip_acquisition.json').read_text())
    if sha(archive_path) != acquisition['observed_sha256']:
        raise ValueError('Archive changed after acquisition')
    with zipfile.ZipFile(archive_path) as z:
        def check(item):
            iid, split = item
            stem = iid.split(':', 1)[1]
            member = 'flickr30k-images/' + stem + '.jpg'
            info = z.getinfo(member)
            content = z.read(member)  # ZIP reader verifies member CRC.
            digest = hashlib.sha256(content).hexdigest()
            if iid in old_train and digest != old_hashes[iid]:
                raise ValueError('Original training image identity mismatch: ' + iid)
            with Image.open(io.BytesIO(content)) as im:
                im.load()
                rgb = im.convert('RGB')
                width, height = rgb.size
                pixel_sha = hashlib.sha256(struct.pack('<II', width, height) + rgb.tobytes()).hexdigest()
            if split == 'new_train':
                dest = ROOT / 'data/official_train_expansion/images' / (stem + '.jpg')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and sha(dest) != digest:
                    raise ValueError('Conflicting existing extracted image')
                dest.write_bytes(content)
            return {'image_id': iid, 'split': split, 'bytes': len(content), 'sha256': digest,
                    'rgb_size_prefixed_sha256': pixel_sha, 'width': width, 'height': height,
                    'crc32': info.CRC, 'reserved_exact_file_match': digest in reserved_hashes,
                    'written_to_disk': split == 'new_train'}
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(check, targets))
    files, pixels = defaultdict(list), defaultdict(list)
    for row in rows:
        files[row['sha256']].append({'id': row['image_id'], 'split': row['split']})
        pixels[row['rgb_size_prefixed_sha256']].append({'id': row['image_id'], 'split': row['split']})
    duplicates = {k: groups for k, groups in pixels.items() if len(groups) > 1}
    crossing = {k: v for k, v in duplicates.items()
                if any(x['split'] == 'fresh_confirmation' for x in v)
                and any(x['split'] != 'fresh_confirmation' for x in v)}
    matches = [r['image_id'] for r in rows if r['reserved_exact_file_match']]
    receipt = {'schema': 'sanw-expanded-train-image-integrity-v1',
               'sample_sha256': sha(HERE / 'FIXED_6000_TRAIN_SAMPLE.json'),
               'confirmation_sha256': sha(HERE / 'FRESH_CONFIRMATION_1500_OWNER_LOCK.json'),
               'archive_sha256': acquisition['observed_sha256'], 'images_decoded': len(rows),
               'old_train': 1200, 'new_train': 4800, 'fresh_confirmation_streamed_only': 1500,
               'reserved_exact_file_hash_count': len(reserved_hashes),
               'reserved_exact_file_matches': matches,
               'selected_exact_file_duplicate_groups': {k: v for k, v in files.items() if len(v) > 1},
               'selected_rgb_pixel_duplicate_groups': duplicates,
               'fresh_confirmation_training_rgb_duplicate_groups': crossing,
               'passed_no_reserved_content_overlap': not matches and not crossing,
               'new_vs_existing_reserved_recompressed_pixel_duplicates': 'Not checked without reserved decoded-pixel hashes; exact existing reserved JPEG hashes were checked.',
               'features_encoded': False, 'rows': rows, 'seconds': time.monotonic() - start}
    out = HERE / 'IMAGE_DECODING_AND_CONTENT_AUDIT.json'
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    summary = {k: v for k, v in receipt.items() if k != 'rows'}
    summary['receipt_sha256'] = sha(out)
    print(json.dumps(summary, indent=2), flush=True)
    if matches or crossing:
        raise RuntimeError('Content overlap found: stop before feature encoding; do not replace fixed IDs.')

if __name__ == '__main__':
    main()

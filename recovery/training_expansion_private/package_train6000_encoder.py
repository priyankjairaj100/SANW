#!/usr/bin/env python3
"""Package one completed expanded cache with exact identity and audit receipts."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['vit_b32', 'rn50'], required=True)
    args = parser.parse_args()
    cache = ROOT / 'results/official_train_expansion' / args.encoder / 'train_6000'
    completion = json.loads((cache / 'completion.json').read_text())
    meta = json.loads((cache / 'metadata.json').read_text())
    manifest = ROOT / 'data/official_train_expansion/train_6000/manifest.json'
    assert completion['complete']
    assert completion['metadata_sha256'] == sha(cache / 'metadata.json')
    assert completion['features_sha256'] == meta['features_sha256'] == sha(cache / 'features.npz')
    assert meta['manifest_sha256'] == sha(manifest)
    assert meta['all_rows_training_only'] and meta['fresh_confirmation_rows'] == 0
    assert meta['old_image_rows_bitwise_preserved'] == 1200
    assert meta['old_text_rows_bitwise_preserved'] == 22192
    assert meta['image_count'] == 6000 and meta['text_count'] == 110516
    audit_names = [
        'FIXED_6000_TRAIN_SAMPLE.json', 'FIXED_4800_CANONICAL_IDS.txt',
        'SAMPLE_CANONICAL_BINDING.json', 'CURRENT_TRAIN_IDS.json',
        'FRESH_CONFIRMATION_1500_OWNER_LOCK.json', 'FRESH_CONFIRMATION_1500_CANONICAL_IDS.txt',
        'RESERVED_IMAGE_ID_INVENTORY.json', 'IMAGE_DECODING_AND_CONTENT_AUDIT.json',
        'TRAIN6000_METADATA_SUMMARY.json', 'TRAIN_SOURCE_ACQUISITION.json',
        'TRAIN_EXPANSION_MANIFEST_RECEIPT.json', 'TRAIN_EXPANSION_STRUCTURAL_AUDIT.json',
        'ASSET_ACQUISITION.json', 'PILOT_AND_FULL_SOURCE_FREEZE.json',
        'RN50_PILOT_LOAD_REPAIR.json', 'encode_expanded_train6000.py',
        'encode_expansion_pilot.py', 'encode_expansion_pilot_v1.py',
        'acquire_expansion_assets.py', 'audit_expansion_images.py',
        'build_additional_train_manifest.py', 'summarize_train6000_metadata.py',
        'package_train6000_encoder.py',
    ]
    files = [cache / name for name in ['features.npz', 'metadata.json', 'completion.json', 'encoding_lock.json']]
    files += [manifest, cache.parent / 'pilot100/receipt.json', cache.parent / 'full_encoding.log']
    files += [HERE / name for name in audit_names]
    archive = ROOT.parent / f'SANW_TRAIN6000_{args.encoder.upper()}_FEATURES_20261005.zip'
    if archive.exists():
        raise FileExistsError(archive)
    rows = []
    with zipfile.ZipFile(archive, 'x', allowZip64=True) as z:
        for path in sorted(files):
            rel = str(path.relative_to(ROOT))
            rows.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)})
            mode = zipfile.ZIP_STORED if path.suffix == '.npz' else zipfile.ZIP_DEFLATED
            z.write(path, rel, compress_type=mode, compresslevel=None if mode == zipfile.ZIP_STORED else 1)
        z.writestr('PACKAGE_CATALOG.json', json.dumps({
            'encoder': args.encoder, 'training_owner_count': 6000,
            'fresh_confirmation_features_included': False,
            'raw_images_or_weights_included': False, 'sqlite_bank_included': False,
            'files': rows,
        }, indent=2) + '\n')
    if archive.stat().st_size >= 500_000_000:
        raise RuntimeError('Archive exceeds requested 500 MB bound; create recorded binary parts.')
    with zipfile.ZipFile(archive) as z:
        for row in rows:
            with z.open(row['path']) as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('Archive member digest differs: ' + row['path'])
    receipt = {'path': str(archive), 'bytes': archive.stat().st_size,
               'sha256': sha(archive), 'member_count': len(rows) + 1,
               'verified_member_hashes': True, 'features_sha256': meta['features_sha256'],
               'metadata_sha256': completion['metadata_sha256'], 'manifest_sha256': meta['manifest_sha256']}
    target = archive.with_name(archive.stem + '_RECEIPT.json')
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

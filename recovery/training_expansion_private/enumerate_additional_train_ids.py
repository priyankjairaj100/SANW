#!/usr/bin/env python3
"""Enumerate official training IDs from already downloaded, pinned inputs only.

This script has no network access or model dependencies. It does not fetch or
read development/test annotations. Execute only after expansion is activated.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

PINNED_SHA256 = '61ad4a26ec8e2b9c5fc69e52a4870d972f7fab92f448b8d069eb9793dba3aeb1'
PINNED_BYTES = 62565577
CURRENT_IDS_SHA256 = '81a501e3318b8b879eafd6a9d1901b314afec43cb72412ce96556f68a713610d'

def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True, type=Path)
    parser.add_argument('--current-train-ids', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose an unused output path; ID lists are immutable.')
    if args.train_csv.stat().st_size != PINNED_BYTES or sha256(args.train_csv) != PINNED_SHA256:
        raise ValueError('Training CSV does not match the pinned official artifact.')
    if sha256(args.current_train_ids) != CURRENT_IDS_SHA256:
        raise ValueError('Current training-ID snapshot identity differs.')
    old = json.loads(args.current_train_ids.read_text())['image_filenames']
    if len(old) != 1200 or len(set(old)) != 1200:
        raise ValueError('Expected exactly 1,200 unique current training filenames.')
    seen_pair_ids = set()
    ids = set()
    rows = 0
    with args.train_csv.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not {'Flickr30kID', 'pairID'}.issubset(reader.fieldnames or []):
            raise ValueError('Official training CSV schema changed.')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed official training annotation.')
            pair = row['pairID']
            if pair in seen_pair_ids:
                raise ValueError('Duplicate official pair identifier.')
            seen_pair_ids.add(pair)
            image = row['Flickr30kID']
            if Path(image).name != image or not image.endswith('.jpg'):
                raise ValueError('Unexpected image filename.')
            ids.add(image)
            rows += 1
    if len(ids) != 29783 or not set(old).issubset(ids):
        raise ValueError('Official train-image pool or current-subset membership differs.')
    additional = sorted(ids - set(old))
    if len(additional) != 28583:
        raise ValueError('Expected 28,583 additional official training images.')
    receipt = {
        'schema': 'sanw-additional-official-train-ids-v1',
        'official_training_csv_sha256': PINNED_SHA256,
        'current_train_ids_sha256': sha256(args.current_train_ids),
        'official_train_image_count': len(ids),
        'training_annotation_rows': rows,
        'current_train_image_count': len(old),
        'additional_image_count': len(additional),
        'additional_image_filenames': additional,
        'development_or_test_annotations_read': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'images': len(additional), 'output_sha256': sha256(args.output)}))

if __name__ == '__main__':
    main()

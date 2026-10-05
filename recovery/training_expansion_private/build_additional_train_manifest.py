#!/usr/bin/env python3
"""Create an additional-official-train manifest without images or model loads.

The only downloaded annotation content read is the pinned official train CSV
and the selected training captions inside the pinned Flickr caption archive.
Frozen benchmark files supply image identities only, never outcomes or text.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data/official_train_expansion/source'
OUT = ROOT / 'data/official_train_expansion'
PRIVATE = ROOT / 'recovery/training_expansion_private'
INPUT_ZIP = ROOT.parent / 'v8_input_recovery/SANW_SCIENTIFIC_INPUTS_20261004_part_05.zip'

def digest(path):
    with path.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def canonical(path, value):
    if path.exists():
        raise FileExistsError(f'Existing immutable output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')

def clean_caption(line):
    words = []
    inside = False
    for word in line.split():
        if word.startswith('[/EN#'):
            if inside:
                raise ValueError('Nested entity markup')
            inside = True
            continue
        if inside and word.endswith(']'):
            word = word[:-1]
            inside = False
        words.append(word)
    if inside:
        raise ValueError('Unterminated entity markup')
    return ' '.join(words)

def main():
    if digest(INPUT_ZIP) != 'cc777bd598620260ee1706b04a169158e71b174e3684f61b26e80fc4db7ba660':
        raise ValueError('Scientific archive identity mismatch')
    if digest(SRC / 'esnlive_train.csv') != '61ad4a26ec8e2b9c5fc69e52a4870d972f7fab92f448b8d069eb9793dba3aeb1':
        raise ValueError('Training CSV identity mismatch')
    if digest(SRC / 'flickr30k_entities_annotations.zip') != '1bdde439e41fa936e31e6d72898f1886d49bb4298f2abcdb50771de4f516b026':
        raise ValueError('Caption archive identity mismatch')
    old_manifest = json.loads((ROOT / 'data/visual_entailment/manifest.json').read_text())
    old_train = {row['id'] for row in old_manifest['images'] if row['split'] == 'train'}
    old_nontrain = {row['id'] for row in old_manifest['images'] if row['split'] != 'train'}
    if len(old_train) != 1200:
        raise ValueError('Current training population differs')
    reserves = {'current_nontraining': old_nontrain}
    reserved_hashes = set()
    inventory = []
    names = ['data/review_followup/e_vil_dev900/manifest.json',
             'data/review_followup/e_vil_test1000/manifest.json',
             'data/coco_karpathy/manifest.json', 'data/sugarcrepe/manifest.json',
             'data/sugarcrepe_pp/manifest.json']
    with zipfile.ZipFile(INPUT_ZIP) as z:
        for name in names:
            raw = z.read(name)
            image_rows = json.loads(raw)['images']
            ids = {row['id'] for row in image_rows}
            if len(ids) != len(image_rows):
                raise ValueError('Duplicate reserved image IDs')
            reserves[name] = ids
            reserved_hashes.update(row['sha256'] for row in image_rows if 'sha256' in row)
            inventory.append({'manifest': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                              'image_count': len(ids), 'image_ids': sorted(ids)})
    reserved = set().union(*reserves.values())
    grouped = {}
    labels = {'entailment': 'supported', 'contradiction': 'contradicted', 'neutral': 'neutral'}
    pair_ids = set()
    with (SRC / 'esnlive_train.csv').open(newline='', encoding='utf-8') as h:
        for number, row in enumerate(csv.DictReader(h), 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed training row {number}')
            if row['gold_label'] not in labels or row['pairID'] in pair_ids:
                raise ValueError(f'Invalid training row {number}')
            pair_ids.add(row['pairID'])
            filename = row['Flickr30kID']
            if Path(filename).name != filename or not filename.endswith('.jpg'):
                raise ValueError('Unsafe official image filename')
            iid = 'flickr:' + Path(filename).stem
            grouped.setdefault(iid, []).append(row)
    official = set(grouped)
    if len(official) != 29783 or not old_train <= official:
        raise ValueError('Official training pool differs from prior provenance')
    excluded = official & reserved
    eligible = sorted(official - old_train - reserved)
    # Existing training images must not overlap either reserved IDs or known
    # reserved image hashes. New raw images are not available for hash checks.
    if old_train & reserved:
        raise ValueError('Existing train/reserved image-ID overlap')
    provenance = json.loads((ROOT / 'data/visual_entailment/provenance.json').read_text())
    old_hashes = {row['sha256'] for name, row in provenance['images'].items()
                  if 'flickr:' + Path(name).stem in old_train}
    if old_hashes & reserved_hashes:
        raise ValueError('Existing train/reserved image-content overlap')
    images, texts, pairs = [], [], []
    relation_counts = Counter()
    selected_ids = set()
    with zipfile.ZipFile(SRC / 'flickr30k_entities_annotations.zip') as z:
        for iid in eligible:
            stem = iid.split(':', 1)[1]
            member = f'Sentences/{stem}.txt'
            caps = [clean_caption(x) for x in z.read(member).decode('utf-8').splitlines() if x.strip()]
            if len(caps) != 5:
                raise ValueError('Training image lacks exactly five captions')
            images.append({'id': iid, 'path': f'data/official_train_expansion/images/{stem}.jpg',
                           'split': 'train', 'acquisition_status': 'not_downloaded'})
            for k, caption in enumerate(caps):
                tid = f'source:{stem}:{k}'
                texts.append({'id': tid, 'text': caption})
                pairs.append({'image_id': iid, 'text_id': tid, 'relation': 'source'})
                relation_counts['source'] += 1
            for row in grouped[iid]:
                tid = 'hypothesis:train:' + row['pairID']
                if tid in selected_ids:
                    raise ValueError('Duplicate new training text ID')
                selected_ids.add(tid)
                relation = labels[row['gold_label']]
                texts.append({'id': tid, 'text': row['hypothesis']})
                pairs.append({'image_id': iid, 'text_id': tid, 'relation': relation})
                relation_counts[relation] += 1
    manifest = {'schema_version': 1, 'dataset': 'official_train_expansion_additional',
                'images': images, 'texts': texts, 'pairs': pairs, 'triplets': []}
    canonical(OUT / 'manifest.json', manifest)
    canonical(OUT / 'additional_image_ids.json', {'image_ids': eligible})
    canonical(PRIVATE / 'RESERVED_IMAGE_ID_INVENTORY.json', {'reserved_sources': inventory,
              'current_manifest_nontraining_ids': sorted(old_nontrain)})
    intersections = {name: len(set(eligible) & ids) for name, ids in reserves.items()}
    receipt = {'schema': 'sanw-additional-train-manifest-v1',
               'official_train_images': len(official), 'already_available_train_images': len(old_train),
               'official_train_reserved_overlap_ids': sorted(excluded),
               'additional_eligible_images': len(eligible), 'text_count': len(texts),
               'relation_counts': dict(relation_counts), 'reserved_union_image_count': len(reserved),
               'disjointness_checks': intersections, 'all_id_disjointness_checks_passed': not any(intersections.values()),
               'existing_train_reserved_content_hash_overlap': len(old_hashes & reserved_hashes),
               'new_images_content_disjointness': 'Not yet checked; image bytes were not downloaded.',
               'manifest_sha256': digest(OUT / 'manifest.json'),
               'additional_image_ids_sha256': digest(OUT / 'additional_image_ids.json'),
               'reserved_id_inventory_sha256': digest(PRIVATE / 'RESERVED_IMAGE_ID_INVENTORY.json'),
               'only_official_train_annotation_rows_used': True, 'benchmark_outcomes_read': False,
               'benchmark_caption_or_triplet_content_used': False,
               'benchmark_data_access': 'Existing archived manifest image IDs, split fields and available hashes only.',
               'images_downloaded': 0, 'model_weights_downloaded': 0,
               'new_annotation_quality_filter_applied': False,
               'note': 'Source captions are joint truths, not assumed equivalent paraphrases. Neutral labels remain distinct.'}
    canonical(PRIVATE / 'TRAIN_EXPANSION_MANIFEST_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()

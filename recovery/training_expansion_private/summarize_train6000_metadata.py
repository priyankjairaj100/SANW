#!/usr/bin/env python3
"""Summarize locked training metadata without loading any feature/outcome arrays."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    path = ROOT / 'data/official_train_expansion/train_6000/manifest.json'
    manifest = json.loads(path.read_text())
    sample_path = HERE / 'FIXED_6000_TRAIN_SAMPLE.json'
    sample = json.loads(sample_path.read_text())
    added = set(sample['additional_image_ids'])
    owners = {row['id'] for row in manifest['images']}
    old = owners - added
    assert len(owners) == 6000 and len(old) == 1200 and len(added) == 4800
    texts = {row['id']: row['text'] for row in manifest['texts']}
    sections = {}
    for name, keep in [('original_train', old), ('additional_train', added), ('combined_train', owners)]:
        pairs = [row for row in manifest['pairs'] if row['image_id'] in keep]
        text_ids = {row['text_id'] for row in pairs}
        role_counts = Counter(row['relation'] for row in pairs)
        source_counts = Counter(row['image_id'] for row in pairs if row['relation'] == 'source')
        source_owners = defaultdict(set)
        for row in pairs:
            if row['relation'] == 'source':
                source_owners[texts[row['text_id']]].add(row['image_id'])
        assert set(source_counts) == keep and set(source_counts.values()) == {5}
        sections[name] = {
            'image_owners': len(keep), 'pairs': len(pairs), 'text_rows': len(text_ids),
            'unique_exact_text_strings': len({texts[tid] for tid in text_ids}),
            'pair_roles': dict(sorted(role_counts.items())),
            'source_captions_per_owner': 5,
            'exact_source_strings_shared_by_multiple_owners': sum(len(value) > 1 for value in source_owners.values()),
        }
    output = {
        'manifest_sha256': digest(path), 'fixed_sample_sha256': digest(sample_path),
        'summary_source_sha256': digest(Path(__file__)), 'sections': sections,
        'sampling': 'Fixed SHA256 ranking of canonical official-training owner IDs, independent of labels and features. Training labels had already been materialized before sample freezing.',
        'fresh_confirmation': '1500 separately locked unused official-training owners; excluded from fitting and selection; no confirmation features encoded.',
        'integrity_scope': 'All 7500 training/confirmation owners decoded. Exact JPEG hashes checked against 8377 reserved owners; decoded RGB duplicates checked within selected7500. Recompressed or perceptual duplicates against old reserved images were not checked.',
        'label_policy': 'Official e-SNLI-VE training labels preserved; no new annotation-quality filter. Neutral labels are not contradictions. Source captions share image ownership and need not be semantic paraphrases.',
        'loaded_feature_arrays': False, 'loaded_benchmark_outcomes': False,
    }
    out = HERE / 'TRAIN6000_METADATA_SUMMARY.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'path': str(out), 'sha256': digest(out), 'sections': sections}))


if __name__ == '__main__':
    main()

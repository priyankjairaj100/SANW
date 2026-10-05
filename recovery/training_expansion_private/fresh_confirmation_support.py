"""Gated, standard-library-only helpers for the frozen fresh1500 input export.

Importing this module opens no dataset, model, owner lock, or feature archive.
Production callers must obtain the final state/source lock before calling any
data-reading helper. Pure helpers also support small synthetic fixtures.
"""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY = 'sanw_practical_v10_fresh_confirmation_state_source_lock'
CONTRACT = {'path': 'results/practical_v10/fresh_confirmation_contract_v1.json',
            'sha256': 'b1f426b284925d863bf717fdafdb11e9bf93f134f47b2deab82e492bfc53eee9'}
OWNER_LOCK = {'path': 'recovery/training_expansion_private/FRESH_CONFIRMATION_1500_OWNER_LOCK.json',
              'sha256': '2193a3cf22b221c02529d5f5914fdace76c3223c7aebedfeb53808962fc7df82'}
EVALUATOR_SOURCE = 'scripts/evaluate_practical_fresh_confirmation_v10.py'
EXPORT_SOURCE = 'recovery/training_expansion_private/encode_fresh_confirmation1500.py'
SUPPORT_SOURCE = 'recovery/training_expansion_private/fresh_confirmation_support.py'
PINNED_ENCODING_SOURCES = {
    'recovery/training_expansion_private/encode_expanded_train6000.py': 'fff13c048416b860458b303de55991f1eef9ed790c4fb30c270c0c3ce3e20e4a',
    'recovery/training_expansion_private/encode_expansion_pilot.py': '087dd3bcf32d2b291898b5cd2ac492ba5a2416ab76555c6313cdfb8a7901b448',
    'recovery/training_expansion_private/build_additional_train_manifest.py': '7a3775a063c7c823bf576d4eda0b4627956701155b8d0be07ec37ca7389bbdbf',
}
INPUT_RECORDS = {
    'train_annotations': {'path': 'data/official_train_expansion/source/esnlive_train.csv', 'bytes': 62565577, 'sha256': '61ad4a26ec8e2b9c5fc69e52a4870d972f7fab92f448b8d069eb9793dba3aeb1'},
    'source_captions': {'path': 'data/official_train_expansion/source/flickr30k_entities_annotations.zip', 'bytes': 29284070, 'sha256': '1bdde439e41fa936e31e6d72898f1886d49bb4298f2abcdb50771de4f516b026'},
    'image_archive': {'path': 'data/official_train_expansion/assets/flickr30k-images.zip', 'bytes': 4390240817, 'sha256': '2ce2420c0d17f0531deaa89ac657b4d5067ec519da16ff1ea12acbf5619c7391'},
    'vit_weights': {'path': 'data/official_train_expansion/assets/open_clip_model.safetensors', 'bytes': 605143316, 'sha256': 'ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6'},
    'rn50_weights': {'path': 'data/official_train_expansion/assets/RN50.pt', 'bytes': 255827503, 'sha256': 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'},
    'fit_owner_sample': {'path': 'recovery/training_expansion_private/FIXED_6000_TRAIN_SAMPLE.json', 'bytes': 122031, 'sha256': '0e19b6577d2a2575d232d76c94e61a54f65cb9363a9be3c36ce6aae7d3d48131'},
    'original_train_ids': {'path': 'recovery/training_expansion_private/CURRENT_TRAIN_IDS.json', 'bytes': 26262, 'sha256': '81a501e3318b8b879eafd6a9d1901b314afec43cb72412ce96556f68a713610d'},
    'reserved_ids': {'path': 'recovery/training_expansion_private/RESERVED_IMAGE_ID_INVENTORY.json', 'bytes': 162065, 'sha256': 'a9f59ee011e50e85a02a2f21d01871085fb530eb1e2822c72bf3859ee1e34f86'},
    'image_integrity_audit': {'path': 'recovery/training_expansion_private/IMAGE_DECODING_AND_CONTENT_AUDIT.json', 'bytes': 3229680, 'sha256': 'aedd1ba1b057c7a7f0055257f1e926d46b2281fc98528efb29a589e1c7d03ca5'},
}
EXPECTED_RUNTIME = {'torch': '2.7.1+cpu', 'torchvision': '0.22.1+cpu', 'open_clip': '2.32.0',
                    'numpy': '2.3.5', 'Pillow': '12.3.0'}
PREPROCESS = {'size': 224, 'mode': 'RGB', 'interpolation': 'bicubic', 'resize_mode': 'shortest',
              'crop': 'center', 'mean': [0.48145466, 0.4578275, 0.40821073],
              'std': [0.26862954, 0.26130258, 0.27577711]}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Record path escapes repository')
    return path


def verify_record(root, entry):
    path = contained(root, entry['path'])
    if ('bytes' in entry and path.stat().st_size != entry['bytes']) or sha(path) != entry['sha256']:
        raise ValueError('File identity mismatch: ' + entry['path'])
    return path


def canonical_json(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    tmp.replace(path)


def record(root, path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(Path(root).resolve())), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def validate_state_grid(lock):
    states = lock.get('states')
    if not isinstance(states, dict) or len(states) != 12:
        raise ValueError('Exactly twelve frozen joint/control states are required')
    seen = set()
    for state in states.values():
        key = state.get('encoder'), state.get('family'), state.get('seed')
        if key in seen or state.get('nonzero') is not True:
            raise ValueError('Duplicate or zero confirmation state')
        seen.add(key)
        for name in ('checkpoint', 'completion', 'ledger'):
            if not isinstance(state.get(name), dict) or not {'path', 'sha256'} <= set(state[name]):
                raise ValueError('Incomplete state binding')
    expected = {(encoder, family, seed) for encoder in ('vit_b32', 'rn50')
                for family in ('joint', 'no_retention') for seed in (17, 29, 43)}
    if seen != expected:
        raise ValueError('Frozen state grid differs from inherited contract')


def require_access(lock_path, expected_sha256, *, root=ROOT):
    """Validate the final lock before loading any owner/annotation/model data.

    Only the lock and source files are opened locally before the evaluator's
    authoritative qualification validator runs. This is not a replacement for
    that validator: it verifies full completed-state semantics and source audit.
    """
    root = Path(root).resolve()
    path = Path(lock_path).resolve()
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256) or sha(path) != expected_sha256:
        raise ValueError('Expected final state/source lock hash is required')
    lock = json.loads(path.read_text())
    if lock.get('study') != STUDY or lock.get('contract') != CONTRACT or lock.get('owner_lock') != OWNER_LOCK:
        raise ValueError('Immutable confirmation contract/owner lock changed')
    validate_state_grid(lock)
    sources = lock.get('source_sha256', {})
    for name in (EVALUATOR_SOURCE, EXPORT_SOURCE, SUPPORT_SOURCE):
        if name not in sources:
            raise ValueError('Missing confirmation source binding: ' + name)
    for name, digest in PINNED_ENCODING_SOURCES.items():
        if sources.get(name) != digest:
            raise ValueError('Inherited encoding recipe identity changed')
    for name, digest in sources.items():
        verify_record(root, {'path': name, 'sha256': digest})
    if lock.get('raw_inputs') != INPUT_RECORDS:
        raise ValueError('Predeclared raw input identities differ')
    # This module is imported only after the final lock and its source hash pass.
    module_path = contained(root, EVALUATOR_SOURCE)
    spec = importlib.util.spec_from_file_location('sanw_fresh_authoritative_lock_validator', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    validated = module.load_input_lock(str(path), expected_sha256)
    if validated != lock:
        raise ValueError('Authoritative validator returned a different lock')
    return lock


def verify_runtime(actual):
    if actual != EXPECTED_RUNTIME:
        raise ValueError('Runtime differs from the successful frozen training encoder: ' + repr(actual))


def load_owner_metadata(root, lock):
    """Called only after require_access succeeds; returns selected metadata."""
    owners = json.loads(verify_record(root, lock['owner_lock']).read_text())
    records = {name: verify_record(root, INPUT_RECORDS[name]) for name in
               ('fit_owner_sample', 'original_train_ids', 'reserved_ids', 'image_integrity_audit')}
    fit = json.loads(records['fit_owner_sample'].read_text())
    original = json.loads(records['original_train_ids'].read_text())
    reserved = json.loads(records['reserved_ids'].read_text())
    audit = json.loads(records['image_integrity_audit'].read_text())
    ids = owners['image_ids']
    selected = set(ids)
    old = {'flickr:' + Path(name).stem for name in original['image_filenames']}
    added = set(fit['additional_image_ids'])
    heldout = set(reserved['current_manifest_nontraining_ids'])
    for source in reserved['reserved_sources']:
        heldout.update(source['image_ids'])
    if (len(ids) != 1500 or len(selected) != 1500 or len(old) != 1200 or len(added) != 4800
            or selected & (old | added | heldout) or old & added
            or owners.get('fit_or_selection_allowed') is not False
            or owners.get('fit_sample_sha256') != INPUT_RECORDS['fit_owner_sample']['sha256']):
        raise ValueError('Fresh owner population or disjointness differs')
    if (audit.get('passed_no_reserved_content_overlap') is not True
            or audit.get('confirmation_sha256') != OWNER_LOCK['sha256']
            or audit.get('archive_sha256') != INPUT_RECORDS['image_archive']['sha256']
            or audit.get('reserved_exact_file_matches') or audit.get('selected_rgb_pixel_duplicate_groups')
            or audit.get('fresh_confirmation_training_rgb_duplicate_groups')):
        raise ValueError('Frozen content audit failed or changed')
    rows = {row['image_id']: row for row in audit['rows'] if row['image_id'] in selected}
    if set(rows) != selected or any(row['split'] != 'fresh_confirmation' for row in rows.values()):
        raise ValueError('Fresh image integrity entries are incomplete')
    return ids, rows


def build_manifest(owners, csv_path, captions_path, image_audit_rows, clean_caption):
    """Build all raw owner-associated rows; no filtering by relation eligibility."""
    if len(owners) != len(set(owners)):
        raise ValueError('Duplicate owner ID')
    selected = set(owners)
    grouped = {owner: [] for owner in owners}
    labels = {'entailment': 'supported', 'contradiction': 'contradicted', 'neutral': 'neutral'}
    pair_ids = set()
    with Path(csv_path).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            filename = row['Flickr30kID']
            if not re.fullmatch(r'[0-9]+\.jpg', filename):
                raise ValueError('Unsafe official image filename')
            owner = 'flickr:' + Path(filename).stem
            if owner not in selected:
                continue
            if (None in row or any(value is None for value in row.values())
                    or row['gold_label'] not in labels or row['pairID'] in pair_ids):
                raise ValueError('Malformed or duplicate selected annotation')
            pair_ids.add(row['pairID'])
            grouped[owner].append((row['pairID'], row['hypothesis'], labels[row['gold_label']]))
    if any(not grouped[owner] for owner in owners):
        raise ValueError('Locked owner missing from official TRAIN annotations')
    images, texts, pairs = [], [], []
    with zipfile.ZipFile(captions_path) as archive:
        for owner in owners:
            if not re.fullmatch(r'flickr:[0-9]+', owner):
                raise ValueError('Unsafe locked owner ID')
            stem = owner.split(':', 1)[1]
            name = f'Sentences/{stem}.txt'
            if archive.namelist().count(name) != 1:
                raise ValueError('Missing or ambiguous source-caption member')
            captions = [clean_caption(line) for line in archive.read(name).decode('utf-8').splitlines() if line.strip()]
            if len(captions) != 5:
                raise ValueError('Exactly five source captions per owner are required')
            audit = image_audit_rows[owner]
            images.append({'id': owner, 'split': 'test', 'path': f'flickr30k-images/{stem}.jpg',
                           'storage': 'zip_member', 'archive': INPUT_RECORDS['image_archive']['path'],
                           'archive_member': f'flickr30k-images/{stem}.jpg',
                           'sha256': audit['sha256'], 'bytes': audit['bytes'],
                           'rgb_size_prefixed_sha256': audit['rgb_size_prefixed_sha256']})
            for k, caption in enumerate(captions):
                tid = f'source:{stem}:{k}'
                texts.append({'id': tid, 'text': caption})
                pairs.append({'image_id': owner, 'text_id': tid, 'relation': 'source'})
            for pair_id, text, relation in grouped[owner]:
                tid = 'hypothesis:train:' + pair_id
                texts.append({'id': tid, 'text': text})
                pairs.append({'image_id': owner, 'text_id': tid, 'relation': relation})
    manifest = {'schema_version': 1, 'dataset': 'fresh_confirmation1500',
                'images': images, 'texts': texts, 'pairs': pairs, 'triplets': []}
    validate_manifest(manifest, owners)
    return manifest


def validate_manifest(manifest, owners):
    if [row['id'] for row in manifest['images']] != list(owners):
        raise ValueError('Ordered manifest owner IDs differ')
    if any(row['split'] != 'test' for row in manifest['images']):
        raise ValueError('Confirmation rows must not be training/selection rows')
    texts = {row['id'] for row in manifest['texts']}
    if len(texts) != len(manifest['texts']):
        raise ValueError('Duplicate text IDs')
    owners = set(owners)
    associations, sources = Counter(), Counter()
    for pair in manifest['pairs']:
        if (pair['image_id'] not in owners or pair['text_id'] not in texts
                or pair['relation'] not in {'source', 'supported', 'contradicted', 'neutral'}):
            raise ValueError('Invalid owner/text/relation association')
        associations[pair['text_id']] += 1
        if pair['relation'] == 'source':
            sources[pair['image_id']] += 1
    if set(associations) != texts or set(associations.values()) != {1}:
        raise ValueError('Every text row must have exactly one annotated owner')
    if set(sources) != owners or set(sources.values()) != {5}:
        raise ValueError('Source ownership differs from five captions per owner')


def decode_verified_jpeg(archive, row):
    from PIL import Image
    member = row['archive_member']
    if not re.fullmatch(r'flickr30k-images/[0-9]+\.jpg', member):
        raise ValueError('Unsafe image archive member')
    if archive.namelist().count(member) != 1:
        raise ValueError('Missing or ambiguous selected JPEG')
    raw = archive.read(member)
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Selected JPEG differs from frozen content audit')
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert('RGB')
    pixels = hashlib.sha256(struct.pack('<II', *rgb.size) + rgb.tobytes()).hexdigest()
    if pixels != row['rgb_size_prefixed_sha256']:
        raise ValueError('Selected decoded RGB differs from frozen content audit')
    return rgb

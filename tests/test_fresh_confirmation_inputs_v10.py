"""Synthetic-only tests; never open the real confirmation owner or annotation rows."""
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / 'recovery/training_expansion_private'
sys.path.insert(0, str(HELPERS))
import fresh_confirmation_support as support
import encode_fresh_confirmation1500 as exporter
from build_additional_train_manifest import clean_caption


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_lock(tmp_path, monkeypatch, reject=False):
    """The evaluator stub isolates ordering; full state semantics have separate tests."""
    evaluator = tmp_path / support.EVALUATOR_SOURCE
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("import json\nfrom pathlib import Path\n"
                         "def load_input_lock(path, expected_sha256):\n" +
                         ("    raise ValueError('authoritative state qualification rejected')\n" if reject else
                          "    return json.loads(Path(path).read_text())\n"))
    for name in (support.EXPORT_SOURCE, support.SUPPORT_SOURCE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source\n')
    monkeypatch.setattr(support, 'PINNED_ENCODING_SOURCES', {})
    states = {}
    for encoder in ('vit_b32', 'rn50'):
        for family in ('joint', 'no_retention'):
            for seed in (17, 29, 43):
                states[f'{encoder}_{family}_{seed}'] = {
                    'encoder': encoder, 'family': family, 'seed': seed, 'nonzero': True,
                    **{name: {'path': f'{name}.dummy', 'sha256': 'a' * 64}
                       for name in ('checkpoint', 'completion', 'ledger')},
                }
    value = {'study': support.STUDY, 'contract': support.CONTRACT,
             'owner_lock': support.OWNER_LOCK, 'raw_inputs': support.INPUT_RECORDS,
             'states': states, 'source_sha256': {name: digest(tmp_path / name)
                                                for name in (support.EVALUATOR_SOURCE, support.EXPORT_SOURCE, support.SUPPORT_SOURCE)}}
    lock = tmp_path / 'decision.json'
    lock.write_text(json.dumps(value))
    return lock, value


def test_pinned_record_hashes_are_full_sha256():
    for row in [*support.INPUT_RECORDS.values(), support.CONTRACT, support.OWNER_LOCK]:
        assert len(row['sha256']) == 64
        assert all(char in '0123456789abcdef' for char in row['sha256'])
    assert support.CONTRACT['sha256'] == 'b1f426b284925d863bf717fdafdb11e9bf93f134f47b2deab82e492bfc53eee9'
    assert support.OWNER_LOCK['sha256'] == '2193a3cf22b221c02529d5f5914fdace76c3223c7aebedfeb53808962fc7df82'


def test_gate_accepts_complete_toy_lock_without_any_owner_file(tmp_path, monkeypatch):
    path, expected = toy_lock(tmp_path, monkeypatch)
    assert support.require_access(path, digest(path), root=tmp_path) == expected
    assert not (tmp_path / support.OWNER_LOCK['path']).exists()


@pytest.mark.parametrize('mutation', ['missing_state', 'duplicate_seed', 'retrieval_family', 'zero_state', 'missing_checkpoint', 'changed_owner', 'changed_contract', 'changed_raw_input'])
def test_gate_rejects_bad_grid_and_identity_before_source_import(tmp_path, monkeypatch, mutation):
    path, lock = toy_lock(tmp_path, monkeypatch)
    first, second = list(lock['states'])[:2]
    if mutation == 'missing_state':
        del lock['states'][first]
    elif mutation == 'duplicate_seed':
        lock['states'][second] = dict(lock['states'][first])
    elif mutation == 'retrieval_family':
        lock['states'][first]['family'] = 'retrieval_only'
    elif mutation == 'zero_state':
        lock['states'][first]['nonzero'] = False
    elif mutation == 'missing_checkpoint':
        del lock['states'][first]['checkpoint']
    elif mutation == 'changed_owner':
        lock['owner_lock'] = {**lock['owner_lock'], 'sha256': '0' * 64}
    elif mutation == 'changed_contract':
        lock['contract'] = {**lock['contract'], 'sha256': '0' * 64}
    else:
        lock['raw_inputs'] = {}
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(importlib.util, 'spec_from_file_location', lambda *args: pytest.fail('premature source import'))
    with pytest.raises(ValueError):
        support.require_access(path, digest(path), root=tmp_path)


def test_gate_rejects_lock_hash_or_changed_eval_source(tmp_path, monkeypatch):
    path, _ = toy_lock(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='lock hash'):
        support.require_access(path, '0' * 64, root=tmp_path)
    (tmp_path / support.EVALUATOR_SOURCE).write_text('raise RuntimeError("must never import tampered source")')
    with pytest.raises(ValueError, match='File identity'):
        support.require_access(path, digest(path), root=tmp_path)


def test_gate_propagates_authoritative_qualification_failure(tmp_path, monkeypatch):
    path, _ = toy_lock(tmp_path, monkeypatch, reject=True)
    with pytest.raises(ValueError, match='authoritative state'):
        support.require_access(path, digest(path), root=tmp_path)


@pytest.mark.parametrize('operation', ['prepare', 'encode'])
def test_no_owner_access_when_authoritative_gate_fails(monkeypatch, operation):
    def reject(args):
        raise ValueError('not frozen')
    monkeypatch.setattr(exporter, 'access', reject)
    monkeypatch.setattr(exporter, 'load_owner_metadata', lambda *args: pytest.fail('owner data opened before gate'))
    with pytest.raises(ValueError, match='not frozen'):
        getattr(exporter, operation)(SimpleNamespace())


def toy_annotations(tmp_path, labels=('entailment', 'contradiction', 'neutral')):
    csv_path = tmp_path / 'train.csv'
    fields = ['Flickr30kID', 'pairID', 'hypothesis', 'gold_label']
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for owner in ('111', '222', '999'):
            for k, label in enumerate(labels):
                writer.writerow({'Flickr30kID': owner + '.jpg', 'pairID': f'{owner}-{k}',
                                 'hypothesis': 'same exact caption' if k == 1 else f'Raw  {label} text ', 'gold_label': label})
    caption_path = tmp_path / 'captions.zip'
    with zipfile.ZipFile(caption_path, 'w') as z:
        for owner in ('111', '222', '999'):
            lines = ['same exact caption', 'same exact caption', '[/EN#1/person a person] walks', 'four', 'five']
            z.writestr(f'Sentences/{owner}.txt', '\n'.join(lines))
    audit = {f'flickr:{owner}': {'sha256': 'b' * 64, 'bytes': 1, 'rgb_size_prefixed_sha256': 'c' * 64}
             for owner in ('111', '222')}
    return csv_path, caption_path, audit


def test_builder_preserves_order_all_raw_labels_and_duplicate_caption_rows(tmp_path):
    csv_path, caption_path, audit = toy_annotations(tmp_path)
    owners = ['flickr:222', 'flickr:111']
    manifest = support.build_manifest(owners, csv_path, caption_path, audit, clean_caption)
    assert [row['id'] for row in manifest['images']] == owners
    assert all(row['split'] == 'test' for row in manifest['images'])
    assert Counter(row['relation'] for row in manifest['pairs']) == {'source': 10, 'supported': 2, 'contradicted': 2, 'neutral': 2}
    texts = {row['id']: row['text'] for row in manifest['texts']}
    assert len(texts) == 16
    assert sum(text == 'same exact caption' for text in texts.values()) == 6
    assert texts['hypothesis:train:222-0'] == 'Raw  entailment text '
    assert texts['source:222:2'] == 'a person walks'
    assert all('999' not in row['id'] for row in manifest['texts'])


def test_builder_refuses_missing_owner_or_bad_label(tmp_path):
    csv_path, caption_path, audit = toy_annotations(tmp_path, labels=('not_a_label',))
    with pytest.raises(ValueError, match='Malformed'):
        support.build_manifest(['flickr:111'], csv_path, caption_path, audit, clean_caption)


def test_manifest_refuses_source_loss_or_training_split(tmp_path):
    csv_path, caption_path, audit = toy_annotations(tmp_path)
    manifest = support.build_manifest(['flickr:111'], csv_path, caption_path, audit, clean_caption)
    manifest['images'][0]['split'] = 'train'
    with pytest.raises(ValueError, match='training'):
        support.validate_manifest(manifest, ['flickr:111'])
    manifest['images'][0]['split'] = 'test'
    manifest['pairs'][0]['relation'] = 'neutral'
    with pytest.raises(ValueError, match='five captions'):
        support.validate_manifest(manifest, ['flickr:111'])


def toy_jpeg(tmp_path):
    image = Image.new('RGB', (5, 4), (21, 84, 126))
    stream = io.BytesIO(); image.save(stream, format='JPEG'); raw = stream.getvalue()
    with Image.open(io.BytesIO(raw)) as decoded:
        rgb = decoded.convert('RGB')
    row = {'archive_member': 'flickr30k-images/111.jpg', 'bytes': len(raw),
           'sha256': hashlib.sha256(raw).hexdigest(),
           'rgb_size_prefixed_sha256': hashlib.sha256(struct.pack('<II', *rgb.size) + rgb.tobytes()).hexdigest()}
    path = tmp_path / 'images.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(row['archive_member'], raw)
    return path, row, rgb


def test_streamed_jpeg_checks_bytes_pixels_and_never_extracts(tmp_path):
    path, row, expected = toy_jpeg(tmp_path)
    before = set(tmp_path.iterdir())
    with zipfile.ZipFile(path) as archive:
        with support.decode_verified_jpeg(archive, row) as actual:
            assert actual.tobytes() == expected.tobytes()
        with pytest.raises(ValueError, match='JPEG differs'):
            support.decode_verified_jpeg(archive, {**row, 'sha256': '0' * 64})
        with pytest.raises(ValueError, match='decoded RGB'):
            support.decode_verified_jpeg(archive, {**row, 'rgb_size_prefixed_sha256': '0' * 64})
        with pytest.raises(ValueError, match='Unsafe'):
            support.decode_verified_jpeg(archive, {**row, 'archive_member': '../111.jpg'})
    assert set(tmp_path.iterdir()) == before


def test_bank_resume_identity_norms_and_distinct_export_rows(tmp_path):
    path = tmp_path / 'toy.sqlite'
    identity = {'state_lock': 'a' * 64}
    bank = exporter.FeatureBank(path, identity, 2, np)
    key = exporter.text_key('identical text')
    bank.put('text', key, np.array([1., 0.], dtype=np.float32)); bank.commit(); bank.close()
    bank = exporter.FeatureBank(path, identity, 2, np)
    rows = [{'id': 'owner1:caption', 'text': 'identical text'}, {'id': 'owner2:caption', 'text': 'identical text'}]
    exported = np.stack([bank.get('text', exporter.text_key(row['text'])) for row in rows])
    assert exported.shape == (2, 2) and rows[0]['id'] != rows[1]['id']
    with pytest.raises(ValueError, match='unnormalized'):
        bank.put('text', 'bad', np.array([0., 0.]))
    with pytest.raises(ValueError, match='committed'):
        bank.put('text', key, np.array([0., 1.]))
    bank.close()
    with pytest.raises(ValueError, match='identity changed'):
        exporter.FeatureBank(path, {'state_lock': 'b' * 64}, 2, np)


def test_runtime_rejects_version_drift():
    support.verify_runtime(dict(support.EXPECTED_RUNTIME))
    with pytest.raises(ValueError, match='Runtime differs'):
        support.verify_runtime({**support.EXPECTED_RUNTIME, 'torch': 'different'})


def test_final_export_rejects_nonfinite_dtype_and_norm_drift():
    images = np.tile(np.array([[1., 0.]], dtype=np.float32), (1500, 1))
    texts = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    manifest = {'texts': [{'id': 'a'}, {'id': 'b'}]}
    exporter.validate_export(images, texts, manifest, np, 2)
    with pytest.raises(ValueError, match='float32'):
        exporter.validate_export(images.astype(np.float64), texts, manifest, np, 2)
    texts[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        exporter.validate_export(images, texts, manifest, np, 2)

"""Publish evaluation input configs only after all five feature caches exist."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'recovery/current_input_recovery'
DATASETS = ['e_vil_test1000', 'visual_entailment', 'sugarcrepe', 'sugarcrepe_pp', 'coco_karpathy']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


receipt = {'scope': 'Verified dataset configurations only. No held-out scoring.', 'encoders': {}}
for encoder in ['vit_b32', 'rn50']:
    datasets = {}
    records = {}
    feature_root = ROOT / ('results/resume_features' if encoder == 'vit_b32' else 'results/strengthen_second_encoder/features')
    training_metadata = json.loads((feature_root / 'visual_entailment/metadata.json').read_text())
    missing = []
    for name in DATASETS:
        manifest = ROOT / 'data' / ('review_followup/e_vil_test1000' if name == 'e_vil_test1000' else name) / 'manifest.json'
        if name == 'e_vil_test1000':
            location = ROOT / ('results/review_followup/features/e_vil_test1000' if encoder == 'vit_b32' else 'results/strengthen_second_encoder/features/review_followup/e_vil_test1000')
        else:
            location = feature_root / name
        features = location / 'features.npz'
        metadata_path = location / 'metadata.json'
        for path in [manifest, features, metadata_path]:
            if not path.is_file():
                missing.append(str(path.relative_to(ROOT)))
        if not features.is_file() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        assert metadata['features_sha256'] == digest(features), str(features)
        assert metadata['manifest_sha256'] == digest(manifest), str(manifest)
        for key in ['model_revision', 'weights_sha256', 'open_clip_version', 'logit_scale']:
            assert metadata[key] == training_metadata[key], (encoder, name, key)
        datasets[name] = {'manifest': str(manifest.relative_to(ROOT)), 'features': str(features.relative_to(ROOT)), 'metadata': str(metadata_path.relative_to(ROOT))}
        records[name] = {'manifest_sha256': digest(manifest), 'features_sha256': metadata['features_sha256'], 'metadata_sha256': digest(metadata_path)}
    if missing:
        receipt['encoders'][encoder] = {'complete': False, 'missing': missing}
        print(encoder, 'pending:', len(missing), 'files', flush=True)
        continue
    assert set(datasets) == set(DATASETS)
    config = {'encoder': encoder, 'datasets': datasets}
    target = OUTPUT / f'datasets_{encoder}.json'
    atomic_json(target, config)
    receipt['encoders'][encoder] = {'complete': True, 'config': str(target.relative_to(ROOT)), 'config_sha256': digest(target), 'inputs': records}
    print('READY', target.relative_to(ROOT), flush=True)
receipt['complete'] = all(value['complete'] for value in receipt['encoders'].values())
atomic_json(OUTPUT / 'dataset_configuration_receipt.json', receipt)

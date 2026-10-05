"""Verify restored and reconstructed feature exports without scoring models."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


report = {'scope': 'Input audit only. No model fitting or held-out scoring.', 'exports': [], 'shared_rows': []}
datasets = ['visual_entailment', 'sugarcrepe', 'sugarcrepe_pp', 'coco_karpathy']
settings = [('vit_b32', ROOT / 'results/resume_features', datasets)]
settings.append(('rn50', ROOT / 'results/strengthen_second_encoder/features', ['visual_entailment', 'review_followup/e_vil_dev900', 'review_followup/e_vil_test1000', 'sugarcrepe', 'sugarcrepe_pp', 'coco_karpathy']))
for encoder, feature_root, names in settings:
    for name in names:
        feature_path = feature_root / name / 'features.npz'
        metadata_path = feature_path.with_name('metadata.json')
        if not feature_path.is_file() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        manifest_path = ROOT / 'data' / name / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        feature_sha = digest(feature_path)
        assert feature_sha == metadata['features_sha256']
        assert digest(manifest_path) == metadata['manifest_sha256']
        entry = {'encoder': encoder, 'dataset': name, 'path': str(feature_path.relative_to(ROOT)), 'bytes': feature_path.stat().st_size, 'sha256': feature_sha, 'metadata_sha256': digest(metadata_path), 'manifest_sha256': digest(manifest_path), 'new_execution': True, 'arrays': {}}
        with np.load(feature_path, allow_pickle=False) as arrays:
            for kind, collection in [('image', 'images'), ('text', 'texts')]:
                ids = arrays[kind + '_ids']
                values = arrays[kind + '_features']
                assert np.array_equal(ids, np.asarray([row['id'] for row in manifest[collection]]))
                assert values.shape == (len(ids), 512 if encoder == 'vit_b32' else 1024)
                assert values.dtype == np.float32
                assert np.isfinite(values).all()
                assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5, rtol=0)
                for key in [kind + '_ids', kind + '_features']:
                    value = arrays[key]
                    entry['arrays'][key] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'c_order_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
            if encoder == 'vit_b32' and name == 'visual_entailment':
                for reference_name in ['e_vil_dev900', 'e_vil_test1000']:
                    reference_path = ROOT / 'results/review_followup/features' / reference_name / 'features.npz'
                    with np.load(reference_path, allow_pickle=False) as reference:
                        for kind in ['image', 'text']:
                            lookup = {str(value): index for index, value in enumerate(reference[kind + '_ids'])}
                            pairs = [(index, lookup[str(value)]) for index, value in enumerate(arrays[kind + '_ids']) if str(value) in lookup]
                            left = arrays[kind + '_features'][[x for x, _ in pairs]]
                            right = reference[kind + '_features'][[y for _, y in pairs]]
                            error = np.abs(left - right)
                            check = {'reference': str(reference_path.relative_to(ROOT)), 'modality': kind, 'rows': len(pairs), 'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean()), 'exact_arrays': bool(np.array_equal(left, right)), 'within_2e_6': bool(np.allclose(left, right, atol=2e-6, rtol=0))}
                            assert check['within_2e_6']
                            report['shared_rows'].append(check)
        report['exports'].append(entry)
report['complete'] = len(report['exports']) == 10
report['verified_export_count'] = len(report['exports'])
target = ROOT / 'recovery/current_input_recovery/feature_export_audit.json'
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'complete': report['complete'], 'verified_export_count': report['verified_export_count'], 'shared_rows': report['shared_rows']}, indent=2))

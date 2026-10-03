#!/usr/bin/env python3
"""Independent saved-array audit of completed relation diagnostic outputs."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', default='results/relation_diagnostic_iter10000')
    args = parser.parse_args()
    root, output = args.root, args.root / args.output
    receipt = json.loads((output / 'FILE_HASHES.json').read_text())
    for name, expected in receipt.items():
        path = output / name
        assert path.stat().st_size == expected['bytes'], name
        assert sha256(path) == expected['sha256'], name
    ledger = json.loads((output / 'protocol_ledger.json').read_text())
    for item in ledger['inputs'].values():
        assert sha256(item['path']) == item['sha256']
    assert sha256(root / 'scripts/relation_diagnostic.py') == ledger['source_sha256']
    manifest = json.loads(Path(ledger['inputs']['manifest']['path']).read_text())
    features = np.load(ledger['inputs']['features']['path'], allow_pickle=False)
    image_lookup = {str(name): index for index, name in enumerate(features['image_ids'])}
    text_lookup = {str(name): index for index, name in enumerate(features['text_ids'])}
    class_names = ['supported', 'contradicted', 'neutral']
    summary = json.loads((output / 'summary.json').read_text())
    findings = {'status': 'passed', 'receipt_files_verified': len(receipt), 'modes': {}}

    def reconstructed_inputs(prediction, mode):
        image = features['image_features'][[image_lookup[str(name)] for name in prediction['image_ids']]]
        text = features['text_features'][[text_lookup[str(name)] for name in prediction['text_ids']]]
        for row, pair_index in enumerate(prediction['manifest_pair_indices']):
            pair = manifest['pairs'][int(pair_index)]
            assert pair['relation'] != 'source'
            assert pair['image_id'] == prediction['image_ids'][row]
            assert pair['text_id'] == prediction['text_ids'][row]
            assert class_names.index(pair['relation']) == prediction['labels'][row]
        if mode == 'image':
            return image.astype(np.float64)
        if mode == 'text':
            return text.astype(np.float64)
        return np.concatenate([image, text, image * text, np.abs(image - text)], axis=1).astype(np.float64)

    def verify_prediction(mode, prediction_path, model_path, gates):
        prediction = np.load(prediction_path, allow_pickle=False)
        model = np.load(model_path, allow_pickle=False)
        x = reconstructed_inputs(prediction, mode)
        logits = ((x - model['scaler_mean']) / model['scaler_scale']) @ model['coefficients'].T + model['intercept']
        probabilities = softmax(logits / model['temperature'].item(), axis=1)
        score_error = float(np.max(np.abs(logits - prediction['logits'])))
        probability_error = float(np.max(np.abs(probabilities - prediction['probabilities'])))
        assert score_error < 1e-10 and probability_error < 1e-10, str(prediction_path)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), prediction['prediction'])
        for name, gate in gates.items():
            accepted = np.zeros(len(x), dtype=bool)
            if gate['threshold'] is not None:
                accepted = (probabilities.argmax(axis=1) == gate['class_index']) & (probabilities[:, gate['class_index']] >= gate['threshold'])
            np.testing.assert_array_equal(accepted, prediction['accepted_' + name])
        return prediction, {'rows': len(x), 'max_logit_error': score_error, 'max_probability_error': probability_error}

    with threadpool_limits(limits=2):
        for mode, result in summary['modes'].items():
            destination = output / mode
            checks = {}
            for partition in ('validation', 'temperature_fit', 'gate_selection', 'test'):
                _, checks[partition] = verify_prediction(mode, destination / (partition + '_predictions.npz'),
                                                         destination / 'classifier.npz', result['gates'])
            oof = np.load(destination / 'train_out_of_fold_predictions.npz', allow_pickle=False)
            assigned = np.zeros(len(oof['labels']), dtype=int)
            image_folds = {}
            for fold_record in result['crossfit']:
                fold = fold_record['fold']
                heldout = set(fold_record['held_out_image_ids'])
                training = set(fold_record['selection']['training_image_ids'])
                assert not heldout & training
                prediction, checks['fold_' + str(fold)] = verify_prediction(
                    mode, destination / f'crossfit_{fold}_predictions.npz', destination / f'crossfit_{fold}_classifier.npz', fold_record['gates'])
                mask = oof['fold_indices'] == fold
                assigned[mask] += 1
                assert set(prediction['image_ids']) == heldout
                assert set(oof['image_ids'][mask]) == heldout
                for image in heldout:
                    assert image not in image_folds
                    image_folds[image] = fold
                for key in ('image_ids', 'text_ids', 'labels', 'manifest_pair_indices', 'logits', 'probabilities', 'prediction',
                            'accepted_supported', 'accepted_contradicted'):
                    np.testing.assert_array_equal(oof[key][mask], prediction[key])
            assert np.all(assigned == 1) and len(image_folds) == 1200
            assert len(oof['labels']) == 16192
            test = np.load(destination / 'test_predictions.npz', allow_pickle=False)
            for name, gate_result in result['test_gates'].items():
                accepted = test['accepted_' + name]
                correct = int(np.sum(test['labels'][accepted] == class_names.index(name)))
                assert int(accepted.sum()) == gate_result['accepted']
                assert correct == gate_result['correct']
                assert accepted.mean() == gate_result['coverage']
                if accepted.any():
                    assert correct / accepted.sum() == gate_result['precision']
            iterations = [entry['iterations'][0] for selection in [result['selection']] + [fold['selection'] for fold in result['crossfit']]
                          for entry in selection['candidates']]
            assert max(iterations) < 10000
            findings['modes'][mode] = {'checks': checks, 'out_of_fold_rows': len(oof['labels']),
                                        'out_of_fold_images': len(image_folds), 'max_solver_iterations': max(iterations)}
    destination = root / 'results/relation_diagnostic_iter10000_audit.json'
    destination.write_text(json.dumps(findings, indent=2, sort_keys=True) + '\n')
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

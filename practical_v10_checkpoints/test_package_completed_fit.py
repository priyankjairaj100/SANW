"""Compact synthetic fixtures only; no trained arrays or actual outcomes loaded."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('preserve_fit', Path(__file__).with_name('package_completed_fit.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def entry(root, path):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': module.digest(path)}


def update_run(run, identity, history, completion):
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for row in history:
        row['checkpoint']['ledger_sha256'] = ledger_sha
    completion['checkpoint_history'] = history
    completion['history'] = [{k: v for k, v in row.items() if k != 'checkpoint'} for row in history]
    completion['ledger_sha256'] = ledger_sha
    write(run / 'ledger.json', {'identity': identity, 'ledger_sha256': ledger_sha})
    write(run / 'history.json', history); write(run / 'completion.json', completion)


def fixture(root, family='joint'):
    source = root / 'src/toy.py'; source.parent.mkdir(); source.write_text('# toy source\n')
    meta = root / 'inputs/metadata.json'; write(meta, {'fixture': True})
    protocol = {'fit_config': {'epochs': 32, 'batch_size': 64}, 'source_sha256': {'src/toy.py': module.digest(source)},
                'streaming': {'fit_threads': 1}, 'training_inputs': {'vit_b32': {'metadata': entry(root, meta)}},
                'original_training_inputs': {'vit_b32': {'metadata': entry(root, meta)}},
                'owner_sample': {}, 'confirmation_owner_lock': {}}
    protocol_path = root / module.PROTOCOL; write(protocol_path, protocol); protocol_sha = module.digest(protocol_path)
    cache = root / 'cache/metadata.json'
    write(cache, {'schema': 'sanw_frozen_score_cache_v10', 'images': 6000, 'texts': 30000,
                 'files': {'image_to_text.npy': {'bytes': 1440000128, 'sha256': 'e' * 64}}})
    run = root / 'results/toy_run'; (run / 'checkpoints').mkdir(parents=True)
    provenance = {'inputs': protocol['training_inputs']['vit_b32'],
                  'original_training_inputs': protocol['original_training_inputs']['vit_b32'],
                  'owner_sample': {}, 'confirmation_owner_lock': {}, 'training_image_manifest_indices': list(range(6000)),
                  'heldout_used': False, 'original_training_feature_bytes_identical': True}
    config = {'epochs': 32, 'batch_size': 64, 'seed': 17}
    identity = {'study': 'sanw_practical_v10', 'mode': 'full', 'encoder': 'vit_b32', 'family': family,
                'config': config, 'protocol_sha256': protocol_sha, 'protocol': entry(root, protocol_path),
                'source_sha256': protocol['source_sha256'], 'streaming': protocol['streaming'],
                'fit_gallery_image_count': 6000, 'fit_gallery_text_count': 30000,
                'official_development_or_benchmarks_used': False, 'training_provenance': provenance,
                'replication_gate': None, 'control_gate': None,
                'frozen_score_cache': {'path': str(cache.parent), 'metadata_sha256': module.digest(cache)}}
    if family == 'no_retention':
        pilots = []
        for encoder in ('vit_b32', 'rn50'):
            p = root / 'pilot_gate' / encoder
            checkpoint = p / 'selected.npz'; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'synthetic checkpoint')
            complete = p / 'completion.json'; result = p / 'result.json'
            write(complete, {'study': 'sanw_practical_v10', 'encoder': encoder, 'protocol_sha256': protocol_sha,
                             'config': {'seed': 17}, 'selected_checkpoint': {'sha256': module.digest(checkpoint)}})
            write(result, {'study': 'sanw_practical_v10_official_development', 'passed': True, 'encoder': encoder,
                           'seed': 17, 'protocol_sha256': protocol_sha, 'checkpoint_sha256': module.digest(checkpoint)})
            pilots.append({'encoder': encoder, 'completion': entry(root, complete), 'checkpoint': entry(root, checkpoint),
                           'result': entry(root, result)})
        gate_path = root / 'gate.json'
        write(gate_path, {'study': 'sanw_practical_v10_replication_gate', 'family': 'joint', 'passed': True,
                          'protocol_sha256': protocol_sha, 'encoders_passed': ['vit_b32', 'rn50'], 'seed': 17, 'pilot_runs': pilots})
        identity['control_gate'] = entry(root, gate_path)
    certificate = {'feasible_with_tolerance': True, 'ranking_checked_canonically': True, 'ranking_preserved': True}
    history = []
    for epoch in range(1, 33):
        checkpoint = run / f'checkpoints/epoch_{epoch:03d}.npz'
        with zipfile.ZipFile(checkpoint, 'w') as z:
            for name in ('schema', 'image_mean', 'text_mean', 'image_basis', 'text_basis', 'coefficient'):
                # Deliberately opaque toy members: preservation compares bytes, not array values.
                z.writestr(name + '.npy', f'synthetic {name} epoch{epoch}'.encode())
        row = {'epoch': epoch, 'optimizer_steps': epoch * 94, 'nonzero': True,
               'training_objective': 1.0 if epoch in (17, 29) else 2.0,
               'checkpoint': {'path': str(checkpoint.relative_to(run)), 'sha256': module.digest(checkpoint)}}
        row['certificate' if family == 'joint' else 'retention_enforced'] = certificate if family == 'joint' else False
        history.append(row)
    selected = run / 'selected.npz'; shutil.copyfile(run / 'checkpoints/epoch_017.npz', selected)
    completion = {'study': 'sanw_practical_v10', 'mode': 'full', 'encoder': 'vit_b32', 'family': family,
                  'config': config, 'protocol_sha256': protocol_sha, 'development_or_test_used': False,
                  'optimizer_steps': 3008, 'selected_epoch': 17, 'selected_training_objective': 1.0,
                  'selected_checkpoint': {'path': 'selected.npz', 'sha256': module.digest(selected)},
                  'selection': 'minimum_feasible_nonzero_training_objective_then_earliest_epoch' if family == 'joint' else
                               'minimum_nonzero_training_objective_then_earliest_epoch_without_feasibility_filter'}
    if family == 'joint':
        completion['final_certificate'] = certificate
    else:
        completion.update(retention_enforced=False, final_training_retention_diagnostic={'ranking_checked_canonically': True,
                                                                                         'ranking_preserved': False})
    update_run(run, identity, history, completion)
    return run, protocol_sha, identity, history, completion


class PreservationTests(unittest.TestCase):
    def test_both_families_package_and_every_tar_member_is_verified(self):
        for family in ('joint', 'no_retention'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); run, protocol_sha, *_ = fixture(root, family)
                output = root / 'out.tar.xz'
                with patch.object(module, 'PROTOCOL_SHA', protocol_sha):
                    result = module.package(root, run, output)
                self.assertEqual(result['selected_epoch'], 17)
                self.assertEqual(result['optimizer_steps'], 3008)
                self.assertFalse(result['numerical_audits_performed'])
                self.assertFalse((root / 'cache/image_to_text.npy').exists())
                manifest = json.loads(Path(result['manifest']).read_text())
                with tarfile.open(output, 'r:xz') as tar:
                    self.assertEqual(len(tar.getmembers()), result['verified_tar_members'])
                    for row in manifest['files']:
                        self.assertEqual(hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest(), row['sha256'])

    def test_middle_update_budget_and_earliest_tie_are_checked(self):
        for damage in ('budget', 'tie'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); run, protocol_sha, identity, history, completion = fixture(root)
                if damage == 'budget':
                    history[10]['optimizer_steps'] -= 1
                else:
                    completion['selected_epoch'] = 29
                update_run(run, identity, history, completion)
                with patch.object(module, 'PROTOCOL_SHA', protocol_sha), self.assertRaises(ValueError):
                    module.inspect_run(root, run)

    def test_checkpoint_source_and_cache_mutation_are_rejected(self):
        for relative in ('src/toy.py', 'results/toy_run/checkpoints/epoch_005.npz', 'cache/metadata.json'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); run, protocol_sha, *_ = fixture(root)
                target = root / relative; target.write_bytes(target.read_bytes() + b'changed')
                with patch.object(module, 'PROTOCOL_SHA', protocol_sha), self.assertRaises(ValueError):
                    module.inspect_run(root, run)

    def test_zero_infeasible_and_control_rules_remain_distinct(self):
        certificate = {'feasible_with_tolerance': True, 'ranking_checked_canonically': True, 'ranking_preserved': True}
        rows = [{'epoch': 1, 'training_objective': 0.0, 'nonzero': False, 'certificate': certificate},
                {'epoch': 2, 'training_objective': 1.0, 'nonzero': True, 'certificate': certificate}]
        self.assertEqual(module.choose_epoch(rows, 'joint')['epoch'], 2)
        damaged = copy.deepcopy(rows); damaged[1]['certificate']['feasible_with_tolerance'] = False
        with self.assertRaises(ValueError):
            module.choose_epoch(damaged, 'joint')
        controls = [{**row, 'retention_enforced': False} for row in damaged]
        self.assertEqual(module.choose_epoch(controls, 'no_retention')['epoch'], 2)

    def test_concurrent_output_and_temporary_file_are_never_clobbered(self):
        for name in ('out.tar.xz', 'out.tar.xz.tmp', 'out.tar.xz.manifest.json', 'out.tar.xz.summary.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); run, protocol_sha, *_ = fixture(root)
                output = root / 'out.tar.xz'; competing = root / name
                original_inspect = module.inspect_run
                def competing_creation(*args):
                    result = original_inspect(*args)
                    competing.write_bytes(b'concurrent writer bytes')
                    return result
                with patch.object(module, 'PROTOCOL_SHA', protocol_sha), \
                        patch.object(module, 'inspect_run', side_effect=competing_creation), \
                        self.assertRaises(FileExistsError):
                    module.package(root, run, output)
                self.assertEqual(competing.read_bytes(), b'concurrent writer bytes')


if __name__ == '__main__':
    unittest.main(verbosity=2)

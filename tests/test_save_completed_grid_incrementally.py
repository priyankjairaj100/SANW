"""Exercise orchestration with disposable checkpoints and a mocked uploader."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import save_completed_grid_incrementally as wrapper
import stream_completed_candidate_checkpoints as archive

SPEC = importlib.util.spec_from_file_location('fixture_archive_tests', Path(__file__).with_name('test_archive_completed_grid_checkpoints.py'))
fixture_tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_tests)


class IncrementalWrapperTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_tests.ArchiveTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.states, self.retained = self.fixture.fixture('ad')
        self.root = self.fixture.root
        self.original = {state['checkpoint']: (self.root / state['checkpoint']).read_bytes() for state in self.states}
        self.helper = self.root / 'mock_upload.py'
        self.helper.write_text('# Only subprocess.run is mocked.\n')
        self.args = SimpleNamespace(grid_root='grid', prefix='campaign', expected_completions=4,
            upload_helper=self.helper, private_output=self.root / 'private', batch_size=1,
            allow_unselected_incomplete_grid=False)
        self.calls = 0
        self.remote = {}
        self.first_members = []

    def uploader(self, command, *, input, text, stdout, stderr):
        self.calls += 1
        request = json.loads(input)
        records = []
        for i, upload in enumerate(request['uploads']):
            path = Path(upload['local_path'])
            self.remote[str(path)] = path.read_bytes()
            records.append({**upload, 'status': 'succeeded', 'file_id': f'file_{self.calls}_{i}',
                            'library_file_id': f'library_{self.calls}_{i}'})
            if self.calls == 1 and path.name.endswith('_MANIFEST.json'):
                self.first_members = json.loads(path.read_text())['snapshot']['members']
        json.dump({'results': records}, stdout)
        return SimpleNamespace(returncode=0)

    def execute(self, uploader=None):
        with patch.object(wrapper.subprocess, 'run', side_effect=uploader or self.uploader), contextlib.redirect_stdout(io.StringIO()):
            return wrapper.run(self.args, repository=self.root)

    def assert_originals_present(self):
        self.assertTrue(all((self.root / name).read_bytes() == value for name, value in self.original.items()))

    def test_success_preserves_selected_states_and_uploads_every_removed_byte(self):
        result = self.execute()
        self.assertTrue(result['complete'])
        self.assertEqual(self.calls, 4)
        self.assertEqual(len(result['batches']), 4)
        selected = {state['checkpoint'] for state in self.retained}
        for state in self.states:
            path = self.root / state['checkpoint']
            if state['checkpoint'] in selected:
                self.assertEqual(path.read_bytes(), self.original[state['checkpoint']])
            else:
                self.assertFalse(path.exists())
        self.assertFalse(list(Path(result['output']).glob('*.zip')))
        self.assertEqual(len(list(self.args.private_output.glob('*_upload_record.json'))), 4)
        self.assertEqual(len(self.remote), 8)
        self.assertTrue(json.loads((Path(result['output']) / 'progress.json').read_text())['complete'])

    def test_nonzero_or_unknown_upload_stops_without_retry_or_prune(self):
        def failed(*pos, **kwargs):
            self.calls += 1
            kwargs['stdout'].write('{"unknown_outcome": true}')
            return SimpleNamespace(returncode=3)
        with self.assertRaisesRegex(RuntimeError, 'No automatic retry'):
            self.execute(failed)
        self.assertEqual(self.calls, 1)
        self.assert_originals_present()
        self.assertFalse(list(self.root.rglob('*PRUNE*')))

    def test_partial_success_stops_without_prune(self):
        def partial(*pos, **kwargs):
            self.calls += 1
            request = json.loads(kwargs['input'])
            records = [{**row, 'status': 'succeeded' if i == 0 else 'failed',
                        'file_id': 'file', 'library_file_id': 'library'}
                       for i, row in enumerate(request['uploads'])]
            json.dump({'results': records}, kwargs['stdout'])
            return SimpleNamespace(returncode=0)
        with self.assertRaisesRegex(ValueError, 'not finalized'):
            self.execute(partial)
        self.assertEqual(self.calls, 1)
        self.assert_originals_present()

    def test_malformed_upload_response_stops_without_retry(self):
        def malformed(*pos, **kwargs):
            self.calls += 1
            kwargs['stdout'].write('incomplete-json')
            return SimpleNamespace(returncode=0)
        with self.assertRaises(json.JSONDecodeError):
            self.execute(malformed)
        self.assertEqual(self.calls, 1)
        self.assert_originals_present()

    def test_existing_batch_and_unknown_outcome_cannot_be_restarted(self):
        def failed(*pos, **kwargs):
            self.calls += 1
            return SimpleNamespace(returncode=5)
        with self.assertRaises(RuntimeError):
            self.execute(failed)
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            self.execute()
        self.assertEqual(self.calls, 1)
        self.assert_originals_present()

    def test_existing_private_upload_receipt_blocks_new_attempt(self):
        self.args.private_output.mkdir()
        (self.args.private_output / 'campaign_batch_001_result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Private batch records'):
            self.execute()
        self.assertEqual(self.calls, 0)
        self.assert_originals_present()

    def test_second_batch_failure_keeps_its_originals(self):
        def second_failed(*pos, **kwargs):
            if self.calls == 1:
                self.calls += 1
                return SimpleNamespace(returncode=7)
            return self.uploader(*pos, **kwargs)
        with self.assertRaises(RuntimeError):
            self.execute(second_failed)
        self.assertEqual(self.calls, 2)
        removed = {x['path'] for x in self.first_members}
        for path, value in self.original.items():
            if path in removed:
                self.assertFalse((self.root / path).exists())
            else:
                self.assertEqual((self.root / path).read_bytes(), value)

    def test_changed_upload_source_prevents_prune(self):
        def changed(*pos, **kwargs):
            result = self.uploader(*pos, **kwargs)
            request = json.loads(kwargs['input'])
            Path(request['uploads'][1]['local_path']).write_bytes(b'changed ZIP after upload')
            return result
        with self.assertRaisesRegex(ValueError, 'source bytes changed'):
            self.execute(changed)
        self.assert_originals_present()

    def test_reappearing_verified_bytes_are_removed_again(self):
        def reappears(*pos, **kwargs):
            result = self.uploader(*pos, **kwargs)
            if self.calls == 2:
                for item in self.first_members:
                    (self.root / item['path']).write_bytes(self.original[item['path']])
            return result
        self.execute(reappears)
        for item in self.first_members:
            self.assertFalse((self.root / item['path']).exists())

    def test_changed_reappearing_bytes_are_preserved_and_stop(self):
        def changed(*pos, **kwargs):
            result = self.uploader(*pos, **kwargs)
            if self.calls == 2:
                (self.root / self.first_members[0]['path']).write_bytes(b'changed reappearance')
            return result
        with self.assertRaisesRegex(ValueError, 'Reappearing retired bytes differ'):
            self.execute(changed)
        self.assertEqual(self.calls, 2)
        self.assertEqual((self.root / self.first_members[0]['path']).read_bytes(), b'changed reappearance')

    def test_incomplete_grid_only_archives_completed_base_candidates(self):
        # Keep only the base completion. Matched runs become unfinished directories.
        for completion in self.fixture.grid.rglob('completion.json'):
            if 'matched_' in str(completion):
                completion.unlink()
        for name in ('state_manifest.json', 'selection_primary.json', 'selection_sensitivity.json', 'archival_keep_manifest.json'):
            (self.fixture.grid / name).unlink()
        self.args.allow_unselected_incomplete_grid = True
        self.args.expected_completions = 1
        partial_originals = {name: value for name, value in self.original.items() if 'matched_' in name}
        result = self.execute()
        self.assertEqual(self.calls, 1)
        self.assertIsNone(result['retained_manifest_sha256'])
        self.assertTrue(all((self.root / name).read_bytes() == value for name, value in partial_originals.items()))
        self.assertTrue((self.fixture.grid / 'candidates/supported/lr_0.001/seed_17/epochs/epoch_02.pt').exists())
        self.assertFalse((self.fixture.grid / 'candidates/supported/lr_0.001/seed_17/epochs/epoch_00.pt').exists())

    def test_verified_resume_skips_saved_batches_without_upload(self):
        original = self.execute()
        self.args.resume_verified = True
        resumed = self.execute()
        self.assertEqual(self.calls, 4)
        self.assertEqual(len(resumed['batches']), len(original['batches']))
        self.assertTrue(resumed['complete'])

    def test_verified_resume_continues_after_preupload_storage_failure(self):
        real_create = archive.create
        creates = []
        def fail_before_second(args):
            creates.append(args.prefix)
            if len(creates) == 2:
                raise ValueError('Resident archive bound exceeded')
            return real_create(args)
        with patch.object(archive, 'create', side_effect=fail_before_second):
            with self.assertRaisesRegex(ValueError, 'Resident archive bound|Prefix already exists'):
                self.execute()
        self.assertEqual(self.calls, 1)
        self.args.resume_verified = True
        result = self.execute()
        self.assertTrue(result['complete'])
        self.assertEqual(self.calls, 4)

    def test_verified_resume_refuses_unknown_upload_outcome(self):
        def second_failed(*pos, **kwargs):
            if self.calls == 1:
                self.calls += 1
                return SimpleNamespace(returncode=7)
            return self.uploader(*pos, **kwargs)
        with self.assertRaises(RuntimeError):
            self.execute(second_failed)
        self.args.resume_verified = True
        with self.assertRaisesRegex(ValueError, 'lacks completed prune/upload'):
            self.execute()
        self.assertEqual(self.calls, 2)

    def test_resume_verifies_upload_record_digest(self):
        self.execute()
        path = self.args.private_output / 'campaign_batch_001_upload_record.json'
        value = json.loads(path.read_text())
        value['results'][0]['file_id'] = 'different'
        path.write_text(json.dumps(value))
        self.args.resume_verified = True
        with self.assertRaisesRegex(ValueError, 'receipt digest mismatch'):
            self.execute()
        self.assertEqual(self.calls, 4)

    def test_resume_rejects_changed_batch_schedule(self):
        self.execute()
        self.args.resume_verified = True
        self.args.batch_size = 2
        with self.assertRaisesRegex(ValueError, 'candidate schedule'):
            self.execute()
        self.assertEqual(self.calls, 4)

    def test_completed_partial_cleanup_ignores_incomplete_bytes_and_unrelated_files(self):
        result = self.execute()
        part = Path(result['batches'][0]['parts'][0]['local_path'])
        partial = Path(str(part) + '.partial')
        partial.write_bytes(b'incomplete synchronized temporary bytes')
        unrelated = part.parent / 'unrelated.zip.partial'
        unrelated.write_bytes(b'not owned by this completed batch')
        self.args.resume_verified = True
        self.execute()
        self.assertFalse(partial.exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual(self.calls, 4)

    def test_resume_checks_all_completed_receipts_before_cleanup_or_new_upload(self):
        result = self.execute()
        first_part = Path(result['batches'][0]['parts'][0]['local_path'])
        partial = Path(str(first_part) + '.partial')
        partial.write_bytes(b'leftover')
        prune_path = Path(result['batches'][1]['prune']['prune_receipt']['local_path'])
        value = json.loads(prune_path.read_text())
        value['status'] = 'in_progress'
        prune_path.write_text(json.dumps(value))
        self.args.resume_verified = True
        with self.assertRaisesRegex(ValueError, 'prune is incomplete'):
            self.execute()
        self.assertTrue(partial.exists())
        self.assertEqual(self.calls, 4)

    def test_cleanup_after_inspection_removes_only_previous_completed_partial(self):
        real_inspect = archive.inspect_candidates
        injected = []
        def inspect_with_reappearance(*args, **kwargs):
            snapshot = real_inspect(*args, **kwargs)
            first_manifest = next((json.loads(data) for path, data in self.remote.items()
                                   if path.endswith('batch_001_MANIFEST.json')), None)
            if (self.calls == 1 and first_manifest and
                snapshot['candidate_names'] != first_manifest['snapshot']['candidate_names']):
                name = next(path for path in self.remote if path.endswith('batch_001_part_0001.zip'))
                partial = Path(name + '.partial')
                with partial.open('wb') as stream:
                    stream.truncate(600 * 1024 * 1024)
                injected.append(partial)
            return snapshot
        with patch.object(archive, 'inspect_candidates', side_effect=inspect_with_reappearance):
            result = self.execute()
        self.assertTrue(result['complete'])
        self.assertEqual(len(injected), 1)
        self.assertFalse(injected[0].exists())
        self.assertEqual(self.calls, 4)

    def test_cleanup_between_parts_preserves_current_unuploaded_archive(self):
        self.args.batch_size = 2
        real_verify = archive.verify_part
        injected = []
        def verify_with_reappearance(path, members, limit):
            total = real_verify(path, members, limit)
            if path.name == 'campaign_batch_002_part_0001.zip.partial':
                previous = next(name for name in self.remote if name.endswith('batch_001_part_0001.zip'))
                partial = Path(previous + '.partial')
                with partial.open('wb') as stream:
                    stream.truncate(600 * 1024 * 1024)
                injected.append(partial)
            return total
        with patch.object(archive, 'verify_part', side_effect=verify_with_reappearance):
            result = self.execute()
        self.assertTrue(result['complete'])
        self.assertEqual(len(injected), 1)
        self.assertFalse(injected[0].exists())
        self.assertEqual(self.calls, 2)
        second_parts = [name for name in self.remote if 'batch_002_part_' in name]
        self.assertEqual(len(second_parts), 2)

    def test_callback_does_not_remove_current_unverified_partial(self):
        real_inspect = archive.inspect_candidates
        injected = []
        def inspect_with_current_copy(*args, **kwargs):
            snapshot = real_inspect(*args, **kwargs)
            output = self.root / 'recovery/incremental_archives_20261004/campaign'
            partial = output / 'campaign_batch_001_part_0001.zip.partial'
            with partial.open('wb') as stream:
                stream.truncate(600 * 1024 * 1024)
            injected.append(partial)
            return snapshot
        with patch.object(archive, 'inspect_candidates', side_effect=inspect_with_current_copy):
            with self.assertRaisesRegex(ValueError, 'Resident archive bound|Prefix already exists'):
                self.execute()
        self.assertEqual(self.calls, 0)
        self.assertTrue(injected[0].exists())
        self.assert_originals_present()

    def test_completed_zip_disappearing_during_hash_is_idempotent(self):
        result = self.execute()
        made = result['batches'][0]
        target = Path(made['parts'][0]['local_path'])
        target.write_bytes(self.remote[str(target)])
        real_receipt = archive.file_receipt
        def disappearing(path):
            if path == target:
                path.unlink()
                raise FileNotFoundError(str(path))
            return real_receipt(path)
        with patch.object(archive, 'file_receipt', side_effect=disappearing):
            wrapper.cleanup_completed_temporary_parts(self.root, made)
        self.assertFalse(target.exists())

    def test_completed_zip_disappearing_before_unlink_is_idempotent(self):
        result = self.execute()
        made = result['batches'][0]
        target = Path(made['parts'][0]['local_path'])
        target.write_bytes(self.remote[str(target)])
        real_unlink = Path.unlink
        def disappearing(path, *args, **kwargs):
            if path == target:
                real_unlink(path, missing_ok=True)
                raise FileNotFoundError(str(path))
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', autospec=True, side_effect=disappearing):
            wrapper.cleanup_completed_temporary_parts(self.root, made)
        self.assertFalse(target.exists())

    def test_retired_original_disappearing_during_digest_is_idempotent(self):
        self.execute()
        item = self.first_members[0]
        target = self.root / item['path']
        target.write_bytes(self.original[item['path']])
        real_digest = archive.digest
        def disappearing(path):
            if path == target:
                path.unlink()
                raise FileNotFoundError(str(path))
            return real_digest(path)
        with patch.object(archive, 'digest', side_effect=disappearing):
            wrapper.cleanup_retired(self.root, [item], [])
        self.assertFalse(target.exists())

    def abandoned_second_creation(self):
        real_create = archive.create
        abandoned = []
        def fail_mid_creation(args):
            if args.prefix.endswith('batch_002'):
                path = args.output_root / (args.prefix + '_part_0001.zip')
                path.write_bytes(b'owned unfinished archive creation')
                abandoned.append(path)
                raise ValueError('Interrupted temporary ZIP creation')
            return real_create(args)
        with patch.object(archive, 'create', side_effect=fail_mid_creation):
            with self.assertRaisesRegex(ValueError, 'Interrupted temporary'):
                self.execute()
        self.assertEqual(self.calls, 1)
        self.args.recover_unuploaded_batch = 2
        return abandoned[0]

    def test_explicit_unuploaded_recovery_preserves_originals_and_allows_resume(self):
        target = self.abandoned_second_creation()
        before = {name: (self.root / name).read_bytes() for name in self.original if (self.root / name).exists()}
        result = self.execute()
        self.assertEqual(result['recovered_unuploaded_batch'], 2)
        self.assertFalse(target.exists())
        self.assertEqual(self.calls, 1)
        self.assertTrue(all((self.root / name).read_bytes() == data for name, data in before.items()))
        self.args.recover_unuploaded_batch = None
        self.args.resume_verified = True
        self.assertTrue(self.execute()['complete'])
        self.assertEqual(self.calls, 4)

    def test_unuploaded_recovery_refuses_any_upload_request(self):
        target = self.abandoned_second_creation()
        (self.args.private_output / 'campaign_batch_002_request.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Upload request or outcome exists'):
            self.execute()
        self.assertTrue(target.exists())
        self.assertEqual(self.calls, 1)

    def test_unuploaded_recovery_requires_unchanged_scientific_originals(self):
        target = self.abandoned_second_creation()
        original = next(self.root.rglob('matched_allocation_distillation_draw_1/lr_0.001/seed_17/epochs/epoch_00.pt'))
        original.write_bytes(b'changed original')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.execute()
        self.assertTrue(target.exists())
        self.assertEqual(self.calls, 1)

    def test_completed_creation_recovery_accepts_exact_resurrection_then_preserves_new_parts(self):
        target = self.abandoned_second_creation()
        old_bytes = target.read_bytes()
        self.execute()  # Explicit recovery records the exact removed bytes.
        target.write_bytes(old_bytes)  # Synchronization restores that same copy.
        self.args.recover_unuploaded_batch = None
        self.args.resume_verified = True
        result = self.execute()
        self.assertTrue(result['complete'])
        self.assertEqual(self.calls, 4)
        # The new ZIP survived its one-shot hook, uploaded, and pruned normally.
        self.assertIn(str(target), self.remote)
        self.assertNotEqual(self.remote[str(target)], old_bytes)
        # Old recovery receipts cannot shadow subsequent completed archive batches.
        self.assertTrue(self.execute()['complete'])
        self.assertEqual(self.calls, 4)

    def test_creation_recovery_hook_handles_resurrection_after_resume_preflight(self):
        target = self.abandoned_second_creation()
        old_bytes = target.read_bytes()
        self.execute()
        self.args.recover_unuploaded_batch = None
        self.args.resume_verified = True
        real_create = archive.create
        injections = []
        def inject_after_preflight(args):
            if args.prefix.endswith('batch_002'):
                target.write_bytes(old_bytes)
                injections.append(str(target))
            return real_create(args)
        with patch.object(archive, 'create', side_effect=inject_after_preflight):
            self.assertTrue(self.execute()['complete'])
        self.assertEqual(injections, [str(target)])
        self.assertEqual(self.calls, 4)

    def test_creation_recovery_reconciliation_refuses_unknown_upload_request(self):
        target = self.abandoned_second_creation()
        old_bytes = target.read_bytes()
        self.execute()
        target.write_bytes(old_bytes)
        (self.args.private_output / 'campaign_batch_002_request.json').write_text('{}')
        self.args.recover_unuploaded_batch = None
        self.args.resume_verified = True
        with self.assertRaisesRegex(ValueError, 'Upload request or outcome exists'):
            self.execute()
        self.assertEqual(target.read_bytes(), old_bytes)
        self.assertEqual(self.calls, 1)

    def test_creation_recovery_reconciliation_refuses_changed_temporary(self):
        target = self.abandoned_second_creation()
        self.execute()
        target.write_bytes(b'changed old temporary')
        self.args.recover_unuploaded_batch = None
        self.args.resume_verified = True
        with self.assertRaisesRegex(ValueError, 'temporary bytes differ'):
            self.execute()
        self.assertEqual(target.read_bytes(), b'changed old temporary')
        self.assertEqual(self.calls, 1)

    def test_creation_recovery_reconciliation_refuses_unknown_temporary(self):
        target = self.abandoned_second_creation()
        self.execute()
        unknown = target.with_name('campaign_batch_002_part_0002.zip')
        unknown.write_bytes(b'not recorded by explicit recovery')
        self.args.recover_unuploaded_batch = None
        self.args.resume_verified = True
        with self.assertRaisesRegex(ValueError, 'Unknown temporary'):
            self.execute()
        self.assertTrue(unknown.exists())
        self.assertEqual(self.calls, 1)


if __name__ == '__main__':
    unittest.main()

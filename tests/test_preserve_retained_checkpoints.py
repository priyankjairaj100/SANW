"""Copy-only preservation tests use disposable checkpoint fixtures."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import preserve_retained_checkpoints as backup

SPEC = importlib.util.spec_from_file_location('backup_fixture', Path(__file__).with_name('test_archive_completed_grid_checkpoints.py'))
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
archive = backup.archive


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ArchiveTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.states, self.retained = self.fixture.fixture()
        self.root = self.fixture.root
        self.originals = {state['checkpoint']: (self.root / state['checkpoint']).read_bytes() for state in self.states}
        records = [archive.record(self.root, self.root / state['checkpoint'], state['checkpoint_sha256'])
                   for state in self.retained]
        inventory = {'purpose': backup.INVENTORY_PURPOSE, 'original_checkpoint_deletion_permitted': False,
            'completed_runs': 4, 'retained_checkpoint_count': len(records),
            'retained_checkpoint_bytes': sum(item['bytes'] for item in records), 'grids': [{
                'completed_runs': 4, 'retained_checkpoints': records,
                'state_manifest': archive.record(self.root, self.root / 'grid/state_manifest.json'),
                'retained_mapping': archive.record(self.root, self.root / 'grid/archival_keep_manifest.json')}]}
        path = self.root / 'inventory.json'
        archive.write_new(path, inventory)
        self.init_args = SimpleNamespace(repository=self.root, inventory='inventory.json',
            inventory_sha256=archive.digest(path), expected_runs=4, prefix='preserved',
            part_limit_bytes=max(archive.zip_upper_bound([item]) for item in records) + 100,
            parts_per_batch=2, resident_limit_bytes=1024 * 1024,
            output_root=self.root / 'backup', private_output=self.root / 'private_backup')

    def initialize(self):
        result = backup.initialize(self.init_args)
        self.args = SimpleNamespace(plan=Path(result['plan']['local_path']), plan_sha256=result['plan']['sha256'])
        self.plan = archive.read_json(self.args.plan)
        return result

    def assert_originals_unchanged(self):
        for name, value in self.originals.items():
            self.assertEqual((self.root / name).read_bytes(), value)

    def results(self, stage, *, status='succeeded'):
        request = archive.read_json(Path(stage['upload_request']))
        response = {'results': [{**item, 'status': status, 'file_id': f'file_{i}', 'library_file_id': f'library_{i}'}
                                for i, item in enumerate(request['uploads'])]}
        Path(stage['expected_result']).write_text(json.dumps(response))
        return response

    def finish(self, stage):
        return backup.finish_upload(SimpleNamespace(**vars(self.args), batch=stage['batch']))

    def test_batched_preservation_saves_all_weights_and_only_cleans_previously_saved_archives(self):
        result = self.initialize()
        self.assertEqual(result['parts'], 4)
        self.assertEqual(result['batches'], 2)
        self.assertFalse(list(self.root.rglob('*.zip')))
        for number in (1, 2):
            stage = backup.create_next(self.args)
            self.assertEqual(stage['batch'], number)
            self.assertEqual(len(stage['parts']), 2)
            self.assertFalse(Path(stage['expected_result']).exists())
            self.results(stage)
            finished = self.finish(stage)
            self.assertFalse(finished['originals_deleted'])
            for item in finished['saved_temporary_parts']:
                self.assertEqual(archive.file_receipt(Path(item['local_path'])), item)
            self.assert_originals_unchanged()
            self.assertEqual(self.finish(stage)['completed_batches'], number)
        self.assertTrue(finished['complete'])
        self.assertTrue(backup.create_next(self.args)['complete'])
        self.assertEqual(len(list(self.init_args.output_root.glob('*.zip'))), 2)

    def test_verified_saved_temporary_zip_may_be_absent_on_resume(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage)
        saved = self.finish(stage)
        # Simulate the root's separate cleanup of verified temporary copies.
        for item in saved['saved_temporary_parts']:
            Path(item['local_path']).unlink()
        self.assertEqual(backup.create_next(self.args)['batch'], 2)
        self.assert_originals_unchanged()

    def test_failed_upload_never_writes_success_or_deletes_anything(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage, status='failed')
        with self.assertRaisesRegex(ValueError, 'Every upload must succeed'):
            self.finish(stage)
        self.assertFalse(backup.locations(self.plan, 1)['verified_upload'].exists())
        self.assertEqual(len(list(self.init_args.output_root.glob('*.zip'))), 2)
        self.assert_originals_unchanged()

    def test_unknown_upload_cannot_be_recreated(self):
        self.initialize()
        stage = backup.create_next(self.args)
        Path(stage['expected_result']).write_text('unknown outcome')
        with self.assertRaisesRegex(ValueError, 'automatic retry are forbidden'):
            backup.create_next(self.args)
        self.assertEqual(Path(stage['expected_result']).read_text(), 'unknown outcome')
        self.assert_originals_unchanged()

    def test_changed_zip_after_upload_prevents_success(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage)
        backup.part_path(self.plan, 1).write_bytes(b'changed upload source')
        with self.assertRaisesRegex(ValueError, 'archive hash or size differs'):
            self.finish(stage)
        self.assertFalse(backup.locations(self.plan, 1)['verified_upload'].exists())
        self.assert_originals_unchanged()

    def test_changed_manifest_after_upload_prevents_false_source_binding(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage)
        path = backup.locations(self.plan, 1)['manifest']
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Pre-upload source bytes or manifest changed'):
            self.finish(stage)
        self.assertFalse(backup.locations(self.plan, 1)['verified_upload'].exists())
        self.assert_originals_unchanged()

    def test_changed_original_prevents_success_but_is_never_removed(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage)
        source = self.root / self.plan['parts'][0][0]['path']
        source.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.finish(stage)
        self.assertEqual(source.read_bytes(), b'changed checkpoint')
        self.assertFalse(backup.locations(self.plan, 1)['verified_upload'].exists())

    def test_upload_digest_disagreement_is_not_overwritten(self):
        self.initialize()
        stage = backup.create_next(self.args)
        response = self.results(stage)
        response['results'][0]['sha256'] = '0' * 64
        Path(stage['expected_result']).write_text(json.dumps(response))
        with self.assertRaisesRegex(ValueError, 'Upload result source digest or size differs'):
            self.finish(stage)
        self.assert_originals_unchanged()

    def test_resident_bound_blocks_allocation_without_removing_existing_files(self):
        self.initialize()
        path = self.init_args.output_root / 'unrelated.zip'
        path.write_bytes(b'x' * self.init_args.resident_limit_bytes)
        with self.assertRaisesRegex(ValueError, 'Resident bound exceeded'):
            backup.create_next(self.args)
        self.assertEqual(path.stat().st_size, self.init_args.resident_limit_bytes)
        self.assertFalse(backup.part_path(self.plan, 1).exists())
        self.assert_originals_unchanged()

    def test_insufficient_disk_blocks_creation_before_any_part(self):
        self.initialize()
        with patch.object(backup.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(ValueError, 'Insufficient free disk'):
                backup.create_next(self.args)
        self.assertFalse(backup.part_path(self.plan, 1).exists())
        self.assert_originals_unchanged()

    def test_init_refuses_missing_source_and_creates_no_output(self):
        (self.root / self.retained[0]['checkpoint']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing regular file'):
            self.initialize()
        self.assertFalse(self.init_args.output_root.exists())
        self.assertFalse(self.init_args.private_output.exists())

    def test_changed_saved_receipt_blocks_later_batches(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage)
        self.finish(stage)
        path = backup.locations(self.plan, 1)['verified_upload']
        record = archive.read_json(path)
        record['sources'][0]['sha256'] = '0' * 64
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Verified upload record differs'):
            backup.create_next(self.args)
        self.assert_originals_unchanged()

    def test_default_240_mib_limit_works_without_imported_128_mib_verifier(self):
        self.init_args.part_limit_bytes = backup.PART_LIMIT
        self.init_args.resident_limit_bytes = 512 * 1024 * 1024
        self.initialize()
        with patch.object(archive, 'verify_part', side_effect=AssertionError('Do not call the pruning verifier')):
            stage = backup.create_next(self.args)
            self.results(stage)
            self.assertTrue(self.finish(stage)['complete'])
        self.assert_originals_unchanged()

    def test_verifier_accepts_real_member_over_128_mib_and_enforces_requested_bound(self):
        # Highly compressible bytes keep this regression small on disk while
        # exercising a real member larger than the imported helper's hard cap.
        block = b'checkpoint-fixture' * 65536
        blocks = (129 * 1024 * 1024) // len(block) + 1
        digest = hashlib.sha256()
        for _ in range(blocks):
            digest.update(block)
        member = {'path': 'weights/large.pt', 'bytes': len(block) * blocks, 'sha256': digest.hexdigest()}
        path = self.root / 'large.zip'
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
            handle.writestr('MANIFEST.json', archive.manifest_bytes([member]))
            with handle.open(member['path'], 'w') as stream:
                for _ in range(blocks):
                    stream.write(block)
        with self.assertRaisesRegex(ValueError, 'ZIP uncompressed size limit exceeded'):
            archive.verify_part(path, [member], backup.PART_LIMIT)
        self.assertGreater(backup.verify_part(path, [member], backup.PART_LIMIT), 128 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError, 'uncompressed size limit exceeded'):
            backup.verify_part(path, [member], 128 * 1024 * 1024)

    def test_copy_verifier_rejects_changed_member_and_unexpected_zip_entries(self):
        value = b'expected checkpoint'
        member = {'path': 'weights/one.pt', 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
        path = self.root / 'changed.zip'
        with zipfile.ZipFile(path, 'x') as handle:
            handle.writestr('MANIFEST.json', archive.manifest_bytes([member]))
            handle.writestr(member['path'], b'changed checkpoint!')
        with self.assertRaisesRegex(ValueError, 'ZIP member digest mismatch'):
            backup.verify_part(path, [member], backup.PART_LIMIT)
        with zipfile.ZipFile(path, 'a') as handle:
            handle.writestr('unexpected.txt', b'unknown')
        with self.assertRaisesRegex(ValueError, 'ZIP membership mismatch'):
            backup.verify_part(path, [member], backup.PART_LIMIT)

    def abandoned_prepared_parts(self):
        self.initialize()
        with patch.object(backup, 'prepare_upload', side_effect=ValueError('simulated final resident failure')):
            with self.assertRaisesRegex(ValueError, 'simulated final resident failure'):
                backup.create_next(self.args)
        self.assertFalse(backup.locations(self.plan, 1)['manifest'].exists())
        self.assertEqual(list(self.init_args.private_output.iterdir()), [])
        return [backup.part_path(self.plan, n) for n in self.plan['batches'][0]]

    def recover(self):
        return backup.recover_prepared(SimpleNamespace(**vars(self.args), batch=1))

    def test_identical_partials_reappearing_after_rename_are_removed_before_next_allocation(self):
        self.initialize()
        rename = Path.rename
        def reappears(path, target):
            result = rename(path, target)
            if path.name.endswith('.zip.partial'):
                path.write_bytes(Path(target).read_bytes())
            return result
        with patch.object(Path, 'rename', autospec=True, side_effect=reappears):
            stage = backup.create_next(self.args)
        self.assertEqual(stage['batch'], 1)
        self.assertFalse(list(self.init_args.output_root.glob('*.partial')))
        self.assertEqual(len(list(self.init_args.output_root.glob('*.zip'))), 2)
        self.assert_originals_unchanged()

    def test_identical_partial_reappearing_during_final_source_check_is_removed(self):
        self.initialize()
        check = archive.check_sources
        def reappears(root, snapshot, **kwargs):
            result = check(root, snapshot, **kwargs)
            first, second = backup.part_path(self.plan, 1), backup.part_path(self.plan, 2)
            if first.exists() and second.exists():
                first.with_suffix('.zip.partial').write_bytes(first.read_bytes())
            return result
        with patch.object(archive, 'check_sources', side_effect=reappears):
            backup.create_next(self.args)
        self.assertFalse(list(self.init_args.output_root.glob('*.partial')))
        self.assert_originals_unchanged()

    def test_recover_prepared_parts_verifies_exact_duplicates_without_recompression(self):
        paths = self.abandoned_prepared_parts()
        before = [archive.file_receipt(path) for path in paths]
        duplicate = paths[0].with_suffix('.zip.partial')
        duplicate.write_bytes(paths[0].read_bytes())
        with patch.object(backup.zipfile, 'ZipFile', wraps=zipfile.ZipFile) as calls:
            stage = self.recover()
        self.assertTrue(all(len(call.args) < 2 or call.args[1] == 'r' for call in calls.call_args_list))
        self.assertTrue(stage['recovered_prepared_parts'])
        self.assertEqual([archive.file_receipt(path) for path in paths], before)
        self.assertFalse(duplicate.exists())
        self.assertFalse(Path(stage['expected_result']).exists())
        self.results(stage)
        self.assertEqual(self.finish(stage)['completed_batches'], 1)
        self.assert_originals_unchanged()

    def test_recovery_rejects_any_upload_preparation_or_unknown_outcome(self):
        paths = self.abandoned_prepared_parts()
        duplicate = paths[0].with_suffix('.zip.partial')
        duplicate.write_bytes(paths[0].read_bytes())
        outcome = backup.locations(self.plan, 1)['result']
        outcome.write_text('unknown outcome')
        with self.assertRaisesRegex(ValueError, 'prepared recovery is forbidden'):
            self.recover()
        self.assertTrue(duplicate.exists())
        self.assertEqual(outcome.read_text(), 'unknown outcome')
        self.assertFalse(backup.locations(self.plan, 1)['manifest'].exists())
        self.assert_originals_unchanged()

    def test_recovery_rejects_different_partial_without_removing_it(self):
        paths = self.abandoned_prepared_parts()
        duplicate = paths[0].with_suffix('.zip.partial')
        duplicate.write_bytes(b'different stale partial')
        with self.assertRaisesRegex(ValueError, 'Partial differs from its verified finalized ZIP'):
            self.recover()
        self.assertEqual(duplicate.read_bytes(), b'different stale partial')
        self.assertFalse(backup.locations(self.plan, 1)['manifest'].exists())
        self.assert_originals_unchanged()

    def test_recovery_rejects_modified_source_before_temporary_cleanup(self):
        paths = self.abandoned_prepared_parts()
        duplicate = paths[0].with_suffix('.zip.partial')
        duplicate.write_bytes(paths[0].read_bytes())
        source = self.root / self.plan['parts'][0][0]['path']
        source.write_bytes(b'modified original')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.recover()
        self.assertTrue(duplicate.exists())
        self.assertEqual(source.read_bytes(), b'modified original')
        self.assertFalse(backup.locations(self.plan, 1)['manifest'].exists())

    def test_recovery_still_enforces_resident_bound(self):
        self.abandoned_prepared_parts()
        unrelated = self.init_args.output_root / 'unrelated.zip'
        unrelated.write_bytes(b'x' * self.init_args.resident_limit_bytes)
        with self.assertRaisesRegex(ValueError, 'Actual resident archive size exceeded'):
            self.recover()
        self.assertTrue(unrelated.exists())
        self.assertFalse(backup.locations(self.plan, 1)['manifest'].exists())
        self.assert_originals_unchanged()

    def test_previously_saved_parts_reappearing_after_new_part_creation_are_cleaned(self):
        self.initialize()
        stage = backup.create_next(self.args)
        old = {backup.part_path(self.plan, n): backup.part_path(self.plan, n).read_bytes() for n in stage['parts']}
        self.results(stage)
        self.finish(stage)
        create = backup.create_part
        def reappears(*args):
            result = create(*args)
            for path, data in old.items():
                path.write_bytes(data)
                path.with_suffix('.zip.partial').write_bytes(b'old interrupted temporary copy')
            return result
        with patch.object(backup, 'create_part', side_effect=reappears):
            stage = backup.create_next(self.args)
        self.assertEqual(stage['batch'], 2)
        self.assertTrue(all(not p.exists() and not p.with_suffix('.zip.partial').exists() for p in old))
        self.assert_originals_unchanged()

    def test_different_previously_saved_temporary_is_not_removed(self):
        self.initialize()
        stage = backup.create_next(self.args)
        self.results(stage)
        self.finish(stage)
        old = backup.part_path(self.plan, 1)
        old.write_bytes(b'changed saved temporary ZIP')
        with self.assertRaisesRegex(ValueError, 'Temporary archive changed'):
            backup.create_next(self.args)
        self.assertEqual(old.read_bytes(), b'changed saved temporary ZIP')
        self.assertFalse(backup.part_path(self.plan, 3).exists())
        self.assert_originals_unchanged()

    def abandoned_one_part(self):
        self.initialize()
        create = backup.create_part
        def fail_second(root, plan, number):
            if number == 2:
                raise ValueError('simulated failure before second part')
            return create(root, plan, number)
        with patch.object(backup, 'create_part', side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, 'simulated failure'):
                backup.create_next(self.args)
        first = backup.part_path(self.plan, 1)
        partial = first.with_suffix('.zip.partial')
        partial.write_bytes(first.read_bytes()[:100])
        receipt = archive.file_receipt(partial)
        args = SimpleNamespace(**vars(self.args), batch=1, complete_missing_parts=True,
            discard_abandoned_partial=[f"1:{receipt['bytes']}:{receipt['sha256']}"])
        return first, partial, args

    def test_explicit_recovery_keeps_verified_part_and_creates_only_missing_part(self):
        first, partial, args = self.abandoned_one_part()
        before = archive.file_receipt(first)
        old_partial = partial.read_bytes()
        check = archive.check_sources
        def reappears(root, snapshot, **kwargs):
            result = check(root, snapshot, **kwargs)
            if backup.part_path(self.plan, 2).exists():
                partial.write_bytes(old_partial)
            return result
        with patch.object(backup, 'create_part', wraps=backup.create_part) as creates, \
             patch.object(archive, 'check_sources', side_effect=reappears):
            stage = backup.recover_prepared(args)
        self.assertEqual([call.args[2] for call in creates.call_args_list], [2])
        self.assertEqual(archive.file_receipt(first), before)
        self.assertFalse(partial.exists())
        self.assertFalse(Path(stage['expected_result']).exists())
        self.results(stage)
        self.assertEqual(self.finish(stage)['completed_batches'], 1)
        self.assert_originals_unchanged()

    def test_wrong_abandoned_partial_hash_blocks_recovery_and_keeps_temporary(self):
        first, partial, args = self.abandoned_one_part()
        value = partial.read_bytes()
        args.discard_abandoned_partial = [f'1:{len(value)}:' + '0' * 64]
        with self.assertRaisesRegex(ValueError, 'explicit recovery receipt'):
            backup.recover_prepared(args)
        self.assertEqual(partial.read_bytes(), value)
        self.assertFalse(backup.part_path(self.plan, 2).exists())
        self.assert_originals_unchanged()

    def test_missing_part_recovery_refuses_prior_upload_request(self):
        first, partial, args = self.abandoned_one_part()
        backup.locations(self.plan, 1)['request'].write_text('{}')
        with self.assertRaisesRegex(ValueError, 'prepared recovery is forbidden'):
            backup.recover_prepared(args)
        self.assertTrue(partial.exists())
        self.assertFalse(backup.part_path(self.plan, 2).exists())
        self.assert_originals_unchanged()


if __name__ == '__main__':
    unittest.main()

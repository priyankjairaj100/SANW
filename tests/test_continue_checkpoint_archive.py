"""Continuation tests use disposable grids and synthetic upload results only."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import continue_checkpoint_archive as continuation
import stream_completed_candidate_checkpoints as archive

SPEC = importlib.util.spec_from_file_location('wrapper_fixtures', Path(__file__).with_name('test_save_completed_grid_incrementally.py'))
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.old = fixtures.IncrementalWrapperTests()
        self.old.setUp()
        self.addCleanup(self.old.doCleanups)
        self.failed_zip = self.old.abandoned_second_creation()
        self.root = self.old.root
        self.prior_output = self.failed_zip.parent
        self.failed_record = self.old.args.private_output / 'campaign_batch_002_result.json'
        self.failed_record.write_text('{"results":[{"status":"failed","error_code":"transfer_failed"}]}')
        self.failed_bytes = self.failed_zip.read_bytes()
        self.failed_result_bytes = self.failed_record.read_bytes()
        self.init_args = SimpleNamespace(repository=self.root, grid_root='grid', prefix='campaignC',
            retained_manifest='grid/archival_keep_manifest.json',
            retained_manifest_sha256=archive.digest(self.root / 'grid/archival_keep_manifest.json'),
            prior_prefix='campaign', prior_output=self.prior_output, prior_private=self.old.args.private_output,
            prior_batches=1, expected_completions=4, batch_size=1,
            output_root=self.root / 'recovery/continuationC', private_output=self.root / 'privateC')

    def initialize(self):
        result = continuation.initialize(self.init_args)
        self.args = SimpleNamespace(plan=Path(result['plan']['local_path']), plan_sha256=result['plan']['sha256'])
        self.plan = archive.read_json(self.args.plan)
        return result

    def upload_result(self, stage, *, status='succeeded'):
        request = archive.read_json(Path(stage['upload_request']))
        result = {'results': [{**item, 'status': status, 'file_id': 'fixture-file', 'library_file_id': 'fixture-library'}
                              for item in request['uploads']]}
        Path(stage['expected_result']).write_text(json.dumps(result))

    def finish(self, stage):
        return continuation.finish_upload(SimpleNamespace(**vars(self.args), batch=stage['batch']))

    def assert_failed_batch_untouched(self):
        self.assertEqual(self.failed_zip.read_bytes(), self.failed_bytes)
        self.assertEqual(self.failed_record.read_bytes(), self.failed_result_bytes)

    def test_three_phases_exclude_only_verified_prior_batch_and_preserve_failed_batch(self):
        initialized = self.initialize()
        self.assertEqual(initialized['excluded_candidates'], 1)
        self.assertEqual(initialized['remaining_candidates'], 3)
        self.assertEqual(self.plan['excluded_candidates'], ['candidates/matched_allocation_distillation_draw_0/lr_0.001/seed_17'])
        for number in range(1, 4):
            stage = continuation.create_next(self.args)
            self.assertEqual(stage['batch'], number)
            self.assertFalse(Path(stage['expected_result']).exists())
            self.upload_result(stage)
            result = self.finish(stage)
            self.assertEqual(result['completed_batches'], number)
            # Reporting a completed finish is safe and creates no second upload.
            self.assertEqual(self.finish(stage)['completed_batches'], number)
        self.assertTrue(result['complete'])
        self.assertTrue(continuation.create_next(self.args)['complete'])
        for state in self.old.retained:
            self.assertEqual((self.root / state['checkpoint']).read_bytes(), self.old.original[state['checkpoint']])
        self.assert_failed_batch_untouched()

    def test_pending_batch_cannot_be_recreated_or_overwritten(self):
        self.initialize()
        stage = continuation.create_next(self.args)
        result_path = Path(stage['expected_result'])
        result_path.write_text('unknown incomplete outcome')
        with self.assertRaisesRegex(ValueError, 'lacks completed prune/upload'):
            continuation.create_next(self.args)
        self.assertEqual(result_path.read_text(), 'unknown incomplete outcome')
        self.assert_failed_batch_untouched()

    def test_failed_upload_never_prunes(self):
        self.initialize()
        stage = continuation.create_next(self.args)
        loc = continuation.locations(self.plan, stage['batch'])
        index = archive.read_json(loc['manifest'])
        self.upload_result(stage, status='failed')
        with self.assertRaisesRegex(ValueError, 'Every upload must succeed'):
            self.finish(stage)
        self.assertTrue(all((self.root / item['path']).exists() for item in index['snapshot']['members']))
        self.assertFalse(loc['upload'].exists())
        self.assertFalse(loc['prune'].exists())

    def test_changed_zip_blocks_finish_before_prune(self):
        self.initialize()
        stage = continuation.create_next(self.args)
        self.upload_result(stage)
        loc = continuation.locations(self.plan, stage['batch'])
        made = archive.read_json(loc['sources'])['made']
        Path(made['parts'][0]['local_path']).write_bytes(b'changed ZIP')
        with self.assertRaisesRegex(ValueError, 'ZIP digest/size mismatch'):
            self.finish(stage)
        self.assertFalse(loc['prune'].exists())
        self.assertFalse(loc['upload'].exists())

    def test_changed_prior_verified_record_blocks_continuation(self):
        self.initialize()
        prior = Path(self.plan['prior_verified_batches'][0]['upload_record']['local_path'])
        prior.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Prior verified record changed'):
            continuation.create_next(self.args)

    def test_immutable_plan_digest_is_required(self):
        self.initialize()
        self.args.plan.write_text(self.args.plan.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'plan digest mismatch'):
            continuation.create_next(self.args)

    def test_init_rejects_failed_prior_batch_without_new_directories(self):
        self.init_args.prior_batches = 2
        with self.assertRaisesRegex(ValueError, 'Incomplete batch manifest'):
            continuation.initialize(self.init_args)
        self.assertFalse(self.init_args.output_root.exists())
        self.assertFalse(self.init_args.private_output.exists())
        self.assert_failed_batch_untouched()


if __name__ == '__main__':
    unittest.main()

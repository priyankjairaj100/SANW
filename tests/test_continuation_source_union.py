"""Run the seven existing continuation tests and focused union-check tests."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


optimized = load('active_continuation', ROOT / 'scripts/continue_checkpoint_archive.py')
existing = load('existing_continuation_tests', ROOT / 'tests/test_continue_checkpoint_archive.py')
existing.continuation = optimized
archive = optimized.archive


class OptimizedSourceChecks(existing.ContinuationTests):
    def complete_two(self):
        self.initialize()
        for _ in range(2):
            stage = optimized.create_next(self.args)
            self.upload_result(stage)
            self.finish(stage)

    def test_shared_sources_checked_once_after_all_receipts(self):
        self.complete_two()
        with patch.object(archive, 'check_sources', wraps=archive.check_sources) as checks, \
             patch.object(optimized.workflow, 'verify_saved_batch', wraps=optimized.workflow.verify_saved_batch) as receipts:
            batches = optimized.existing_batches(self.root, self.plan)
        self.assertEqual(len(batches), 2)
        self.assertEqual(receipts.call_count, 2)
        self.assertTrue(all(call.kwargs['check_sources'] is False for call in receipts.call_args_list))
        self.assertEqual(checks.call_count, 1)
        snapshot = checks.call_args.args[1]
        names = [item['path'] for item in snapshot['protected_files'] + snapshot['members']]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(checks.call_args.kwargs['allow_missing_members'])

    def test_missing_protected_checkpoint_fails_union_check(self):
        self.complete_two()
        protected = self.root / self.old.retained[0]['checkpoint']
        protected.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing regular file'):
            optimized.existing_batches(self.root, self.plan)

    def test_missing_archived_members_remain_allowed(self):
        self.complete_two()
        batches = optimized.existing_batches(self.root, self.plan)
        members = [item for batch in batches for item in batch['index']['snapshot']['members']]
        self.assertTrue(members)
        self.assertTrue(all(not (self.root / item['path']).exists() for item in members))

    def test_conflicting_records_fail_before_any_byte_check(self):
        self.complete_two()
        verify = optimized.workflow.verify_saved_batch
        count = 0
        def conflicting(*args, **kwargs):
            nonlocal count
            result = copy.deepcopy(verify(*args, **kwargs))
            count += 1
            if count == 2:
                item = copy.deepcopy(self.plan['protected_files'][0])
                item['sha256'] = '0' * 64
                result[1]['snapshot']['protected_files'].append(item)
            return result
        with patch.object(optimized.workflow, 'verify_saved_batch', side_effect=conflicting), \
             patch.object(archive, 'check_sources', wraps=archive.check_sources) as checks:
            with self.assertRaisesRegex(ValueError, 'Conflicting record'):
                optimized.existing_batches(self.root, self.plan)
        self.assertEqual(checks.call_count, 0)

    def test_protected_status_wins_for_a_path_also_listed_as_member(self):
        self.complete_two()
        batches = optimized.existing_batches(self.root, self.plan)
        member = batches[0]['index']['snapshot']['members'][0]
        plan = copy.deepcopy(self.plan)
        plan['protected_files'].append(member)
        self.assertFalse((self.root / member['path']).exists())
        with self.assertRaisesRegex(ValueError, 'Missing regular file'):
            optimized.existing_batches(self.root, plan)

    def test_cross_class_digest_conflict_is_rejected_globally(self):
        self.complete_two()
        batches = optimized.existing_batches(self.root, self.plan)
        member = copy.deepcopy(batches[0]['index']['snapshot']['members'][0])
        member['sha256'] = '1' * 64
        plan = copy.deepcopy(self.plan)
        plan['protected_files'].append(member)
        with self.assertRaisesRegex(ValueError, 'Conflicting record'):
            optimized.existing_batches(self.root, plan)


if __name__ == '__main__':
    unittest.main()

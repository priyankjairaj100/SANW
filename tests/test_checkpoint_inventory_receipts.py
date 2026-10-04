"""Receipt adapters use only disposable byte fixtures and synthetic uploads."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import checkpoint_inventory_receipts as receipts


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_fixtures = fixture_module('test_archive_completed_grid_checkpoints')
stream_fixtures = fixture_module('test_save_completed_grid_incrementally')
retained_fixtures = fixture_module('test_preserve_retained_checkpoints')
archive = receipts.archive


class LegacyReceiptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy_fixtures.ArchiveTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.fixture()
        self.root = self.fixture.root
        with contextlib.redirect_stderr(io.StringIO()):
            made = receipts.legacy.archive_grid(self.fixture.args)
        self.manifest = self.fixture.args.output_root / made['manifest']['name']
        self.index = archive.read_json(self.manifest)
        self.upload = self.root / 'upload.json'
        self.prune = self.manifest.with_name('fixture_LOCAL_PRUNE_RECEIPT.json')
        self.fixture.write(self.upload, {'results': [
            {'local_path': str(self.manifest.parent / row['name']),
             'bytes': row['bytes'], 'sha256': row['sha256'], 'status': 'succeeded',
             'purpose': 'create_library_file', 'file_id': f'file_{i}', 'library_file_id': f'library_{i}'}
            for i, row in enumerate([made['manifest'], *made['parts']])]})
        receipts.legacy.prune_grid(argparse.Namespace(repository=self.root, manifest=self.manifest,
            manifest_sha256=made['manifest']['sha256'], upload_receipt=self.upload,
            confirm_uploaded_and_authorize_deletion=True))
        for part in self.index['parts']:
            (self.manifest.parent / part['name']).unlink()

    def verify(self):
        return receipts.verify_legacy(self.root, self.manifest, self.upload, self.prune)

    def change_upload(self, transform):
        value = archive.read_json(self.upload)
        transform(value)
        self.fixture.write(self.upload, value)
        prune = archive.read_json(self.prune)
        prune['upload_receipt_sha256'] = archive.digest(self.upload)
        self.fixture.write(self.prune, prune)

    def test_completed_metadata_succeeds_without_zips_or_deleted_originals(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(self.verify(), self.index)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse(list(self.root.rglob('*.zip')))

    def test_changed_manifest_or_upload_breaks_linked_hash(self):
        for path in (self.manifest, self.upload):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(ValueError, 'linked receipt digest'):
                    self.verify()
                path.write_bytes(original)

    def test_failed_or_unfinalized_upload_is_rejected_even_with_valid_link(self):
        original = self.upload.read_bytes()
        original_prune = self.prune.read_bytes()
        for key, value in [('status', 'failed'), ('file_id', ''), ('library_file_id', ''),
                           ('purpose', 'replace_library_file'), ('sha256', '0' * 64), ('bytes', 1)]:
            with self.subTest(key=key):
                self.change_upload(lambda doc: doc['results'][0].update({key: value}))
                with self.assertRaises(ValueError):
                    self.verify()
                self.upload.write_bytes(original)
                self.prune.write_bytes(original_prune)

    def test_duplicate_or_missing_upload_path_is_rejected(self):
        self.change_upload(lambda doc: doc['results'].__setitem__(1, copy.deepcopy(doc['results'][0])))
        with self.assertRaisesRegex(ValueError, 'duplicate upload path'):
            self.verify()

    def test_incomplete_prune_or_member_coverage_is_rejected(self):
        original = self.prune.read_bytes()
        for key, value in [('status', 'in_progress'), ('retained_files_verified', False),
                           ('deleted_members', []), ('deleted_bytes', 0)]:
            with self.subTest(key=key):
                doc = json.loads(original)
                doc[key] = value
                self.fixture.write(self.prune, doc)
                with self.assertRaises(ValueError):
                    self.verify()
        self.prune.write_bytes(original)

    def test_unsafe_part_or_member_paths_are_rejected(self):
        original = copy.deepcopy(self.index)
        for mutation in [lambda doc: doc['parts'][0].update(name='../escape.zip'),
                         lambda doc: doc['snapshot']['members'][0].update(path='../escape.pt')]:
            with self.subTest(mutation=mutation):
                doc = copy.deepcopy(original)
                mutation(doc)
                self.fixture.write(self.manifest, doc)
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    self.verify()

    def test_duplicate_part_members_cannot_inflate_coverage(self):
        self.index['parts'][0]['members'].append(copy.deepcopy(self.index['parts'][0]['members'][0]))
        self.fixture.write(self.manifest, self.index)
        with self.assertRaisesRegex(ValueError, 'Duplicate checkpoint'):
            self.verify()

    def test_embedded_manifest_bytes_are_required_in_uncompressed_total(self):
        row = self.index['parts'][0]
        row['uncompressed_bytes'] = sum(item['bytes'] for item in row['members'])
        self.fixture.write(self.manifest, self.index)
        with self.assertRaisesRegex(ValueError, 'uncompressed part byte count'):
            self.verify()


class StreamedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = stream_fixtures.IncrementalWrapperTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        result = self.fixture.execute()
        self.root = self.fixture.root
        self.manifest = Path(result['output']) / 'campaign_batch_001_MANIFEST.json'
        self.upload = self.fixture.args.private_output / 'campaign_batch_001_upload_record.json'
        self.index = archive.read_json(self.manifest)

    def verify(self, **overrides):
        snapshot = self.index['snapshot']
        kwargs = dict(root=self.root, manifest_path=self.manifest, upload_path=self.upload,
            grid_root='grid', candidates=snapshot['candidate_names'],
            retained_manifest=snapshot['retained_manifest']['path'],
            retained_sha256=snapshot['retained_manifest']['sha256'])
        return receipts.verify_streamed(**{**kwargs, **overrides})

    def test_streamed_receipts_verify_without_any_source_hashing(self):
        with patch.object(archive, 'check_sources', side_effect=AssertionError('Source read')):
            self.assertEqual(self.verify(), self.index)
        self.assertFalse(list(self.root.rglob('*.zip')))

    def test_wrong_candidate_schedule_and_changed_upload_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'candidate schedule'):
            self.verify(candidates=['wrong'])
        self.upload.write_text(self.upload.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'receipt digest mismatch'):
            self.verify()


class RetainedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = retained_fixtures.PreservationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.initialize()
        stage = receipts.backup.create_next(self.fixture.args)
        self.fixture.results(stage)
        self.fixture.finish(stage)
        self.plan = self.fixture.plan
        self.sha = self.fixture.args.plan_sha256
        self.loc = receipts.backup.locations(self.plan, 1)
        self.index = archive.read_json(self.loc['manifest'])
        for number in self.plan['batches'][0]:
            receipts.backup.part_path(self.plan, number).unlink()

    def verify(self, **overrides):
        return receipts.verify_retained(**{**dict(plan=self.plan, plan_sha=self.sha, batch=1), **overrides})

    def test_retained_receipts_verify_without_zips_or_source_reads(self):
        with patch.object(archive, 'check_sources', side_effect=AssertionError('Source read')):
            self.assertEqual(self.verify(), self.index)
        self.fixture.assert_originals_unchanged()

    def test_changed_source_snapshot_or_manifest_fails(self):
        for name in ('manifest', 'sources'):
            with self.subTest(name=name):
                path = self.loc[name]
                original = path.read_bytes()
                if name == 'manifest':
                    path.write_bytes(original + b' ')
                else:
                    doc = json.loads(original)
                    doc['sources'][0]['sha256'] = '0' * 64
                    path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, 'Pre-upload source bytes'):
                    self.verify()
                path.write_bytes(original)

    def test_verified_retained_evidence_requires_exact_binding_and_finalized_results(self):
        path = self.loc['verified_upload']
        original = path.read_bytes()
        mutations = [lambda doc: doc.update(plan_sha256='0' * 64),
                     lambda doc: doc.update(batch=2),
                     lambda doc: doc.update(all_members_verified_locally=False),
                     lambda doc: doc.update(originals_deleted=True),
                     lambda doc: doc['sources'][0].update(bytes=0),
                     lambda doc: doc['results'][0].update(status='failed'),
                     lambda doc: doc['results'][0].update(file_id=None),
                     lambda doc: doc['results'][0].update(sha256='0' * 64)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                doc = json.loads(original)
                mutate(doc)
                path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    self.verify()
        path.write_bytes(original)

    def test_invalid_batch_or_plan_hash_is_rejected(self):
        for change in ({'batch': 0}, {'batch': 99}, {'plan_sha': 'unknown'}, {'plan_sha': '0' * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(**change)


if __name__ == '__main__':
    unittest.main()

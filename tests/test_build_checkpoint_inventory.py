"""Public inventory classification must not overstate preservation or disclose receipts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import build_checkpoint_inventory as builder


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.expected = {f'grid/{name}.pt': {'sha256': str(i) * 64, 'grid_root': 'grid',
            'roles': {'candidate_epoch'}, 'state_ids': {name}}
            for i, name in enumerate(('required', 'intermediate', 'extra', 'missing'), 1)}
        def record(name):
            path = f'grid/{name}.pt'
            return {'path': path, 'bytes': 123, 'sha256': self.expected[path]['sha256']}
        self.required = {'grid/required.pt': record('required')}
        self.extras = {'grid/extra.pt': record('extra')}
        self.missing = {'grid/missing.pt': {'path': 'grid/missing.pt', 'sha256': '4' * 64}}
        self.coverage = {}

    def classify(self):
        return builder.classify(self.expected, self.required, self.extras, self.missing, self.coverage)

    def saved(self, name, kind):
        path = f'grid/{name}.pt'
        self.coverage[path] = {'record': {'path': path, 'bytes': 123, 'sha256': self.expected[path]['sha256']},
                               'kind': kind, 'location': {'manifest_sha256': 'a' * 64,
                               'part_name': name + '.zip', 'part_sha256': 'b' * 64}}

    def test_pending_sources_are_not_marked_saved(self):
        rows = {row['path']: row for row in self.classify()}
        self.assertEqual(rows['grid/required.pt']['status'], 'required_retained_pending_save')
        self.assertEqual(rows['grid/intermediate.pt']['status'], 'intermediate_archive_pending')
        self.assertEqual(rows['grid/extra.pt']['status'], 'replication_extra_local_only')
        self.assertEqual(rows['grid/missing.pt']['status'], 'discarded_by_fitter_unavailable_not_archived')
        self.assertTrue(all('saved_archive' not in row for row in rows.values()))

    def test_only_verified_coverage_changes_saved_status(self):
        self.saved('required', 'retained_checkpoint_copy')
        self.saved('intermediate', 'intermediate_epoch_archive')
        rows = {row['path']: row for row in self.classify()}
        self.assertEqual(rows['grid/required.pt']['status'], 'required_retained_saved')
        self.assertEqual(rows['grid/intermediate.pt']['status'], 'intermediate_archived_verified')
        self.assertEqual(rows['grid/extra.pt']['status'], 'replication_extra_local_only')

    def test_digest_conflicts_are_rejected(self):
        self.saved('intermediate', 'intermediate_epoch_archive')
        self.coverage['grid/intermediate.pt']['record']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Archived checkpoint digest differs'):
            self.classify()

    def test_required_checkpoint_cannot_be_classified_as_pruned_intermediate(self):
        self.saved('required', 'intermediate_epoch_archive')
        with self.assertRaisesRegex(ValueError, 'pruning archive'):
            self.classify()

    def test_overlapping_unavailable_and_retained_categories_are_rejected(self):
        self.missing.update(self.required)
        with self.assertRaisesRegex(ValueError, 'categories overlap'):
            self.classify()

    def test_deterministic_order_ignores_input_dictionary_order(self):
        first = builder.archive.canonical(self.classify())
        self.expected = dict(reversed(list(self.expected.items())))
        self.assertEqual(builder.archive.canonical(self.classify()), first)

    def test_public_archive_whitelist_drops_remote_identifiers_and_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example_MANIFEST.json'
            path.write_text('{}')
            index = {'parts': [{'archive': {'local_path': '/private/secret/example.zip',
                'bytes': 456, 'sha256': 'a' * 64, 'library_file_id': 'PRIVATE_LIBRARY_ID'},
                'members': [{'path': 'grid/required.pt', 'bytes': 123, 'sha256': '1' * 64}],
                'private_url': 'https://private.example/secret'}],
                'results': [{'file_id': 'PRIVATE_FILE_ID', 'library_file_id': 'PRIVATE_LIBRARY_ID'}]}
            value = builder.public_archive(path, index, 'retained_checkpoint_copy')
        text = json.dumps(value)
        for forbidden in ('PRIVATE_', '/private/', 'https://', 'library_file_id', 'results'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(value['parts'][0]['name'], 'example.zip')

    def test_duplicate_archive_coverage_cannot_inflate_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example_MANIFEST.json'
            path.write_text('{}')
            index = {'parts': [{'name': 'one.zip', 'bytes': 456, 'sha256': 'a' * 64,
                'members': [copy.deepcopy(self.required['grid/required.pt'])]}]}
            archives, coverage = [], {}
            builder.add_coverage(coverage, archives, path, index, 'intermediate_epoch_archive')
            with self.assertRaisesRegex(ValueError, 'multiple preservation entries'):
                builder.add_coverage(coverage, archives, path, index, 'intermediate_epoch_archive')


if __name__ == '__main__':
    unittest.main()

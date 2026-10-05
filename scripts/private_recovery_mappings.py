"""Collect private restore metadata without copying files or changing archives."""
from pathlib import Path
import re

import build_checkpoint_inventory as inventory
import checkpoint_inventory_receipts as receipts
import continue_checkpoint_archive as continuation
import preserve_retained_checkpoints as backup
import stream_completed_candidate_checkpoints as archive


MAPPING_FIELDS = frozenset({'filename', 'file_id', 'library_file_id', 'sha256', 'bytes'})


def _merge_rows(rows):
    by_name = {}
    for row in rows:
        archive.require(set(row) == MAPPING_FIELDS, 'Recovery mapping contains unexpected fields')
        previous = by_name.setdefault(row['filename'], row)
        archive.require(previous == row, 'Conflicting recovery mapping for filename: ' + row['filename'])
    return [by_name[name] for name in sorted(by_name)]


def deduplicate_mappings(rows):
    """Public entry point for merging already sanitized artifact groups."""
    return _merge_rows(rows)


def sanitize_results(results, expected_sources):
    """Whitelist finalized create/replace results bound to exact source receipts.

    expected_sources maps absolute local paths to their verified bytes and SHA256.
    This helper does not establish how those source receipts were verified.
    """
    archive.require(isinstance(results, list) and len(results) == len(expected_sources),
                    'Recovery results must cover exactly the expected sources')
    for path, source in expected_sources.items():
        archive.require(isinstance(path, str) and Path(path).is_absolute(),
                        'Expected recovery source path must be absolute')
        archive.require(type(source.get('bytes')) is int and source['bytes'] >= 0 and
                        isinstance(source.get('sha256'), str) and
                        re.fullmatch(r'[0-9a-f]{64}', source['sha256']), 'Invalid expected source receipt')
    rows, seen = [], set()
    for result in results:
        name = result.get('local_path')
        archive.require(name in expected_sources and name not in seen, 'Unexpected or duplicate recovery source')
        archive.require(result.get('status') == 'succeeded' and
                        result.get('purpose') in ('create_library_file', 'replace_library_file'),
                        'Recovery result must be a successful create or replace')
        archive.require(all(isinstance(result.get(key), str) and result[key].strip()
                            for key in ('file_id', 'library_file_id')), 'Recovery result lacks finalized identifiers')
        expected = expected_sources[name]
        archive.require(type(result.get('bytes')) is int and
                        all(result.get(key) == expected[key] for key in ('bytes', 'sha256')),
                        'Recovery result source digest or size differs')
        rows.append({'filename': Path(name).name, 'file_id': result['file_id'],
                     'library_file_id': result['library_file_id'],
                     'sha256': expected['sha256'], 'bytes': expected['bytes']})
        seen.add(name)
    return _merge_rows(rows)


def _regular_receipt(path):
    path = Path(path)
    archive.require(path.is_absolute() and path.is_file() and not path.is_symlink(),
                    'Missing regular recovery receipt')
    return path.resolve()


def _stream_sources(path, expected, *, plan_sha=None, batch=None):
    """Check optional continuation source metadata before including its path."""
    doc = archive.read_json(path)
    if plan_sha is not None:
        archive.require(doc.get('plan_sha256') == plan_sha, 'Continuation source plan differs')
    if batch is not None:
        archive.require(doc.get('batch') == batch, 'Continuation source batch differs')
    made = doc['made']
    rows = [made['manifest'], *made['parts']]
    sources = {row['local_path']: row for row in rows}
    archive.require(len(sources) == len(rows) and set(sources) == set(expected),
                    'Continuation source paths differ')
    archive.require(all(all(sources[name].get(key) == source[key] for key in ('bytes', 'sha256'))
                        for name, source in expected.items()), 'Continuation source metadata differs')


def collect_archive_recovery(root, config):
    """Return sanitized mappings and existing restore-receipt Paths.

    The complete inventory check establishes checkpoint coverage before any
    mapping is returned. Raw requests, outcomes, logs, and ZIPs are excluded.
    """
    root = Path(root).resolve()
    checked = inventory.build(root, config, verify_local=False, require_complete=True)
    archive.require(checked.get('status') == 'complete', 'Recovery requires complete preservation campaigns')
    mapping_rows, paths = [], set()

    def add_path(path):
        checked_path = _regular_receipt(path)
        paths.add(checked_path)
        return checked_path

    def add_ref(ref):
        path, value = inventory.pinned(ref)
        add_path(path)
        return path, value

    def append_results(path, expected):
        mapping_rows.extend(sanitize_results(archive.read_json(path)['results'], expected))

    def streamed(manifest, upload, prune, plan, candidates, source_path=None, plan_sha=None, batch=None):
        keep = plan['retained_manifest']
        index = receipts.verify_streamed(root, manifest, upload, plan['grid_root'], candidates,
                                        keep['path'], keep['sha256'])
        expected = archive.upload_sources(manifest, index)
        append_results(upload, expected)
        for path in (manifest, upload, prune):
            add_path(path)
        if source_path is not None and source_path.exists():
            _stream_sources(source_path, expected, plan_sha=plan_sha, batch=batch)
            add_path(source_path)

    add_ref(config['retained_inventory'])
    add_ref(config['replication_status'])
    for legacy in config['legacy_archives']:
        manifest, _ = add_ref(legacy['manifest'])
        upload, _ = add_ref(legacy['upload_receipt'])
        prune, _ = add_ref(legacy['prune_receipt'])
        index = receipts.verify_legacy(root, manifest, upload, prune)
        append_results(upload, archive.upload_sources(manifest, index))
    for ref in config['continuation_plans']:
        _, plan = add_ref(ref)
        for prior in plan['prior_verified_batches']:
            manifest, _ = add_ref(prior['manifest'])
            upload, _ = add_ref(prior['upload_record'])
            prune, _ = add_ref(prior['prune_receipt'])
            source_path = upload.with_name(upload.name.removesuffix('_upload_record.json') + '_sources.json')
            streamed(manifest, upload, prune, plan, prior['candidates'], source_path, batch=prior['batch'])
        for number in range(1, continuation.batch_count(plan) + 1):
            loc = continuation.locations(plan, number)
            streamed(loc['manifest'], loc['upload'], loc['prune'], plan,
                     continuation.batch_names(plan, number), loc['sources'], ref['sha256'], number)
    _, plan = add_ref(config['retained_plan'])
    for number in range(1, len(plan['batches']) + 1):
        receipts.verify_retained(plan, config['retained_plan']['sha256'], number)
        loc = backup.locations(plan, number)
        _, sources = backup.load_batch(plan, config['retained_plan']['sha256'], number,
                                       verify_zips=False, require_sources=True)
        append_results(loc['verified_upload'], {row['local_path']: row for row in sources})
        for key in ('manifest', 'sources', 'verified_upload'):
            add_path(loc[key])
    return _merge_rows(mapping_rows), sorted(paths, key=str)

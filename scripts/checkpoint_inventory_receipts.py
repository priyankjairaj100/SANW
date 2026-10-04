"""Read-only adapters for historical checkpoint preservation receipts.

These functions validate linked metadata after temporary ZIPs leave disk. They
do not recheck remote availability or rehash original checkpoint files.
"""
from pathlib import Path
import re

import archive_completed_grid_checkpoints as legacy
import preserve_retained_checkpoints as backup
import save_completed_grid_incrementally as workflow
import stream_completed_candidate_checkpoints as archive


def _metadata_path(root, value, *, repository_local=False):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if repository_local:
        archive.require(path.is_relative_to(root), 'Receipt path is outside the repository')
        return archive.safe_path(root, path.relative_to(root).as_posix())
    archive.require(path.is_file() and not path.is_symlink(), 'Missing regular receipt file')
    return path.resolve()


def _records(root, records):
    archive.require(isinstance(records, list), 'Expected a list of checkpoint records')
    for item in records:
        archive.safe_path(root, item['path'], exists=False)
        archive.require(type(item.get('bytes')) is int and item['bytes'] >= 0,
                        'Invalid recorded byte count')
        archive.require(isinstance(item.get('sha256'), str) and
                        re.fullmatch(r'[0-9a-f]{64}', item['sha256']), 'Invalid recorded SHA256')
    ordered = archive.exact_records(records)
    archive.require(len(ordered) == len(records), 'Duplicate checkpoint record paths')
    return ordered


def verify_legacy(root, manifest_path, upload_path, prune_path):
    """Verify a completed whole-grid archive using its linked upload evidence."""
    root = Path(root).resolve()
    manifest = _metadata_path(root, manifest_path, repository_local=True)
    upload = _metadata_path(root, upload_path)
    prune = _metadata_path(root, prune_path, repository_local=True)
    index = archive.read_json(manifest)
    archive.require(index.get('schema_version') == 1 and
                    index.get('purpose') == 'lossless_completed_grid_epoch_archive' and
                    index.get('path_base') == 'repository' and index.get('originals_deleted') is False,
                    'Unsupported legacy archive manifest')
    prefix = index.get('archive_prefix')
    archive.require(isinstance(prefix, str) and re.fullmatch(r'[A-Za-z0-9_-]+', prefix),
                    'Unsafe legacy archive prefix')
    archive.require(manifest.name == prefix + '_MANIFEST.json' and
                    prune == manifest.with_name(prefix + '_LOCAL_PRUNE_RECEIPT.json'),
                    'Legacy manifest or prune receipt path differs')
    limit = index.get('part_limit_bytes')
    archive.require(type(limit) is int and 0 < limit <= legacy.MAX_PART_BYTES,
                    'Invalid legacy part limit')
    snapshot = index['snapshot']
    grid = snapshot['grid_root']
    archive.safe_path(root, grid, exists=False)
    members = _records(root, snapshot['members'])
    protected = _records(root, snapshot['protected_files'])
    retained = _records(root, snapshot['retained_checkpoints'])
    archive.exact_records(protected + retained)
    protected_paths = {row['path'] for row in protected + retained}
    archive.require(members and not ({row['path'] for row in members} & protected_paths),
                    'Legacy archive includes protected files or has no members')
    archive.require(all(row['path'].startswith(grid + '/candidates/') and
                        legacy.EPOCH.search(row['path']) for row in members),
                    'Legacy members are outside candidate epoch checkpoints')
    parts = index['parts']
    archive.require(parts and len(parts) == len({part['name'] for part in parts}),
                    'Missing or duplicate legacy parts')
    all_members = []
    sources = {str(manifest): archive.file_receipt(manifest)}
    for part in parts:
        name = part['name']
        archive.require(Path(name).name == name and name.endswith('.zip'), 'Unsafe legacy ZIP name')
        archive.safe_path(root, (manifest.parent / name).relative_to(root).as_posix(), exists=False)
        rows = _records(root, part['members'])
        archive.require(rows and type(part.get('bytes')) is int and part['bytes'] > 0 and
                        isinstance(part.get('sha256'), str) and
                        re.fullmatch(r'[0-9a-f]{64}', part['sha256']), 'Invalid legacy part source record')
        expected_total = sum(row['bytes'] for row in rows) + len(legacy.manifest_bytes(part['members']))
        archive.require(part.get('uncompressed_bytes') == expected_total <= limit,
                        'Legacy uncompressed part byte count differs')
        all_members.extend(rows)
        sources[str(manifest.parent / name)] = part
    archive.require(_records(root, all_members) == members, 'Legacy part membership differs from snapshot')
    receipt = archive.read_json(prune)
    archive.require(receipt.get('status') == 'completed' and receipt.get('retained_files_verified') is True,
                    'Legacy prune lacks completed retained verification')
    archive.require(receipt.get('archive_manifest') == manifest.name and
                    receipt.get('archive_manifest_sha256') == archive.digest(manifest) and
                    receipt.get('upload_receipt_sha256') == archive.digest(upload),
                    'Legacy linked receipt digest mismatch')
    archive.require(_records(root, receipt['deleted_members']) == members and
                    receipt.get('deleted_bytes') == sum(row['bytes'] for row in members),
                    'Legacy prune member coverage differs')
    archive.validate_upload_results(archive.read_json(upload)['results'], sources)
    return index


def verify_streamed(root, manifest_path, upload_path, grid_root, candidates,
                    retained_manifest, retained_sha256):
    """Reuse the completed streaming workflow's checks without source reads."""
    root = Path(root).resolve()
    manifest = _metadata_path(root, manifest_path, repository_local=True)
    upload = _metadata_path(root, upload_path)
    _, index, _ = workflow.verify_saved_batch(
        root, manifest, upload, grid_root=grid_root, candidates=candidates,
        retained_manifest=retained_manifest, retained_sha256=retained_sha256,
        check_sources=False)
    return index


def verify_retained(plan, plan_sha, batch):
    """Verify one copy-only batch; the caller must first pin the loaded plan."""
    archive.require(type(batch) is int and 1 <= batch <= len(plan['batches']),
                    'Invalid retained preservation batch')
    archive.require(isinstance(plan_sha, str) and re.fullmatch(r'[0-9a-f]{64}', plan_sha),
                    'Invalid retained preservation plan SHA256')
    archive.require(plan.get('schema_version') == 1 and plan.get('purpose') == backup.PURPOSE and
                    plan.get('original_checkpoint_deletion_permitted') is False,
                    'Unsupported retained preservation plan')
    index, sources = backup.load_batch(plan, plan_sha, batch, verify_zips=False, require_sources=True)
    root = Path(plan['repository']).resolve()
    location = backup.locations(plan, batch)['verified_upload']
    saved = archive.read_json(_metadata_path(root, location))
    archive.require(saved.get('purpose') == backup.PURPOSE + '_verified_upload' and
                    saved.get('plan_sha256') == plan_sha and saved.get('batch') == batch and
                    saved.get('all_members_verified_locally') is True and
                    saved.get('originals_deleted') is False and saved.get('sources') == sources,
                    'Retained verified upload differs from its immutable sources')
    archive.require(len(sources) == len({item['local_path'] for item in sources}),
                    'Duplicate retained upload source paths')
    archive.validate_upload_results(saved['results'], {item['local_path']: item for item in sources})
    return index

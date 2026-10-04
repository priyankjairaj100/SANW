#!/usr/bin/env python3
"""Copy retained checkpoints into bounded parts; never delete any source file.

init -> create-next -> separate prepared upload -> finish-upload.
Identical rename leftovers are removed after checking their finalized ZIP.
Saved finalized ZIP cleanup remains a separate root action after verification.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

import stream_completed_candidate_checkpoints as archive
import save_completed_grid_incrementally as workflow

PURPOSE = 'copy_only_retained_checkpoint_preservation'
INVENTORY_PURPOSE = 'proposed_522_run_retained_checkpoint_persistence'
PART_LIMIT = 240 * 1024 * 1024


def verify_part(path, members, limit):
    """Verify this copy-only format without the pruning helper's 128 MiB cap."""
    archive.require(0 < limit <= PART_LIMIT, 'Preservation ZIP limit exceeds 240 MiB')
    archive.require(path.stat().st_size <= limit, 'Preservation ZIP compressed size limit exceeded')
    archive.require(len(archive.exact_records(members)) == len(members), 'Duplicate preservation member records')
    with zipfile.ZipFile(path) as handle:
        names = handle.namelist()
        archive.require(len(names) == len(set(names)), 'Duplicate ZIP members')
        archive.require(set(names) == {'MANIFEST.json', *[item['path'] for item in members]},
                        'ZIP membership mismatch')
        total = sum(info.file_size for info in handle.infolist())
        archive.require(total <= limit, 'Preservation ZIP uncompressed size limit exceeded')
        archive.require(handle.read('MANIFEST.json') == archive.manifest_bytes(members), 'ZIP MANIFEST mismatch')
        for item in members:
            archive.require(handle.getinfo(item['path']).file_size == item['bytes'], 'ZIP member size differs')
            value, size = hashlib.sha256(), 0
            with handle.open(item['path']) as stream:
                # Reading every member to EOF checks its CRC through zipfile.
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
                    size += len(block)
            archive.require(size == item['bytes'] and value.hexdigest() == item['sha256'],
                            f"ZIP member digest mismatch: {item['path']}")
    return total


def partition(records, limit):
    parts, current = [], []
    for item in records:
        archive.require(archive.zip_upper_bound([item]) <= limit, 'One checkpoint exceeds the part bound')
        if current and archive.zip_upper_bound(current + [item]) > limit:
            parts.append(current)
            current = []
        current.append(item)
    if current:
        parts.append(current)
    return parts


def initialize(args):
    root = args.repository.resolve()
    archive.require(re.fullmatch(r'[A-Za-z0-9_-]+', args.prefix), 'Unsafe preservation prefix')
    archive.require(0 < args.part_limit_bytes <= PART_LIMIT, 'Part bound must be at most 240 MiB')
    archive.require(1 <= args.parts_per_batch <= 4, 'Use one to four parts per batch')
    archive.require(args.resident_limit_bytes >= args.part_limit_bytes, 'Resident bound is smaller than a part')
    inventory_path = archive.safe_path(root, args.inventory)
    inventory_record = archive.record(root, inventory_path, args.inventory_sha256)
    inventory = archive.read_json(inventory_path)
    archive.require(inventory.get('purpose') == INVENTORY_PURPOSE and
                    inventory.get('original_checkpoint_deletion_permitted') is False,
                    'A pinned copy-only retained inventory is required')
    archive.require(inventory['completed_runs'] == args.expected_runs and
                    sum(row['completed_runs'] for row in inventory['grids']) == args.expected_runs,
                    'Completed-run inventory differs')
    records = archive.exact_records([item for row in inventory['grids'] for item in row['retained_checkpoints']])
    archive.require(records and len(records) == inventory['retained_checkpoint_count'] and
                    sum(item['bytes'] for item in records) == inventory['retained_checkpoint_bytes'],
                    'Retained inventory totals differ')
    protected = archive.exact_records([inventory_record] + [row[key] for row in inventory['grids']
                                                           for key in ('state_manifest', 'retained_mapping')])
    archive.exact_records(protected + records)  # Reject cross-class conflicts.
    archive.check_sources(root, {'protected_files': protected, 'members': records})
    parts = partition(records, args.part_limit_bytes)
    batches = [list(range(i + 1, min(i + args.parts_per_batch, len(parts)) + 1))
               for i in range(0, len(parts), args.parts_per_batch)]
    archive.require(all(sum(archive.zip_upper_bound(parts[n - 1]) for n in batch) <= args.resident_limit_bytes
                        for batch in batches), 'A batch exceeds the resident bound')
    output, private = args.output_root.resolve(), args.private_output.resolve()
    archive.require(output.is_relative_to(root) and not output.is_relative_to(root / 'results'),
                    'Preservation output must be outside scientific results')
    archive.require(output != private and not output.exists() and not private.exists(),
                    'Fresh distinct preservation directories are required')
    plan = {'schema_version': 1, 'purpose': PURPOSE, 'repository': str(root), 'prefix': args.prefix,
            'output_root': str(output), 'private_output': str(private), 'inventory': inventory_record,
            'protected_files': protected, 'parts': parts, 'batches': batches,
            'part_limit_bytes': args.part_limit_bytes, 'resident_limit_bytes': args.resident_limit_bytes,
            'original_checkpoint_deletion_permitted': False}
    output.mkdir(parents=True)
    private.mkdir(parents=True)
    path = output / 'PRESERVATION_PLAN.json'
    archive.write_new(path, plan)
    return {'plan': archive.file_receipt(path), 'checkpoint_count': len(records),
            'checkpoint_bytes': sum(item['bytes'] for item in records), 'parts': len(parts), 'batches': len(batches)}


def load_plan(args):
    archive.require(archive.digest(args.plan) == args.plan_sha256, 'Preservation plan digest mismatch')
    plan = archive.read_json(args.plan)
    archive.require(plan.get('schema_version') == 1 and plan.get('purpose') == PURPOSE and
                    plan.get('original_checkpoint_deletion_permitted') is False, 'Unsupported preservation plan')
    archive.require(args.plan.resolve().parent == Path(plan['output_root']), 'Plan output directory differs')
    root = Path(plan['repository'])
    archive.check_sources(root, {'protected_files': plan['protected_files'], 'members': []})
    archive.require([n for batch in plan['batches'] for n in batch] == list(range(1, len(plan['parts']) + 1)),
                    'Part schedule is incomplete or duplicated')
    members = [item for part in plan['parts'] for item in part]
    archive.require(len(archive.exact_records(members)) == len(members), 'Checkpoint appears in multiple parts')
    return root, plan


def locations(plan, number):
    prefix = f"{plan['prefix']}_batch_{number:03d}"
    output, private = Path(plan['output_root']), Path(plan['private_output'])
    return {'manifest': output / (prefix + '_MANIFEST.json'),
            **{name: private / (prefix + '_' + name + '.json')
               for name in ('sources', 'request', 'result', 'verified_upload')},
            'stderr': private / (prefix + '_stderr.txt')}


def part_path(plan, number):
    return Path(plan['output_root']) / f"{plan['prefix']}_part_{number:04d}.zip"


@contextmanager
def campaign_lock(plan):
    with (Path(plan['output_root']) / '.preservation.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Preservation campaign is busy') from error
        yield


def load_batch(plan, plan_sha, number, *, verify_zips=False, require_sources=True):
    loc = locations(plan, number)
    index = archive.read_json(loc['manifest'])
    archive.require(index.get('purpose') == PURPOSE and index.get('plan_sha256') == plan_sha and
                    index.get('batch') == number and index.get('originals_deleted') is False,
                    'Batch manifest belongs to another preservation plan')
    numbers = plan['batches'][number - 1]
    archive.require([row['number'] for row in index['parts']] == numbers, 'Batch part schedule differs')
    for row, part_number in zip(index['parts'], numbers):
        members = plan['parts'][part_number - 1]
        archive.require(row['members'] == members and row['archive']['local_path'] == str(part_path(plan, part_number)),
                        'Batch members or archive path differ')
        if verify_zips:
            path = part_path(plan, part_number)
            archive.require(archive.file_receipt(path) == row['archive'], 'Uploaded archive hash or size differs')
            verify_part(path, members, plan['part_limit_bytes'])
    sources = [archive.file_receipt(loc['manifest'])] + [row['archive'] for row in index['parts']]
    if require_sources:
        pinned = archive.read_json(loc['sources'])
        archive.require(pinned == {'plan_sha256': plan_sha, 'batch': number, 'sources': sources},
                        'Pre-upload source bytes or manifest changed')
    return index, sources


def completed_batches(plan, plan_sha, *, pending=None):
    completed, gap = [], False
    for number, numbers in enumerate(plan['batches'], 1):
        loc = locations(plan, number)
        present = any(path.exists() for path in loc.values()) or any(
            part_path(plan, n).exists() or part_path(plan, n).with_suffix('.zip.partial').exists() for n in numbers)
        if not present:
            gap = True
            continue
        if number == pending and not loc['verified_upload'].exists():
            archive.require(not gap, 'Pending batch follows an unfinished batch')
            gap = True
            continue
        archive.require(not gap and loc['verified_upload'].is_file(),
                        'Pending or unknown upload exists; recreation and automatic retry are forbidden')
        index, sources = load_batch(plan, plan_sha, number)
        saved = archive.read_json(loc['verified_upload'])
        archive.require(saved.get('purpose') == PURPOSE + '_verified_upload' and
                        saved.get('plan_sha256') == plan_sha and saved.get('batch') == number and
                        saved.get('all_members_verified_locally') is True and saved.get('sources') == sources,
                        'Verified upload record differs from its immutable sources')
        archive.validate_upload_results(saved['results'], {item['local_path']: item for item in sources})
        completed.append(number)
    return completed


def progress(plan, completed):
    return {'complete': len(completed) == len(plan['batches']), 'completed_batches': len(completed),
            'completed_parts': sum(len(plan['batches'][n - 1]) for n in completed), 'originals_deleted': False}


def cleanup_identical_partials(root, parts):
    """Remove only exact copies of verified finalized parts under campaign lock."""
    duplicates = []
    for row in parts:
        final = archive.safe_path(root, Path(row['archive']['local_path']).relative_to(root).as_posix())
        partial = archive.safe_path(root, final.relative_to(root).as_posix() + '.partial', exists=False)
        try:
            if not partial.exists():
                continue
            archive.require(archive.file_receipt(final) == row['archive'], 'Finalized part changed before partial cleanup')
            receipt = archive.file_receipt(partial)
            archive.require(all(receipt[key] == row['archive'][key] for key in ('bytes', 'sha256')),
                            'Partial differs from its verified finalized ZIP; inspect before recovery')
            duplicates.append((partial, receipt))
        except FileNotFoundError:
            # An already absent temporary copy needs no cleanup.
            archive.require(final.is_file(), 'Finalized part disappeared during partial cleanup')
    removed = []
    for path, receipt in duplicates:
        try:
            archive.require(archive.file_receipt(path) == receipt, 'Partial changed before cleanup')
            path.unlink(missing_ok=True)
            removed.append(receipt)
        except FileNotFoundError:
            pass
    return removed


def require_no_preparation(plan, number):
    prefix = f"{plan['prefix']}_batch_{number:03d}_"
    archive.require(not list(Path(plan['private_output']).glob(prefix + '*')) and
                    not list(Path(plan['output_root']).glob(prefix + '*')),
                    'Upload preparation or outcome exists; prepared recovery is forbidden')


def saved_temporary_parts(plan, plan_sha, completed):
    """Caller has verified all completed upload records under the campaign lock."""
    return [row['archive'] for number in completed for row in load_batch(plan, plan_sha, number)[0]['parts']]


def cleanup_saved_temporaries(root, parts):
    # The reused cleanup accepts finalized copies only at their exact saved hash.
    # Completed batches' own partial copies are disposable after verified saving.
    workflow.cleanup_completed_temporary_parts(root, {'parts': parts})


def parse_abandoned_partials(plan, batch, values, verified_parts):
    verified = {row['number']: row for row in verified_parts}
    result = []
    for value in values:
        fields = value.split(':')
        archive.require(len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit() and
                        re.fullmatch(r'[0-9a-f]{64}', fields[2]), 'Invalid abandoned partial receipt')
        number, size, sha = int(fields[0]), int(fields[1]), fields[2]
        archive.require(number in plan['batches'][batch - 1] and number in verified,
                        'An abandoned partial requires its verified finalized ZIP')
        result.append({'local_path': str(part_path(plan, number).with_suffix('.zip.partial')),
                       'bytes': size, 'sha256': sha, 'final': verified[number]['archive']})
    archive.require(len({item['local_path'] for item in result}) == len(result), 'Duplicate abandoned partial receipt')
    return result


def cleanup_abandoned_partials(root, records):
    """Explicit recovery only, after intact originals and final ZIP verification."""
    for item in records:
        path = archive.safe_path(root, Path(item['local_path']).relative_to(root).as_posix(), exists=False)
        try:
            if not path.exists():
                continue
            actual = archive.file_receipt(path)
            if all(actual[key] == item['final'][key] for key in ('bytes', 'sha256')):
                continue  # The ordinary exact-duplicate cleanup handles this copy.
            archive.require(all(actual[key] == item[key] for key in ('local_path', 'bytes', 'sha256')),
                            'Abandoned partial differs from the explicit recovery receipt')
            archive.require(archive.file_receipt(Path(item['final']['local_path'])) == item['final'],
                            'Finalized ZIP changed before abandoned partial cleanup')
            path.unlink(missing_ok=True)
        except FileNotFoundError:
            archive.require(Path(item['final']['local_path']).is_file(), 'Finalized ZIP disappeared')


def create_part(root, plan, number):
    members, path = plan['parts'][number - 1], part_path(plan, number)
    partial = path.with_suffix('.zip.partial')
    archive.require(not path.exists() and not partial.exists(), 'Part path already exists')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as handle:
        handle.writestr('MANIFEST.json', archive.manifest_bytes(members))
        for item in members:
            handle.write(archive.safe_path(root, item['path']), arcname=item['path'])
    verify_part(partial, members, plan['part_limit_bytes'])
    with partial.open('rb') as handle:
        os.fsync(handle.fileno())
    partial.rename(path)
    return {'number': number, 'members': members, 'archive': archive.file_receipt(path)}


def prepare_upload(root, plan, plan_sha, number, parts, *, recovered=False, cleanup=None):
    """Create immutable upload inputs only after the complete byte audit passes."""
    require_no_preparation(plan, number)
    members = [item for row in parts for item in row['members']]
    archive.check_sources(root, {'protected_files': plan['protected_files'], 'members': members})
    if cleanup is not None:
        cleanup()
    removed = cleanup_identical_partials(root, parts)
    archive.require(archive.resident_bytes(Path(plan['output_root'])) <= plan['resident_limit_bytes'],
                    'Actual resident archive size exceeded the bound')
    require_no_preparation(plan, number)
    loc = locations(plan, number)
    archive.write_new(loc['manifest'], {'schema_version': 1, 'purpose': PURPOSE,
        'plan_sha256': plan_sha, 'batch': number, 'parts': parts, 'originals_deleted': False,
        'recovered_prepared_parts': recovered})
    _, sources = load_batch(plan, plan_sha, number, require_sources=False)
    archive.write_new(loc['sources'], {'plan_sha256': plan_sha, 'batch': number, 'sources': sources})
    archive.write_new(loc['request'], {'uploads': [{'local_path': item['local_path'],
        'purpose': 'create_library_file', 'library_artifact_type': 'other'} for item in sources]})
    return {'phase': 'awaiting_root_upload', 'batch': number, 'parts': plan['batches'][number - 1],
            'upload_request': str(loc['request']), 'expected_result': str(loc['result']),
            'expected_stderr': str(loc['stderr']), 'source_receipts': archive.file_receipt(loc['sources']),
            'recovered_prepared_parts': recovered, 'removed_identical_partials': removed, 'originals_deleted': False}


def create_next(args):
    root, plan = load_plan(args)
    with campaign_lock(plan):
        completed = completed_batches(plan, args.plan_sha256)
        if len(completed) == len(plan['batches']):
            return progress(plan, completed)
        number = len(completed) + 1
        numbers = plan['batches'][number - 1]
        planned = [item for n in numbers for item in plan['parts'][n - 1]]
        archive.check_sources(root, {'protected_files': plan['protected_files'], 'members': planned})
        saved = saved_temporary_parts(plan, args.plan_sha256, completed)
        cleanup_saved_temporaries(root, saved)
        needed = sum(archive.zip_upper_bound(plan['parts'][n - 1]) for n in numbers)
        archive.require(archive.resident_bytes(Path(plan['output_root'])) + needed <= plan['resident_limit_bytes'],
                        'Resident bound exceeded; inspect and remove only verified saved temporary copies')
        archive.require(shutil.disk_usage(plan['output_root']).free >= needed + 64 * 1024 * 1024,
                        'Insufficient free disk for this preservation batch')
        parts = []
        def cleanup():
            cleanup_saved_temporaries(root, saved)
            cleanup_identical_partials(root, parts)
        for n in numbers:
            cleanup()
            archive.require(archive.resident_bytes(Path(plan['output_root'])) + archive.zip_upper_bound(plan['parts'][n - 1]) <=
                            plan['resident_limit_bytes'], 'Resident bound exceeded before part allocation')
            parts.append(create_part(root, plan, n))
            cleanup()
            archive.require(archive.resident_bytes(Path(plan['output_root'])) <= plan['resident_limit_bytes'],
                            'Actual resident archive size exceeded the bound')
        return prepare_upload(root, plan, args.plan_sha256, number, parts, cleanup=cleanup)


def recover_prepared(args):
    """Verify existing complete ZIPs without recompressing or retrying uploads."""
    root, plan = load_plan(args)
    archive.require(1 <= args.batch <= len(plan['batches']), 'Invalid preservation batch')
    with campaign_lock(plan):
        completed = completed_batches(plan, args.plan_sha256, pending=args.batch)
        archive.require(args.batch == len(completed) + 1, 'Recover only the next unfinished batch')
        require_no_preparation(plan, args.batch)
        parts, missing = [], []
        for number in plan['batches'][args.batch - 1]:
            path = archive.safe_path(root, part_path(plan, number).relative_to(root).as_posix(), exists=False)
            if not path.exists():
                archive.require(getattr(args, 'complete_missing_parts', False), 'Prepared finalized ZIP is missing')
                missing.append(number)
                continue
            members = plan['parts'][number - 1]
            verify_part(path, members, plan['part_limit_bytes'])
            parts.append({'number': number, 'members': members, 'archive': archive.file_receipt(path)})
        archive.require(parts, 'Recovery requires at least one complete verified part')
        planned = [item for n in plan['batches'][args.batch - 1] for item in plan['parts'][n - 1]]
        archive.check_sources(root, {'protected_files': plan['protected_files'], 'members': planned})
        abandoned = parse_abandoned_partials(plan, args.batch, getattr(args, 'discard_abandoned_partial', []), parts)
        saved = saved_temporary_parts(plan, args.plan_sha256, completed)
        def cleanup():
            require_no_preparation(plan, args.batch)
            cleanup_saved_temporaries(root, saved)
            cleanup_abandoned_partials(root, abandoned)
            cleanup_identical_partials(root, parts)
        cleanup()
        needed = sum(archive.zip_upper_bound(plan['parts'][n - 1]) for n in missing)
        archive.require(archive.resident_bytes(Path(plan['output_root'])) + needed <= plan['resident_limit_bytes'],
                        'Actual resident archive size exceeded the bound')
        if missing:
            archive.require(shutil.disk_usage(plan['output_root']).free >= needed + 64 * 1024 * 1024,
                            'Insufficient free disk to complete prepared parts')
        for number in missing:
            cleanup()
            archive.require(archive.resident_bytes(Path(plan['output_root'])) +
                            archive.zip_upper_bound(plan['parts'][number - 1]) <= plan['resident_limit_bytes'],
                            'Resident bound exceeded before recovery allocation')
            parts.append(create_part(root, plan, number))
            cleanup()
        parts.sort(key=lambda row: row['number'])
        return prepare_upload(root, plan, args.plan_sha256, args.batch, parts, recovered=True, cleanup=cleanup)


def finish_upload(args):
    root, plan = load_plan(args)
    archive.require(1 <= args.batch <= len(plan['batches']), 'Invalid preservation batch')
    with campaign_lock(plan):
        completed = completed_batches(plan, args.plan_sha256, pending=args.batch)
        if args.batch in completed:
            return progress(plan, completed)
        archive.require(args.batch == len(completed) + 1, 'Verify upload batches in order')
        loc = locations(plan, args.batch)
        archive.require(loc['request'].is_file() and loc['result'].is_file(),
                        'Upload request or outcome is missing; do not retry an unknown outcome')
        index, sources = load_batch(plan, args.plan_sha256, args.batch, verify_zips=True)
        expected_request = {'uploads': [{'local_path': item['local_path'], 'purpose': 'create_library_file',
                                        'library_artifact_type': 'other'} for item in sources]}
        archive.require(archive.read_json(loc['request']) == expected_request, 'Upload request changed')
        expected = {item['local_path']: item for item in sources}
        response = copy.deepcopy(archive.read_json(loc['result']))
        for item in response['results']:
            archive.require(item.get('local_path') in expected, 'Unexpected upload result path')
            for key in ('bytes', 'sha256'):
                archive.require(key not in item or item[key] == expected[item['local_path']][key],
                                'Upload result source digest or size differs')
            item.update(bytes=expected[item['local_path']]['bytes'], sha256=expected[item['local_path']]['sha256'])
        archive.validate_upload_results(response['results'], expected)
        members = [item for row in index['parts'] for item in row['members']]
        archive.check_sources(root, {'protected_files': plan['protected_files'], 'members': members})
        archive.write_new(loc['verified_upload'], {'purpose': PURPOSE + '_verified_upload',
            'plan_sha256': args.plan_sha256, 'batch': args.batch, 'sources': sources,
            'results': response['results'], 'all_members_verified_locally': True, 'originals_deleted': False})
        return {'batch_finished': args.batch, 'verified_upload': archive.file_receipt(loc['verified_upload']),
                'saved_temporary_parts': [row['archive'] for row in index['parts']],
                **progress(plan, completed + [args.batch])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    init = commands.add_parser('init')
    init.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('inventory', 'inventory-sha256', 'prefix'):
        init.add_argument('--' + name, required=True)
    for name in ('output-root', 'private-output'):
        init.add_argument('--' + name, type=Path, required=True)
    init.add_argument('--expected-runs', type=int, default=522)
    init.add_argument('--part-limit-bytes', type=int, default=PART_LIMIT)
    init.add_argument('--parts-per-batch', type=int, default=2)
    init.add_argument('--resident-limit-bytes', type=int, default=512 * 1024 * 1024)
    for phase in ('create-next', 'finish-upload', 'recover-prepared'):
        command = commands.add_parser(phase)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--plan-sha256', required=True)
        if phase in ('finish-upload', 'recover-prepared'):
            command.add_argument('--batch', type=int, required=True)
        if phase == 'recover-prepared':
            command.add_argument('--complete-missing-parts', action='store_true')
            command.add_argument('--discard-abandoned-partial', action='append', default=[],
                                 metavar='PART:BYTES:SHA256')
    args = parser.parse_args()
    result = {'init': initialize, 'create-next': create_next, 'finish-upload': finish_upload,
              'recover-prepared': recover_prepared}[args.phase](args)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()

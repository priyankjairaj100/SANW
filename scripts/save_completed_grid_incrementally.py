#!/usr/bin/env python3
"""Archive completed selected grids through verified, bounded private saves.

Storage orchestration only. This does not modify scientific training sources.
An interrupted or uncertain upload stops the run without an automatic retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import SimpleNamespace

import stream_completed_candidate_checkpoints as archive


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_saved_batch(root, manifest, upload_path, *, grid_root, candidates,
                       retained_manifest, retained_sha256, check_sources=True):
    """Accept only a fully uploaded batch with a completed, linked prune receipt."""
    archive.require(manifest.is_file() and not manifest.is_symlink(),
                    'Incomplete batch manifest; inspect prior upload outcome before recovery')
    manifest_sha = archive.digest(manifest)
    index = archive.load_index(manifest, manifest_sha)
    prune_path = manifest.with_name(index['prefix'] + '_LOCAL_PRUNE_RECEIPT.json')
    archive.require(prune_path.is_file() and upload_path.is_file(),
                    'Batch lacks completed prune/upload records; automatic retry is forbidden')
    prune = archive.read_json(prune_path)
    upload = archive.read_json(upload_path)
    archive.require(prune.get('status') == 'completed',
                    'Batch prune is incomplete; inspect its receipt before recovery')
    archive.require(prune.get('archive_manifest_sha256') == manifest_sha and
                    prune.get('verified_upload_record_sha256') == archive.digest(upload_path),
                    'Completed batch receipt digest mismatch')
    archive.require(upload.get('purpose') == archive.PURPOSE + '_verified_uploads' and
                    upload.get('all_parts_verified_locally') is True,
                    'Batch lacks a verified upload record')
    archive.require(upload.get('archive_manifest_sha256') == manifest_sha and
                    upload.get('archive_manifest_bytes') == manifest.stat().st_size,
                    'Verified upload record binds another manifest')
    sources = archive.upload_sources(manifest, index)
    archive.require(upload['sources'] == sources, 'Saved upload source records differ')
    archive.validate_upload_results(upload['results'], sources)
    snapshot = index['snapshot']
    archive.require(snapshot['grid_root'] == grid_root and snapshot['candidate_names'] == sorted(candidates),
                    'Saved batch uses another grid or candidate schedule')
    keep = snapshot.get('retained_manifest')
    archive.require((keep is None and retained_manifest is None and retained_sha256 is None) or
                    (keep is not None and keep['path'] == retained_manifest and keep['sha256'] == retained_sha256),
                    'Saved batch retained manifest differs')
    archive.require(prune.get('deleted_members') == snapshot['members'] and
                    prune.get('deleted_bytes') == sum(x['bytes'] for x in snapshot['members']),
                    'Completed prune receipt has incomplete member coverage')
    if check_sources:
        archive.check_sources(root, snapshot, allow_missing_members=True)
    made = {'manifest': archive.file_receipt(manifest),
            'parts': [{'local_path': str(manifest.parent / part['name']), 'bytes': part['bytes'],
                       'sha256': part['sha256']} for part in index['parts']],
            'member_count': len(snapshot['members']), 'candidate_count': len(candidates),
            'original_bytes': sum(x['bytes'] for x in snapshot['members'])}
    pruned = {'prune_receipt': archive.file_receipt(prune_path), 'deleted_count': len(snapshot['members']),
              'deleted_bytes': prune['deleted_bytes']}
    return made, index, pruned


def cleanup_completed_temporary_parts(root, made):
    """Caller must first verify a completed prune and its exact upload records."""
    removed = []
    for item in made['parts']:
        path = archive.safe_path(root, Path(item['local_path']).relative_to(root).as_posix(), exists=False)
        try:
            if path.exists():
                archive.require(archive.file_receipt(path) == item, 'Temporary archive changed before removal')
                path.unlink(missing_ok=True)
                removed.append(str(path))
        except FileNotFoundError:
            # Another authorized cleanup may remove a verified copy during stat,
            # hashing, or unlink. Absence is already the desired final state.
            pass
        # ZIP creation is the only writer of this exact temporary path. A completed
        # batch cannot still have its create operation running. Its saved ZIP is
        # authoritative, so a synchronized incomplete temporary copy is disposable.
        partial = archive.safe_path(root, path.relative_to(root).as_posix() + '.partial', exists=False)
        try:
            archive.require(stat.S_ISREG(partial.stat().st_mode), 'Expected a regular temporary ZIP')
            partial.unlink(missing_ok=True)
            removed.append(str(partial))
        except FileNotFoundError:
            pass
    return removed


def cleanup_retired(root, retired, completed_batches):
    for made in completed_batches:
        cleanup_completed_temporary_parts(root, made)
    for item in retired:
        path = archive.safe_path(root, item['path'], exists=False)
        try:
            if path.exists():
                archive.require(path.stat().st_size == item['bytes'] and archive.digest(path) == item['sha256'],
                                'Reappearing retired bytes differ from the verified upload')
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def recover_unuploaded_creation(root, args, output, private, candidates, keep_path, keep_hash):
    """Explicitly discard only abandoned ZIP creation with every original intact."""
    number = args.recover_unuploaded_batch
    count = (len(candidates) + args.batch_size - 1) // args.batch_size
    archive.require(isinstance(number, int) and 1 <= number <= count, 'Recovery batch is outside this campaign')
    prefix = f'{args.prefix}_batch_{number:03d}'
    selected = candidates[(number - 1) * args.batch_size:number * args.batch_size]
    keep_name = keep_path.relative_to(root).as_posix() if keep_path else None
    for previous in range(1, number):
        prior = f'{args.prefix}_batch_{previous:03d}'
        verify_saved_batch(root, output / (prior + '_MANIFEST.json'), private / (prior + '_upload_record.json'),
            grid_root=args.grid_root, candidates=candidates[(previous - 1) * args.batch_size:previous * args.batch_size],
            retained_manifest=keep_name, retained_sha256=keep_hash)
    for later in range(number + 1, count + 1):
        later_prefix = f'{args.prefix}_batch_{later:03d}'
        archive.require(not list(output.glob(later_prefix + '_*')) and not list(private.glob(later_prefix + '_*')),
                        'Later batch records exist; inspect campaign history before recovery')
    expected = {f'{prefix}_part_{i:04d}.zip{suffix}' for i in range(1, len(selected) + 1)
                for suffix in ('', '.partial')}
    receipt_path = private / f'{args.prefix}_creation_recovery_batch_{number:03d}.json'
    archive.require(not receipt_path.exists(), 'Creation recovery receipt exists; inspect its prior status')
    with archive.candidate_locks(root, args.grid_root, selected):
        # The same candidate locks guard ZIP creation. A live create operation
        # cannot race this recovery. An uploader requires a final manifest first.
        archive.require(not list(private.glob(prefix + '_*')),
                        'Upload request or outcome exists; unuploaded recovery is forbidden')
        found = sorted(output.glob(prefix + '_*'))
        archive.require(found and all(path.name in expected for path in found),
                        'Recovery accepts only expected ZIP temporaries, without a final manifest or prune receipt')
        snapshot = archive.inspect_candidates(root, args.grid_root, selected, keep_name, keep_hash)
        observed = []
        for path in found:
            path = archive.safe_path(root, path.relative_to(root).as_posix())
            archive.require(path.is_file(), 'Recovery temporary path is not a regular file')
            observed.append(archive.file_receipt(path))
        # Repeat absence checks after the complete original-file audit.
        archive.require(not list(private.glob(prefix + '_*')) and
                        all(path.name in expected for path in output.glob(prefix + '_*')),
                        'Upload or final archive records appeared during recovery inspection')
        archive.write_new(receipt_path, {'status': 'in_progress', 'batch': number,
            'purpose': 'discard_unuploaded_temporary_creation', 'snapshot': snapshot,
            'temporary_files': observed, 'upload_records_present': False})
        removed = []
        for item in observed:
            path = archive.safe_path(root, Path(item['local_path']).relative_to(root).as_posix(), exists=False)
            try:
                archive.require(archive.file_receipt(path) == item, 'Temporary creation changed during recovery')
                path.unlink(missing_ok=True)
                removed.append(item)
            except FileNotFoundError:
                pass
        archive.check_sources(root, snapshot)
        write(receipt_path, {'status': 'completed', 'batch': number,
            'purpose': 'discard_unuploaded_temporary_creation', 'snapshot': snapshot,
            'removed_temporary_files': removed, 'upload_records_present': False,
            'scientific_originals_verified_intact': True})
    result = {'recovered_unuploaded_batch': number, 'temporary_files_removed': len(removed),
              'recovery_receipt': archive.file_receipt(receipt_path), 'uploads_performed': False}
    print(json.dumps(result), flush=True)
    return result


def verify_completed_creation_recovery(root, args, output, private, number, candidates, keep_path, keep_hash,
                                       *, snapshot=None, expected_receipt_sha256=None, cleanup=False):
    """Recognize exact returned temporaries from a completed explicit recovery."""
    prefix = f'{args.prefix}_batch_{number:03d}'
    receipt_path = private / f'{args.prefix}_creation_recovery_batch_{number:03d}.json'
    archive.require(receipt_path.is_file() and not receipt_path.is_symlink(), 'Completed creation recovery receipt is missing')
    receipt_sha = archive.digest(receipt_path)
    archive.require(expected_receipt_sha256 is None or receipt_sha == expected_receipt_sha256,
                    'Creation recovery receipt changed after preflight')
    receipt = archive.read_json(receipt_path)
    archive.require(receipt.get('status') == 'completed' and receipt.get('batch') == number and
                    receipt.get('purpose') == 'discard_unuploaded_temporary_creation' and
                    receipt.get('upload_records_present') is False and
                    receipt.get('scientific_originals_verified_intact') is True,
                    'Creation recovery receipt is not complete')
    archive.require(not list(private.glob(prefix + '_*')),
                    'Upload request or outcome exists; recovered creation cannot be reused')
    keep_name = keep_path.relative_to(root).as_posix() if keep_path else None
    if snapshot is None:
        with archive.candidate_locks(root, args.grid_root, candidates):
            snapshot = archive.inspect_candidates(root, args.grid_root, candidates, keep_name, keep_hash)
    archive.require(snapshot['grid_root'] == args.grid_root and snapshot['candidate_names'] == sorted(candidates),
                    'Recovered creation candidate schedule differs')
    keep = snapshot.get('retained_manifest')
    archive.require((keep is None and keep_name is None and keep_hash is None) or
                    (keep is not None and keep['path'] == keep_name and keep['sha256'] == keep_hash),
                    'Recovered creation retained manifest differs')
    archive.require(snapshot == receipt['snapshot'], 'Recovered creation snapshot or source bytes changed')
    expected_names = {f'{prefix}_part_{i:04d}.zip{suffix}' for i in range(1, len(candidates) + 1)
                      for suffix in ('', '.partial')}
    recorded = {}
    for item in receipt['removed_temporary_files']:
        path = Path(item['local_path'])
        archive.require(path.parent == output and path.name in expected_names and item['local_path'] not in recorded,
                        'Recovery receipt has an unsafe or duplicate temporary path')
        recorded[item['local_path']] = item
    archive.require(recorded, 'Recovery receipt has no exact removed temporary records')
    current = []
    for path in sorted(output.glob(prefix + '_*')):
        archive.require(str(path) in recorded, 'Unknown temporary or final record appeared for recovered creation')
        path = archive.safe_path(root, path.relative_to(root).as_posix())
        archive.require(archive.file_receipt(path) == recorded[str(path)], 'Returned creation temporary bytes differ')
        current.append((path, recorded[str(path)]))
    archive.require(not list(private.glob(prefix + '_*')), 'Upload records appeared during recovery reconciliation')
    if cleanup:
        # The caller holds the current candidate locks through this hook and ZIP
        # creation. Do not call this again after the new creation starts writing.
        for path, item in current:
            try:
                archive.require(archive.file_receipt(path) == item, 'Returned temporary changed before cleanup')
                path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
    return {'receipt_sha256': receipt_sha, 'batch': number, 'returned_temporaries': len(current)}


def run(args, *, repository=None):
    root = (repository or Path(__file__).resolve().parents[1]).resolve()
    archive.require(re.fullmatch(r'[A-Za-z0-9_-]+', args.prefix), 'Unsafe archive prefix')
    archive.require(1 <= args.batch_size <= 19, 'Batch size must be between 1 and 19')
    archive.require(args.upload_helper.is_file(), 'Upload helper is missing')
    grid = archive.safe_path(root, args.grid_root, exists=False)
    archive.require(grid.is_dir(), 'Grid is missing')
    output = root / 'recovery' / 'incremental_archives_20261004' / args.prefix
    recover = getattr(args, 'recover_unuploaded_batch', None)
    resume = getattr(args, 'resume_verified', False) or recover is not None
    archive.require(resume or not output.exists(), 'Output already exists; inspect prior upload outcomes before any restart')
    private = args.private_output.resolve()
    archive.require(resume or not private.exists() or not list(private.glob(args.prefix + '_batch_*')),
                    'Private batch records already exist; automatic upload retry is forbidden')
    state_path = grid / 'state_manifest.json'
    completions = sorted((grid / 'candidates').rglob('completion.json'))
    archive.require(len(completions) == args.expected_completions and completions,
                    'Completion count differs from the requested snapshot')
    terminal = set()
    terminal_states = []
    for path in completions:
        receipt = json.loads(path.read_text())
        relative = f"epochs/epoch_{receipt['epochs_completed']:02d}.pt"
        name = (path.parent / relative).relative_to(root).as_posix()
        terminal.add(name)
        terminal_states.append({'checkpoint': name, 'checkpoint_sha256': receipt['artifact_sha256'][relative]})
    if getattr(args, 'allow_unselected_incomplete_grid', False):
        archive.require(not any((grid / name).exists() for name in
                        ('state_manifest.json', 'selection_primary.json', 'selection_sensitivity.json')),
                        'Incomplete-grid mode requires a grid before selection')
        retained, keep_path, keep_hash = terminal_states, None, None
    else:
        state = json.loads(state_path.read_text())
        archive.require(state.get('matched_controls_complete') is True, 'Matched controls are incomplete')
        ids = {x['state_id'] for key in ('selections', 'decomposition_selections', 'factorial_selections')
               for x in state.get(key, [])}
        retained = [x for x in state['states'] if x['state_id'] in ids or x.get('checkpoint') in terminal]
        archive.require(terminal <= {x.get('checkpoint') for x in retained}, 'A terminal checkpoint is absent')
        keep = {**state, 'states': retained,
                'full_state_manifest': {'path': state_path.relative_to(root).as_posix(),
                                        'sha256': archive.digest(state_path)}}
        keep_path = grid / 'archival_keep_manifest.json'
        if keep_path.exists():
            archive.require(json.loads(keep_path.read_text()) == keep, 'Existing retained manifest differs')
        else:
            archive.write_new(keep_path, keep)
        keep_hash = archive.digest(keep_path)
    kept_paths = {x.get('checkpoint') for x in retained}
    candidates = []
    for path in completions:
        receipt = json.loads(path.read_text())
        remaining = [name for name in receipt['artifact_sha256'] if re.fullmatch(r'epochs/epoch_\d{2,}\.pt', name)
                     and (path.parent / name).relative_to(root).as_posix() not in kept_paths]
        if remaining:
            candidates.append(path.parent.relative_to(grid).as_posix())
    private.mkdir(parents=True, exist_ok=True)
    if recover is not None:
        return recover_unuploaded_creation(root, args, output, private, candidates, keep_path, keep_hash)
    summaries = []
    retired = []
    completed_batches = []
    saved = {}
    recovered = {}
    if resume:
        seen_missing = False
        expected_prefixes = set()
        for offset in range(0, len(candidates), args.batch_size):
            number = offset // args.batch_size + 1
            prefix = f'{args.prefix}_batch_{number:03d}'
            expected_prefixes.add(prefix)
            recovery_receipt = private / f'{args.prefix}_creation_recovery_batch_{number:03d}.json'
            if (recovery_receipt.exists() and not (output / (prefix + '_MANIFEST.json')).exists() and
                not (output / (prefix + '_LOCAL_PRUNE_RECEIPT.json')).exists()):
                recovered[number] = verify_completed_creation_recovery(root, args, output, private, number,
                    candidates[offset:offset + args.batch_size], keep_path, keep_hash)
                seen_missing = True
                # A completed explicit recovery proves these exact old copies are
                # abandoned. Do not reinterpret a synchronization rescan as upload
                # evidence. The one-shot locked hook checks them again before create.
                continue
            exists = bool(list(output.glob(prefix + '_*')) or list(private.glob(prefix + '_*')))
            if not exists:
                seen_missing = True
                continue
            archive.require(not seen_missing, 'Saved batches have a gap; inspect campaign history before recovery')
            manifest = output / (prefix + '_MANIFEST.json')
            saved[number] = verify_saved_batch(root, manifest, private / (prefix + '_upload_record.json'),
                grid_root=args.grid_root, candidates=candidates[offset:offset + args.batch_size],
                retained_manifest=keep_path.relative_to(root).as_posix() if keep_path else None,
                retained_sha256=keep_hash)
        # Refuse a changed schedule which would silently ignore old later batches.
        pattern = re.compile(re.escape(args.prefix) + r'_batch_\d{3,}')
        for path in [*output.glob(args.prefix + '_batch_*'), *private.glob(args.prefix + '_batch_*')]:
            match = pattern.match(path.name)
            archive.require(match and match.group() in expected_prefixes, 'Unscheduled prior batch requires manual review')
    for offset in range(0, len(candidates), args.batch_size):
        number = offset // args.batch_size + 1
        prefix = f'{args.prefix}_batch_{number:03d}'
        cleanup_retired(root, retired, completed_batches)
        if number in saved:
            made, index, pruned = saved[number]
            retired.extend(index['snapshot']['members'])
            retired.extend({'path': Path(part['local_path']).relative_to(root).as_posix(),
                            'bytes': part['bytes'], 'sha256': part['sha256']} for part in made['parts'])
            completed_batches.append(made)
            cleanup_retired(root, retired, completed_batches)
            summaries.append({'batch': number, 'manifest': made['manifest'], 'parts': made['parts'],
                              'member_count': made['member_count'], 'original_bytes': made['original_bytes'],
                              'prune': pruned})
            print(json.dumps({'batch_resumed_verified': number, 'candidates': made['candidate_count']}), flush=True)
            continue
        made = archive.create(SimpleNamespace(repository=root, grid_root=args.grid_root,
            candidate=candidates[offset:offset + args.batch_size], output_root=output, prefix=prefix,
            retained_manifest=keep_path.relative_to(root).as_posix() if keep_path else None,
            retained_manifest_sha256=keep_hash,
            part_limit_bytes=100*1024*1024, resident_limit_bytes=512*1024*1024,
            pre_allocation_cleanup=lambda previous_retired=tuple(retired), previous_batches=tuple(completed_batches):
                cleanup_retired(root, previous_retired, previous_batches),
            pre_creation_cleanup=(lambda snapshot, batch=number, batch_candidates=tuple(candidates[offset:offset + args.batch_size]),
                                         receipt_sha=recovered[number]['receipt_sha256']:
                verify_completed_creation_recovery(root, args, output, private, batch, list(batch_candidates),
                    keep_path, keep_hash, snapshot=snapshot, expected_receipt_sha256=receipt_sha, cleanup=True))
                if number in recovered else None))
        manifest = Path(made['manifest']['local_path'])
        index = json.loads(manifest.read_text())
        sources = [made['manifest'], *made['parts']]
        request = {'uploads': [{'local_path': x['local_path'], 'purpose': 'create_library_file',
                                'library_artifact_type': 'other'} for x in sources]}
        request_path = private / f'{prefix}_request.json'
        write(request_path, request)
        raw_path = private / f'{prefix}_result.json'
        error_path = private / f'{prefix}_stderr.txt'
        with raw_path.open('x') as stdout, error_path.open('x') as stderr:
            result = subprocess.run([sys.executable, str(args.upload_helper)],
                input=json.dumps(request), text=True, stdout=stdout, stderr=stderr)
        if result.returncode != 0:
            raise RuntimeError(f'Upload returned {result.returncode}; inspect {raw_path}. No automatic retry.')
        response = json.loads(raw_path.read_text())
        records = response['results']
        expected = {x['local_path']: x for x in sources}
        archive.require(len(records) == len(expected), 'Upload returned an incomplete result set')
        archive.require(len({x['local_path'] for x in records}) == len(records), 'Upload returned duplicate paths')
        for item in records:
            archive.require(item['local_path'] in expected, 'Upload returned an unknown path')
            source = expected[item['local_path']]
            archive.require(item['status'] == 'succeeded' and item.get('library_file_id') and item.get('file_id'),
                            'Upload is not finalized; automatic retry is forbidden')
            archive.require(archive.file_receipt(Path(item['local_path'])) == source, 'Uploaded source bytes changed')
            item.update(bytes=source['bytes'], sha256=source['sha256'])
        enriched = private / f'{prefix}_verified_sources.json'
        write(enriched, response)
        upload_path = private / f'{prefix}_upload_record.json'
        verified = archive.record_uploads(SimpleNamespace(manifest=manifest,
            manifest_sha256=made['manifest']['sha256'], upload_receipt=enriched, output_receipt=upload_path))
        pruned = archive.prune(SimpleNamespace(repository=root, manifest=manifest,
            manifest_sha256=made['manifest']['sha256'], upload_record=upload_path,
            upload_record_sha256=verified['verified_upload_record']['sha256'],
            confirm_uploaded_and_authorize_deletion=True))
        retired.extend(index['snapshot']['members'])
        for part in made['parts']:
            path = Path(part['local_path'])
            retired.append({'path': path.relative_to(root).as_posix(), 'bytes': part['bytes'], 'sha256': part['sha256']})
        completed_batches.append(made)
        cleanup_retired(root, retired, completed_batches)
        summaries.append({'batch': number, 'manifest': made['manifest'],
                          'parts': made['parts'], 'member_count': made['member_count'],
                          'original_bytes': made['original_bytes'], 'prune': pruned})
        write(output / 'progress.json', {'grid_root': args.grid_root, 'complete': False, 'batches': summaries})
        write(private / f'{args.prefix}_retired_files.json', {'files': retired})
        print(json.dumps({'batch_saved': number, 'candidates': made['candidate_count'],
                          'original_bytes': made['original_bytes']}), flush=True)
    for item in retained:
        if item.get('checkpoint'):
            archive.require(archive.digest(archive.safe_path(root, item['checkpoint'])) == item['checkpoint_sha256'],
                            'Retained checkpoint changed')
    write(output / 'progress.json', {'grid_root': args.grid_root, 'complete': True,
          'archived_candidates': len(candidates), 'retained_manifest_sha256': keep_hash, 'batches': summaries})
    write(private / f'{args.prefix}_retired_files.json', {'files': retired,
          'completed_temporary_partials': [part['local_path'] + '.partial'
                                          for made in completed_batches for part in made['parts']]})
    print(json.dumps({'complete': True, 'batches': len(summaries),
                      'original_bytes': sum(x['original_bytes'] for x in summaries)}), flush=True)
    return {'complete': True, 'batches': summaries, 'retained_manifest_sha256': keep_hash,
            'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-root', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--expected-completions', type=int, required=True)
    parser.add_argument('--upload-helper', type=Path, required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--allow-unselected-incomplete-grid', action='store_true',
                        help='Archive only current completed candidates before selection; preserve all terminals')
    parser.add_argument('--resume-verified', action='store_true',
                        help='Skip only batches with verified uploads and completed linked prune receipts')
    parser.add_argument('--recover-unuploaded-batch', type=int,
                        help='Discard only this abandoned temporary creation; require no upload records and intact originals; then exit')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

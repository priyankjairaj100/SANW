#!/usr/bin/env python3
"""Start or continue an archive, with a separate upload process.

init-fresh (or init) -> create-next -> ROOT PREPARED UPLOAD -> finish-upload.
This helper never uploads files. It never retries or overwrites an upload result.
"""
import argparse
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import stream_completed_candidate_checkpoints as archive
import save_completed_grid_incrementally as workflow

PURPOSE = 'verified_archive_continuation_plan'


def batch_names(plan, number):
    start = (number - 1) * plan['batch_size']
    return plan['remaining_candidates'][start:start + plan['batch_size']]


def batch_count(plan):
    return (len(plan['remaining_candidates']) + plan['batch_size'] - 1) // plan['batch_size']


def locations(plan, number):
    prefix = f"{plan['prefix']}_batch_{number:03d}"
    output, private = Path(plan['output_root']), Path(plan['private_output'])
    return {'prefix': prefix, 'manifest': output / (prefix + '_MANIFEST.json'),
            'prune': output / (prefix + '_LOCAL_PRUNE_RECEIPT.json'),
            'sources': private / (prefix + '_sources.json'),
            'request': private / (prefix + '_request.json'),
            'result': private / (prefix + '_result.json'),
            'stderr': private / (prefix + '_stderr.txt'),
            'enriched': private / (prefix + '_enriched_results.json'),
            'upload': private / (prefix + '_upload_record.json')}


def initialize(args, *, fresh=False):
    root = args.repository.resolve()
    archive.require(re.fullmatch(r'[A-Za-z0-9_-]+', args.prefix), 'Unsafe fresh prefix')
    archive.require(1 <= args.batch_size <= 19, 'Invalid batch schedule')
    if fresh:
        archive.require(all(getattr(args, key, None) is None for key in
                            ('prior_prefix', 'prior_output', 'prior_private', 'prior_batches')),
                        'A fresh campaign cannot exclude prior batches')
    else:
        archive.require(args.prefix != args.prior_prefix, 'Continuation requires a fresh prefix')
        archive.require(args.prior_batches > 0, 'Invalid batch schedule')
    grid = archive.safe_path(root, args.grid_root, exists=False)
    keep_path = archive.safe_path(root, args.retained_manifest)
    keep_record = archive.record(root, keep_path, args.retained_manifest_sha256)
    keep = archive.read_json(keep_path)
    completions = sorted((grid / 'candidates').rglob('completion.json'))
    archive.require(len(completions) == args.expected_completions, 'Unexpected completed grid size')
    kept = {state.get('checkpoint') for state in keep['states']}
    candidates = []
    for path in completions:
        receipt = archive.read_json(path)
        remaining = [name for name in receipt['artifact_sha256']
                     if re.fullmatch(r'epochs/epoch_\d{2,}\.pt', name) and
                     (path.parent / name).relative_to(root).as_posix() not in kept]
        if remaining:
            candidates.append(path.parent.relative_to(grid).as_posix())
    output, private = args.output_root.resolve(), args.private_output.resolve()
    archive.require(output.is_relative_to(root) and not output.is_relative_to(grid), 'Unsafe continuation output directory')
    archive.require(output != private, 'Archive and private directories must differ')
    if not fresh:
        archive.require(output != args.prior_output.resolve() and private != args.prior_private.resolve(),
                        'Continuation must use fresh output and private directories')
    archive.require(not output.exists() and not private.exists(), 'Continuation directories already exist')
    excluded, prior_records, protected = set(), [], []
    initial_bounds = None
    if fresh:
        # A new campaign has no preservation evidence for missing originals.
        # Verify every candidate now, without writing an archive or deleting bytes.
        with archive.candidate_locks(root, args.grid_root, candidates):
            snapshot = archive.inspect_candidates(root, args.grid_root, candidates,
                                                  args.retained_manifest, args.retained_manifest_sha256)
        protected.extend(snapshot['protected_files'])
        bounds = [archive.zip_upper_bound(item['members']) for item in snapshot['candidates']]
        archive.require(bounds and all(item['members'] for item in snapshot['candidates']),
                        'A fresh candidate has no unretained checkpoints')
        archive.require(max(bounds) <= archive.PART_LIMIT, 'One candidate exceeds the part limit')
        batch_bounds = [sum(bounds[i:i + args.batch_size]) for i in range(0, len(bounds), args.batch_size)]
        archive.require(max(batch_bounds) <= 512 * 1024 * 1024, 'Fresh batch exceeds the resident archive bound')
        initial_bounds = {'member_count': len(snapshot['members']),
                          'original_bytes': sum(item['bytes'] for item in snapshot['members']),
                          'maximum_part_upper_bound_bytes': max(bounds),
                          'maximum_batch_upper_bound_bytes': max(batch_bounds)}
    for number in range(1, (0 if fresh else args.prior_batches) + 1):
        prefix = f'{args.prior_prefix}_batch_{number:03d}'
        manifest = args.prior_output.resolve() / (prefix + '_MANIFEST.json')
        upload = args.prior_private.resolve() / (prefix + '_upload_record.json')
        expected = candidates[(number - 1) * args.batch_size:number * args.batch_size]
        archive.require(expected, 'Prior batch schedule exceeds current candidate list')
        made, index, pruned = workflow.verify_saved_batch(root, manifest, upload,
            grid_root=args.grid_root, candidates=expected, retained_manifest=args.retained_manifest,
            retained_sha256=args.retained_manifest_sha256)
        names = index['snapshot']['candidate_names']
        archive.require(not excluded.intersection(names), 'Prior batches overlap')
        excluded.update(names)
        protected.extend(index['snapshot']['protected_files'])
        prior_records.append({'batch': number, 'manifest': made['manifest'],
            'upload_record': archive.file_receipt(upload), 'prune_receipt': pruned['prune_receipt'],
            'candidates': names})
    plan = {'schema_version': 1, 'purpose': PURPOSE, 'repository': str(root),
            'campaign_mode': 'fresh' if fresh else 'continuation',
            'grid_root': args.grid_root, 'retained_manifest': keep_record,
            'expected_completions': args.expected_completions, 'prefix': args.prefix,
            'output_root': str(output), 'private_output': str(private), 'batch_size': args.batch_size,
            'all_archive_candidates': candidates, 'excluded_candidates': sorted(excluded),
            'remaining_candidates': [name for name in candidates if name not in excluded],
            'prior_verified_batches': prior_records, 'protected_files': archive.exact_records(protected)}
    if initial_bounds is not None:
        plan['initial_archive_bounds'] = initial_bounds
    archive.require(plan['remaining_candidates'], 'There are no remaining candidates')
    output.mkdir(parents=True)
    private.mkdir(parents=True)
    path = output / 'CONTINUATION_PLAN.json'
    archive.write_new(path, plan)
    return {'plan': archive.file_receipt(path), 'verified_prior_batches': len(prior_records),
            'excluded_candidates': len(excluded), 'remaining_candidates': len(plan['remaining_candidates']),
            'new_batches': batch_count(plan)}


def initialize_fresh(args):
    return initialize(args, fresh=True)


def load_plan(args):
    path = args.plan.resolve()
    archive.require(archive.digest(path) == args.plan_sha256, 'Continuation plan digest mismatch')
    plan = archive.read_json(path)
    archive.require(plan.get('schema_version') == 1 and plan.get('purpose') == PURPOSE, 'Unsupported continuation plan')
    root = Path(plan['repository'])
    archive.require(path.parent == Path(plan['output_root']), 'Plan resides outside its pinned output directory')
    archive.require(len(plan['remaining_candidates']) == len(set(plan['remaining_candidates'])) and
                    not set(plan['remaining_candidates']).intersection(plan['excluded_candidates']), 'Invalid remaining candidate partition')
    archive.check_sources(root, {'protected_files': plan['protected_files'], 'members': []})
    for prior in plan['prior_verified_batches']:
        for key in ('manifest', 'upload_record', 'prune_receipt'):
            archive.require(archive.file_receipt(Path(prior[key]['local_path'])) == prior[key], 'Prior verified record changed')
    return root, plan


def existing_batches(root, plan, *, pending_number=None):
    completed = []
    missing_seen = False
    for number in range(1, batch_count(plan) + 1):
        loc = locations(plan, number)
        present = bool(list(Path(plan['output_root']).glob(loc['prefix'] + '_*')) or
                       list(Path(plan['private_output']).glob(loc['prefix'] + '_*')))
        if not present:
            missing_seen = True
            continue
        if number == pending_number and not loc['prune'].exists():
            archive.require(not missing_seen, 'Pending batch follows a missing batch')
            missing_seen = True
            continue
        archive.require(not missing_seen, 'Existing continuation batches are out of order')
        made, index, pruned = workflow.verify_saved_batch(root, loc['manifest'], loc['upload'],
            grid_root=plan['grid_root'], candidates=batch_names(plan, number),
            retained_manifest=plan['retained_manifest']['path'], retained_sha256=plan['retained_manifest']['sha256'],
            check_sources=False)
        completed.append({'number': number, 'made': made, 'index': index, 'pruned': pruned})
    # Receipt checks above stay per batch. Merge their byte checks only after all
    # receipts pass, including records shared with the pinned continuation plan.
    protected = list(plan['protected_files'])
    members = []
    for batch in completed:
        protected.extend(batch['index']['snapshot']['protected_files'])
        members.extend(batch['index']['snapshot']['members'])
    # Check conflicts across both classes before allowing any missing member.
    combined = archive.exact_records(protected + members)
    mandatory_paths = {item['path'] for item in protected}
    archive.check_sources(root, {
        'protected_files': [item for item in combined if item['path'] in mandatory_paths],
        'members': [item for item in combined if item['path'] not in mandatory_paths],
    }, allow_missing_members=True)
    return completed


def cleanup_completed(root, completed):
    retired = [item for batch in completed for item in batch['index']['snapshot']['members']]
    workflow.cleanup_retired(root, retired, [batch['made'] for batch in completed])


def progress(plan, completed, pending=None):
    value = {'complete': len(completed) == batch_count(plan), 'completed_batches': len(completed),
             'completed_candidates': sum(len(batch_names(plan, item['number'])) for item in completed),
             'original_bytes_saved': sum(item['made']['original_bytes'] for item in completed), 'pending_batch': pending}
    workflow.write(Path(plan['output_root']) / 'progress.json', value)
    return value


def create_next(args):
    root, plan = load_plan(args)
    completed = existing_batches(root, plan)
    cleanup_completed(root, completed)
    number = len(completed) + 1
    if number > batch_count(plan):
        return progress(plan, completed)
    loc = locations(plan, number)
    made = archive.create(SimpleNamespace(repository=root, grid_root=plan['grid_root'],
        candidate=batch_names(plan, number), output_root=Path(plan['output_root']), prefix=loc['prefix'],
        retained_manifest=plan['retained_manifest']['path'], retained_manifest_sha256=plan['retained_manifest']['sha256'],
        part_limit_bytes=100 * 1024 * 1024, resident_limit_bytes=512 * 1024 * 1024,
        pre_allocation_cleanup=lambda: cleanup_completed(root, completed)))
    sources = [made['manifest'], *made['parts']]
    archive.write_new(loc['sources'], {'plan_sha256': args.plan_sha256, 'batch': number, 'made': made})
    archive.write_new(loc['request'], {'uploads': [{'local_path': item['local_path'], 'purpose': 'create_library_file',
                                                   'library_artifact_type': 'other'} for item in sources]})
    progress(plan, completed, number)
    return {'phase': 'awaiting_root_upload', 'batch': number, 'candidates': batch_names(plan, number),
            'upload_request': str(loc['request']), 'expected_result': str(loc['result']),
            'expected_stderr': str(loc['stderr']), 'source_receipts': archive.file_receipt(loc['sources'])}


def finish_upload(args):
    root, plan = load_plan(args)
    number = args.batch
    archive.require(1 <= number <= batch_count(plan), 'Invalid continuation batch')
    completed = existing_batches(root, plan, pending_number=number)
    if any(batch['number'] == number for batch in completed):
        return progress(plan, completed)  # Completed batches are safe to report again.
    archive.require(number == len(completed) + 1, 'Finish uploads in campaign order')
    loc = locations(plan, number)
    archive.require(loc['sources'].is_file() and loc['request'].is_file() and loc['result'].is_file(),
                    'Upload input or result is missing; do not retry an unknown outcome')
    archive.require(not loc['enriched'].exists() and not loc['upload'].exists(),
                    'Partial upload verification exists; inspect before manual recovery')
    source_document = archive.read_json(loc['sources'])
    archive.require(source_document['plan_sha256'] == args.plan_sha256 and source_document['batch'] == number,
                    'Upload sources belong to another plan or batch')
    made = source_document['made']
    archive.require(Path(made['manifest']['local_path']) == loc['manifest'], 'Unexpected batch manifest path')
    index = archive.load_index(loc['manifest'], made['manifest']['sha256'], verify_zips=True)
    archive.require(index['snapshot']['candidate_names'] == sorted(batch_names(plan, number)), 'Upload candidate schedule differs')
    sources = [made['manifest'], *made['parts']]
    expected_request = {'uploads': [{'local_path': item['local_path'], 'purpose': 'create_library_file',
                                    'library_artifact_type': 'other'} for item in sources]}
    archive.require(archive.read_json(loc['request']) == expected_request, 'Upload request changed')
    for item in sources:
        archive.require(archive.file_receipt(Path(item['local_path'])) == item, 'Uploaded source hash or size differs')
    response = copy.deepcopy(archive.read_json(loc['result']))
    expected = {item['local_path']: item for item in sources}
    for result in response['results']:
        archive.require(result.get('local_path') in expected, 'Unexpected upload result path')
        result.update(bytes=expected[result['local_path']]['bytes'], sha256=expected[result['local_path']]['sha256'])
    archive.validate_upload_results(response['results'], expected)
    archive.write_new(loc['enriched'], response)
    accepted = archive.record_uploads(SimpleNamespace(manifest=loc['manifest'], manifest_sha256=made['manifest']['sha256'],
        upload_receipt=loc['enriched'], output_receipt=loc['upload']))
    pruned = archive.prune(SimpleNamespace(repository=root, manifest=loc['manifest'], manifest_sha256=made['manifest']['sha256'],
        upload_record=loc['upload'], upload_record_sha256=accepted['verified_upload_record']['sha256'],
        confirm_uploaded_and_authorize_deletion=True))
    completed.append({'number': number, 'made': made, 'index': index, 'pruned': pruned})
    cleanup_completed(root, completed)
    return {'batch_finished': number, **progress(plan, completed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    for phase in ('init', 'init-fresh'):
        init = sub.add_parser(phase)
        init.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
        for name in ('grid-root', 'prefix', 'retained-manifest', 'retained-manifest-sha256'):
            init.add_argument('--' + name, required=True)
        for name in ('output-root', 'private-output'):
            init.add_argument('--' + name, type=Path, required=True)
        if phase == 'init':
            init.add_argument('--prior-prefix', required=True)
            for name in ('prior-output', 'prior-private'):
                init.add_argument('--' + name, type=Path, required=True)
            init.add_argument('--prior-batches', type=int, required=True)
        init.add_argument('--expected-completions', type=int, required=True)
        init.add_argument('--batch-size', type=int, default=3)
    for name in ('create-next', 'finish-upload'):
        command = sub.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--plan-sha256', required=True)
        if name == 'finish-upload':
            command.add_argument('--batch', type=int, required=True)
    args = parser.parse_args()
    result = {'init': initialize, 'init-fresh': initialize_fresh,
              'create-next': create_next, 'finish-upload': finish_upload}[args.phase](args)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()

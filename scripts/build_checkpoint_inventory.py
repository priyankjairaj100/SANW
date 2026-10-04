#!/usr/bin/env python3
"""Build a deterministic public checkpoint inventory from private verified receipts.

The input configuration stays private. Output contains no remote identifiers,
URLs, private paths, or upload outcome payloads. No checkpoint or archive changes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import checkpoint_inventory_receipts as receipts
import continue_checkpoint_archive as continuation
import preserve_retained_checkpoints as backup
import stream_completed_candidate_checkpoints as archive


def pinned(ref):
    path = Path(ref['local_path'])
    archive.require(archive.file_receipt(path) == ref, 'Pinned input receipt changed')
    return path, archive.read_json(path)


def public_file(path):
    return {'name': path.name, 'bytes': path.stat().st_size, 'sha256': archive.digest(path)}


def public_archive(path, index, kind):
    """Whitelist public fields; never copy an upload result or source payload."""
    parts = []
    for row in index['parts']:
        source = row['archive'] if kind == 'retained_checkpoint_copy' else row
        name = Path(source['local_path']).name if kind == 'retained_checkpoint_copy' else source['name']
        parts.append({'name': name, 'bytes': source['bytes'], 'sha256': source['sha256'],
                      'checkpoint_count': len(row['members']),
                      'checkpoint_bytes': sum(item['bytes'] for item in row['members'])})
    return {'kind': kind, 'status': 'saved_verified', 'manifest': public_file(path), 'parts': parts}


def add_coverage(coverage, archives, manifest, index, kind):
    public = public_archive(manifest, index, kind)
    archives.append(public)
    for part, record in zip(index['parts'], public['parts']):
        for item in part['members']:
            archive.require(item['path'] not in coverage, 'Checkpoint covered by multiple preservation entries')
            coverage[item['path']] = {'record': item, 'kind': kind,
                'location': {'manifest_sha256': public['manifest']['sha256'], 'part_name': record['name'],
                             'part_sha256': record['sha256']}}


def classify(expected, required, extras, missing, coverage):
    categories = [set(required), set(extras), set(missing)]
    archive.require(all(not categories[i].intersection(categories[j]) for i in range(3) for j in range(i)),
                    'Required, local-only, and unavailable checkpoint categories overlap')
    archive.require(set().union(*categories) <= set(expected), 'Classified checkpoint is outside the frozen universe')
    archive.require(set(coverage) <= set(expected), 'Saved archive contains an unexpected checkpoint')
    rows = []
    for path, source in sorted(expected.items()):
        row = {'path': path, 'sha256': source['sha256'], 'grid_root': source['grid_root'],
               'roles': sorted(source['roles']), 'state_ids': sorted(source['state_ids'])}
        known = required.get(path) or extras.get(path) or missing.get(path)
        if known:
            archive.require(known['sha256'] == source['sha256'], 'Classified checkpoint digest differs')
            if 'bytes' in known:
                row['bytes'] = known['bytes']
        saved = coverage.get(path)
        if saved:
            archive.require(saved['record']['sha256'] == source['sha256'], 'Archived checkpoint digest differs')
            if 'bytes' in row:
                archive.require(row['bytes'] == saved['record']['bytes'], 'Archived checkpoint size differs')
            row['bytes'] = saved['record']['bytes']
            row['saved_archive'] = saved['location']
        if path in required:
            archive.require(saved is None or saved['kind'] == 'retained_checkpoint_copy', 'Required checkpoint used a pruning archive')
            row['status'] = 'required_retained_saved' if saved else 'required_retained_pending_save'
            row['roles'] = sorted(set(row['roles']) | {'required_retained'})
        elif path in extras:
            archive.require(saved is None, 'Local-only extra unexpectedly has saved archive coverage')
            row['status'] = 'replication_extra_local_only'
        elif path in missing:
            archive.require(saved is None, 'Unavailable checkpoint unexpectedly has saved archive coverage')
            row['status'] = 'discarded_by_fitter_unavailable_not_archived'
        else:
            archive.require(saved is None or saved['kind'] == 'intermediate_epoch_archive', 'Unexpected retained-copy member')
            row['status'] = 'intermediate_archived_verified' if saved else 'intermediate_archive_pending'
        rows.append(row)
    return rows


def checkpoint_universe(root, inventory):
    expected, required, bindings = {}, {}, []
    def add(path, sha, grid, role, sid=None):
        archive.safe_path(root, path, exists=False)
        row = expected.setdefault(path, {'sha256': sha, 'grid_root': grid, 'roles': set(), 'state_ids': set()})
        archive.require(row['sha256'] == sha and row['grid_root'] == grid, 'Conflicting frozen checkpoint mapping')
        row['roles'].add(role)
        if sid:
            row['state_ids'].add(sid)
    for grid in inventory['grids']:
        grid_name = grid['grid_root']
        full_ref = grid['state_manifest']
        full_path = archive.safe_path(root, full_ref['path'])
        archive.require(archive.record(root, full_path, full_ref['sha256']) == full_ref, 'Full state manifest changed')
        full = archive.read_json(full_path)
        states = {state['state_id']: state for state in full['states']}
        archive.require(len(states) == len(full['states']), 'Duplicate frozen state IDs')
        keep_ref = grid['retained_mapping']
        keep_path = archive.safe_path(root, keep_ref['path'])
        archive.require(archive.record(root, keep_path, keep_ref['sha256']) == keep_ref, 'Retained mapping changed')
        keep = archive.read_json(keep_path)
        retained = archive.exact_records(grid['retained_checkpoints'])
        archive.require(len(retained) == grid['checkpoint_count'] and
                        sum(item['bytes'] for item in retained) == grid['checkpoint_bytes'], 'Grid retained totals differ')
        for item in retained:
            archive.require(item['path'] not in required, 'Required checkpoint assigned to multiple grids')
            required[item['path']] = item
        for sid, state in states.items():
            if state.get('checkpoint'):
                add(state['checkpoint'], state['checkpoint_sha256'], grid_name, 'manifest_state', sid)
        for key, role in (('selections', 'development_selected'), ('decomposition_selections', 'decomposition_selected'),
                          ('factorial_selections', 'factorial')):
            for choice in full.get(key, []) + (keep.get(key, []) if key == 'factorial_selections' else []):
                state = states[choice['state_id']]
                if state.get('checkpoint'):
                    archive.require(state['checkpoint'] in required, 'Selected or factorial checkpoint is not retained')
                    add(state['checkpoint'], state['checkpoint_sha256'], grid_name, role, state['state_id'])
        completions = sorted((root / grid_name / 'candidates').rglob('completion.json'))
        archive.require(len(completions) == grid['completed_runs'], 'Completed-run count changed')
        for path in completions:
            complete = archive.read_json(path)
            archive.require(complete['ledger_sha256'] == full['ledger_sha256'], 'Candidate ledger differs')
            hashes = complete.get('checkpoint_sha256', complete['artifact_sha256'])
            for relative, sha in hashes.items():
                if not relative.endswith('.pt'):
                    continue
                name = (path.parent / relative).relative_to(root).as_posix()
                add(name, sha, grid_name, 'candidate_epoch')
                if relative == f"epochs/epoch_{complete['epochs_completed']:02d}.pt":
                    archive.require(name in required, 'Candidate terminal checkpoint is not retained')
                    add(name, sha, grid_name, 'candidate_terminal')
        bindings.append({'grid_root': grid_name, 'completed_runs': len(completions),
                         'state_manifest': full_ref, 'retained_mapping': keep_ref})
    archive.require(len(required) == inventory['retained_checkpoint_count'] and
                    sum(item['bytes'] for item in required.values()) == inventory['retained_checkpoint_bytes'],
                    'Required checkpoint totals differ')
    return expected, required, bindings


def collect_streamed(root, ref, coverage, archives, campaigns, protected):
    plan_path, plan = pinned(ref)
    archive.require(plan['purpose'] == continuation.PURPOSE and plan['repository'] == str(root), 'Archive plan identity differs')
    keep = plan['retained_manifest']
    protected.extend(plan['protected_files'])
    prior_candidates = []
    completed, batches = 0, []
    for old in plan['prior_verified_batches']:
        manifest, _ = pinned(old['manifest'])
        upload, _ = pinned(old['upload_record'])
        pinned(old['prune_receipt'])
        index = receipts.verify_streamed(root, manifest, upload, plan['grid_root'], old['candidates'], keep['path'], keep['sha256'])
        prior_candidates.extend(old['candidates'])
        protected.extend(index['snapshot']['protected_files'])
        add_coverage(coverage, archives, manifest, index, 'intermediate_epoch_archive')
    archive.require(sorted(prior_candidates) == sorted(plan['excluded_candidates']) and
                    set(plan['excluded_candidates']).isdisjoint(plan['remaining_candidates']) and
                    sorted(plan['excluded_candidates'] + plan['remaining_candidates']) == sorted(plan['all_archive_candidates']),
                    'Continuation candidate partition differs')
    for number in range(1, continuation.batch_count(plan) + 1):
        loc = continuation.locations(plan, number)
        status = 'pending'
        if loc['prune'].is_file() and archive.read_json(loc['prune']).get('status') == 'completed':
            index = receipts.verify_streamed(root, loc['manifest'], loc['upload'], plan['grid_root'],
                continuation.batch_names(plan, number), keep['path'], keep['sha256'])
            protected.extend(index['snapshot']['protected_files'])
            add_coverage(coverage, archives, loc['manifest'], index, 'intermediate_epoch_archive')
            status = 'saved_verified'
            completed += 1
        batches.append({'batch': number, 'status': status, 'planned_parts': len(continuation.batch_names(plan, number))})
    campaigns.append({'prefix': plan['prefix'], 'kind': 'intermediate_epoch_archive', 'grid_root': plan['grid_root'],
        'plan_sha256': ref['sha256'], 'prior_verified_batches': len(plan['prior_verified_batches']),
        'completed_batches': completed, 'planned_batches': len(batches), 'batches': batches})


def build(root, config, *, verify_local=True, require_complete=False):
    inventory_path, inventory = pinned(config['retained_inventory'])
    _, replication = pinned(config['replication_status'])
    expected, required, bindings = checkpoint_universe(root, inventory)
    extras, missing = {}, {}
    for grid in replication['grids']:
        manifest = archive.safe_path(root, grid['manifest']['path'])
        archive.require(archive.record(root, manifest, grid['manifest']['sha256']) == grid['manifest'], 'Replication manifest changed')
        extras.update({item['path']: item for item in grid['present_extra_records']})
        missing.update({item['path']: item for item in grid['missing_unretained_records']})
    coverage, archives, campaigns, protected = {}, [], [], []
    for legacy in config['legacy_archives']:
        manifest, _ = pinned(legacy['manifest'])
        upload, _ = pinned(legacy['upload_receipt'])
        prune, _ = pinned(legacy['prune_receipt'])
        index = receipts.verify_legacy(root, manifest, upload, prune)
        protected.extend(index['snapshot']['protected_files'])
        add_coverage(coverage, archives, manifest, index, 'intermediate_epoch_archive')
        campaigns.append({'prefix': index['archive_prefix'], 'kind': 'intermediate_epoch_archive',
                          'grid_root': index['snapshot']['grid_root'], 'status': 'saved_verified',
                          'completed_parts': len(index['parts']), 'planned_parts': len(index['parts'])})
    for ref in config['continuation_plans']:
        collect_streamed(root, ref, coverage, archives, campaigns, protected)
    _, plan = pinned(config['retained_plan'])
    archive.require(plan['purpose'] == backup.PURPOSE and plan['repository'] == str(root) and
                    plan['inventory']['sha256'] == config['retained_inventory']['sha256'], 'Retained plan identity differs')
    all_planned = archive.exact_records([item for part in plan['parts'] for item in part])
    archive.require(all_planned == archive.exact_records(list(required.values())), 'Retained copy plan does not cover required closure')
    protected.extend(plan['protected_files'])
    batches = []
    for number, parts in enumerate(plan['batches'], 1):
        loc = backup.locations(plan, number)
        status = 'pending'
        if loc['verified_upload'].is_file():
            index = receipts.verify_retained(plan, config['retained_plan']['sha256'], number)
            add_coverage(coverage, archives, loc['manifest'], index, 'retained_checkpoint_copy')
            status = 'saved_verified'
        batches.append({'batch': number, 'status': status, 'planned_parts': len(parts)})
    campaigns.append({'prefix': plan['prefix'], 'kind': 'retained_checkpoint_copy',
        'plan_sha256': config['retained_plan']['sha256'], 'planned_parts': len(plan['parts']),
        'planned_batches': len(batches), 'completed_batches': sum(row['status'] == 'saved_verified' for row in batches),
        'batches': batches})
    rows = classify(expected, required, extras, missing, coverage)
    counts = dict(sorted(Counter(row['status'] for row in rows).items()))
    universe_counts = {'checkpoint_paths': len(rows), 'required_retained': len(required),
        'replication_extra_local_only': len(extras), 'discarded_unavailable': len(missing),
        'archive_eligible': len(rows) - len(required) - len(extras) - len(missing),
        'completed_runs': sum(grid['completed_runs'] for grid in inventory['grids'])}
    archive.require(universe_counts == config['expected_counts'], 'Checkpoint universe differs from reviewed scope')
    complete = not (counts.get('required_retained_pending_save', 0) or counts.get('intermediate_archive_pending', 0))
    archive.require(not require_complete or complete, 'Preservation still has pending checkpoint saves')
    if verify_local:
        archive.check_sources(root, {'protected_files': archive.exact_records(protected + list(required.values()) + list(extras.values())),
                                     'members': []})
        archive.require(all(not archive.safe_path(root, path, exists=False).exists() for path in missing),
                        'A previously unavailable checkpoint reappeared; update its classification')
    result = {'schema_version': 1, 'purpose': 'research_checkpoint_preservation_inventory',
        'status': 'complete' if complete else 'pending', 'checkpoint_universe': universe_counts,
        'status_counts': counts, 'required_checkpoint_closure_saved': not counts.get('required_retained_pending_save', 0),
        'all_epoch_checkpoint_bytes_recovered': False, 'remote_availability_rechecked': False,
        'local_required_and_extra_sources_verified': verify_local,
        'verification_basis': 'Exact source hashes, archive membership, finalized upload receipts, and linked completed prune receipts where applicable.',
        'scope_note': 'Required retained closure covers reported evaluation, selection, audit, and terminal-fit checkpoints. Extra replication weights remain local-only. Discarded intermediate weights retain hashes but are not archived.',
        'grid_bindings': sorted(bindings, key=lambda row: row['grid_root']),
        'campaigns': sorted(campaigns, key=lambda row: row['prefix']),
        'saved_archives': sorted(archives, key=lambda row: row['manifest']['name']), 'checkpoints': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    archive.require(archive.digest(args.config) == args.config_sha256, 'Private configuration digest differs')
    value = build(args.repository.resolve(), archive.read_json(args.config), require_complete=args.require_complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        old = archive.read_json(args.output)
        archive.require(old.get('status') != 'complete' or value['status'] == 'complete', 'Do not replace complete inventory with pending state')
    temporary = args.output.with_suffix(args.output.suffix + '.new')
    archive.write_new(temporary, value)
    temporary.replace(args.output)
    print(json.dumps({'inventory': archive.file_receipt(args.output), 'status': value['status'],
                      'counts': value['status_counts']}, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()

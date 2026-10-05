#!/usr/bin/env python3
"""Preserve one completed V10 joint/control fit using metadata and byte checks only.

This unbound preservation utility never loads NumPy arrays, recomputes scores,
changes selection, transfers Library files, or modifies fitting sources.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import zipfile

PROTOCOL = 'results/practical_v10/expanded_training_protocol_v1.json'
PROTOCOL_SHA = 'ce04032395f185519f3f5012a82d403ff41a7e99b8662e998c99028cf719a826'
INDEX_NAME = 'COMPLETED_FIT_ARCHIVE_INDEX.json'
UTILITY_MEMBER = 'preservation/package_completed_fit_v2.py'
MAX_SMALL_FILE = 64 * 1024 * 1024


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained(root, value):
    root = Path(root).resolve(); path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise ValueError('Symlink is not an immutable preservation source: ' + str(value))
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError('Expected regular repository file: ' + str(value))
    return path


def verify_entry(root, entry):
    path = contained(root, entry['path'])
    if ('bytes' in entry and path.stat().st_size != entry['bytes']) or digest(path) != entry['sha256']:
        raise ValueError('Bound file identity changed: ' + str(path))
    return path


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def choose_epoch(history, family):
    """Reconstruct the inherited rule without inspecting checkpoint arrays."""
    if family not in ('joint', 'no_retention'):
        raise ValueError('Unsupported fit family')
    candidates = []
    for row in history:
        if not isinstance(row['nonzero'], bool) or not math.isfinite(row['training_objective']):
            raise ValueError('Nonfinite objective or invalid nonzero flag')
        if family == 'joint':
            certificate = row.get('certificate', {})
            if any(certificate.get(key) is not True for key in
                   ('feasible_with_tolerance', 'ranking_checked_canonically', 'ranking_preserved')):
                raise ValueError('A joint epoch lacks its recorded feasible canonical certificate')
        elif row.get('retention_enforced') is not False:
            raise ValueError('Matched no-retention history imposed retention')
        if row['nonzero']:
            candidates.append(row)
    if not candidates:
        raise ValueError('No nonzero epoch is selectable')
    return min(candidates, key=lambda row: (row['training_objective'], row['epoch']))


def npz_payload_hashes(path):
    """Read ZIP member bytes, not NumPy values; ignore irrelevant ZIP timestamps."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        required = {'schema.npy', 'image_mean.npy', 'text_mean.npy', 'image_basis.npy', 'text_basis.npy', 'coefficient.npy'}
        if set(names) != required or len(names) != len(required):
            raise ValueError('Unexpected selected-model archive members')
        result = {}
        for member in archive.infolist():
            if member.file_size > MAX_SMALL_FILE:
                raise ValueError('Model member unexpectedly large')
            with archive.open(member) as handle:
                result[member.filename] = hashlib.file_digest(handle, 'sha256').hexdigest()
        return result


def gate_files(root, entry, protocol_sha):
    """Verify existing gate identity only; do not recompute heldout intervals."""
    gate_path = verify_entry(root, entry); gate = read(gate_path); files = {gate_path}
    if (gate.get('study') != 'sanw_practical_v10_replication_gate' or gate.get('family') != 'joint'
            or gate.get('passed') is not True or gate.get('protocol_sha256') != protocol_sha
            or sorted(gate.get('encoders_passed', [])) != ['rn50', 'vit_b32'] or gate.get('seed') != 17):
        raise ValueError('Replica/control gate scope differs')
    pilots = gate.get('pilot_runs', [])
    if sorted(row.get('encoder', '') for row in pilots) != ['rn50', 'vit_b32']:
        raise ValueError('Gate must bind both seed17 pilots')
    for pilot in pilots:
        paths = {name: verify_entry(root, pilot[name]) for name in ('completion', 'checkpoint', 'result')}
        files.update(paths.values())
        completion, result = read(paths['completion']), read(paths['result'])
        if (completion.get('study') != 'sanw_practical_v10' or completion.get('encoder') != pilot['encoder']
                or completion.get('protocol_sha256') != protocol_sha or completion.get('config', {}).get('seed') != 17
                or completion.get('selected_checkpoint', {}).get('sha256') != pilot['checkpoint']['sha256']
                or result.get('study') != 'sanw_practical_v10_official_development' or result.get('passed') is not True
                or result.get('encoder') != pilot['encoder'] or result.get('seed') != 17
                or result.get('protocol_sha256') != protocol_sha
                or result.get('checkpoint_sha256') != pilot['checkpoint']['sha256']):
            raise ValueError('Gate pilot/result identity differs')
    # Keep its small source/state lock and source-audit receipt for future review.
    if isinstance(gate.get('lock'), dict):
        lock_path = verify_entry(root, gate['lock']); files.add(lock_path); lock = read(lock_path)
        for name, expected in lock.get('source_sha256', {}).items():
            files.add(verify_entry(root, {'path': name, 'sha256': expected}))
        for name in ('source_audit', 'protocol'):
            if isinstance(lock.get(name), dict):
                files.add(verify_entry(root, lock[name]))
    return files


def inspect_run(root, run):
    root = Path(root).resolve(); run = Path(run)
    if not run.is_absolute():
        run = root / run
    run = run.resolve()
    if not run.is_relative_to(root) or not run.is_dir():
        raise ValueError('Run must be a repository directory')
    completion, ledger, history = (read(run / name) for name in ('completion.json', 'ledger.json', 'history.json'))
    identity = ledger['identity']
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if ledger_sha != ledger['ledger_sha256'] or completion['ledger_sha256'] != ledger_sha:
        raise ValueError('Ledger identity mismatch')
    protocol_path = contained(root, PROTOCOL)
    if digest(protocol_path) != PROTOCOL_SHA:
        raise ValueError('Immutable v10 protocol changed')
    protocol = read(protocol_path)
    encoder, family, seed = identity.get('encoder'), identity.get('family'), identity.get('config', {}).get('seed')
    if encoder not in ('vit_b32', 'rn50') or family not in ('joint', 'no_retention') or seed not in (17, 29, 43):
        raise ValueError('Run is not a prescribed core joint/control state')
    config = {**protocol['fit_config'], 'seed': seed}
    if (config['epochs'] != 32 or config['batch_size'] != 64
            or any(item.get('study') != 'sanw_practical_v10' or item.get('mode') != 'full'
                   or item.get('encoder') != encoder or item.get('family') != family
                   or item.get('config') != config or item.get('protocol_sha256') != PROTOCOL_SHA
                   for item in (identity, completion))
            or identity.get('source_sha256') != protocol['source_sha256']
            or verify_entry(root, identity['protocol']) != protocol_path
            or identity.get('streaming') != protocol['streaming']
            or identity.get('fit_gallery_image_count') != 6000 or identity.get('fit_gallery_text_count') != 30000
            or identity.get('official_development_or_benchmarks_used') is not False
            or completion.get('development_or_test_used') is not False):
        raise ValueError('Immutable run configuration/scope changed')
    provenance = identity['training_provenance']
    if (provenance.get('inputs') != protocol['training_inputs'][encoder]
            or provenance.get('original_training_inputs') != protocol['original_training_inputs'][encoder]
            or provenance.get('owner_sample') != protocol['owner_sample']
            or provenance.get('confirmation_owner_lock') != protocol['confirmation_owner_lock']
            or provenance.get('training_image_manifest_indices') != list(range(6000))
            or provenance.get('heldout_used') is not False
            or provenance.get('original_training_feature_bytes_identical') is not True):
        raise ValueError('Training input provenance changed')
    files = {protocol_path}
    for name, expected in protocol['source_sha256'].items():
        files.add(verify_entry(root, {'path': name, 'sha256': expected}))
    if (history != completion.get('checkpoint_history') or len(history) != 32
            or [row['epoch'] for row in history] != list(range(1, 33))
            or any(row['optimizer_steps'] != 94 * row['epoch'] for row in history)
            or completion.get('optimizer_steps') != 3008
            or [{key: value for key, value in row.items() if key != 'checkpoint'} for row in history] != completion.get('history')):
        raise ValueError('Complete 32epoch/3008update history is required')
    chosen = choose_epoch(history, family)
    selection = ('minimum_feasible_nonzero_training_objective_then_earliest_epoch' if family == 'joint' else
                 'minimum_nonzero_training_objective_then_earliest_epoch_without_feasibility_filter')
    if (completion.get('selection') != selection or completion.get('selected_epoch') != chosen['epoch']
            or completion.get('selected_training_objective') != chosen['training_objective']):
        raise ValueError('Selected epoch differs from the inherited earliest-minimum rule')
    if family == 'joint':
        if any(completion.get('final_certificate', {}).get(key) is not True for key in
               ('feasible_with_tolerance', 'ranking_checked_canonically', 'ranking_preserved')):
            raise ValueError('Selected joint state lacks its final canonical certificate')
    elif (completion.get('retention_enforced') is not False
          or completion.get('final_training_retention_diagnostic', {}).get('ranking_checked_canonically') is not True):
        raise ValueError('No-retention control lacks its recorded final diagnostic')
    for row in history:
        verify_entry(run, row['checkpoint'])
        if row['checkpoint'].get('ledger_sha256') != ledger_sha:
            raise ValueError('Checkpoint ledger binding differs')
    selected = verify_entry(run, completion['selected_checkpoint'])
    selected_epoch = verify_entry(run, chosen['checkpoint'])
    if npz_payload_hashes(selected) != npz_payload_hashes(selected_epoch):
        raise ValueError('Selected NPZ payload differs from its chosen epoch')
    gate_name = 'control_gate' if family == 'no_retention' else 'replication_gate'
    gate_required = family == 'no_retention' or seed != 17
    if bool(identity.get(gate_name)) != gate_required or identity.get('replication_gate' if gate_name == 'control_gate' else 'control_gate') is not None:
        raise ValueError('Run gate provenance differs from family/seed rule')
    if gate_required:
        files.update(gate_files(root, identity[gate_name], PROTOCOL_SHA))
    cache = identity['frozen_score_cache']
    cache_metadata = contained(root, Path(cache['path']) / 'metadata.json')
    if digest(cache_metadata) != cache['metadata_sha256']:
        raise ValueError('Bound cache metadata changed')
    meta = read(cache_metadata)
    if meta.get('schema') != 'sanw_frozen_score_cache_v10' or meta.get('images') != 6000 or meta.get('texts') != 30000:
        raise ValueError('Frozen-score cache metadata scope differs')
    files.add(cache_metadata)
    for input_set in (protocol['training_inputs'][encoder], protocol['original_training_inputs'][encoder]):
        files.add(verify_entry(root, input_set['metadata']))
    run_files = [path for path in run.rglob('*') if path.is_file()]
    for path in run_files:
        if path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError('Symlink/escaped run file is not archived')
    files.update(run_files)
    for path in files:
        if path.stat().st_size > MAX_SMALL_FILE or path.name in ('image_to_text.npy', 'text_to_image.npy', 'features.npz', 'resumable_bank.sqlite'):
            raise ValueError('Unexpected large cache/input payload in preservation scope: ' + str(path))
    summary = {'study': 'sanw_practical_v10_completed_fit_preservation_v2', 'encoder': encoder,
        'family': family, 'seed': seed, 'selected_epoch': chosen['epoch'], 'selected_training_objective': chosen['training_objective'],
        'selection': selection, 'epochs': 32, 'optimizer_steps': 3008, 'protocol_sha256': PROTOCOL_SHA,
        'ledger_sha256': ledger_sha, 'run': str(run.relative_to(root)), 'historical_repository_root': str(root),
        'run_file_count': len(run_files), 'checkpoint_payload_selection_identity_verified': True,
        'gate_required': gate_required, 'gate_metadata_verified': gate_required,
        'numerical_audits_performed': False, 'source_or_state_changes': False,
        'large_cache_arrays_read_or_included': False,
        'scope': 'Byte and recorded-metadata preservation checks only; not an independent numerical fit audit or heldout success claim.',
        'separately_preserved_dependencies': 'Training feature arrays/manifests and input provenance remain bound by protocol and separately preserved feature/prefit archives.',
        'excluded_frozen_score_array_records': meta['files']}
    return run, sorted(files), summary


def package(root, run, output):
    root = Path(root).resolve(); output = Path(output).resolve()
    if output.exists() or not str(output).endswith('.tar.xz'):
        raise ValueError('A new .tar.xz output path is required')
    manifest = output.with_name(output.name + '.manifest.json')
    receipt = output.with_name(output.name + '.summary.json')
    if manifest.exists() or receipt.exists():
        raise FileExistsError('Preservation sidecars already exist')
    run, files, summary = inspect_run(root, run)
    if output.is_relative_to(run):
        raise ValueError('Archive output must not be inside its run')
    payloads = [(str(path.relative_to(root)), path) for path in files]
    payloads.append((UTILITY_MEMBER, Path(__file__).resolve()))
    if len({name for name, _ in payloads}) != len(payloads):
        raise ValueError('Duplicate logical archive member')
    inventory = [{'path': name, 'sha256': digest(path), 'bytes': path.stat().st_size} for name, path in sorted(payloads)]
    index = {**summary, 'files': inventory}
    index_bytes = json_bytes(index)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.tmp')
    if temp.exists():
        raise FileExistsError('Incomplete archive exists; inspect it before retrying')
    # Exclusive creation prevents a competing invocation from truncating bytes.
    with temp.open('xb') as target, tarfile.open(fileobj=target, mode='w:xz', preset=3) as archive:
        for name, path in sorted(payloads):
            info = tarfile.TarInfo(name); info.size = path.stat().st_size; info.mode = 0o644
            with path.open('rb') as handle:
                archive.addfile(info, handle)
        info = tarfile.TarInfo(INDEX_NAME); info.size = len(index_bytes); info.mode = 0o644
        archive.addfile(info, io.BytesIO(index_bytes))
    expected = {row['path']: row for row in inventory}
    expected[INDEX_NAME] = {'bytes': len(index_bytes), 'sha256': hashlib.sha256(index_bytes).hexdigest()}
    with tarfile.open(temp, 'r:xz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError('Tar member inventory differs')
        for member in members:
            row = expected[member.name]
            if not member.isfile() or member.size != row['bytes']:
                raise ValueError('Tar member type/size mismatch')
            with archive.extractfile(member) as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('Tar member hash mismatch: ' + member.name)
    # Detect any mutation while source bytes were being copied or verified.
    for name, path in payloads:
        if digest(path) != expected[name]['sha256'] or path.stat().st_size != expected[name]['bytes']:
            raise ValueError('Source changed while packaging: ' + name)
    if {str(path.relative_to(root)) for path in run.rglob('*') if path.is_file()} != {row['path'] for row in inventory if row['path'].startswith(str(run.relative_to(root)) + '/')}:
        raise ValueError('Run file population changed during preservation')
    # link() is an atomic no-clobber publication on this same filesystem.
    # Keep a failed temporary archive available for inspection; never overwrite.
    os.link(temp, output)
    temp.unlink()
    with manifest.open('xb') as handle:
        handle.write(index_bytes)
    result = {**summary, 'archive': str(output), 'archive_bytes': output.stat().st_size, 'archive_sha256': digest(output),
              'verified_tar_members': len(expected), 'manifest': str(manifest), 'manifest_sha256': digest(manifest),
              'utility_sha256': digest(Path(__file__)), 'library_transfers_performed': False}
    with receipt.open('xb') as handle:
        handle.write(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1] / 'SANW_practical_work')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.repository, args.run, args.output)), flush=True)


if __name__ == '__main__':
    main()

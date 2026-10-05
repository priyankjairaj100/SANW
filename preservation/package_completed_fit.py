"""Archive one completed immutable V10 fit; never touch model selection."""
import argparse
import hashlib
import io
import json
import pathlib
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repository', type=pathlib.Path, required=True)
    p.add_argument('--run', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    root = args.repository.resolve()
    run = (root / args.run).resolve()
    run.relative_to(root)
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing archive')
    completion = json.loads((run / 'completion.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    ledger = json.loads((run / 'ledger.json').read_text())
    if completion['checkpoint_history'] != history:
        raise ValueError('Completion/history mismatch')
    identity = ledger['identity']
    ledger_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if ledger_hash != ledger['ledger_sha256'] or ledger_hash != completion['ledger_sha256']:
        raise ValueError('Ledger identity mismatch')
    protocol = root / 'results/practical_v10/expanded_training_protocol_v1.json'
    if digest(protocol) != completion['protocol_sha256']:
        raise ValueError('Protocol mismatch')
    specification = json.loads(protocol.read_text())
    source_paths = []
    for name, expected in specification['source_sha256'].items():
        source = root / name
        if digest(source) != expected:
            raise ValueError('Changed source: ' + name)
        source_paths.append(source)
    if len(history) != 32 or [r['epoch'] for r in history] != list(range(1, 33)):
        raise ValueError('Incomplete epoch history')
    if history[-1]['optimizer_steps'] != 3008:
        raise ValueError('Incomplete optimizer budget')
    for row in history:
        checkpoint = row['checkpoint']
        if digest(run / checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Checkpoint mismatch')
        if checkpoint['ledger_sha256'] != ledger_hash:
            raise ValueError('Checkpoint ledger mismatch')
    selected = completion['selected_checkpoint']
    if digest(run / selected['path']) != selected['sha256']:
        raise ValueError('Selected state mismatch')
    members = sorted(set([protocol, *source_paths, *[q for q in run.rglob('*') if q.is_file()]]))
    inventory = [{'path': str(q.relative_to(root)), 'sha256': digest(q), 'bytes': q.stat().st_size} for q in members]
    note = {
        'study': 'sanw_practical_v10_completed_fit_preservation',
        'encoder': completion['encoder'], 'family': completion['family'],
        'selected_epoch': completion['selected_epoch'],
        'protocol_sha256': completion['protocol_sha256'],
        'scope': 'Completed training evidence only; no held-out success claim.',
        'dependencies': 'Feature banks and input evidence are separately preserved in TRAIN6000 feature archives and V10 PREFIT checkpoint. Frozen-score matrices are reproducible intermediates and excluded.',
        'historical_repository_root': str(root), 'files': inventory,
    }
    with tarfile.open(args.output, 'w:xz', preset=3) as archive:
        for q in members:
            archive.add(q, arcname=str(q.relative_to(root)), recursive=False)
        body = (json.dumps(note, indent=2, sort_keys=True) + '\n').encode()
        info = tarfile.TarInfo('COMPLETED_FIT_ARCHIVE_INDEX.json')
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
        archive.add(pathlib.Path(__file__), arcname='preservation/package_completed_fit.py')
    with tarfile.open(args.output, 'r:xz') as archive:
        for row in inventory:
            if hashlib.sha256(archive.extractfile(row['path']).read()).hexdigest() != row['sha256']:
                raise ValueError('Archive verification failed: ' + row['path'])
    print(json.dumps({'archive': str(args.output.resolve()), 'sha256': digest(args.output), 'bytes': args.output.stat().st_size, 'verified_files': len(inventory)}))


if __name__ == '__main__':
    main()

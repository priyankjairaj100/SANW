#!/usr/bin/env python3
"""Package the complete project, omitting only documented runtime duplicates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from inventory_recovery import inventory
from verify_recovery import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    temporary = output.with_suffix('.partial.zip')
    if output.exists() or temporary.exists() or output.is_relative_to(ROOT):
        raise SystemExit('Use a new output path outside the project directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory_path = ROOT / 'results/archive_inventory.json'
    state = inventory(ROOT, inventory_path)
    paths = sorted({row['path'] for row in state['eligible_files']} | {'results/archive_inventory.json'})
    manifest = {'schema_version': 1,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'execution_id': 'relation-labels-reconstruction-2026-10-03',
                'scope': 'Complete reconstructed project: sources, raw data, pinned model, all feature caches, candidate and selected states, predictions, analysis, manuscript, and PDFs. Exclusions are documented runtime caches and incomplete downloads.',
                'files': []}
    fingerprints = {}
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for index, relative in enumerate(paths):
            path = ROOT / relative
            before = path.stat()
            digest = hashlib.sha256()
            size = 0
            info = zipfile.ZipInfo('project/' + relative, (2026, 10, 3, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.npz', '.zip', '.gz'} else zipfile.ZIP_DEFLATED
            info._compresslevel = 1
            with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
                raise RuntimeError(f'Project changed during packaging: {relative}')
            fingerprints[path] = (after.st_size, after.st_mtime_ns)
            manifest['files'].append({'path': info.filename, 'bytes': size, 'sha256': digest.hexdigest()})
            if index % 1000 == 0:
                print(json.dumps({'packaged_files': index + 1, 'total_files': len(paths)}), flush=True)
        for local, name in [(ROOT / 'scripts/verify_recovery.py', 'verify_recovery.py'),
                            (ROOT / 'docs/RECOVERY_README.md', 'RECOVERY_README.md')]:
            if local.exists():
                data = local.read_bytes()
                archive.writestr(name, data)
                manifest['files'].append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        archive.writestr('RECOVERY_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    print('Archive closed; verifying all member hashes and CRCs', flush=True)
    report, status = verify(archive_path=temporary)
    if status:
        raise RuntimeError(json.dumps(report))
    for path, expected in fingerprints.items():
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != expected:
            raise RuntimeError(f'Project changed before archive publication: {path.relative_to(ROOT)}')
    temporary.replace(output)
    digest = hashlib.sha256()
    with output.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    report.update({'path': str(output), 'sha256': digest.hexdigest()})
    output.with_suffix('.receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

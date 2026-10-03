#!/usr/bin/env python3
"""Make a verified compact paper/code/results companion to the complete archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from inventory_recovery import exclusion_reason
from verify_recovery import verify

ROOT = Path(__file__).resolve().parents[1]
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.log'}


def keep(relative: Path) -> bool:
    if exclusion_reason(relative, bank_redundant=True):
        return False
    parts = relative.parts
    if len(parts) == 1:
        return relative.suffix in {'.md', '.txt', '.toml', '.py'}
    if parts[0] in {'src', 'scripts', 'tests', 'configs', 'docs', 'historical_context', 'manuscript', 'output'}:
        return True
    if parts[0] == 'data':
        return relative.suffix in TEXT and 'coco_images' not in parts and 'images' not in parts
    if parts[0] == 'results':
        if relative.suffix in TEXT:
            return True
        if relative.suffix == '.npz':
            return parts[1] != 'features' and relative.name != 'coco_karpathy.npz'
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complete-archive-name', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    temporary = output.with_suffix('.partial.zip')
    if output.is_relative_to(ROOT) or output.exists() or temporary.exists():
        raise SystemExit('Use a new archive path outside the project')
    output.parent.mkdir(parents=True, exist_ok=True)
    scope = (
        'This compact companion contains the paper, flat Overleaf source ZIP, all '
        'scientific code, reproduction instructions, input manifests and annotation '
        'records, reported aggregates and audits, candidate training histories, raw '
        'discrimination predictions, and the complete relation diagnostic. It excludes '
        'raw image bytes, pretrained and adapter weights, frozen feature matrices, and '
        'full COCO item-level prediction arrays. Those artifacts are retained in '
        + args.complete_archive_name + '. Historical records are separate from fresh evidence.\n'
    )
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': scope, 'files': []}
    fingerprints = {}
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or not path.resolve().is_relative_to(ROOT):
                continue
            relative = path.relative_to(ROOT)
            if not keep(relative):
                continue
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'File changed during packaging: {relative}')
            fingerprints[path] = (after.st_size, after.st_mtime_ns)
            name = 'project/' + relative.as_posix()
            archive.writestr(name, content)
            manifest['files'].append({'path': name, 'bytes': len(content),
                                      'sha256': hashlib.sha256(content).hexdigest()})
        for name, content in [('PACKAGE_SCOPE.txt', scope.encode()),
                              ('verify_recovery.py', (ROOT / 'scripts/verify_recovery.py').read_bytes())]:
            archive.writestr(name, content)
            manifest['files'].append({'path': name, 'bytes': len(content),
                                      'sha256': hashlib.sha256(content).hexdigest()})
        archive.writestr('RECOVERY_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    report, status = verify(archive_path=temporary)
    if status:
        raise RuntimeError(json.dumps(report))
    for path, expected in fingerprints.items():
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != expected:
            raise RuntimeError(f'Project changed before publication: {path}')
    temporary.replace(output)
    report.update({'path': str(output), 'bytes': output.stat().st_size,
                   'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    output.with_suffix('.receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

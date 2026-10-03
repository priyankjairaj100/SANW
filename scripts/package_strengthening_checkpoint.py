#!/usr/bin/env python3
"""Make a small, independently hash-verifiable incremental project checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a delivered checkpoint.')
    paths = set()
    for pattern in args.paths:
        matches = list(ROOT.glob(pattern))
        if not matches:
            raise SystemExit(f'No matches: {pattern}')
        for path in matches:
            candidates = path.rglob('*') if path.is_dir() else [path]
            paths.update(p for p in candidates if p.is_file() and '__pycache__' not in p.parts)
    rows = []
    for path in sorted(paths):
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise SystemExit(f'Unsafe member: {path}')
        content = path.read_bytes()
        rows.append({'path': str(path.relative_to(ROOT)), 'bytes': len(content), 'sha256': sha(content)})
    manifest = {'stage': args.stage, 'kind': 'incremental_project_checkpoint',
                'repository': 'https://github.com/priyankjairaj100/SANW', 'files': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        for row in rows:
            archive.write(ROOT / row['path'], 'project/' + row['path'])
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for row in rows:
            content = archive.read('project/' + row['path'])
            assert len(content) == row['bytes'] and sha(content) == row['sha256']
    receipt = {'stage': args.stage, 'archive': str(args.output.resolve()),
               'bytes': args.output.stat().st_size, 'sha256': sha(args.output.read_bytes()),
               'file_count': len(rows), 'status': 'member_hashes_and_crc_verified'}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

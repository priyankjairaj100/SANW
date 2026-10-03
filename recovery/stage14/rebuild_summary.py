#!/usr/bin/env python3
"""Verify stage14's Git-backed files, or recreate its scoped recovery ZIP."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify saved payload without writing a ZIP.')
    parser.add_argument('--output', type=Path, help='New ZIP destination; existing paths are never overwritten.')
    args = parser.parse_args()
    manifest_bytes = (HERE / 'RECOVERY_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads((HERE / 'receipt.json').read_text())
    payload = {'RECOVERY_MANIFEST.json': manifest_bytes}
    for entry in manifest['files']:
        name = entry['path']
        parts = Path(name).parts
        if Path(name).is_absolute() or '..' in parts:
            raise ValueError(f'Unsafe manifest path: {name}')
        if name.startswith('project/'):
            source = ROOT / name.removeprefix('project/')
        elif name == 'PACKAGE_SCOPE.txt':
            source = HERE / name
        elif name == 'verify_recovery.py':
            source = ROOT / 'scripts/verify_recovery.py'
        else:
            raise ValueError(f'Unknown support path: {name}')
        content = source.read_bytes()
        if len(content) != entry['bytes'] or digest(content) != entry['sha256']:
            raise ValueError(f'Stage14 payload changed or unavailable: {source}')
        payload[name] = content
    result = {'status': 'verified', 'payload_files': len(manifest['files']),
              'project_files': manifest['source_file_count'], 'protocol_sha256': manifest['protocol_sha256']}
    if not args.verify_only:
        destination = args.output or ROOT / 'output/recovery' / receipt['archive']
        if destination.exists():
            raise FileExistsError(f'Existing destination preserved: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        config = receipt['rebuild']
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=config['compression_level']) as archive:
            for name, content in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=tuple(config['entry_timestamp']))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = config['create_system']
                info.external_attr = config['entry_external_attr']
                archive.writestr(info, content, compresslevel=config['compression_level'])
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Recreated ZIP failed CRC verification')
            for name, content in payload.items():
                if archive.read(name) != content:
                    raise RuntimeError(f'Recreated ZIP payload differs: {name}')
        actual_hash = digest(destination.read_bytes())
        result.update({'output': str(destination), 'bytes': destination.stat().st_size,
                       'sha256': actual_hash, 'matches_original_zip_bytes': actual_hash == receipt['sha256'],
                       'all_payload_bytes_verified': True})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Restore exact assets without evaluating features or rewriting frozen receipts.

Every write requires --execute. Benchmark arrays stay inside exact archives.
The frozen preparation command restores them. Scoring requires the frozen release.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 1024 * 1024


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def safe_path(root, relative):
    """Refuse absolute paths, traversal, and symlink destinations."""
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or '..' in part.parts or '\\' in relative:
        raise ValueError('Unsafe relative path: ' + relative)
    root = root.resolve()
    result = root.joinpath(*part.parts)
    for ancestor in (result, *result.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError('Symlink destination refused: ' + relative)
    if not result.resolve().is_relative_to(root):
        raise ValueError('Path escapes repository: ' + relative)
    return result


def matches(path, record):
    return (path.is_file() and path.stat().st_size == record['bytes']
            and sha256(path) == record['sha256'])


def require_match(path, record):
    if not matches(path, record):
        raise ValueError('Missing or changed asset: ' + str(path))


@contextmanager
def atomic_destination(path):
    """Publish with an atomic no-overwrite link; retain existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    partial = Path(name)
    try:
        with os.fdopen(fd, 'wb') as out:
            yield out, partial
            out.flush()
            os.fsync(out.fileno())
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


class Assets:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'local/assets_manifest.json').read_text())
        if self.manifest['schema'] != 'sanw-local-assets-v1':
            raise ValueError('Unknown asset manifest schema')

    def archives(self, group):
        if group == 'all':
            return self.manifest['archives']
        if group == 'benchmark':
            return [a for a in self.manifest['archives']
                    if a['name'].startswith('SANW_SCIENTIFIC_INPUTS_')]
        return [a for a in self.manifest['archives']
                if any(m['restore_group'] == group for m in a['members'])]

    def archive_path(self, record):
        return safe_path(self.root, record.get('assembled_path', record.get('path')))

    def verify(self, group='all'):
        total = 0
        for record in self.archives(group):
            for chunk in record.get('chunks', []):
                require_match(safe_path(self.root, chunk['path']), chunk)
                total += chunk['bytes']
            if 'chunks' not in record:
                require_match(self.archive_path(record), record)
                total += record['bytes']
            path = self.archive_path(record)
            if path.exists():
                require_match(path, record)
        return {'verified': True, 'group': group, 'transport_bytes': total,
                'archives': len(self.archives(group))}

    def assemble(self, group='all', execute=False):
        records = self.archives(group)
        actions = []
        # Validate every transport file and all existing destinations first.
        self.verify(group)
        for record in records:
            path = self.archive_path(record)
            if 'chunks' not in record or path.exists():
                continue
            actions.append({'action': 'assemble', 'path': str(path.relative_to(self.root)),
                            'bytes': record['bytes']})
        if execute:
            for record in records:
                path = self.archive_path(record)
                if 'chunks' not in record or path.exists():
                    continue
                with atomic_destination(path) as (out, partial):
                    for chunk in record['chunks']:
                        with safe_path(self.root, chunk['path']).open('rb') as inp:
                            while data := inp.read(BLOCK):
                                out.write(data)
                    out.flush()
                    require_match(partial, record)
        return {'execute': execute, 'actions': actions}

    def restore(self, group='train-dev', execute=False):
        if group not in ('train-dev', 'fits', 'all'):
            raise ValueError('Use the frozen preparation command for benchmark extraction')
        accepted = {'train-dev', 'fits'} if group == 'all' else {group}
        records = self.archives(group)
        targets = []
        seen = {}
        # Refuse conflicting destinations before writing any files.
        for record in records:
            for member in record['members']:
                if member['restore_group'] not in accepted:
                    continue
                path = safe_path(self.root, member['path'])
                key = str(path)
                identity = (member['bytes'], member['sha256'])
                if key in seen and seen[key] != identity:
                    raise ValueError('Conflicting manifest members: ' + member['path'])
                if key in seen:
                    continue
                seen[key] = identity
                if path.exists():
                    require_match(path, member)
                else:
                    targets.append((record, member, path))
        plan = self.assemble(group, execute=execute)
        plan['restores'] = [{'path': m['path'], 'bytes': m['bytes']} for _, m, _ in targets]
        if not execute:
            return plan
        by_archive = {}
        for record, member, path in targets:
            by_archive.setdefault(record['name'], []).append((member, path))
        for record in records:
            selected = by_archive.get(record['name'], [])
            if not selected:
                continue
            archive = self.archive_path(record)
            require_match(archive, record)
            if record['format'] == 'zip':
                with zipfile.ZipFile(archive) as z:
                    if len(z.namelist()) != len(set(z.namelist())):
                        raise ValueError('Duplicate archive members: ' + record['name'])
                    for member, path in selected:
                        info = z.getinfo(member['path'])
                        if info.is_dir() or info.file_size != member['bytes']:
                            raise ValueError('Invalid ZIP member: ' + member['path'])
                        with z.open(info) as inp:
                            self.copy_member(inp, path, member)
            elif record['format'] == 'tar.xz':
                with tarfile.open(archive) as tar:
                    if len(tar.getnames()) != len(set(tar.getnames())):
                        raise ValueError('Duplicate archive members: ' + record['name'])
                    for member, path in selected:
                        info = tar.getmember(member['path'])
                        if not info.isfile() or info.size != member['bytes']:
                            raise ValueError('Invalid TAR member: ' + member['path'])
                        with tar.extractfile(info) as inp:
                            self.copy_member(inp, path, member)
            else:
                raise ValueError('Unknown archive format')
        return plan

    @staticmethod
    def copy_member(inp, path, member):
        with atomic_destination(path) as (out, partial):
            total = 0
            while data := inp.read(BLOCK):
                total += len(data)
                if total > member['bytes']:
                    raise ValueError('Archive member exceeds expected size')
                out.write(data)
            out.flush()
            require_match(partial, member)

    def acquire_raw(self, execute=False):
        records = self.manifest['raw_assets']
        actions = []
        for record in records:
            path = safe_path(self.root, record['path'])
            if path.exists():
                require_match(path, record)
            else:
                actions.append({'path': record['path'], 'bytes': record['bytes'], 'url': record['url']})
        if not execute:
            return {'execute': False, 'actions': actions}
        for record in records:
            path = safe_path(self.root, record['path'])
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = safe_path(self.root, record['path'] + '.local-download')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > record['bytes']:
                raise ValueError('Partial download exceeds expected size: ' + record['path'])
            if offset < record['bytes']:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                request = urllib.request.Request(record['url'], headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status == 206:
                        expected_range = f'bytes {offset}-'
                        if not response.headers.get('Content-Range', '').startswith(expected_range):
                            raise ValueError('Download resume offset mismatch')
                        mode = 'ab'
                    elif response.status == 200:
                        mode, offset = 'wb', 0
                    else:
                        raise ValueError('Unexpected download response')
                    with partial.open(mode) as out:
                        while data := response.read(BLOCK):
                            offset += len(data)
                            if offset > record['bytes']:
                                raise ValueError('Downloaded file exceeds expected size')
                            out.write(data)
            require_match(partial, record)
            os.link(partial, path)
            partial.unlink()
        receipt = {'schema': 'sanw-local-raw-acquisition-v1', 'verified': True,
                   'files': [{k: r[k] for k in ('path', 'bytes', 'sha256', 'url')} for r in records]}
        dest = safe_path(self.root, 'local/state/raw_acquisition.json')
        data = (json.dumps(receipt, indent=2) + '\n').encode()
        if dest.exists():
            if dest.read_bytes() != data:
                raise ValueError('Existing local acquisition receipt differs')
        else:
            with atomic_destination(dest) as (out, _):
                out.write(data)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'assemble', 'restore', 'train-dev', 'completed-fits', 'acquire-raw'])
    parser.add_argument('--group', choices=['all', 'train-dev', 'fits', 'benchmark'], default=None)
    parser.add_argument('--execute', action='store_true', help='Write verified files; otherwise show the plan')
    args = parser.parse_args(argv)
    assets = Assets()
    if args.command == 'verify':
        result = assets.verify(args.group or 'all')
    elif args.command == 'assemble':
        result = assets.assemble(args.group or 'all', args.execute)
    elif args.command == 'restore':
        result = assets.restore(args.group or 'train-dev', args.execute)
    elif args.command in ('train-dev', 'completed-fits'):
        result = assets.restore('train-dev' if args.command == 'train-dev' else 'fits', args.execute)
    else:
        result = assets.acquire_raw(args.execute)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, tarfile.TarError) as error:
        print('Asset operation stopped: ' + str(error), file=sys.stderr)
        sys.exit(2)

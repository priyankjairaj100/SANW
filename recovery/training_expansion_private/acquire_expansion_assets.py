#!/usr/bin/env python3
"""Acquire only pinned image archive and the established CLIP backbones."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def digest(path):
    with path.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def save(path, value):
    part = path.with_suffix(path.suffix + '.tmp')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(path)

def acquire(item):
    name, meta = item
    path = ROOT / 'data/official_train_expansion/assets' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    receipt = HERE / (name.replace('.', '_') + '_acquisition.json')
    expected = meta['bytes']
    start = time.monotonic()
    row = {'name': name, 'source_url': meta['url'], 'path': str(path.relative_to(ROOT)),
           'expected_bytes': expected, 'expected_sha256': meta.get('sha256'),
           'pinned_object_etag': meta.get('object_etag'), 'attempts': []}
    if path.exists():
        actual = digest(path)
        if path.stat().st_size != expected or (meta.get('sha256') and actual != meta['sha256']):
            raise ValueError('Existing asset identity mismatch: ' + name)
        row.update({'complete': True, 'bytes': expected, 'observed_sha256': actual, 'reused': True})
        save(receipt, row)
        return row
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError('Partial asset exceeds pinned byte count: ' + name)
        if offset == expected:
            break
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(meta['url'], headers=headers), timeout=120) as response:
                etag = response.headers.get('ETag', '').strip('"')
                if meta.get('object_etag') and etag != meta['object_etag']:
                    raise ValueError('Pinned archive ETag changed')
                if offset and response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unsafe HTTP resume offset')
                    mode = 'ab'
                elif response.status == 200:
                    mode, offset = 'wb', 0
                else:
                    raise ValueError('Unexpected HTTP response')
                row['attempts'].append({'attempt': attempt + 1, 'resume_offset': offset,
                                        'http_status': response.status, 'etag': etag})
                save(receipt, row)
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(16 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        offset += len(block)
                        if offset > expected:
                            raise ValueError('Downloaded payload exceeds pinned size')
            if partial.stat().st_size == expected:
                break
        except Exception as error:
            row['last_error'] = str(error)
            save(receipt, row)
            if attempt == 2:
                raise
    if partial.stat().st_size != expected:
        raise ValueError('Incomplete pinned asset: ' + name)
    actual = digest(partial)
    if meta.get('sha256') and actual != meta['sha256']:
        raise ValueError('Pinned payload SHA256 mismatch: ' + name)
    partial.replace(path)
    row.update({'complete': True, 'bytes': expected, 'observed_sha256': actual,
                'payload_sha256_matched_pin': bool(meta.get('sha256')),
                'seconds': time.monotonic() - start})
    save(receipt, row)
    print(json.dumps({k: row[k] for k in ('name', 'bytes', 'observed_sha256', 'seconds')}), flush=True)
    return row

def main():
    evidence = json.loads((HERE / 'OFFICIAL_TRAIN_EXPANSION_EVIDENCE.json').read_text())['expansion']
    files = [('flickr30k-images.zip', evidence['image_archive'])]
    for weight in evidence['weights']:
        filename = 'open_clip_model.safetensors' if weight['name'] == 'vit_b32_laion2b' else 'RN50.pt'
        files.append((filename, weight))
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(acquire, files))
    save(HERE / 'ASSET_ACQUISITION.json', {'schema': 'sanw-expanded-train-assets-v1',
                                         'files': rows, 'training_sample_sha256': digest(HERE / 'FIXED_6000_TRAIN_SAMPLE.json')})

if __name__ == '__main__':
    main()

"""Recover only declared Flickr members with pooled, verified HTTP ranges.

Acquisition optimization only. Scientific source files and manifests stay fixed.
Every published image must match the saved byte count, CRC32, and SHA256.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import struct
import threading
import time
import zlib

import requests

ROOT = Path(__file__).resolve().parents[2]
INDEX = json.loads((ROOT / 'data/visual_entailment/raw/flickr30k_zip_selected_index.json').read_text())
STATE = threading.local()
ROWS = {}
for source in ['data/visual_entailment/provenance.json', 'data/review_followup/e_vil_dev900/provenance.json', 'data/review_followup/e_vil_test1000/provenance.json']:
    for row in json.loads((ROOT / source).read_text())['images'].values():
        ROWS.setdefault(row['path'], row)


def valid_response(response, start, end):
    response.raise_for_status()
    assert response.status_code == 206
    assert response.headers['Content-Range'] == f"bytes {start}-{end}/{INDEX['archive_size']}"
    assert response.headers['ETag'].strip('"') == INDEX['archive_etag']
    assert len(response.content) == end - start + 1


# Resolve the public immutable archive once. Do not persist temporary CDN URLs.
end = INDEX['archive_size'] - 1
response = requests.get(INDEX['archive_url'], headers={'Range': f'bytes={end-21}-{end}'}, timeout=90)
valid_response(response, end - 21, end)
TRANSFER_URL = response.url


def get_range(start, end):
    if not hasattr(STATE, 'session'):
        STATE.session = requests.Session()
    for attempt in range(4):
        try:
            response = STATE.session.get(TRANSFER_URL, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
            valid_response(response, start, end)
            return response.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def one(row):
    target = ROOT / row['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        data = target.read_bytes()
        status = 'existing'
    else:
        item = INDEX['images'][target.name]
        packet = get_range(item['offset'], min(item['offset'] + item['compressed_size'] + 2047, INDEX['archive_size'] - 1))
        assert packet[:4] == b'PK\x03\x04'
        name_length, extra_length = struct.unpack_from('<HH', packet, 26)
        offset = 30 + name_length + extra_length
        if offset + item['compressed_size'] <= len(packet):
            raw = packet[offset:offset + item['compressed_size']]
        else:
            start = item['offset'] + offset
            raw = get_range(start, start + item['compressed_size'] - 1)
        assert item['compression'] in (0, 8)
        data = zlib.decompress(raw, -15) if item['compression'] == 8 else raw
        status = 'restored'
    assert len(data) == row['size_bytes']
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert zlib.crc32(data) & 0xffffffff == row['crc32']
    if status == 'restored':
        temporary = target.with_suffix('.pooled.part')
        temporary.write_bytes(data)
        temporary.replace(target)
    return {'path': row['path'], 'bytes': len(data), 'sha256': row['sha256'], 'status': status}


started = time.monotonic()
receipt = {'source': INDEX['archive_url'], 'archive_etag': INDEX['archive_etag'], 'archive_bytes': INDEX['archive_size'], 'total': len(ROWS), 'verified': [], 'errors': []}
with concurrent.futures.ThreadPoolExecutor(max_workers=96) as pool:
    futures = {pool.submit(one, row): row['path'] for row in ROWS.values()}
    for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
        try:
            receipt['verified'].append(future.result())
        except Exception as error:
            receipt['errors'].append({'path': futures[future], 'error_type': type(error).__name__})
        if count % 100 == 0 or count == len(ROWS):
            print(count, '/', len(ROWS), 'errors', len(receipt['errors']), 'elapsed', round(time.monotonic() - started, 1), flush=True)
            (ROOT / 'recovery/current_input_recovery/flickr_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert not receipt['errors'], receipt['errors'][:5]

"""Recover pinned Flickr images by a single archive transfer, verify per-image hashes."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,os,time
ROOT=Path(__file__).resolve().parents[2]
index=json.loads((ROOT/'data/visual_entailment/raw/flickr30k_zip_selected_index.json').read_text())
archive=ROOT/'recovery/resume_20261004/flickr30k-images.zip'
if not archive.exists() and not archive.with_suffix('.zip.part').exists():
 part=archive.with_suffix('.zip.part')
 start=time.monotonic();total=0
 with urllib.request.urlopen(index['archive_url'],timeout=180) as response,part.open('wb') as out:
  etag=response.headers.get('ETag','').strip('"');assert etag==index['archive_etag'],etag
  while True:
   b=response.read(16*1024*1024)
   if not b:break
   out.write(b);total+=len(b)
   if total%(256*1024*1024)==0:print('Downloaded',total,'seconds',round(time.monotonic()-start,1),flush=True)
 assert total==index['archive_size'],total
 with part.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 part.replace(archive)
 print('Observed archive SHA256',digest,flush=True)
elif not archive.exists():
 part=archive.with_suffix('.zip.part')
 assert part.stat().st_size==index['archive_size']
 # Hugging Face's observed ETag is an object identity, not assumed SHA256.
 # Archive members remain bound by original CRC32, size and image SHA256.
 part.replace(archive)
rows={}
for f in ['data/visual_entailment/provenance.json','data/review_followup/e_vil_dev900/provenance.json','data/review_followup/e_vil_test1000/provenance.json']:
 for r in json.loads((ROOT/f).read_text())['images'].values():rows.setdefault(r['path'],r)
restored=[]
with archive.open('rb') as f:archive_sha=hashlib.file_digest(f,'sha256').hexdigest()
with zipfile.ZipFile(archive) as z:
 for i,r in enumerate(rows.values(),1):
  path=ROOT/r['path'];path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():
   with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256'],str(path)
  else:
   info=z.getinfo(r['archive_member']);assert info.CRC==r['crc32'] and info.file_size==r['size_bytes'],str(path)
   content=z.read(r['archive_member']);assert len(content)==r['size_bytes'] and hashlib.sha256(content).hexdigest()==r['sha256'],str(path)
   part=path.with_suffix('.bulk.part');part.write_bytes(content);part.replace(path)
  restored.append({'path':r['path'],'sha256':r['sha256'],'bytes':r['size_bytes']})
  if i%100==0:print('Restored',i,'/',len(rows),flush=True)
(ROOT/'recovery/resume_20261004/flickr_bulk_receipt.json').write_text(json.dumps({'total':len(rows),'verified':restored,'observed_archive_sha256':archive_sha,'pinned_archive_etag':index['archive_etag'],'pinned_archive_bytes':index['archive_size']},indent=2)+'\n')
print('COMPLETE',len(rows),flush=True)

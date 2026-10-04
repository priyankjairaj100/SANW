"""Pooled HTTP recovery of existing frozen COCO image manifest entries."""
from pathlib import Path
import json,hashlib,requests,threading,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];state=threading.local();rows={}
for ds in ['sugarcrepe','sugarcrepe_pp','coco_karpathy']:
 for r in json.loads((ROOT/'data'/ds/'manifest.json').read_text())['images']:rows.setdefault(r['path'],r)
def one(r):
 p=ROOT/r['path'];p.parent.mkdir(exist_ok=True,parents=True)
 if p.exists():
  assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];return
 if not hasattr(state,'session'):state.session=requests.Session()
 rel=r.get('official_filename');rel=('val2014/'+rel) if rel else ('val2017/'+f"{r['coco_id']:012d}.jpg")
 for k in range(3):
  try:
   response=state.session.get('https://s3.amazonaws.com/images.cocodataset.org/'+rel,timeout=90);response.raise_for_status();b=response.content
   assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
   part=p.with_suffix('.pooled.part');part.write_bytes(b);part.replace(p);return
  except Exception:
   if k==2:raise
   time.sleep(2)
errors=[];start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=96) as pool:
 futures={pool.submit(one,r):r['path'] for r in rows.values()}
 for i,f in enumerate(concurrent.futures.as_completed(futures),1):
  try:f.result()
  except Exception as e:errors.append({'path':futures[f],'error':str(e)})
  if i%100==0:print(i,'/',len(rows),'errors',len(errors),'seconds',round(time.monotonic()-start),flush=True)
receipt={'total':len(rows),'errors':errors,'verified_sha256':{r['path']:r['sha256'] for r in rows.values()},'elapsed_seconds':time.monotonic()-start}
(ROOT/'recovery/resume_20261004/coco_pooled_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if errors:raise RuntimeError(errors[:5])
print('COMPLETE',len(rows),flush=True)

"""Recover only images named by existing manifests, verify original SHA256s.

Does not regenerate/modify manifests, frozen protocols, or training sources.
"""
from pathlib import Path
import sys,json,hashlib,urllib.request,concurrent.futures,time,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_visual_entailment import archive_index,acquire_image

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=['flickr','coco'],required=True);p.add_argument('--workers',type=int,default=32);args=p.parse_args()
 if args.group=='flickr':
  files=['data/visual_entailment/provenance.json','data/review_followup/e_vil_dev900/provenance.json','data/review_followup/e_vil_test1000/provenance.json']
  rows={}
  for f in files:
   for r in json.loads((ROOT/f).read_text())['images'].values():rows.setdefault(r['path'],r)
  raw=ROOT/'data/visual_entailment/raw';raw.mkdir(parents=True,exist_ok=True)
  print('Acquiring pinned Flickr ZIP index',flush=True);idx=archive_index(raw)['images'];print('Index ready',len(idx),flush=True)
 else:
  rows={}
  for n in ['coco_karpathy','sugarcrepe','sugarcrepe_pp']:
   for r in json.loads((ROOT/'data'/n/'manifest.json').read_text())['images']:rows.setdefault(r['path'],r)
 def one(r):
  path=ROOT/r['path'];path.parent.mkdir(parents=True,exist_ok=True);size=r.get('size_bytes',r.get('bytes'))
  if path.exists():
   assert path.stat().st_size==size and sha(path)==r['sha256'],str(path)
   return {'path':r['path'],'status':'existing'}
  if args.group=='flickr':acquire_image(path.name,idx[path.name],path.parent,ROOT)
  else:
   cid=r['coco_id'];rel='val2014/COCO_val2014_'+f'{cid:012d}.jpg' if r.get('official_filename') else 'val2017/'+f'{cid:012d}.jpg'
   url='https://s3.amazonaws.com/images.cocodataset.org/'+rel
   part=path.with_suffix('.resume.part')
   for attempt in range(4):
    try:
     with urllib.request.urlopen(url,timeout=90) as response:content=response.read()
     assert len(content)==size and hashlib.sha256(content).hexdigest()==r['sha256'],str(path)
     part.write_bytes(content);part.replace(path);break
    except Exception:
     if attempt==3:raise
     time.sleep(min(2**attempt,4))
  assert path.stat().st_size==size and sha(path)==r['sha256'],str(path)
  return {'path':r['path'],'status':'restored','sha256':r['sha256'],'bytes':size}
 result={'group':args.group,'total':len(rows),'restored':[],'errors':[]}
 print('Recovering',args.group,len(rows),'images',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  pending={pool.submit(one,r):r['path'] for r in rows.values()}
  for i,f in enumerate(concurrent.futures.as_completed(pending),1):
   try:result['restored'].append(f.result())
   except Exception as e:result['errors'].append({'path':pending[f],'error':str(e)})
   if i%50==0 or i==len(rows):
    print(json.dumps({'done':i,'total':len(rows),'errors':len(result['errors'])}),flush=True)
    (ROOT/'recovery/resume_20261004'/f'{args.group}_acquisition_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 if result['errors']:raise RuntimeError(result['errors'][:5])
if __name__=='__main__':main()

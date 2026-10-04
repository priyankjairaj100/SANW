from pathlib import Path
import hashlib,json,urllib.request,os,time
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'data/model/acquisition_receipt.json'
r=json.loads(p.read_text())
for name,pin in r['identity']['files'].items():
 path=ROOT/'data/model'/name
 def check(p):
  return p.stat().st_size==pin['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==pin['sha256']
 if path.exists():
  assert check(path),path
  continue
 part=path.with_suffix(path.suffix+'.resume.part')
 print('Downloading',name,flush=True)
 with urllib.request.urlopen(r['source_urls'][name],timeout=180) as response,part.open('wb') as out:
  total=0
  while True:
   b=response.read(8*1024*1024)
   if not b:break
   out.write(b);total+=len(b)
   if total%(64*1024*1024)==0:print('Downloaded bytes',total,flush=True)
 assert check(part),(part,'hash/size mismatch')
 os.link(part,path);part.unlink()
 print('Restored exact model',name,pin['sha256'],flush=True)

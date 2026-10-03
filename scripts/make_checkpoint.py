#!/usr/bin/env python3
"""Create an immutable, verified recovery ZIP for one completed work stage."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--name',required=True); p.add_argument('--include',action='append',default=[])
 p.add_argument('--max-file-mib',type=float,default=40)
 a=p.parse_args(); out=ROOT.parent/'deliveries'/f'{a.name}.zip'; out.parent.mkdir(exist_ok=True)
 if out.exists(): raise SystemExit('Refusing to overwrite an earlier checkpoint')
 files=[]
 if not a.include:
  for folder in ['src','scripts','tests','configs','docs','historical_context','manuscript']:
   files+=list((ROOT/folder).rglob('*'))
  files+=list(ROOT.glob('*.toml'))+list(ROOT.glob('README*'))
  for folder in ['results','data']:
   files += [x for x in (ROOT/folder).rglob('*') if x.suffix in {'.json','.csv','.txt','.md','.jsonl'}]
 else:
  for pattern in a.include: files+=list(ROOT.glob(pattern))
 files=sorted({x for x in files if x.is_file() and not any(t in x.parts for t in ['__pycache__','.cache','.git','build','tex-runtime']) and x.stat().st_size<a.max_file_mib*1024**2 and x.suffix not in {'.part','.tmp','.incomplete'}})
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':a.name,'scope':'Explicit listed files; see protocol and status for completed versus pending work','files':[]}
 with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:
   before=f.stat(); content=f.read_bytes(); after=f.stat()
   if before.st_mtime_ns!=after.st_mtime_ns: raise RuntimeError(f'File changed during checkpoint: {f}')
   rel=f.relative_to(ROOT).as_posix(); z.writestr('project/'+rel,content)
   manifest['files'].append({'path':'project/'+rel,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
  z.writestr('CHECKPOINT_MANIFEST.json',json.dumps(manifest,indent=2))
  z.writestr('RESTORE.txt','Extract this ZIP. Merge later checkpoints into the same project folder. Every listed file has a SHA256 in CHECKPOINT_MANIFEST.json. Public raw data/model sources and revisions are in provenance and protocol records. Historical context is not newly computed evidence.\n')
 with zipfile.ZipFile(out) as z:
  if z.testzip() is not None: raise RuntimeError('ZIP integrity failed')
 receipt={'path':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'files':len(files),'crc_verified':True}
 out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Restore the exact scientific snapshot. Refuse changes to unrecognized local files."""
from pathlib import Path
import argparse, hashlib, json, os, tarfile, tempfile
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git_digest(path):
 p=Path(path);h=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0')
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def safe(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe snapshot path: '+name)
 out=root/p
 cursor=root
 for piece in p.parts:
  cursor=cursor/piece
  if cursor.is_symlink():raise ValueError('Symbolic links are not permitted: '+name)
 if not out.resolve().is_relative_to(root.resolve()):raise ValueError('Path leaves project: '+name)
 return out

def restore(root,manifest,check=False):
 rows={r['path']:r for r in manifest['files']}
 if len(rows)!=len(manifest['files']):raise ValueError('Duplicate manifest paths')
 before={};pending=[]
 for name,row in rows.items():
  p=safe(root,name)
  if p.exists():
   if not p.is_file() or p.is_symlink():raise ValueError('Existing path is not an ordinary file: '+name)
   h=digest(p)
   if h==row['sha256']:continue
   if not row.get('previous_git_blob') or git_digest(p)!=row['previous_git_blob']:
    raise ValueError('Unrecognized local edits; preserve and resolve before restoration: '+name)
   before[name]=h
  else:before[name]=None
  pending.append(name)
 if check:return {'verified':not pending,'pending_files':len(pending),'file_count':len(rows)}
 tempdir=safe(root,'local/assets/science_snapshot');tempdir.mkdir(parents=True,exist_ok=True)
 archive=safe(root,'local/assets/science_snapshot/scientific_snapshot.tar.xz')
 if not archive.exists():
  fd,tmp=tempfile.mkstemp(prefix='assembly.',dir=tempdir)
  try:
   with os.fdopen(fd,'wb') as out:
    for part in manifest['parts']:
     p=safe(root,part['path'])
     if p.stat().st_size!=part['bytes'] or digest(p)!=part['sha256']:raise ValueError('Snapshot chunk differs: '+str(p))
     with p.open('rb') as f:
      while b:=f.read(1<<20):out.write(b)
   if Path(tmp).stat().st_size!=manifest['archive_bytes'] or digest(tmp)!=manifest['archive_sha256']:raise ValueError('Snapshot archive differs')
   os.replace(tmp,archive)
  finally:Path(tmp).unlink(missing_ok=True)
 if archive.stat().st_size!=manifest['archive_bytes'] or digest(archive)!=manifest['archive_sha256']:raise ValueError('Snapshot archive identity failed')
 with tarfile.open(archive,'r:xz') as tar:
  members=tar.getmembers();names=[m.name for m in members]
  if len(names)!=len(set(names)) or set(names)!=set(rows):raise ValueError('Snapshot members differ')
  for m in members:
   if not m.isfile() or m.size!=rows[m.name]['bytes']:raise ValueError('Invalid snapshot member: '+m.name)
  pending=set(pending)
  for m in members:
   if m.name not in pending:continue
   dest=safe(root,m.name);dest.parent.mkdir(parents=True,exist_ok=True)
   expected=before[m.name]
   if (expected is None and dest.exists()) or (expected is not None and (not dest.exists() or digest(dest)!=expected)):
    raise ValueError('File changed during restoration: '+m.name)
   fd,tmp=tempfile.mkstemp(prefix=dest.name+'.',dir=dest.parent)
   try:
    with os.fdopen(fd,'wb') as out,tar.extractfile(m) as f:
     while b:=f.read(1<<20):out.write(b)
    if digest(tmp)!=rows[m.name]['sha256']:raise ValueError('Member hash differs: '+m.name)
    os.replace(tmp,dest)
   finally:Path(tmp).unlink(missing_ok=True)
 return {'verified':True,'restored_files':len(pending),'file_count':len(rows),'numerical_experiments':0}

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--execute',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
 a=p.parse_args();manifest=json.loads((ROOT/'local/science_snapshot_manifest.json').read_text())
 try:
  result=restore(a.root.resolve(),manifest,not a.execute);result['executed']=a.execute;print(json.dumps(result,indent=2));return 2 if a.check and not result['verified'] else 0
 except (OSError,ValueError,tarfile.TarError) as e:print('STOP:',e);return 2
if __name__=='__main__':raise SystemExit(main())

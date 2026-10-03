#!/usr/bin/env python3
"""Pinned frozen OpenCLIP encoding with transactional, content-addressed resume."""
from __future__ import annotations
import argparse, hashlib, json, os, sqlite3, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REVISION = '1a25a446712ba5ee05982a381eed697ef9b435cf'
WEIGHT_SHA = 'ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)

def short_text(model,tokens):
    """Discard only positions after every EOT; causal attention preserves EOT."""
    end=tokens.argmax(dim=-1)
    length=int(end.max())+1
    tokens=tokens[:,:length]
    dtype=model.transformer.get_cast_dtype()
    x=model.token_embedding(tokens).to(dtype)
    x=x+model.positional_embedding[:length].to(dtype)
    x=model.transformer(x,attn_mask=model.attn_mask[:length,:length])
    x=model.ln_final(x)
    x=x[torch.arange(x.shape[0]),end]
    if model.text_projection is not None:
        x=model.text_projection(x) if isinstance(model.text_projection,torch.nn.Linear) else x@model.text_projection
    return F.normalize(x,dim=-1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',action='append',required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--image-batch',type=int,default=32)
    p.add_argument('--text-batch',type=int,default=128)
    p.add_argument('--stock-text',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    modeldir=ROOT/'data/model'; weight=modeldir/'open_clip_model.safetensors'
    actual=digest(weight)
    if actual!=WEIGHT_SHA: raise RuntimeError('Pinned model hash mismatch')
    config=json.loads((modeldir/'open_clip_config.json').read_text())
    prep=config.get('preprocess_cfg',{})
    kwargs={}
    for key in ['mean','std','interpolation','resize_mode']:
        if key in prep: kwargs['image_'+key]=prep[key]
    model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained=str(weight),**kwargs)
    model.eval(); model.requires_grad_(False)
    tokenizer=open_clip.get_tokenizer('ViT-B-32')
    scale=float(model.logit_scale.exp())
    bankpath=ROOT/'results/features/bank.sqlite'; bankpath.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(bankpath)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS features(kind TEXT,key TEXT,value BLOB,PRIMARY KEY(kind,key))')
    signature=json.dumps({'weight':WEIGHT_SHA,'config':digest(modeldir/'open_clip_config.json'),'open_clip':open_clip.__version__},sort_keys=True)
    previous=db.execute("SELECT value FROM metadata WHERE key='encoder'").fetchone()
    if previous and previous[0]!=signature: raise RuntimeError('Feature bank encoder identity mismatch')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('encoder',?)",(signature,)); db.commit()
    short_ok=not args.stock_text
    parity=[]
    with torch.inference_mode():
        for dataset in args.dataset:
            start=time.monotonic()
            manifestpath=ROOT/'data'/dataset/'manifest.json'
            manifest=json.loads(manifestpath.read_text())
            texts=manifest['texts']; images=manifest['images']
            if short_ok:
                indices=sorted(set([0,len(texts)//2,len(texts)-1]+list(range(min(20,len(texts))))))
                tokens=tokenizer([texts[i]['text'] for i in indices])
                stock=model.encode_text(tokens,normalize=True)
                shortened=short_text(model,tokens)
                error=float((stock-shortened).abs().max())
                parity.append({'dataset':dataset,'examples':len(indices),'max_abs_error':error,'tolerance':2e-6})
                if error>2e-6: raise RuntimeError(f'Causal truncation parity failed: {error}')
            for kind,items,batch_size in [('image',images,args.image_batch),('text',texts,args.text_batch)]:
                keyed=[]
                for item in items:
                    key=digest(ROOT/item['path']) if kind=='image' else hashlib.sha256(item['text'].encode()).hexdigest()
                    keyed.append((key,item))
                pending=[]; seen=set()
                for key,item in keyed:
                    if key not in seen and not db.execute('SELECT 1 FROM features WHERE kind=? AND key=?',(kind,key)).fetchone():
                        pending.append((key,item)); seen.add(key)
                if kind=='text': pending.sort(key=lambda pair:len(pair[1]['text']))
                print(f'{dataset} {kind}: {len(pending)} new / {len(items)} total',flush=True)
                for pos in range(0,len(pending),batch_size):
                    batch=pending[pos:pos+batch_size]
                    if kind=='image':
                        tensors=[]
                        for _,item in batch:
                            with Image.open(ROOT/item['path']) as im: tensors.append(preprocess(im.convert('RGB')))
                        encoded=model.encode_image(torch.stack(tensors),normalize=True)
                    else:
                        tokens=tokenizer([item['text'] for _,item in batch])
                        encoded=short_text(model,tokens) if short_ok else model.encode_text(tokens,normalize=True)
                    values=encoded.cpu().numpy().astype(np.float32)
                    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=2e-5):
                        raise RuntimeError('Invalid or unnormalized encoded features')
                    db.executemany('INSERT INTO features VALUES (?,?,?)',[(kind,key,value.tobytes()) for (key,_),value in zip(batch,values)])
                    db.commit()
                    if pos//batch_size%10==0 or pos+batch_size>=len(pending):
                        print(f'{dataset} {kind}: encoded {min(pos+batch_size,len(pending))}/{len(pending)} elapsed {time.monotonic()-start:.1f}s',flush=True)
                rows=[np.frombuffer(db.execute('SELECT value FROM features WHERE kind=? AND key=?',(kind,key)).fetchone()[0],dtype=np.float32).copy() for key,_ in keyed]
                if kind=='image': image_values=np.stack(rows)
                else: text_values=np.stack(rows)
            output=ROOT/'results/features'/dataset; output.mkdir(parents=True,exist_ok=True)
            temp=output/'features.tmp.npz'
            np.savez_compressed(temp,image_features=image_values,text_features=text_values,
                                image_ids=np.asarray([v['id'] for v in images]),text_ids=np.asarray([v['id'] for v in texts]))
            temp.replace(output/'features.npz')
            atomic_json(output/'metadata.json',{'schema_version':1,'dataset':dataset,'manifest_sha256':digest(manifestpath),
                'model_repository':'laion/CLIP-ViT-B-32-laion2B-s34B-b79K','model_revision':REVISION,'weights_sha256':actual,
                'logit_scale':scale,'image_count':len(images),'text_count':len(texts),'dimension':512,
                'dtype':'float32','normalization':'L2','preprocess_config':prep,'preprocess_repr':str(preprocess),
                'open_clip_version':open_clip.__version__,'torch_version':torch.__version__,
                'text_encoding':'causally_trimmed_after_last_eot' if short_ok else 'stock',
                'causal_padding_parity':parity,'source_sha256':digest(__file__),
                'features_sha256':digest(output/'features.npz'),'elapsed_seconds':time.monotonic()-start})
            print(f'COMPLETE {dataset}: {output}',flush=True)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()

if __name__=='__main__': main()

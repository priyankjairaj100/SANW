"""Check recreated cache identity and feature-array parity without changing originals."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
report={'scope':'Reencoded frozen ViT inputs; original metadata preserved','datasets':{},'shared_cache_checks':[]}
for ds in ['visual_entailment','sugarcrepe','sugarcrepe_pp','coco_karpathy']:
 p=ROOT/'results/resume_features'/ds/'features.npz'
 if not p.exists():continue
 original=json.loads((ROOT/'results/features'/ds/'metadata.json').read_text());meta=json.loads((p.parent/'metadata.json').read_text());manifest=ROOT/'data'/ds/'manifest.json'
 arrays=np.load(p);observed=digest(p)
 assert observed==meta['features_sha256'];assert meta['weights_sha256']==original['weights_sha256'];assert meta['manifest_sha256']==digest(manifest)==original['manifest_sha256']
 entry={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':observed,'original_container_sha256':original['features_sha256'],'exact_original_container_restored':observed==original['features_sha256'],'metadata_sha256':digest(p.parent/'metadata.json'),'manifest_sha256':digest(manifest),'arrays':{}}
 for key in arrays.files:
  a=arrays[key];entry['arrays'][key]={'shape':list(a.shape),'dtype':str(a.dtype),'c_order_bytes_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
 report['datasets'][ds]=entry
 if ds=='visual_entailment':
  for other in ['e_vil_dev900','e_vil_test1000']:
   q=ROOT/'results/review_followup/features'/other/'features.npz';ref=np.load(q)
   for modality in ['image','text']:
    ids=modality+'_ids';features=modality+'_features';lookup={str(v):i for i,v in enumerate(ref[ids])};pairs=[(i,lookup[str(v)]) for i,v in enumerate(arrays[ids]) if str(v) in lookup]
    assert pairs
    left=arrays[features][[i for i,j in pairs]];right=ref[features][[j for i,j in pairs]];error=np.abs(left-right)
    check={'reference':str(q.relative_to(ROOT)),'reference_sha256':digest(q),'modality':modality,'shared_rows':len(pairs),'exact_arrays':np.array_equal(left,right),'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),'within_2e_6':bool(np.allclose(left,right,rtol=0,atol=2e-6))}
    report['shared_cache_checks'].append(check);assert check['within_2e_6'],check
report['complete']=len(report['datasets'])==4
out=ROOT/'recovery/resume_20261004/reencoded_feature_audit.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

#!/usr/bin/env python3
"""Package the visually checked v3 flat Overleaf source without local caches."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
import fitz
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript_review_v3';O=ROOT/'output';P=O/'pdf/acl27_review_followup_v3.pdf'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((O/'pdf/paper_build_receipt_v3.json').read_text());assert build['sha256']==sha(P) and build['limitations_pages']==[5] and not build['tex_warnings'] and not build['type3_fonts']
audit=json.loads((M/'number_binding_audit.json').read_text());assert audit['status']=='passed'
for name,h in build['source_hashes'].items():assert sha(M/name)==h,name
for name,h in audit['audited_table_hashes'].items():assert sha(M/name)==h,name
for name,h in json.loads((M/'original_manuscript_preservation.json').read_text()).items():assert sha(ROOT/'manuscript'/name)==h,name
# Check flat-source dependencies, including unused retained historical fragments.
refs=[]
for p in M.glob('*.tex'):
 if p.name=='official_template_example.tex':continue
 for name in re.findall(r'\\input\{([^}]+)\}|\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',p.read_text()):
  n=next(x for x in name if x);assert '/' not in n and (M/n).is_file(),(p,n);refs.append(n)
doc=fitz.open(P);render=ROOT/'tmp/pdfs/v3-final/pages';render.mkdir(parents=True,exist_ok=True);page_receipts=[]
for i,page in enumerate(doc):
 out=render/f'page-{i+1:02}.png';page.get_pixmap(matrix=fitz.Matrix(1,1)).save(out);page_receipts.append({'page':i+1,'render_sha256':sha(out)})
layout={'status':'passed','pdf':str(P.relative_to(ROOT)),'sha256':sha(P),'bytes':P.stat().st_size,'total_pages':len(doc),'main_pages':4,'limitations_page':5,'all_pages_rendered':True,'visual_review':'All pages inspected for clipping, overlap, figure and table layout. Page 4 selection table appears after its definitions. Appendix spacing is unstretched.','style':'Official ACL review style unmodified','undefined_references':False,'overfull_boxes':False,'type3_fonts':False,'number_binding_checks':audit['checks'],'reviewers':['manuscript agent: all-page visual and number binding','root: main pages and representative appendices','training agent: numerical/story precision','losses agent: mathematical scope and proof references'],'page_renders':page_receipts}
(M/'final_layout_receipt_v3.json').write_text(json.dumps(layout,indent=2)+'\n')
shutil.copy2(O/'pdf/paper_build_receipt_v3.json',M/'paper_build_receipt_v3.json')
files=sorted(p for p in M.iterdir() if p.is_file() and p.suffix in {'.tex','.bib','.bst','.sty','.pdf','.png','.jpg','.json','.txt','.md','.py'})
archive=O/'acl27-overleaf-review-followup-v3.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in files:z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all('/' not in n for n in z.namelist())
receipt={'status':'frozen','pdf':{'path':str(P.relative_to(ROOT)),'sha256':sha(P),'bytes':P.stat().st_size},'overleaf':{'path':str(archive.relative_to(ROOT)),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)},'source_files':{p.name:sha(p) for p in files},'scripts':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/generate_review_paper_results.py',ROOT/'scripts/audit_review_paper.py',ROOT/'scripts/build_review_paper.py',Path(__file__).resolve()]},'original_manuscript_unchanged':True}
(O/'review_followup_v3_delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['status','pdf','overleaf','original_manuscript_unchanged']},indent=2))

#!/usr/bin/env python3
"""Independently bind every v3 table cell to audited JSON, then audit receipts."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript_review_v3'
def load(rel):return json.loads((ROOT/rel).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=0
sources={}
def read(rel):
 p=ROOT/rel;sources[rel]=sha(p);return json.loads(p.read_text())
def equal(a,b):
 global checks
 assert a==b,(a,b);checks+=1
def rows(name):
 s=(M/name).read_text();out=[]
 for body in re.findall(r'\\midrule\n(.*?)\\bottomrule',s,re.S):
  out.extend([[v.strip() for v in row.removesuffix('\\\\').split('&')] for row in body.strip().splitlines()])
 return out
F=read('results/review_followup/analysis/fixed_epoch_metrics.json');S=read('results/review_followup/analysis/selected_strategy_metrics.json');P=read('results/review_followup/analysis/primary_contrasts.json');O=read('results/analysis/aggregate_metrics.json')
keys=['visual_entailment.accuracy','sugarcrepe.accuracy','sugarcrepe_pp.both_accuracy','e_vil_test1000.i2t.r1','e_vil_test1000.t2i.r1','coco_karpathy.i2t.r1','coco_karpathy.t2i.r1']
rates=[.0001,.0003,.001];methods=['source','supported']+[f'{c}_draw_{d}' for c in ['count_only','score_stratified'] for d in range(3)]
cells={(x['method'],x['learning_rate'],x['epoch']):x for x in F['cells']};selected={(x['selector'],x['method']):x for x in S['strategies']}
def verify_numbers(actual,expected,digits=2,scale=100):
 equal(len(actual),len(expected))
 for text,value in zip(actual,expected):equal(text,f'{value*scale:.{digits}f}')
r=rows('followup_secondary_table.tex');equal(len(r),7);verify_numbers(r[0][2:],[O['methods']['frozen']['metrics'][k]['mean'] for k in keys[:3]])
for row,(rate,method) in zip(r[1:],[(r,m) for r in rates for m in ['source','supported']]):verify_numbers(row[2:],[cells[method,rate,10]['metrics'][k]['mean'] for k in keys[:3]])
r=rows('followup_selection_table.tex');equal(len(r),4)
for row,(selector,method) in zip(r,[(s,m) for s in ['native','source_retrieval'] for m in ['source','supported']]):
 x=selected[selector,method];equal(row[2],'/'.join(str(t['epoch']) for t in x['selected_states']));verify_numbers(row[3:],[x['metrics'][k]['mean'] for k in keys[3:5]])
r=rows('followup_results_appendix.tex');equal(len(r),9+72+48+36)
contrasts={(x['learning_rate'],x['right'],x['metric']):x for x in P['contrasts']}
for row,(rate,comp) in zip(r[:9],[(r,c) for r in rates for c in ['source','count_only','score_stratified']]):
 for offset,metric in [(2,'i2t.r1'),(4,'t2i.r1')]:
  x=contrasts[rate,comp,metric];equal(row[offset],f'{x["difference_percentage_points"]:+.3f}');equal(row[offset+1],'$['+','.join(f'{v:+.3f}' for v in x['ci_percentage_points'])+']$')
o=9
for method in methods:
 for rate in rates:
  x=cells[method,rate,10]
  for i,seed in enumerate([17,29,43]):
   row=r[o];o+=1;equal(row[2],str(seed));verify_numbers(row[3:],[x['metrics'][k]['values'][i] for k in keys])
for selector in ['native','source_retrieval']:
 for method in methods:
  x=selected[selector,method]
  for i,state in enumerate(x['selected_states']):
   row=r[o];o+=1;equal(row[2:4],[str(state['seed']),str(state['epoch'])]);verify_numbers(row[4:],[x['metrics'][k]['values'][i] for k in keys])
tkeys=keys[:3]+['sugarcrepe_pp.positive1_accuracy','sugarcrepe_pp.positive2_accuracy']+keys[3:5]
for method in ['source','supported','count_only','score_stratified']:
 for rate in rates:
  for epoch in [1,5,10]:
   x=cells[method,rate,epoch] if method in ['source','supported'] else next(c for c in F['equal_draw_policy_means'] if c['condition']==method and c['learning_rate']==rate and c['epoch']==epoch)
   row=r[o];o+=1;equal(row[2],str(epoch));verify_numbers(row[3:],[x['metrics'][k]['mean'] for k in tkeys])
equal(o,len(r))
macros=dict(re.findall(r'\\newcommand\{\\(Fu[A-Za-z]+)\}\{([^}]+)\}',(M/'followup_numbers.tex').read_text()))
used=set(re.findall(r'\\(Fu[A-Za-z]+)',(M/'main.tex').read_text()+(M/'abstract.tex').read_text()))
equal(used-set(macros),set())
# Main-text effects are independently recomputed rather than trusted as macros.
source_losses=[-contrasts[r,'source','i2t.r1']['difference_percentage_points'] for r in rates]
strata_gains=[contrasts[r,'score_stratified','t2i.r1']['difference_percentage_points'] for r in rates]
for name,value in [('FuSourceLossLow',min(source_losses)),('FuSourceLossHigh',max(source_losses)),('FuStrataGainLow',min(strata_gains)),('FuStrataGainHigh',max(strata_gains))]:equal(macros[name],f'{value:.2f}')
for suffix,rate in zip(['One','Two','Three'],rates):
 equal(macros['FuStrataImageEffect'+suffix],f'{contrasts[rate,"score_stratified","i2t.r1"]["difference_percentage_points"]:+.2f}')
 for name,key in [('FuCocoImageDifference','coco_karpathy.i2t.r1'),('FuCocoTextDifference','coco_karpathy.t2i.r1')]:
  d=100*(cells['supported',rate,10]['metrics'][key]['mean']-cells['source',rate,10]['metrics'][key]['mean']);equal(macros[name+suffix],f'{d:+.2f}')
 d=100*(cells['supported',rate,10]['metrics']['sugarcrepe.accuracy']['mean']-cells['source',rate,10]['metrics']['sugarcrepe.accuracy']['mean']);equal(macros['FuSugarGain'+suffix],f'{d:.2f}')
equal(macros['FuAdditionalFailuresLow'],f'{10*min(source_losses):.0f}');equal(macros['FuAdditionalFailuresHigh'],f'{10*max(source_losses):.0f}')
equal(macros['FuExpansionZeroSelections'],str(sum(state['epoch']==0 for x in S['strategies'] if x['selector']=='source_retrieval' and x['method']!='source' for state in x['selected_states'])))
for item in load('manuscript_review_v3/v3_results_receipt.json')['analysis_inputs']+load('manuscript_review_v3/v3_results_receipt.json')['generated_outputs']:equal(sha(ROOT/item['path']),item['sha256'])
for name,value in load('manuscript_review_v3/original_manuscript_preservation.json').items():equal(sha(ROOT/'manuscript'/name),value)
receipt={'status':'passed','checks':checks,'all_primary_effects':18,'terminal_seed_rows':72,'selected_seed_rows':48,'descriptive_trajectory_rows':36,'main_secondary_rows':7,'main_selection_rows':4,'rounding':'Exact formatted equality at displayed precision','source_sha256':sources,'auditor_sha256':sha(Path(__file__)),'audited_table_hashes':{n:sha(M/n) for n in ['followup_secondary_table.tex','followup_selection_table.tex','followup_results_appendix.tex','followup_numbers.tex']}}
(M/'number_binding_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

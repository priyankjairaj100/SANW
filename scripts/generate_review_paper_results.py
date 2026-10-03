#!/usr/bin/env python3
"""Generate v3 paper numbers/tables from audited follow-up JSON; preserve v2."""
from pathlib import Path
import re
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
SHA='3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34'
LRS=[.0001,.0003,.001]
METHODS=['source','supported']+[f'{c}_draw_{d}' for c in ['count_only','score_stratified'] for d in range(3)]
KEYS=['visual_entailment.accuracy','sugarcrepe.accuracy','sugarcrepe_pp.both_accuracy','e_vil_test1000.i2t.r1','e_vil_test1000.t2i.r1','coco_karpathy.i2t.r1','coco_karpathy.t2i.r1']
LABELS={'source':'Source','supported':'Supported',**{f'count_only_draw_{d}':f'Count {d+1}' for d in range(3)},**{f'score_stratified_draw_{d}':f'Strata {d+1}' for d in range(3)}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def pp(v):return f'{100*v:.2f}'
def lr(v):return {1e-4:r'$10^{-4}$',3e-4:r'$3\cdot10^{-4}$',1e-3:r'$10^{-3}$'}[v]
def table(headers,rows,caption,label,wide=False,small=False,placement="t"):
 env='table*' if wide else 'table'
 size='\\small\n' if small else ''
 return '\n'.join([f'\\begin{{{env}}}[{placement}]','\\nolinenumbers','\\centering',size+r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',*[' & '.join(map(str,r))+r'\\' for r in rows],r'\bottomrule',r'\end{tabular}',f'\\caption{{{caption}}}\\label{{{label}}}',f'\\end{{{env}}}',''])
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'manuscript_review_v3');a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True)
 inputs=[]
 def read(rel):
  path=ROOT/rel;inputs.append({'path':rel,'sha256':sha(path)});return load(path)
 protocol=read('docs/REVIEW_FOLLOWUP_PROTOCOL.json');assert sha(ROOT/'docs/REVIEW_FOLLOWUP_PROTOCOL.json')==SHA
 aggregate=read('results/review_followup/analysis/fixed_epoch_metrics.json');sel=read('results/review_followup/analysis/selected_strategy_metrics.json');primary=read('results/review_followup/analysis/primary_contrasts.json');audit=read('results/review_followup/analysis/analysis_audit.json');independent=read('results/review_followup/analysis/independent_primary_audit.json');training=read('results/review_followup/training_final_audit.json')
 assert audit['status']=='passed' and audit['primary_effects']==18 and primary['protocol_sha256']==SHA
 original=read('results/analysis/aggregate_metrics.json')
 figure_receipt=read('results/review_followup/figures/figure_provenance.json')
 cells={(x['method'],x['learning_rate'],x['epoch']):x for x in aggregate['cells']}
 summaries={(x['selector'],x['method']):x for x in sel['strategies']}
 def mean(m,rate,k,epoch=10):return cells[m,rate,epoch]['metrics'][k]['mean']
 def delta(m,b,rate,k):return mean(m,rate,k)-mean(b,rate,k)
 effects={(x['learning_rate'],x['right'],x['metric']):x for x in primary['contrasts']}
 outputs=[]
 def write(name,s):
  p=out/name;p.write_text('% Generated from audited follow-up artifacts. Do not hand-edit numbers.\n'+s);outputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 macros={'FuFits':72,'FuAssignmentDraws':3,'FuTrainImages':'1,200','FuDevImages':900,'FuDevCaptions':'4,500','FuTestImages':'1,000','FuTestCaptions':'5,000','FuPrimaryFamily':18,'FuConfidence':'99.72','FuStates':792,'FuExpansionZeroSelections':21}
 source_effects=[effects[r,'source','i2t.r1']['difference_percentage_points'] for r in LRS]
 strat_effects=[effects[r,'score_stratified','t2i.r1']['difference_percentage_points'] for r in LRS]
 macros.update(FuSourceLossLow=f'{min(-x for x in source_effects):.2f}',FuSourceLossHigh=f'{max(-x for x in source_effects):.2f}',FuStrataGainLow=f'{min(strat_effects):.2f}',FuStrataGainHigh=f'{max(strat_effects):.2f}',FuAdditionalFailuresLow=f'{min(-10*x for x in source_effects):.0f}',FuAdditionalFailuresHigh=f'{max(-10*x for x in source_effects):.0f}')
 for i,rate in enumerate(LRS,1):
  for name,k in [('Sugar','sugarcrepe.accuracy'),('Both','sugarcrepe_pp.both_accuracy')]:macros[f'Fu{name}Gain{i}']=pp(delta('supported','source',rate,k))
  for direction,key in [('Image','i2t.r1'),('Text','t2i.r1')]:
   e=effects[rate,'score_stratified',key];macros[f'FuStrata{direction}Effect{i}']=f'{e["difference_percentage_points"]:+.2f}';macros[f'FuStrata{direction}CI{i}']='['+','.join(f'{v:.2f}' for v in e['ci_percentage_points'])+']'
 macros['FuEarlyBoth']=pp(mean('supported',1e-4,'sugarcrepe_pp.both_accuracy',epoch=1))
 macros['FuEarlySugar']=pp(mean('supported',1e-4,'sugarcrepe.accuracy',epoch=1))
 for i,r in enumerate(LRS,1):
  macros[f'FuCocoImageDifference{i}']=f"{100*delta('supported','source',r,'coco_karpathy.i2t.r1'):+.2f}"
  macros[f'FuCocoTextDifference{i}']=f"{100*delta('supported','source',r,'coco_karpathy.t2i.r1'):+.2f}"
 coco_losses=[-100*delta('supported','source',r,'coco_karpathy.i2t.r1') for r in LRS]
 macros['FuCocoLossLow']=f'{min(coco_losses):.2f}';macros['FuCocoLossHigh']=f'{max(coco_losses):.2f}'
 frozen=original['methods']['frozen']['metrics']
 # Frozen figures are checked against the selected epoch-zero records, not typed baselines.
 frozen_sel=summaries['source_retrieval','supported'];assert all(r['epoch']==0 for r in frozen_sel['selected_states'])
 macros['FuFrozenImageRecall']=pp(frozen_sel['metrics'][KEYS[3]]['mean']);macros['FuFrozenTextRecall']=pp(frozen_sel['metrics'][KEYS[4]]['mean'])
 write('followup_numbers.tex','\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for raw_k,v in macros.items() for k in [re.sub(r'[123]$', lambda m: {'1':'One','2':'Two','3':'Three'}[m.group()], raw_k)])+'\n')
 rows=[['--','Frozen']+[pp(frozen[k]['mean']) for k in KEYS[:3]]]
 for rate in LRS:
  for method in ['source','supported']:rows.append([lr(rate),LABELS[method]]+[pp(mean(method,rate,k)) for k in KEYS[:3]])
 write('followup_secondary_table.tex',table(['Rate','Positives','Relation','Edited','Both'],rows,'Mean accuracy (\\%) at matched epoch ten. Relation is e-SNLI-VE; Edited is SugarCrepe; Both requires both valid SugarCrepe++ captions above the foil. These are descriptive endpoints.','tab:secondary'))
 rows=[]
 for selector,label in [('native','Native'),('source_retrieval','Retrieval')]:
  for method in ['source','supported']:
   x=summaries[selector,method];states=x['selected_states'];rates={r['learning_rate'] for r in states};assert len(rates)==1
   rows.append([label,LABELS[method],'/'.join(str(r['epoch']) for r in states)]+[pp(x['metrics'][k]['mean']) for k in KEYS[3:5]])
 write('followup_selection_table.tex',table(['Selection','Positives','Epochs',r'I$\to$T',r'T$\to$I'],rows,'Selected procedures on the full source-caption test pool. Epochs follow seeds 17/29/43. Native supported uses learning rate $3\\cdot10^{-4}$; the other rows use $10^{-4}$. Epoch zero is frozen initialization.','tab:selection',placement='!b'))
 rows=[]
 for rate in LRS:
  for c,label in [('source','Source'),('count_only','Count'),('score_stratified','Strata')]:
   vals=[]
   for metric in ['i2t.r1','t2i.r1']:
    e=effects[rate,c,metric];vals += [f'{e["difference_percentage_points"]:+.3f}', '$['+','.join(f'{v:+.3f}' for v in e['ci_percentage_points'])+']$']
   rows.append([lr(rate),label]+vals)
 full=r'\section{Matched Assignment Study: Complete Results}\label{app:furesults}'+'\n'+table(['Rate','Comparator',r'$\Delta$ I$\to$T','99.7222\\% interval',r'$\Delta$ T$\to$I','99.7222\\% interval'],rows,'All 18 primary effects. Differences are supported minus comparator, in percentage points. Randomized comparators average all three assignment draws within each training seed before seed averaging. Intervals resample images and condition on the fitted seeds and draws.','tab:fuprimary',True,True)
 full+='\\subsection{Every terminal draw and training seed}\nEach row below is one epoch-ten fit. Count and Strata numbers identify assignment draws; no draw is selected for reporting. All entries are percentages.\\n'.replace('\\n','\n')
 rows=[]
 for method in METHODS:
  for rate in LRS:
   x=cells[method,rate,10];assert x['seeds']==[17,29,43]
   for i,seed in enumerate(x['seeds']):rows.append([LABELS[method],lr(rate),str(seed)]+[pp(x['metrics'][k]['values'][i]) for k in KEYS])
 for start in range(0,len(rows),18):
  full+=table(['Policy','Rate','Seed','Rel.','Edited','Both',r'e I$\to$T',r'e T$\to$I',r'C I$\to$T',r'C T$\to$I'],rows[start:start+18],'Terminal per-seed results. e denotes the full e-ViL source-caption pool and C denotes COCO. Other endpoints follow Table~\\ref{tab:secondary}.',f'tab:futerm{start//18}',True,True)
 full+='\\subsection{Both complete selection strategies}\n'
 rows=[]
 for selector in ['native','source_retrieval']:
  for method in METHODS:
   x=summaries[selector,method]
   for i,state in enumerate(x['selected_states']):rows.append(['Native' if selector=='native' else 'Retrieval',LABELS[method],str(state['seed']),str(state['epoch'])]+[pp(x['metrics'][k]['values'][i]) for k in KEYS])
 for start in range(0,len(rows),16):
  full+=table(['Selector','Policy','Seed','Ep.','Rel.','Edited','Both',r'e I$\to$T',r'e T$\to$I',r'C I$\to$T',r'C T$\to$I'],rows[start:start+16],'Every selected training-seed state. Each random draw has its own learning-rate and epoch selection; there is no winning-draw search. The source manifest records exact state IDs and learning rates.',f'tab:fusel{start//16}',True,True)
 trajectory_keys=KEYS[:3]+['sugarcrepe_pp.positive1_accuracy','sugarcrepe_pp.positive2_accuracy']+KEYS[3:5]
 rows=[]
 for method in ['source','supported','count_only','score_stratified']:
  for rate in LRS:
   for epoch in [1,5,10]:
    if method in ['source','supported']:x=cells[method,rate,epoch]
    else:x=next(z for z in aggregate['equal_draw_policy_means'] if z['condition']==method and z['learning_rate']==rate and z['epoch']==epoch)
    label={'source':'Source','supported':'Supported','count_only':'Count mean','score_stratified':'Strata mean'}[method]
    rows.append([label,lr(rate),epoch]+[pp(x['metrics'][k]['mean']) for k in trajectory_keys])
 full+='\\subsection{Complete descriptive trajectory means}\n'
 for start in range(0,len(rows),18):
  full+=table(['Policy','Rate','Ep.','Rel.','Edited','Both','First','Alternative',r'e I$\to$T',r'e T$\to$I'],rows[start:start+18],'All declared descriptive epoch means. First and Alternative are the separate valid-caption SugarCrepe++ accuracies. Random families average all three assignments equally within seed.',f'tab:futraj{start//18}',True,True)
 full+='\\subsection{Prespecified trajectories}\n\\begin{figure*}[t]\n\\centering\\includegraphics[width=\\textwidth]{followup_trajectories.pdf}\n\\caption{Descriptive source-pool retrieval at epochs one, five, and ten. Every learning rate is displayed. Randomized families average all assignment draws equally within seed. The primary family concerns epoch ten only.}\\label{fig:futrajectories}\n\\end{figure*}\n'
 write('followup_results_appendix.tex',full)
 for srcname,dstname in [('primary_effects_paper.pdf','followup_primary.pdf'),('primary_effects_paper.png','followup_primary.png'),('fixed_trajectories.pdf','followup_trajectories.pdf')]:
  src=ROOT/'results/review_followup/figures'/srcname;inputs.append({'path':str(src.relative_to(ROOT)),'sha256':sha(src)});dst=out/dstname;shutil.copy2(src,dst);outputs.append({'path':str(dst.relative_to(ROOT)),'sha256':sha(dst)})
 write('followup_primary_figure.tex',r'''\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{followup_primary.pdf}
\caption{All 18 primary comparisons at matched epoch ten. Points are supported promotion minus the named comparator. Bars are 99.7222\% paired image-cluster intervals, adjusted for the 18 comparisons and conditional on the fitted seeds and assignment draws. Random controls average three fixed assignments within each of three training seeds.}\label{fig:followupprimary}
\end{figure*}
''')
 receipt={'execution_kind':'review_followup','protocol_sha256':SHA,'analysis_inputs':inputs,'generator_sha256':sha(__file__),'generated_outputs':outputs,'original_manuscript_modified':False,'historical_aggregates_used':False}
 (out/'v3_results_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'outputs':len(outputs),'receipt':str((out/'v3_results_receipt.json').relative_to(ROOT))}))
if __name__=='__main__':main()

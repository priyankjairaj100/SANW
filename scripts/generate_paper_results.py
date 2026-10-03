#!/usr/bin/env python3
"""Generate ACL paper numbers, tables, and figures from the audited fresh rerun.

Usage from the repository root:
  python scripts/generate_paper_results.py \
    --analysis-dir results/analysis \
    --diagnostic results/relation_diagnostic_iter10000/summary.json
  python scripts/build_paper.py

Historical aggregates are never read. Output paths stay flat for Overleaf.
Every loaded input and generated output is hash-bound in paper_results_receipt.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['frozen', 'clip', 'multipositive', 'grounded_no_abstention',
         'grounded_no_hardening', 'grounded', 'random_exclusion', 'smoothing',
         'pairwise_rank', 'sanw_fixed', 'shuffled', 'sanw_median', 'constant']
LABELS = {'frozen': 'Frozen', 'clip': 'Source positives', 'multipositive': 'Positive expansion',
          'grounded_no_abstention': '+ contradiction weighting',
          'grounded_no_hardening': '+ neutral exclusion', 'grounded': '+ both',
          'random_exclusion': 'Random exclusion', 'smoothing': 'Target smoothing',
          'pairwise_rank': 'Pairwise ranking', 'sanw_fixed': 'Fixed similarity',
          'shuffled': 'Shuffled weights', 'sanw_median': 'Median similarity', 'constant': 'Constant weights'}
SHORT = {'frozen':'Frozen','clip':'Source','multipositive':'Expanded','grounded_no_abstention':'Contradiction',
         'grounded_no_hardening':'Neutral','grounded':'Combined','random_exclusion':'Random',
         'smoothing':'Smoothing','pairwise_rank':'Ranking','sanw_fixed':'Fixed','shuffled':'Shuffled',
         'sanw_median':'Median','constant':'Constant'}
KEYS = ['visual_entailment.accuracy','sugarcrepe.accuracy','sugarcrepe_pp.both_accuracy',
        'sugarcrepe_pp.positive1_accuracy','sugarcrepe_pp.positive2_accuracy','coco_karpathy.i2t.r1','coco_karpathy.t2i.r1']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(x, signed=False):
    return f'{100*x:+.2f}' if signed else f'{100*x:.2f}'


def number(x):
    return f'{x:,}' if isinstance(x, int) else f'{x:g}'


def table(headers, rows, caption, label, wide=False, spacing=4):
    env = 'table*' if wide else 'table'
    cols = 'l' + 'r' * (len(headers)-1)
    lines = [rf'\begin{{{env}}}[t]', r'\centering', rf'\setlength{{\tabcolsep}}{{{spacing}pt}}',
             rf'\begin{{tabular}}{{{cols}}}', r'\toprule', ' & '.join(headers) + r'\\', r'\midrule']
    lines += [' & '.join(map(str,row)) + r'\\' if row is not None else r'\midrule' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', rf'\caption{{{caption}}}', rf'\label{{{label}}}', rf'\end{{{env}}}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=ROOT/'results/analysis')
    parser.add_argument('--diagnostic', type=Path, default=ROOT/'results/relation_diagnostic_iter10000/summary.json')
    parser.add_argument('--selection', type=Path, default=ROOT/'results/study/selection.json')
    parser.add_argument('--output', type=Path, default=ROOT/'manuscript')
    parser.add_argument('--allow-incomplete-diagnostic', action='store_true', help='Draft generation only; diagnostic remains pending.')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    inputs={};outputs=[]
    def load(path):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT) or 'historical_context' in path.parts:
            raise ValueError('Inputs must be current project outputs, never historical aggregates.')
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    def write(name,text):
        path=out/name;path.write_text('% Generated from audited fresh results. Do not edit numeric content.\n'+text)
        outputs.append(path)
    audit=load(args.analysis_dir/'analysis_audit.json')
    if audit['status']!='passed' or audit['historical_aggregates_used'] or audit['run_count']!=37:
        raise ValueError('A complete audited fresh study is required.')
    aggregate=load(args.analysis_dir/'aggregate_metrics.json')
    contrasts=load(args.analysis_dir/'primary_contrasts.json')
    categories=load(args.analysis_dir/'sugarcrepe_category_descriptive.json')['groups']
    retention=load(args.analysis_dir/'retrieval_and_caption_retention.json')
    selection=load(args.selection)
    index=load(ROOT/'results/evaluation/index.json')
    if not index['complete_selected_study'] or index['evidence_type']!='new_execution':
        raise ValueError('Incomplete or non-research evaluation index.')
    if aggregate['source_index_sha256']!=sha(ROOT/'results/evaluation/index.json'):
        raise ValueError('Analysis is not bound to the current evaluation index.')
    if index['selection_sha256']!=sha(args.selection):
        raise ValueError('Evaluation is not bound to the current selected states.')
    if contrasts['source_index_sha256']!=aggregate['source_index_sha256']:
        raise ValueError('Primary intervals and aggregate metrics use different evaluations.')
    metrics=aggregate['methods']
    if set(metrics)!=set(ORDER):raise ValueError('Unexpected policy set.')
    protocol=load(ROOT/'docs/RERUN_PROTOCOL.json')
    ve=load(ROOT/'data/visual_entailment/provenance.json')
    input_audit=load(ROOT/'results/input_audit.json')
    overlap=load(ROOT/'data/benchmark_overlaps.json')
    source_audit=load(ROOT/'data/visual_entailment/official_release_audit.json')
    benchmark_prov={n:load(ROOT/f'data/{n}/provenance.json') for n in ['sugarcrepe','sugarcrepe_pp','coco_karpathy']}
    for n in ['visual_entailment','sugarcrepe','sugarcrepe_pp','coco_karpathy']:
        load(ROOT/f'results/features/{n}/metadata.json')
    def mean(m,k):return metrics[m]['metrics'][k]['mean']
    def cell(m,k,sd=False):
        d=metrics[m]['metrics'][k];val=percent(d['mean'])
        return f'${val}_{{\\pm {percent(d["seed_std"])}}}$' if sd and d['seed_std'] is not None else val
    def delta(a,b,k):return mean(a,k)-mean(b,k)
    macros={
      'PositiveRelationGain':percent(delta('multipositive','clip',KEYS[0])),
      'ConstantSugar':percent(mean('constant',KEYS[1])),
      'ConstantRetrieval':percent(mean('constant',KEYS[5])),
      'ConstantExpandedRetrievalGain':percent(delta('constant','multipositive',KEYS[5])),
      'ConstantRelation':percent(mean('constant',KEYS[0])),
      'ExpandedSugar':percent(mean('multipositive',KEYS[1])),
      'ExpandedRetrieval':percent(mean('multipositive',KEYS[5])),
      'ExpandedRelation':percent(mean('multipositive',KEYS[0])),
      'PositiveSugarGain':percent(delta('multipositive','clip',KEYS[1])),
      'PositiveRetrievalLoss':percent(-delta('multipositive','clip',KEYS[5])),
      'PositiveTextRetrievalLoss':percent(-delta('multipositive','clip',KEYS[6])),
      'ExtraFailedImageQueries':f"{sum(r['i2t']['net_additional_failures'] for r in retention['multipositive_vs_source_query_transitions'])/3:.0f}",
      'AddGain':percent(categories['added_content']['multipositive_minus_source']['mean']),
      'ReplaceSwapLoss':percent(-categories['combined_replace_swap']['multipositive_minus_source']['mean']),
      'NeutralRandomRelationGain':percent(delta('grounded_no_hardening','random_exclusion',KEYS[0])),
      'NeutralRandomSugarGain':percent(delta('grounded_no_hardening','random_exclusion',KEYS[1])),
      'ShuffleFixedRelationGain':percent(delta('shuffled','sanw_fixed',KEYS[0])),
      'ShuffleFixedSugarGain':percent(delta('shuffled','sanw_fixed',KEYS[1])),
      'MedianConstantSugarGap':percent(delta('sanw_median','constant',KEYS[1])),
      'ConstantRetrievalLoss':percent(-delta('constant','clip',KEYS[5])),
      'NonzeroAdapterCount':str(retention['nonzero_selected_adapter_count']),
      'ZeroAdapterCount':str(sum(r['epoch']==0 for r in retention['selected_adapters_vs_frozen'])),
      'AlternativeDeclineCount':str(sum(r['epoch']!=0 and r['p2_delta']<0 for r in retention['selected_adapters_vs_frozen'])),
      'OriginalImproveCount':str(sum(r['epoch']!=0 and r['p1_delta']>0 for r in retention['selected_adapters_vs_frozen'])),
      'FrozenPone':percent(mean('frozen',KEYS[3])), 'FrozenPtwo':percent(mean('frozen',KEYS[4])),
      'ExpandedPone':percent(mean('multipositive',KEYS[3])), 'ExpandedPtwo':percent(mean('multipositive',KEYS[4])),
      'RawAggregateChecks':str(audit['aggregate_checks']),
      'TrainImageCount':number(ve['statistics']['train']['images']),
      'CalibrationImageCount':number(ve['statistics']['calibration']['images']),
      'ValidationImageCount':number(ve['statistics']['validation']['images']),
      'TestImageCount':number(ve['statistics']['test']['images']),
      'SugarPairCount':number(benchmark_prov['sugarcrepe']['counts']['triplets']),
      'SugarImageCount':number(benchmark_prov['sugarcrepe']['counts']['images']),
      'SugarPpTripletCount':number(benchmark_prov['sugarcrepe_pp']['counts']['triplets']),
      'SugarPpImageCount':number(benchmark_prov['sugarcrepe_pp']['counts']['images']),
      'CocoImageCount':number(benchmark_prov['coco_karpathy']['counts']['images']),
      'CocoTextCount':number(benchmark_prov['coco_karpathy']['counts']['texts']),
      'EpochCount':str(protocol['training']['epochs']),
      'CandidateRunCount':str(len(protocol['training']['methods'])*len(protocol['training']['seeds'])*len(protocol['training']['learning_rates'])),
      'SelectedRunCount':str(selection['selected_checkpoint_count']),
      'TrainableParameterCount':number(protocol['adapter']['trainable_parameters']),
    }
    for contrast in contrasts['contrasts']:
        if contrast['method_b']=='grounded_no_abstention' and contrast['dataset']=='visual_entailment':
            lo,hi=contrast['ci_percentage_points']
            macros['NeutralUnderHardeningDelta']=f"{contrast['difference_percentage_points']:+.2f}"
            macros['NeutralUnderHardeningInterval']=f'[{lo:.2f},{hi:.2f}]'
        if contrast['method_b']=='multipositive':
            suffix='Relation' if contrast['dataset']=='visual_entailment' else 'Sugar'
            lo,hi=contrast['ci_percentage_points']
            macros['Combined'+suffix+'Delta']=f"{contrast['difference_percentage_points']:+.2f}"
            macros['Combined'+suffix+'Interval']=f'[{lo:.2f},{hi:.2f}]'
    write('numbers.tex','\n'.join(rf'\newcommand{{\{k}}}{{{v}}}' for k,v in macros.items())+'\n')
    core=ORDER[:6]+['constant']
    rows=[]
    for m in core:
        rows.append([LABELS[m],cell(m,KEYS[0],True),cell(m,KEYS[1],True),cell(m,KEYS[2]),cell(m,KEYS[5]),cell(m,KEYS[6])])
        if m=='frozen':rows.append(None)
    write('main_results_table.tex',table(['Policy','e-SNLI-VE','SugarCrepe','SC++ both',r'COCO I$\to$T',r'COCO T$\to$I'],rows,
      'Relation interventions with the same candidate pool and tuning budget. Scores are percentages; subscripts show sample standard deviation across three selected seeds for the primary endpoints. COCO columns report full-pool R@1. The three rows marked + add rules to positive expansion. Constant weights use source positives.',
      'tab:main',wide=True,spacing=4))
    rows=[]
    for group,label in [('added_content','Added content'),('combined_replace_swap','Replace/swap'),('all','All edits')]:
        g=categories[group];rows.append([label,number(g['items']),percent(g['methods']['clip']['mean']),percent(g['methods']['multipositive']['mean']),percent(g['multipositive_minus_source']['mean'],True)])
    write('edit_groups_table.tex',table(['Edit group','$n$','Source','Expand.',r'$\Delta$'],rows,
      'SugarCrepe accuracy by edit group. Differences are positive expansion minus source positives, in percentage points. Groups partition the official pairs.', 'tab:edits',spacing=3))
    rows=[[LABELS[m],cell(m,KEYS[0]),cell(m,KEYS[1]),cell(m,KEYS[5])] for m in ['sanw_fixed','shuffled','sanw_median']]
    write('weight_controls_table.tex',table(['Weights','e-SNLI-VE','Sugar',r'I$\to$T'],rows,
      r'Similarity weights and their shuffle, using source positives. Entries are mean percentages; the final column is COCO R@1. The constant-weight reference appears in Table~\ref{tab:main}.', 'tab:weights',spacing=3.5))
    split_rows=[]
    for s in ['train','calibration','validation','test']:
        d=ve['statistics'][s];split_rows.append([s.capitalize(),number(d['images']),number(d['pairs']['supported']),number(d['pairs']['contradicted']),number(d['pairs']['neutral'])])
    write('split_counts_table.tex',table(['Split','Images','Support','Contradict','Neutral'],split_rows,
       'Retained hypothesis annotations. Each image also has five source captions. All validation and test images, and 1,192 of 1,200 training images, have both supported and contradicted hypotheses.', 'tab:split-counts',spacing=2.3))
    # Comprehensive appendix tables preserve per-seed outcomes and selection.
    chunks=[]
    rows=[]
    for m in ORDER:
        rows.append([SHORT[m]]+[cell(m,k,True) for k in KEYS[:3]])
    chunks.append(table(['Policy','e-SNLI-VE','SugarCrepe','SC++ both'],rows,
      'All selected policies on the three discrimination endpoints. Entries are percentages with sample seed standard deviations. Source, Expanded, Contradiction, Neutral, and Combined correspond to the first five trainable policies in Table~\\ref{tab:main}. Other names follow Table~\\ref{tab:policies}.','tab:all-discrimination',wide=True))
    rows=[[SHORT[m]]+[cell(m,k,True) for k in KEYS[3:]] for m in ORDER]
    chunks.append(table(['Policy','SC++ first','SC++ alternative',r'COCO I$\to$T',r'COCO T$\to$I'],rows,
      'Valid-caption and retrieval retention. Entries are percentages with sample seed standard deviations.','tab:all-retention',wide=True,spacing=3))
    for start in (1,7):
        chosen=ORDER[start:start+6];rows=[]
        for m in chosen:
            for pos,run in enumerate(metrics[m]['runs']):
                rows.append([SHORT[m],run['seed'],f"{run['learning_rate']:g}",run['epoch']]+[percent(metrics[m]['metrics'][k]['seed_values'][pos]) for k in [KEYS[0],KEYS[1],KEYS[2],KEYS[5],KEYS[6]]])
        chunks.append(table(['Policy','Seed','Rate','Epoch','e-SNLI','Sugar','SC++',r'I$\to$T',r'T$\to$I'],rows,
          'Individual selected checkpoints. SC++ requires both positives to beat the foil; retrieval columns are full-pool R@1. All metric entries are percentages.',f'tab:seeds-{start}',wide=True,spacing=3))
    rows=[]
    for c in contrasts['contrasts']:
        endpoint='e-SNLI-VE' if c['dataset']=='visual_entailment' else 'SugarCrepe'
        lo,hi=c['ci_percentage_points'];rows.append([SHORT[c['method_b']],endpoint,f"{c['difference_percentage_points']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]']+[f'{100*x:+.3f}' for x in c['seed_differences']])
    chunks.append(table(['Comparator','Endpoint',r'$\Delta$',r'99.17\% interval','Seed 17','Seed 29','Seed 43'],rows,
      'Six primary contrasts: Combined minus the comparator, in percentage points. Intervals use paired image-cluster resampling and the fixed six-comparison Bonferroni family.', 'tab:primary',wide=True,spacing=3))
    cat_keys=['add_att','add_obj','replace_att','replace_obj','replace_rel','swap_att','swap_obj']
    rows=[[SHORT[m]]+[percent(categories[c]['methods'][m]['mean']) for c in cat_keys] for m in ORDER]
    chunks.append(table(['Policy','Add A','Add O','Repl. A','Repl. O','Repl. R','Swap A','Swap O'],rows,
      'SugarCrepe category means (percent). A, O, and R denote attribute, object, and relation. Category denominators are listed in Appendix~\\ref{app:provenance}.','tab:categories',wide=True,spacing=3))
    rows=[]
    for r in retention['multipositive_vs_source_query_transitions']:
        for direction in ['i2t','t2i']:
            d=r[direction];rows.append([r['seed'],r'I$\to$T' if direction=='i2t' else r'T$\to$I',number(d['queries']),number(d['lost_successes']),number(d['new_successes']),number(d['net_additional_failures'])])
    chunks.append(table(['Seed','Queries','Pool','Lost','New','Net loss'],rows,
      'Retrieval success transitions from source positives to positive expansion. Lost and new count changed rank-one successes on the identical query pool.','tab:transitions',wide=True,spacing=5))
    rows=[]
    for m in ORDER:
        rows.append([SHORT[m]]+[cell(m,'coco_karpathy.'+direction+'.r'+str(rank)) for direction in ['i2t','t2i'] for rank in [1,5,10]])
    chunks.append(table(['Policy',r'I$\to$T R1','R5','R10',r'T$\to$I R1','R5','R10'],rows,
      'Full-pool COCO recall at ranks one, five, and ten, averaged across selected seeds (percent).','tab:retrieval-full',wide=True,spacing=4))
    chunks.insert(0,r'Tables~\ref{tab:all-discrimination}--\ref{tab:retrieval-full} give all selected-policy means, seed outcomes, primary intervals, category scores, and retrieval transitions. Five selected states retain epoch zero: all three ranking runs and fixed-similarity seeds 29 and 43. Those states have predictions identical to the frozen reference.'+'\n\n')
    write('results_appendix.tex','\n'.join(chunks))

    # Illustrative retrieval query, chosen deterministically after scoring.
    import numpy as np
    coco_manifest=load(ROOT/'data/coco_karpathy/manifest.json')
    text_by_id={t['id']:t['text'] for t in coco_manifest['texts']}
    image_by_id={i['id']:i['path'] for i in coco_manifest['images']}
    def prediction(run_id):
        record=next(r for r in index['runs'] if r['run_id']==run_id)['datasets']['coco_karpathy']
        path=ROOT/record['predictions']
        if sha(path)!=record['predictions_sha256']:raise ValueError('Query example prediction hash mismatch.')
        inputs[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    source_predictions=[prediction(f'clip_seed_{s}') for s in [17,29,43]]
    expanded_predictions=[prediction(f'multipositive_seed_{s}') for s in [17,29,43]]
    candidate=np.logical_and.reduce([(a['i2t_ranks']==1)&(b['i2t_ranks']>1) for a,b in zip(source_predictions,expanded_predictions)])
    candidate &= expanded_predictions[0]['i2t_ranks']>5
    chosen=np.flatnonzero(candidate)
    if not len(chosen):raise ValueError('Declared illustrative query selection returned no example.')
    row=int(chosen[0]);sp=source_predictions[0];mp=expanded_predictions[0]
    iid=str(sp['image_ids'][row]);source_top=str(sp['text_ids'][sp['i2t_top_indices'][row,0]]);expanded_top=str(mp['text_ids'][mp['i2t_top_indices'][row,0]])
    original=ROOT/image_by_id[iid];inputs[str(original.relative_to(ROOT))]=sha(original)
    image_output=out/'retrieval_query.jpg';image_output.write_bytes(original.read_bytes());outputs.append(image_output)
    example={'image_id':iid,'seed':17,'selection':'First COCO manifest-order image correct under source positives in all three seeds, incorrect under positive expansion in all three seeds, and outside the top five under positive expansion in seed 17. Illustrative post hoc example; no statistical test.',
             'source_caption':text_by_id[source_top].strip(),'expanded_caption':text_by_id[expanded_top].strip(),
             'source_caption_id':source_top,'expanded_caption_id':expanded_top,
             'source_rank':int(sp['i2t_ranks'][row]),'expanded_rank':int(mp['i2t_ranks'][row]),'image_sha256':sha(original)}
    example_path=out/'retrieval_example_data.json';example_path.write_text(json.dumps(example,indent=2)+'\n');outputs.append(example_path)
    def tex_escape(text):
        for a,b in [('&',r'\&'),('%',r'\%'),('#',r'\#'),('_',r'\_'),('$',r'\$')]:text=text.replace(a,b)
        return text
    example_tex=(r'\begin{figure}[t]'+'\n'+r'\centering\includegraphics[width=0.78\columnwidth]{retrieval_query.jpg}'+'\n'+
      r'\par\raggedright\textbf{Source positives:} '+tex_escape(example['source_caption'])+'\n\n'+
      r'\textbf{Positive expansion:} '+tex_escape(example['expanded_caption'])+'\n'+
      r'\caption{Top retrieved captions for the same image query (seed 17). The best annotated caption moves from rank '+str(example['source_rank'])+' to '+str(example['expanded_rank'])+r'. Both models search the full caption pool.}\label{fig:query}'+'\n'+r'\end{figure}'+'\n')
    write('retrieval_example.tex',example_tex)
    write('retrieval_example_provenance.tex','The illustrative query in Figure~\\ref{fig:query} is '+iid.replace(':',r'\,')+'. It is the first image in COCO manifest order that source-positive adaptation retrieves correctly at rank one in all three seeds, while positive expansion fails at rank one in every seed and falls outside the top five in seed 17. The displayed captions are the two seed-17 top results. This deterministic post hoc choice illustrates a recorded failure and is not a representative sample or an extra significance test.\n')

    # Figure uses every nonzero selected state; no jitter hides exact locations.
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    colors={'clip':'#4b5563','multipositive':'#b91c1c','grounded':'#b91c1c','grounded_no_abstention':'#b91c1c','grounded_no_hardening':'#b91c1c','random_exclusion':'#b91c1c'}
    fig,ax=plt.subplots(figsize=(3.3,2.35),layout='constrained')
    points=[]
    for r in retention['selected_adapters_vs_frozen']:
        if r['epoch']==0:continue
        x,y=100*r['p1_delta'],100*r['p2_delta'];points.append({**r,'x_percentage_points':x,'y_percentage_points':y})
        ax.scatter(x,y,s=25,c=colors.get(r['method'],'#2563a6'),marker='s' if r['method']=='multipositive' else 'o',alpha=.82,linewidths=.4,edgecolors='white',zorder=3)
    ax.axhline(0,color='#777777',lw=.7);ax.axvline(0,color='#777777',lw=.7)
    ax.set_xlabel('First valid caption: change (pp)');ax.set_ylabel('Alternative caption: change (pp)')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,zorder=0)
    handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#4b5563',label='Source'),Line2D([0],[0],marker='o',color='w',markerfacecolor='#b91c1c',label='Expanded targets'),Line2D([0],[0],marker='o',color='w',markerfacecolor='#2563a6',label='Other controls')]
    ax.legend(handles=handles,loc='lower left',fontsize=6.5,frameon=False)
    for suffix in ['pdf','png']:
        path=out/('caption_retention.'+suffix);fig.savefig(path,dpi=200,metadata={'CreationDate':None} if suffix=='pdf' else None);outputs.append(path)
    plt.close(fig)
    figure_data=out/'figure_data.json';figure_data.write_text(json.dumps({'source':'results/analysis/retrieval_and_caption_retention.json','points':points},indent=2)+'\n');outputs.append(figure_data)
    # Validation trajectories expose endpoint/epoch-zero selection in the appendix.
    fig,axes=plt.subplots(4,3,figsize=(7,7.4),layout='constrained',sharex=True)
    for ax,m in zip(axes.ravel(),ORDER[1:]):
        lr=selection['methods'][m]['learning_rate']
        for run in selection['methods'][m]['runs']:
            path=ROOT/Path(run['candidate_checkpoint']).parent/'history.json'
            history=load(path)
            entries=history if isinstance(history,list) else history.get('epochs',history.get('history',[]))
            if entries:
                xs=[r['epoch'] for r in entries];ys=[r.get('validation',r).get('score') for r in entries]
                ax.plot(xs,ys,lw=1,label=str(run['seed']))
                hit=[i for i,x in enumerate(xs) if x==run['epoch']]
                if hit:ax.scatter([xs[hit[0]]],[ys[hit[0]]],s=12,zorder=3)
        ax.set_title(SHORT[m]+f' (lr={lr:g})',fontsize=8);ax.grid(alpha=.2);ax.tick_params(labelsize=7)
    axes[-1,0].set_xlabel('Epoch');axes[-1,1].set_xlabel('Epoch');axes[-1,2].set_xlabel('Epoch')
    axes[0,0].legend(fontsize=6,frameon=False);fig.supylabel('Validation selection score',fontsize=9)
    path=out/'validation_trajectories.pdf';fig.savefig(path,metadata={'CreationDate':None});outputs.append(path);plt.close(fig)
    write('validation_figure.tex',r'\begin{figure*}[t]'+'\n'+r'\centering\includegraphics[width=\textwidth]{validation_trajectories.pdf}'+'\n'+r'\caption{Validation trajectories at the selected learning rate of each policy. Lines are seeds; dots mark retained epochs. Every candidate evaluated epoch zero through ten. These are selection data, not held-out outcomes.}\label{fig:validation}'+'\n'+r'\end{figure*}'+'\n')
    # Diagnostic is separately hash-bound and must contain every mode for final output.
    diagnostic=load(args.diagnostic)
    complete=set(diagnostic.get('modes',{}))=={'image','text','multimodal'}
    if not complete and not args.allow_incomplete_diagnostic:
        raise ValueError('Diagnostic has not completed image, text and multimodal modes.')
    if complete:
        dchunks=[];rows=[]
        for mode in ['image','text','multimodal']:
            d=diagnostic['modes'][mode]
            for calibration in ['uncalibrated','calibrated']:
                m=d['metrics']['test'][calibration]
                rows.append([mode.capitalize(),calibration,percent(m['accuracy']),percent(m['macro_f1']),percent(m['ece_10_bins']),f"{m['brier_sum_classes']:.3f}",f"{m['nll']:.3f}"])
        dchunks.append(table(['Input','Probabilities','Acc.','Macro-F1','ECE','Brier','NLL'],rows,
          'Held-out relation classification on the newly sampled test set. Accuracy, macro-F1, and ECE are percentages; Brier and NLL retain their native scale.','tab:diagnostic',wide=True))
        rows=[]
        for mode in ['image','text','multimodal']:
            d=diagnostic['modes'][mode]
            for label in ['supported','contradicted']:
                g=d['test_gates'][label];interval=g['precision_image_bootstrap']['interval']
                rows.append([mode.capitalize(),label,str(g['accepted']),str(g['correct']),percent(g['coverage']),'n/a' if g['precision'] is None else percent(g['precision']),'n/a' if interval is None else '['+', '.join(percent(v) for v in interval)+']'])
        dchunks.append(table(['Input','Gate','Accept','Correct','Coverage','Precision',r'95\% interval'],rows,
          'Selective test predictions. Precision is undefined when no pair is accepted. Intervals resample whole test images and are descriptive.','tab:gates',wide=True,spacing=3))
        rows=[]
        for mode in ['image','text','multimodal']:
            d=diagnostic['modes'][mode];iterations=[i for c in d['selection']['candidates'] for i in c['iterations']]
            rows.append([mode.capitalize(),f"{d['selection']['selected_C']:g}",f"{d['temperature']['temperature']:.3f}",str(max(iterations)),percent(d['out_of_fold_metrics']['macro_f1'])])
        dchunks.append(table(['Input','$C$','Temp.','Max. iter.','OOF F1'],rows,
          'Classifier selection and completion. Maximum iteration count is over the full-training grid; OOF is image-disjoint out-of-fold macro-F1 (percent).','tab:diagnostic-selection',wide=True))
        write('diagnostic_results.tex','\n'.join(dchunks))
    else:
        write('diagnostic_results.tex',r'\pending{The amended diagnostic is still running. No incomplete diagnostic numbers are presented.}'+'\n')
    receipt={'schema_version':1,'execution_kind':'fresh_rerun','analysis_audit_status':audit['status'],
             'diagnostic_complete':complete,'historical_results_read':False,
             'generator_sha256':sha(__file__),'analysis_inputs':[{'path':p,'sha256':h} for p,h in sorted(inputs.items())],
             'generated_outputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in outputs]}
    (out/'paper_results_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'generated':len(outputs),'diagnostic_complete':complete,'receipt':str((out/'paper_results_receipt.json').relative_to(ROOT))},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Portable orchestration only. Frozen scientific programs remain authoritative."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = Path('/workspace/scratch/81995298881b/SANW_practical_work')
V = 'results/practical_v10/'
PROTOCOL = V + 'expanded_training_protocol_v1.json'
PROTOCOL_SHA = 'ce04032395f185519f3f5012a82d403ff41a7e99b8662e998c99028cf719a826'
GATE17 = V + 'replication_gate_seed17_v2.json'
GATE17_SHA = '882262b6db49d9e159bfd3824a4cbae83d717eadc52441cbe8f26f6a3606acc9'
AGGROOT = V + 'official_development_fixed_three_seed'
AGG = AGGROOT + '/result.json'
RET = V + 'retrieval_only_control_contract_v1.json'
LAB = V + 'expanded_labclip_control_contract_recovery_v2.json'
CORE = V + 'benchmark_core_lock_local_v1.json'
RETL = V + 'benchmark_retrieval_lock_local_v1.json'
LABL = V + 'benchmark_labclip_lock_local_v1.json'
RELEASE = V + 'benchmark_common_release_local_v1.json'
INPUTS = V + 'benchmark_inputs_local_v1.json'
COUT, ROUT, LOUT = [V + 'benchmark_local_' + x for x in ('core','retrieval','labclip')]
FSTATE = V + 'fresh_confirmation_state_source_lock_v1.json'
FFEATURE = V + 'fresh_confirmation_feature_lock_v1.json'
FRELEASE = V + 'fresh_confirmation_scoring_release_v1.json'
FOUT = V + 'fresh_confirmation_outcomes'
ENCODERS = ('vit_b32', 'rn50')
SEEDS = (17, 29, 43)
DATASETS = ('e_vil_test1000', 'coco_karpathy', 'sugarcrepe_pp')
PHASES = ('status','preflight','prepare','rn-replicas','audit-replicas','lock-replicas',
          'preserve-development','evaluate-replicas','aggregate','no-retention','retrieval-only','labclip',
          'prepare-benchmark','core-lock','audit-labclip','supplementary-locks','release-benchmark',
          'preserve-benchmark','score-benchmark','analyze-benchmark','fresh-lock','preserve-fresh-inputs','fresh-prepare',
          'fresh-encode','fresh-feature-lock','fresh-release','preserve-fresh','fresh-score','report')
SCRIPTS = {
 'fit':'scripts/run_practical_streaming_v10.py',
 'audit':'scripts/audit_practical_streaming_fullfits_v10.py',
 'dev':'scripts/evaluate_practical_official_development_v10.py',
 'agg':'scripts/aggregate_practical_official_development_v10.py',
 'retfit':'scripts/run_practical_retrieval_only_v10.py',
 'labfit':'scripts/run_practical_labclip_v10.py',
 'core':'scripts/evaluate_practical_benchmark_v10.py',
 'ret':'scripts/evaluate_practical_retrieval_only_benchmark_v10.py',
 'lab':'scripts/evaluate_practical_labclip_benchmark_v10.py',
 'fresh':'scripts/evaluate_practical_fresh_confirmation_v10.py',
 'encode':'recovery/training_expansion_private/encode_fresh_confirmation1500.py',
}


def path(value):
    p = (ROOT / value).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Path escapes repository: ' + str(value))
    return p


def digest(value):
    with path(value).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(value):
    return json.loads(path(value).read_text())


def rec(value):
    p = path(value)
    return {'path': str(p.relative_to(ROOT)), 'sha256': digest(value), 'bytes': p.stat().st_size}


def save_new(value, data):
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write('\n')


def run_dir(family, encoder, seed):
    group = 'full_pilots' if family == 'joint' and seed == 17 else 'full_replications' if family == 'joint' else 'full_controls'
    return V + f'{group}/{family}_{encoder}_seed{seed}'


def runs(family):
    return [run_dir(family,e,s) for e in ENCODERS for s in SEEDS]


def audit_path(encoder, seed):
    return V + f'independent_fullfit_audit_{encoder}_seed{seed}_v1.json'


def dev_lock(seed):
    return V + f'official_development_seed{seed}_lock_v2.json'


def dev_result(encoder, seed):
    return V + f'official_development/joint_{encoder}_seed{seed}/result.json'


def source_checks():
    checks = []
    for name,key in ((PROTOCOL,'source_sha256'),(V+'supplement_sources_ready_v1.json','source_sha256'),
                     (V+'fresh_confirmation_sources_ready_v2.json','source_sha256'),(V+'evaluation_sources_ready_v1.json','sources')):
        checks.extend((p,h) for p,h in read(name)[key].items())
    checks.extend(((PROTOCOL,PROTOCOL_SHA),(GATE17,GATE17_SHA)))
    return checks


def check_sources():
    checks = source_checks()
    bad = [p for p,h in checks if not path(p).is_file() or digest(p) != h]
    if bad:
        raise ValueError('Missing or changed frozen sources: ' + ', '.join(bad))
    return len(checks)


def require_pass(value):
    if read(value).get('passed') is not True:
        raise ValueError('Failed or absent gate: ' + value)


def require_compatibility():
    if not HISTORICAL.exists() or HISTORICAL.resolve() != ROOT:
        raise ValueError('Historical absolute paths are not mapped to this checkout. Use local/docker.sh.')


def require_runtime(encoding=False):
    import importlib.metadata as md
    want = {'numpy':'2.3.5','torch':'2.7.1+cpu','torchvision':'0.22.1+cpu','open_clip_torch':'2.32.0','Pillow':'12.3.0'} if encoding else {'numpy':'2.5.2','torch':'2.6.0+cpu'}
    actual = {name: md.version(name) for name in want}
    if platform.machine().lower() not in ('x86_64','amd64'):
        raise ValueError('Frozen local execution requires x86_64. Use the linux/amd64 Docker image.')
    if actual != want:
        raise ValueError('Runtime mismatch: ' + json.dumps({'required':want,'actual':actual}))
    if not encoding and sys.version_info[:3] != (3,12,14):
        raise ValueError('Fitting and evaluation require Python 3.12.14.')


@contextlib.contextmanager
def execution_lock():
    # Advisory process locks release automatically after interruption.
    import fcntl
    target = path('local/state/execution.lock')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another local workflow command is running.') from error
        yield


class Runner:
    def __init__(self, execute=False):
        self.execute = execute
        self.commands = []

    def sha(self, value):
        return digest(value) if path(value).is_file() else 'SHA256(' + value + ')'

    def call(self, key, *args, threads=1, output=None):
        command = [sys.executable, SCRIPTS.get(key,key), *map(str,args)]
        self.commands.append(command)
        print(f'[{threads} thread(s)] ' + shlex.join(command), flush=True)
        if self.execute:
            if output is not None and path(output).exists():
                raise FileExistsError('Output already exists; inspect before continuing: ' + output)
            env = os.environ.copy()
            env.update({name:str(threads) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
            env['PYTHONPATH'] = str(ROOT / 'src')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            log_name = 'local/logs/' + stamp + '.log'
            receipt_name = 'local/logs/' + stamp + '.json'
            path(log_name).parent.mkdir(parents=True, exist_ok=True)
            receipt = {'command': command, 'cwd': str(ROOT), 'threads': threads,
                       'started_at_utc': datetime.now(timezone.utc).isoformat(),
                       'runtime': environment_record(), 'log': log_name}
            with path(log_name).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    returncode = process.wait()
                    raise
                finally:
                    process.stdout.close()
                    receipt['returncode'] = process.returncode
                    receipt['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                    save_new(receipt_name, receipt)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)

    def gate(self, value):
        print('REQUIRE passed: ' + value)
        if self.execute:
            require_pass(value)

    def fit(self, family, encoder, seed, key, args):
        output = run_dir(family,encoder,seed)
        if path(output + '/completion.json').is_file():
            print('REUSE completed run; later audits must verify it: ' + output)
            return
        if path(output).exists():
            print('RETIRE incomplete output without deletion: ' + output)
            if self.execute:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                dest = path('local/retired/' + path(output).name + '_' + stamp)
                dest.parent.mkdir(parents=True, exist_ok=True)
                path(output).rename(dest)
        self.call(key,*args,'--output',output,threads=3,output=output)


def tree_files(values):
    found = set()
    for value in values:
        p = path(value)
        if not p.exists():
            raise FileNotFoundError(value)
        for child in p.rglob('*') if p.is_dir() else (p,):
            if child.is_file() and '__pycache__' not in child.parts:
                found.add(str(child.relative_to(ROOT)))
    return sorted(found)


def preservation_scope(stage):
    common = sorted({p for p,_ in source_checks()}) + [V+'official_development_source_audit_v2.json', V+'independent_fullfit_audit_seed17_combined_v1.json', V+'official_development_seed17_lock_v2.json', V+'official_development/independent_actual_result_audit_seed17_v1.json']
    if stage == 'development':
        return common + runs('joint') + [dev_lock(s) for s in (29,43)] + [audit_path(e,s) for e in ENCODERS for s in (29,43)]
    if stage == 'fresh-inputs':
        return common + runs('joint') + runs('no_retention') + [FSTATE, AGG, V+'fresh_confirmation_source_audit_v1.json']
    if stage == 'benchmark':
        return common + sum([runs(f) for f in ('joint','no_retention','retrieval_only','labclip')],[]) + [CORE,RETL,LABL,RELEASE]
    return common + [FSTATE,FFEATURE,FRELEASE,RELEASE,CORE,RETL,LABL] + sum([runs(f) for f in ('joint','no_retention','retrieval_only','labclip')],[])


def preserve(stage, execute):
    receipt = 'local/state/preserved_' + stage + '.json'
    archive = 'local/state/preserved_' + stage + '.tar.gz'
    print('PRESERVE exact states and locks: ' + archive)
    print('Copy this archive and its receipt to separate durable storage before proceeding.')
    if not execute:
        return
    if path(receipt).exists() or path(archive).exists():
        raise FileExistsError('Preservation already exists; verify it instead: ' + receipt)
    files = tree_files(preservation_scope(stage))
    rows = [rec(p) for p in files]
    path(archive).parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path(archive),'x:gz') as tar:
        for p in files:
            tar.add(path(p),arcname=p,recursive=False)
    save_new(receipt,{'stage':stage,'files':rows,'archive':rec(archive),
                     'remote_persistence_verified':False,'created_at_utc':datetime.now(timezone.utc).isoformat()})


def require_preserved(stage, execute):
    print('REQUIRE unchanged preserved states: ' + stage)
    if not execute:
        return
    value = read('local/state/preserved_' + stage + '.json')
    for row in [value['archive'],*value['files']]:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Preserved artifact changed: ' + row['path'])
    expected = tree_files(preservation_scope(stage))
    if {r['path'] for r in value['files']} != set(expected):
        raise ValueError('Preservation does not cover the current required state set.')


def build_phase(phase, r):
    protocol_args = ['--protocol',PROTOCOL,'--protocol-sha256',PROTOCOL_SHA]
    ret_args = ['--release',RELEASE,'--release-sha256',r.sha(RELEASE)]
    fresh_args = ['--release',FRELEASE,'--release-sha256',r.sha(FRELEASE)]
    if phase in ('rn-replicas','audit-replicas','lock-replicas','evaluate-replicas'):
        r.gate(GATE17)
    if phase in ('no-retention','retrieval-only','labclip','prepare-benchmark','core-lock','audit-labclip',
                  'supplementary-locks','release-benchmark','score-benchmark','analyze-benchmark',
                  'fresh-lock','fresh-prepare','fresh-encode','fresh-feature-lock','fresh-release','fresh-score'):
        r.gate(AGG)
    if phase == 'prepare':
        r.call('local/restore_science.py','--execute')
        r.call('local/assets.py','assemble','--group','all','--execute')
        r.call('local/assets.py','restore','--group','train-dev','--execute')
        r.call('local/assets.py','restore','--group','fits','--execute')
    elif phase == 'rn-replicas':
        for seed in (29,43):
            r.fit('joint','rn50',seed,'fit',['--protocol',PROTOCOL,'--family','joint','--encoder','rn50','--seed',seed,
                   '--replication-gate',GATE17,'--cache','local/cache/rn50'])
    elif phase == 'audit-replicas':
        for e in ENCODERS:
            for s in (29,43):
                out = audit_path(e,s)
                r.call('audit','--protocol',PROTOCOL,'--runs',run_dir('joint',e,s),'--output',out,output=out)
    elif phase == 'lock-replicas':
        for s in (29,43):
            for e in ENCODERS:
                r.gate(audit_path(e,s))
            out = dev_lock(s)
            r.call('dev','lock',*protocol_args,'--runs',*[run_dir('joint',e,s) for e in ENCODERS],'--output',out,output=out)
    elif phase == 'evaluate-replicas':
        require_preserved('development',r.execute)
        for s in (29,43):
            for e in ENCODERS:
                out = str(Path(dev_result(e,s)).parent)
                r.call('dev','evaluate','--lock',dev_lock(s),'--lock-sha256',r.sha(dev_lock(s)),
                       '--encoder',e,'--output',out,output=out)
    elif phase == 'aggregate':
        r.call('agg',*protocol_args,'--results',*[dev_result(e,s) for s in SEEDS for e in ENCODERS],
               '--output',AGGROOT,output=AGGROOT)
        r.gate(AGG)
    elif phase == 'no-retention':
        for e in ENCODERS:
            for s in SEEDS:
                r.fit('no_retention',e,s,'fit',['--protocol',PROTOCOL,'--family','no_retention','--encoder',e,'--seed',s,
                   '--control-gate',GATE17,'--cache','local/cache/'+e])
    elif phase == 'retrieval-only':
        completions = [p+'/completion.json' for p in runs('no_retention')]
        for e in ENCODERS:
            for s in SEEDS:
                r.fit('retrieval_only',e,s,'retfit',['fit','--contract',RET,'--encoder',e,'--seed',s,
                   '--development-gate',AGG,'--matched-control-completions',*completions,'--cache','local/cache/'+e])
    elif phase == 'labclip':
        for e in ENCODERS:
            for s in SEEDS:
                r.fit('labclip',e,s,'labfit',['fit','--contract',LAB,'--development-gate',AGG,'--encoder',e,'--seed',s])
    elif phase == 'prepare-benchmark':
        r.call('scripts/prepare_practical_benchmark_inputs_v10.py','restore','--archives','local/assets/archives',
               '--output',INPUTS,output=INPUTS)
    elif phase == 'core-lock':
        r.call('core','lock',*protocol_args,'--development-gate',AGG,'--inputs',INPUTS,
               '--source-audit',V+'benchmark_source_audit_v1.json','--runs',*runs('joint'),*runs('no_retention'),
               '--output',CORE,output=CORE)
    elif phase == 'audit-labclip':
        for e in ENCODERS:
            for s in SEEDS:
                out = V+f'labclip_training_audit_{e}_seed{s}_local_v1.json'
                r.call('lab','audit-training','--core-lock',CORE,'--core-lock-sha256',r.sha(CORE),
                       '--contract',LAB,'--run',run_dir('labclip',e,s),'--output',out,output=out)
    elif phase == 'supplementary-locks':
        core_args = ['--core-lock',CORE,'--core-lock-sha256',r.sha(CORE)]
        r.call('ret','lock',*core_args,'--contract',RET,'--source-audit',V+'retrieval_only_benchmark_source_audit_v1.json',
               '--runs',*runs('retrieval_only'),'--output',RETL,output=RETL)
        r.call('lab','lock',*core_args,'--contract',LAB,'--source-audit',V+'labclip_benchmark_source_audit_v1.json',
               '--runs',*runs('labclip'),'--training-audits',*[V+f'labclip_training_audit_{e}_seed{s}_local_v1.json' for e in ENCODERS for s in SEEDS],
               '--output-root',LOUT,'--output',LABL,output=LABL)
    elif phase == 'release-benchmark':
        r.call('ret','release','--lock',RETL,'--lock-sha256',r.sha(RETL),'--core-output-root',COUT,
               '--supplement-output-root',ROUT,'--additional-locks',LABL,'--output',RELEASE,output=RELEASE)
    elif phase == 'score-benchmark':
        require_preserved('benchmark',r.execute)
        for e in ENCODERS:
            for d in DATASETS:
                for key,mode in (('ret','score-core'),('ret','score'),('lab','score')):
                    r.call(key,mode,*ret_args,'--encoder',e,'--dataset',d)
    elif phase == 'analyze-benchmark':
        require_preserved('benchmark',r.execute)
        out = COUT+'/analysis'
        r.call('core','analyze','--lock',CORE,'--lock-sha256',r.sha(CORE),'--indices',
               *[COUT+'/'+e+'/'+d+'/index.json' for e in ENCODERS for d in DATASETS],'--output',out,output=out)
        for key,out in (('ret',ROUT+'/analysis'),('lab',LOUT+'/analysis')):
            r.call(key,'analyze',*ret_args,'--output',out,output=out)
    elif phase == 'fresh-lock':
        r.call('fresh','state-lock',*protocol_args,'--development-gate',AGG,'--source-audit',V+'fresh_confirmation_source_audit_v1.json',
               '--runs',*runs('joint'),*runs('no_retention'),'--output',FSTATE,output=FSTATE)
    elif phase in ('fresh-prepare','fresh-encode'):
        require_preserved('fresh-inputs',r.execute)
        args = ['--state-source-lock',FSTATE,'--lock-sha256',r.sha(FSTATE)]
        if phase == 'fresh-prepare':
            r.call('local/assets.py','acquire-raw','--execute')
            r.call('encode','prepare',*args)
        else:
            for e in ENCODERS:
                r.call('encode','encode',*args,'--encoder',e,'--threads',4)
    elif phase == 'fresh-feature-lock':
        r.call('fresh','feature-lock','--lock',FSTATE,'--lock-sha256',r.sha(FSTATE),'--completions',
               *['results/fresh_confirmation1500/'+e+'/completion.json' for e in ENCODERS],'--output',FFEATURE,output=FFEATURE)
    elif phase == 'fresh-release':
        r.call('fresh','release','--lock',FFEATURE,'--lock-sha256',r.sha(FFEATURE),'--planned-states-release',RELEASE,
               '--planned-states-release-sha256',r.sha(RELEASE),'--output-root',FOUT,'--output',FRELEASE,output=FRELEASE)
    elif phase == 'fresh-score':
        require_preserved('fresh',r.execute)
        for e in ENCODERS:
            r.call('fresh','score',*fresh_args,'--encoder',e,'--output',FOUT+'/'+e,output=FOUT+'/'+e)
        r.call('fresh','analyze',*fresh_args,'--indices',*[FOUT+'/'+e+'/index.json' for e in ENCODERS],
               '--output',FOUT+'/analysis',output=FOUT+'/analysis')
    elif phase.startswith('preserve-'):
        preserve(phase.removeprefix('preserve-'),r.execute)
    else:
        raise ValueError('Unknown execution phase: '+phase)
    return r.commands


def status():
    rows=[]
    for f in ('joint','no_retention','retrieval_only','labclip'):
        for e in ENCODERS:
            for s in SEEDS:
                p=run_dir(f,e,s)
                state='complete' if path(p+'/completion.json').is_file() else 'partial' if path(p).exists() else 'absent'
                rows.append({'family':f,'encoder':e,'seed':s,'state':state})
    print(json.dumps({'run_states':rows,'aggregate_present':path(AGG).exists(),
                      'historical_path_maps_here':HISTORICAL.exists() and HISTORICAL.resolve()==ROOT,
                      'completion_is_not_a_numerical_audit':True},indent=2))


def preflight():
    count=check_sources()
    missing=[]
    protocol=read(PROTOCOL)
    for key in ('training_inputs','original_training_inputs','development_inputs'):
        for entries in protocol[key].values():
            for record in entries.values():
                p,h=record['path'],record['sha256']
                if not path(p).exists() or digest(p)!=h:
                    missing.append(p)
    print(json.dumps({'frozen_hash_checks_passed':count,'missing_or_changed_inputs':sorted(set(missing)),
                      'historical_path_maps_here':HISTORICAL.exists() and HISTORICAL.resolve()==ROOT},indent=2))
    if missing:
        raise ValueError('Run prepare to restore exact training and development inputs.')
    require_compatibility()
    require_runtime()


def environment_record():
    import importlib.metadata as md
    packages = {}
    for name in ('numpy','torch','torchvision','open_clip_torch','Pillow','pytest'):
        try: packages[name] = md.version(name)
        except md.PackageNotFoundError: pass
    git = {}
    for name, args in (('commit',['rev-parse','HEAD']),('branch',['branch','--show-current']),
                       ('tracked_status',['status','--porcelain','--untracked-files=no'])):
        try:
            result = subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=False)
            git[name] = result.stdout.strip() if result.returncode == 0 else None
        except OSError: git[name] = None
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'machine':platform.machine(),'cpu_count':os.cpu_count(),'packages':packages,'git':git,
            'free_disk_bytes':shutil.disk_usage(ROOT).free}


def report_files(full_evidence=False):
    names=[]
    for p in path(V).rglob('*'):
        if not p.is_file() or any(part in ('frozen_cache','cache','__pycache__') for part in p.parts):
            continue
        if p.name.startswith('.') or p.suffix in ('.tmp','.part') or p.name == 'features.npz':
            continue
        rel=str(p.relative_to(ROOT))
        if full_evidence:
            if p.suffix in ('.json','.jsonl','.npz','.npy','.csv','.tsv','.txt','.log'):
                names.append(rel)
        elif 'checkpoints' not in p.parts and p.suffix == '.json':
            if p.name in ('result.json','analysis.json','completion.json','ledger.json','index.json','history.json') or any(x in p.name for x in ('audit','lock','release','gate')):
                names.append(rel)
    # Fresh input identities and annotations live outside the V10 outcome directory.
    # Feature matrices remain local; their metadata and completion hashes travel.
    for folder in ('results/fresh_confirmation1500','data/fresh_confirmation1500'):
        names.extend(str(p.relative_to(ROOT)) for p in path(folder).rglob('*.json') if p.is_file())
    names += [str(p.relative_to(ROOT)) for p in path('local/state').glob('*.json')] if path('local/state').exists() else []
    names += [str(p.relative_to(ROOT)) for p in path('local/logs').glob('*') if p.is_file()] if path('local/logs').exists() else []
    return sorted(set(names))


def report(execute, full_evidence=False):
    names=report_files(full_evidence)
    mode='FULL_EVIDENCE' if full_evidence else 'RESULTS'
    out='local/reports/SANW_'+mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.tar.gz'
    print('REPORT '+out)
    print(json.dumps({'mode':mode,'file_count':len(names),'input_bytes':sum(path(p).stat().st_size for p in names),
                      'fresh_features_raw_images_weights_and_caches_included':False},sort_keys=True))
    if execute:
        path(out).parent.mkdir(parents=True,exist_ok=True)
        environment_path = out + '.environment.json'
        save_new(environment_path,environment_record())
        names.append(environment_path)
        rows=[rec(p) for p in sorted(set(names))]
        with tarfile.open(path(out),'x:gz') as tar:
            for row in rows:
                tar.add(path(row['path']),arcname=row['path'],recursive=False)
        # Never describe a changing campaign snapshot as internally consistent.
        for row in rows:
            if digest(row['path']) != row['sha256']:
                raise ValueError('Evidence changed while packaging: '+row['path'])
        manifest={'archive':rec(out),'files':rows,'mode':mode,
                  'raw_prediction_and_bootstrap_arrays_included':full_evidence,
                  'fresh_feature_arrays_included':False,'raw_images_weights_and_caches_included':False,
                  'note':('Full stored V10 evidence, including selected states, checkpoints, predictions, paired arrays, and bootstrap arrays. Fresh input metadata and manifests included; feature matrices remain local.' if full_evidence else
                          'Compact review bundle. Retain complete raw outputs locally for independent numerical review.')}
        save_new(out+'.json',manifest)
        print(json.dumps(manifest['archive'],indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=PHASES)
    parser.add_argument('--execute',action='store_true',help='Run the printed commands. Default only prints the plan.')
    parser.add_argument('--full-evidence',action='store_true',help='Report only: include V10 numerical evidence and checkpoints.')
    args=parser.parse_args(argv)
    if args.full_evidence and args.phase != 'report':
        parser.error('--full-evidence is only valid for report')
    phase_started=datetime.now(timezone.utc)
    exit_code=0
    error_text=None
    try:
        if args.phase=='status': status()
        elif args.phase=='preflight': preflight()
        elif args.phase=='report':
            if args.execute:
                with execution_lock(): report(True,args.full_evidence)
            else: report(False,args.full_evidence)
        elif not args.execute: build_phase(args.phase,Runner())
        else:
            with execution_lock():
                if args.phase!='prepare':
                    check_sources()
                    require_compatibility()
                    require_runtime(encoding=args.phase=='fresh-encode')
                build_phase(args.phase,Runner(True))
    except (OSError,ValueError,RuntimeError,ImportError,subprocess.CalledProcessError) as error:
        error_text=str(error)
        print('STOP: '+error_text,file=sys.stderr)
        exit_code=1
    except BaseException as error:
        exit_code=130 if isinstance(error,KeyboardInterrupt) else 1
        error_text=repr(error)
        raise
    finally:
        if args.execute:
            receipt_name='local/logs/'+phase_started.strftime('%Y%m%dT%H%M%S%fZ')+'_phase.json'
            save_new(receipt_name,{'phase':args.phase,'full_evidence':args.full_evidence,
                                  'started_at_utc':phase_started.isoformat(),
                                  'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                                  'exit_code':exit_code,'error':error_text,
                                  'runtime':environment_record()})
    return exit_code

if __name__=='__main__':
    raise SystemExit(main())

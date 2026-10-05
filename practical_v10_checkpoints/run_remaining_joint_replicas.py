#!/usr/bin/env python3
"""Unbound, training-only queue for existing ViT29/43 followed by RN29/43.

No imports of research numerical code; no scoring, audits, packaging, restarts,
checkpoint selection, or source edits. Run only after reviewing this controller.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

BASE=Path(__file__).resolve().parent
REPO=BASE.parent/'SANW_practical_work'
RUNTIME=BASE.parent/'practical_python'
RUNNER=REPO/'scripts/run_practical_streaming_v10.py'
PROTOCOL=REPO/'results/practical_v10/expanded_training_protocol_v1.json'
PROTOCOL_SHA='ce04032395f185519f3f5012a82d403ff41a7e99b8662e998c99028cf719a826'
GATE=REPO/'results/practical_v10/replication_gate_seed17_v2.json'
GATE_SHA='882262b6db49d9e159bfd3824a4cbae83d717eadc52441cbe8f26f6a3606acc9'
RUN_BASE=REPO/'results/practical_v10/full_replications'
CACHE=REPO/'results/practical_v10/frozen_cache/rn50'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')
GIB=1<<30
POLICY={'samples':3,'poll_seconds':15,'rn_startup_reserve_bytes':int(4.5*GIB),'safety_bytes':int(.75*GIB),
        'accounting':'raw_cgroup_memory_current_includes_reclaimable_file_cache; no_cache_subtraction',
        'fallback_interpretation':'conservative_overlap_guard_not_proof_of_inadequate_RAM'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def run_path(encoder,seed):return RUN_BASE/f'joint_{encoder}_seed{seed}'
def record(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
def bound(entry,base=REPO):
    path=Path(entry['path']);path=(path if path.is_absolute() else base/path).resolve()
    if not path.is_relative_to(REPO) or sha(path)!=entry['sha256']:raise ValueError('Changed or escaping bound artifact')
    if 'bytes' in entry and path.stat().st_size!=entry['bytes']:raise ValueError('Changed bound byte count')
    return path


def fixed_inputs():
    if sha(PROTOCOL)!=PROTOCOL_SHA or sha(GATE)!=GATE_SHA:raise ValueError('Frozen protocol or approved seed17 gate changed')
    protocol,gate=read(PROTOCOL),read(GATE)
    if (protocol.get('study')!='sanw_practical_v10' or protocol.get('seeds')!=[17,29,43]
            or protocol['streaming']['fit_threads']!=3 or protocol['fit_config']['epochs']!=32
            or protocol['fit_config']['batch_size']!=64):raise ValueError('Frozen fit identity differs')
    for path,digest in protocol['source_sha256'].items():
        if sha(REPO/path)!=digest:raise ValueError('Frozen fitting source changed: '+path)
    if (gate.get('study')!='sanw_practical_v10_replication_gate' or gate.get('family')!='joint'
            or gate.get('passed') is not True or gate.get('seed')!=17 or gate.get('protocol_sha256')!=PROTOCOL_SHA
            or sorted(gate.get('encoders_passed',[]))!=['rn50','vit_b32']):raise ValueError('Wrong replication prerequisite')
    # Identity/byte validation only; the independent development audit is external.
    for item in gate['pilot_runs']:
        for key in ('completion','checkpoint','result'):bound(item[key])
    return protocol


def expected(encoder,seed):
    return {'repository':str(REPO),'protocol':str(PROTOCOL),'encoder':encoder,'seed':str(seed),'family':'joint',
            'output':str(run_path(encoder,seed)),'cache':str(CACHE if encoder=='rn50' else CACHE.parent/'vit_b32'),
            'replication-gate':str(GATE)}


def parse_runner(argv,cwd):
    positions=[i for i,x in enumerate(argv) if x.endswith('/run_practical_streaming_v10.py') or x=='run_practical_streaming_v10.py']
    if not positions:return None
    if len(positions)!=1:raise ValueError('Ambiguous runner process')
    pos=positions[0];path=Path(argv[pos]);path=(path if path.is_absolute() else cwd/path).resolve()
    if path!=RUNNER:return None
    values={'repository':str(REPO),'family':'joint','seed':'17'};seen=set();i=pos+1
    while i<len(argv):
        word=argv[i]
        if not word.startswith('--'):raise ValueError('Unexpected live runner argument')
        if '=' in word:key,value=word[2:].split('=',1);i+=1
        else:
            key=word[2:]
            if i+1>=len(argv):raise ValueError('Missing live runner option value')
            value=argv[i+1];i+=2
        if key in seen or key not in {'repository','protocol','encoder','seed','family','output','cache','replication-gate','control-gate'}:
            raise ValueError('Duplicate or unknown live runner option')
        seen.add(key);values[key]=value
    for key in ('repository','protocol','output','cache','replication-gate','control-gate'):
        if key in values:
            path=Path(values[key]);values[key]=str((path if path.is_absolute() else cwd/path).resolve())
    return values


def load_existing_attestation(path,digest,protocol):
    path=Path(path).resolve()
    if path.parent!=BASE or sha(path)!=digest:raise ValueError('Existing-process attestation identity differs')
    value=read(path)
    if (value.get('study')!='sanw_existing_vit_process_attestation_v1' or value.get('protocol_sha256')!=PROTOCOL_SHA
            or value.get('gate_sha256')!=GATE_SHA or len(value.get('processes',[]))!=2):
        raise ValueError('Existing-process attestation scope differs')
    result={};seeds=set()
    for item in value['processes']:
        seed=item['seed'];pid=item['pid']
        if (item.get('encoder')!='vit_b32' or seed not in (29,43) or seed in seeds or pid in result
                or type(pid) is not int or pid<=0 or type(item['start_ticks']) is not int
                or item.get('recorded_cwd')!=str(REPO)
                or item.get('recorded_threads')!={key:'3' for key in THREADS}
                or item.get('recorded_provenance')!='parent_launch_and_frozen_ledger_not_proc_observed'
                or parse_runner(item['argv'],REPO)!=expected('vit_b32',seed)):
            raise ValueError('Existing-process attestation is not the two fixed ViT jobs')
        ledger_path=run_path('vit_b32',seed)/'ledger.json'
        if item['ledger']!=record(ledger_path):raise ValueError('Attested ViT ledger bytes changed')
        validate_ledger(run_path('vit_b32',seed),'vit_b32',seed,protocol)
        raw=b'\0'.join(os.fsencode(x) for x in item['argv'])+b'\0'
        if hashlib.sha256(raw).hexdigest()!=item['cmdline_sha256']:raise ValueError('Attested raw argv hash differs')
        result[pid]=item;seeds.add(seed)
    if seeds!={29,43}:raise ValueError('Both existing ViT identities must be attested')
    return result


def attested_process(pid,start_ticks,argv,raw,attestations):
    item=(attestations or {}).get(int(pid))
    if (item is None or item['start_ticks']!=start_ticks or item['argv']!=argv
            or item['cmdline_sha256']!=hashlib.sha256(raw).hexdigest()
            or item['ledger']!=record(run_path('vit_b32',item['seed'])/'ledger.json')):
        raise PermissionError('Denied /proc access has no exact immutable existing-ViT attestation')
    return {'pid':int(pid),'start_ticks':start_ticks,'options':expected('vit_b32',item['seed']),
            'threads':item['recorded_threads'],'cwd_thread_provenance':'recorded_parent_launch_and_validated_ledger',
            'process_identity_provenance':'observed_proc_cmdline_and_start_ticks',
            'ledger_sha256':item['ledger']['sha256']}


def process_info(pid,attestations=None):
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    if fields[0]=='Z':return None
    raw=(proc/'cmdline').read_bytes();argv=[os.fsdecode(x) for x in raw.split(b'\0') if x]
    if not any(x.endswith('/run_practical_streaming_v10.py') or x=='run_practical_streaming_v10.py' for x in argv):return None
    try:
        cwd=Path(os.readlink(proc/'cwd'));environment=(proc/'environ').read_bytes()
    except PermissionError:
        return attested_process(pid,int(fields[19]),argv,raw,attestations)
    values=parse_runner(argv,cwd)
    if values is None:return None
    env={}
    for word in environment.split(b'\0'):
        key,sep,value=word.partition(b'=')
        if os.fsdecode(key) in THREADS:env[os.fsdecode(key)]=os.fsdecode(value)
    return {'pid':int(pid),'start_ticks':int(fields[19]),'options':values,'threads':env,
            'cwd_thread_provenance':'observed_proc_cwd_and_environ',
            'process_identity_provenance':'observed_proc_cmdline_and_start_ticks'}


def live_runners(attestations=None):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:item=process_info(path.name,attestations)
        except (FileNotFoundError,ProcessLookupError):continue
        if item:result.append(item)
    return result


def match_target(runners,encoder,seed):
    wanted=expected(encoder,seed);matches=[p for p in runners if p['options'].get('output')==wanted['output']]
    if len(matches)>1:raise RuntimeError('Duplicate runner processes share an output')
    if matches and (matches[0]['options']!=wanted or matches[0]['threads']!={key:'3' for key in THREADS}):
        raise RuntimeError('Live runner differs from frozen requested command or three-thread environment')
    return matches[0] if matches else None


def validate_ledger(path,encoder,seed,protocol):
    ledger=read(path/'ledger.json');identity=ledger['identity'];config={**protocol['fit_config'],'seed':seed}
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest()!=ledger['ledger_sha256']:raise ValueError('Ledger identity hash differs')
    if (identity.get('study')!='sanw_practical_v10' or identity.get('family')!='joint' or identity.get('mode')!='full'
            or identity.get('encoder')!=encoder or identity.get('config')!=config or identity.get('protocol_sha256')!=PROTOCOL_SHA
            or identity.get('source_sha256')!=protocol['source_sha256'] or identity.get('replication_gate',{}).get('sha256')!=GATE_SHA
            or identity.get('streaming')!=protocol['streaming'] or identity.get('environment',{}).get('blas_threads')!={key:'3' for key in THREADS}
            or identity.get('fit_gallery_image_count')!=6000 or identity.get('fit_gallery_text_count')!=30000
            or identity.get('official_development_or_benchmarks_used') is not False):raise ValueError('Run ledger violates fixed scope')
    bound(identity['protocol']);bound(identity['replication_gate'])
    provenance=identity['training_provenance']
    if provenance['inputs']!=protocol['training_inputs'][encoder] or provenance['heldout_used'] is not False:
        raise ValueError('Run training provenance differs')
    cache=identity['frozen_score_cache']
    if Path(cache['path']).resolve()!=Path(expected(encoder,seed)['cache']) or sha(Path(cache['path'])/'metadata.json')!=cache['metadata_sha256']:
        raise ValueError('Run cache metadata identity differs')
    return ledger


def validate_history(path,rows,ledger,*,complete):
    if not rows or [r['epoch'] for r in rows]!=list(range(1,len(rows)+1)) or len(rows)>32 or (complete and len(rows)!=32):
        raise ValueError('Training history has missing, reordered or excess epochs')
    for row in rows:
        if (row['optimizer_steps']!=94*row['epoch'] or not math.isfinite(row['training_objective'])
                or type(row['nonzero']) is not bool or not all(row['certificate'].get(k) is True for k in
                    ('ranking_checked_canonically','ranking_preserved','feasible_with_tolerance'))):raise ValueError('Invalid training epoch receipt')
        entry=row['checkpoint'];checkpoint=(path/entry['path']).resolve()
        if (not checkpoint.is_relative_to(path) or entry['ledger_sha256']!=ledger['ledger_sha256']
                or sha(checkpoint)!=entry['sha256']):raise ValueError('Epoch checkpoint identity differs')


def same_npz_arrays(left,right):
    with zipfile.ZipFile(left) as a,zipfile.ZipFile(right) as b:
        if sorted(a.namelist())!=sorted(b.namelist()):return False
        return all(a.read(name)==b.read(name) for name in a.namelist())


def completion(encoder,seed,protocol):
    path=run_path(encoder,seed)
    if not (path/'completion.json').exists():return None
    ledger=validate_ledger(path,encoder,seed,protocol);value=read(path/'completion.json');rows=value['checkpoint_history']
    validate_history(path,rows,ledger,complete=True)
    if (read(path/'history.json')!=rows or [{k:v for k,v in r.items() if k!='checkpoint'} for r in rows]!=value['history']
            or value.get('study')!='sanw_practical_v10' or value.get('family')!='joint' or value.get('mode')!='full'
            or value.get('encoder')!=encoder or value.get('config')!={**protocol['fit_config'],'seed':seed}
            or value.get('protocol_sha256')!=PROTOCOL_SHA or value.get('ledger_sha256')!=ledger['ledger_sha256']
            or value.get('optimizer_steps')!=3008):raise ValueError('Completion does not match full fixed fit')
    chosen=min((r for r in rows if r['nonzero']),key=lambda r:(r['training_objective'],r['epoch']))
    if (value['selection']!='minimum_feasible_nonzero_training_objective_then_earliest_epoch'
            or value['selected_epoch']!=chosen['epoch'] or value['selected_training_objective']!=chosen['training_objective']
            or not all(value['final_certificate'].get(k) is True for k in
                ('ranking_checked_canonically','ranking_preserved','feasible_with_tolerance'))):raise ValueError('Completion selection or certificate differs')
    selected=(path/value['selected_checkpoint']['path']).resolve()
    if (not selected.is_relative_to(path) or sha(selected)!=value['selected_checkpoint']['sha256']
            or not same_npz_arrays(selected,path/chosen['checkpoint']['path'])):raise ValueError('Selected bytes differ from selected training epoch')
    return {'encoder':encoder,'seed':seed,'completion':record(path/'completion.json'),'selected':record(selected),
            'epoch':chosen['epoch'],'validation_scope':'manifest_and_checkpoint_bytes_only_no_numerical_audit'}


def first_epoch(protocol):
    path=run_path('rn50',29)
    if not (path/'history.json').exists():return False
    ledger=validate_ledger(path,'rn50',29,protocol);rows=read(path/'history.json')
    validate_history(path,rows[:1],ledger,complete=False)
    return rows[0]['epoch']==1


def cache_status():
    if not CACHE.exists():return {'state':'absent'}
    if not CACHE.is_dir() or not (CACHE/'metadata.json').is_file():raise RuntimeError('Existing RN cache is incomplete; no overwrite or repair')
    value=read(CACHE/'metadata.json')
    if (value.get('schema')!='sanw_frozen_score_cache_v10' or value.get('images')!=6000 or value.get('texts')!=30000
            or value.get('dimension')!=1024 or value.get('dtype')!='float64'
            or set(value.get('files',{}))!={'image_to_text.npy','text_to_image.npy','ownership.npz'}):raise ValueError('RN cache metadata scope differs')
    for name,item in value['files'].items():
        if not (CACHE/name).is_file() or (CACHE/name).stat().st_size!=item['bytes']:raise RuntimeError('RN cache file absent or incomplete')
    return {'state':'complete_manifest','metadata':record(CACHE/'metadata.json'),
            'validation_scope':'metadata_and_file_sizes; frozen_runner_revalidates_full_content_hashes'}


def cgroup_path():
    entries=[line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    if len(entries)!=1:raise ValueError('Expected cgroup v2')
    path=Path('/sys/fs/cgroup')/entries[0].lstrip('/')
    if not (path/'memory.current').exists():raise ValueError('Cannot resolve own memory cgroup')
    return path


def memory_sample():
    try:
        cg=cgroup_path();limit=(cg/'memory.max').read_text().strip()
        if limit=='max':raise ValueError('No finite cgroup limit')
        stat={k:int(v) for k,v in (line.split() for line in (cg/'memory.stat').read_text().splitlines())}
        events={k:int(v) for k,v in (line.split() for line in (cg/'memory.events').read_text().splitlines())}
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        cpu=(cg/'cpu.max').read_text().split();cores=None if cpu[0]=='max' else int(cpu[0])/int(cpu[1])
        return {'valid':True,'cgroup':str(cg),'limit':int(limit),'current':int((cg/'memory.current').read_text()),
                'host_available':available,'swap_current':int((cg/'memory.swap.current').read_text()),'cpu_quota_cores':cores,
                'anon':stat.get('anon'),'file':stat.get('file'),'kernel':stat.get('kernel'),'events':events}
    except (OSError,ValueError,StopIteration) as error:return {'valid':False,'reason':str(error)}


def overlap_decision(samples):
    required=POLICY['rn_startup_reserve_bytes']+POLICY['safety_bytes'];details=[]
    if len(samples)!=POLICY['samples']:return False,{'reason':'insufficient_memory_samples','samples':samples,'policy':POLICY}
    for item in samples:
        if not item.get('valid'):return False,{'reason':'telemetry_unavailable_use_sequential','samples':samples,'policy':POLICY}
        headroom=min(item['limit']-item['current'],item['host_available'])
        details.append({**item,'raw_headroom':headroom,'required_additional_headroom':required})
        if headroom<required or item['swap_current']!=0 or item['cpu_quota_cores'] is None or item['cpu_quota_cores']<6:
            return False,{'reason':'conservative_raw_headroom_or_swap_or_cpu_guard_use_sequential',
                         'not_proof_of_inadequate_RAM':True,'samples':details,'policy':POLICY}
    return True,{'reason':'all_conservative_overlap_samples_passed','samples':details,'policy':POLICY,
                 'startup_reserve_is_planning_bound_not_measured_guarantee':True}


class Journal:
    def __init__(self,directory):self.directory=directory;self.log=(directory/'events.jsonl').open('x')
    def emit(self,event,**data):
        item={'at_utc':utc(),'event':event,**data}
        self.log.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');self.log.flush();os.fsync(self.log.fileno())
        tmp=self.directory/'status.tmp';tmp.write_text(json.dumps(item,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(self.directory/'status.json')
        print(json.dumps(item,sort_keys=True),flush=True)


def launch(seed,allow_build,journal,attestations=None):
    fixed_inputs();path=run_path('rn50',seed)
    if path.exists():raise FileExistsError('Refusing to overwrite or adopt RN output: '+str(path))
    cache=cache_status()
    if cache['state']=='absent' and (seed!=29 or not allow_build):raise RuntimeError('Completed RN cache required; explicit RN29-only cache-build permission missing')
    minimum_free=(2*6000*30000*8 if cache['state']=='absent' else 0)+(512<<20)
    if shutil.disk_usage(REPO).free<minimum_free:raise RuntimeError('Insufficient disk reserve for unchanged fit/cache writes')
    if match_target(live_runners(attestations),'rn50',seed):raise RuntimeError('RN target runner already live')
    command=[sys.executable,str(RUNNER)]
    for key,value in expected('rn50',seed).items():command.extend(['--'+key,value])
    env=os.environ.copy();env.update({key:'3' for key in THREADS});env['PYTHONPATH']=str(RUNTIME)+os.pathsep+str(REPO/'src')
    path.mkdir(parents=True,exist_ok=False)  # Reserve fresh output exclusively; runner permits an empty directory.
    log=journal.directory/f'rn50_seed{seed}.log'
    journal.emit('launch_intent',seed=seed,command=command,cache=cache,threads={key:env[key] for key in THREADS},
                 pythonpath=env['PYTHONPATH'],output=str(path),log=str(log),memory=memory_sample())
    with log.open('xb') as stream:child=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    child._controller_launched_at=time.monotonic()
    journal.emit('launched',seed=seed,pid=child.pid,output=str(path),log=str(log))
    return child


def run(args):
    directory=Path(args.state_directory).resolve()
    if directory.parent!=BASE:raise ValueError('Controller state directory must be a direct child of checkpoint root')
    with (BASE/'joint_replica_controller.lock').open('a+') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        directory.mkdir(exist_ok=False);journal=Journal(directory)
        children={};validated={};seen={};overlap=None;samples=[];rn_cache_sha=None
        try:
            protocol=fixed_inputs()
            attestations=load_existing_attestation(args.existing_vit_attestation,args.attestation_sha256,protocol)
            for seed in (29,43):
                if run_path('rn50',seed).exists():raise FileExistsError('Pre-existing RN target output; no overwrite or automatic adoption')
            initial_memory=memory_sample()
            journal.emit('started',controller=record(__file__),protocol=record(PROTOCOL),gate=record(GATE),
                         existing_vit_attestation=record(args.existing_vit_attestation),
                         allowed_jobs=[['vit_b32',29],['vit_b32',43],['rn50',29],['rn50',43]],memory=initial_memory,
                         policy=POLICY,allow_rn29_cache_build=args.allow_rn29_cache_build,
                         scope='training_only; no_evaluation_audits_packaging_or_Library_calls')
            while True:
                runners=live_runners(attestations);current=memory_sample()
                allowed={str(run_path(e,s)) for e,s in [('vit_b32',29),('vit_b32',43)]+[('rn50',s) for s in children]}
                if any(p['options'].get('output') not in allowed for p in runners):raise RuntimeError('Unexpected additional frozen-runner fit is live')
                if current.get('valid') and initial_memory.get('valid'):
                    if any(current['events'].get(k,0)>initial_memory['events'].get(k,0) for k in ('oom','oom_kill','oom_group_kill')):
                        raise RuntimeError('New cgroup OOM event; stop queue without launching further fits')
                active={}
                for encoder,seed in [('vit_b32',29),('vit_b32',43)]+[('rn50',s) for s in children]:
                    key=(encoder,seed);live=match_target(runners,encoder,seed);active[key]=live
                    if live:
                        process_key=live['pid'],live['start_ticks']
                        if key in seen and seen[key]!=process_key:raise RuntimeError('Runner PID/start-time identity changed; no automatic restart')
                        seen[key]=process_key
                    if encoder=='rn50':
                        code=children[seed].poll()
                        if code is not None and code!=0:raise RuntimeError(f'RN seed{seed} exited with code{code}; no retry')
                    if key not in validated:
                        value=completion(encoder,seed,protocol)
                        if value:validated[key]=value;journal.emit('completion_validated',**value)
                        elif not live:
                            # For newly launched direct children, /proc may briefly precede exec.
                            if (encoder!='rn50' or children[seed].poll() is not None
                                    or time.monotonic()-children[seed]._controller_launched_at>10):
                                raise RuntimeError(f'Incomplete {encoder} seed{seed} has no live frozen runner')
                if not children:
                    if all(('vit_b32',s) in validated and active[('vit_b32',s)] is None for s in (29,43)):
                        children[29]=launch(29,args.allow_rn29_cache_build,journal,attestations)
                elif 43 not in children:
                    done=('rn50',29) in validated and children[29].poll()==0
                    if done:
                        if rn_cache_sha is not None and sha(CACHE/'metadata.json')!=rn_cache_sha:
                            raise RuntimeError('Shared cache metadata changed before sequential second launch')
                        journal.emit('rn43_schedule',mode='sequential_after_validated_rn29',overlap_decision=overlap)
                        children[43]=launch(43,False,journal,attestations)
                    elif overlap is None and first_epoch(protocol):
                        cache=cache_status()
                        if cache['state']!='complete_manifest':raise RuntimeError('RN first epoch lacks complete shared cache')
                        if rn_cache_sha is not None and cache['metadata']['sha256']!=rn_cache_sha:
                            raise RuntimeError('Shared cache metadata changed during overlap sampling')
                        rn_cache_sha=cache['metadata']['sha256'];samples.append(current)
                        if len(samples)==POLICY['samples']:
                            allowed_overlap,decision=overlap_decision(samples);overlap=decision
                            journal.emit('overlap_decision',allowed=allowed_overlap,**decision)
                            if allowed_overlap:
                                if sha(CACHE/'metadata.json')!=rn_cache_sha:raise RuntimeError('Shared cache metadata changed before second launch')
                                children[43]=launch(43,False,journal,attestations)
                if set(children)=={29,43} and all(('rn50',s) in validated and children[s].poll()==0 for s in (29,43)):
                    journal.emit('complete',fits=[validated[e,s] for e,s in [('vit_b32',29),('vit_b32',43),('rn50',29),('rn50',43)]],
                                 evaluation_or_control_started=False,overlap_decision=overlap)
                    return
                journal.emit('waiting',validated=[f'{e}:{s}' for e,s in validated],live=runners,
                             child_returncodes={str(s):p.poll() for s,p in children.items()},memory=current,overlap_decision=overlap)
                time.sleep(POLICY['poll_seconds'])
        except BaseException as error:
            journal.emit('failed_closed',error_type=type(error).__name__,error=str(error),
                         child_pids={str(s):p.pid for s,p in children.items()},
                         children_left_running=True,no_more_jobs_will_launch=True)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Explicitly authorize the reviewed fixed training queue')
    parser.add_argument('--allow-rn29-cache-build',action='store_true',help='Allow only RN29 frozen runner to create an absent cache')
    parser.add_argument('--existing-vit-attestation',required=True,help='Reviewed immutable existing ViT-only process records')
    parser.add_argument('--attestation-sha256',required=True)
    parser.add_argument('--state-directory',default=str(BASE/'joint_replica_controller_run_v1'))
    args=parser.parse_args()
    if not args.execute:parser.error('Review first; --execute is required. No controller work has started.')
    run(args)

if __name__=='__main__':main()

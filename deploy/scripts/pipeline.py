#!/usr/bin/env python3
"""Login-node-safe submission; all installation and benchmark work is allocated."""
import argparse, datetime, hashlib, math, json, fcntl, os, pathlib, re, shutil, signal, socket, subprocess, sys, tempfile, time, uuid
ROOT = pathlib.Path(__file__).resolve().parents[2]

def required_path(value):
    if not value.strip(): raise argparse.ArgumentTypeError('Path cannot be empty')
    return pathlib.Path(value)

def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def atomic(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(obj, f, indent=2); f.write('\n'); name=f.name
    os.replace(name, path)
def read(path): return json.loads(pathlib.Path(path).read_text())
def receipt(run, stage): return run/'receipts'/f'{stage}.json'
def update(run, stage, **fields):
    p=receipt(run,stage); p.parent.mkdir(parents=True,exist_ok=True)
    with (p.parent/f'{stage}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        obj=read(p) if p.exists() else {}; obj.update(fields, updated=stamp()); atomic(p,obj)
def command(args, **kw):
    # Keep child cleanup inside the allocation, including uv's application process.
    capture=kw.pop('capture_output',False)
    if capture: kw.update(stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    with subprocess.Popen(args,text=True,start_new_session=True,**kw) as process:
        try:
            stdout,stderr=process.communicate()
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
            raise
        result=subprocess.CompletedProcess(args,process.returncode,stdout,stderr)
        result.check_returncode(); return result
def scheduler(job):
    p=subprocess.run(['squeue','-h','-j',job,'-o','%T|%R'],capture_output=True,text=True)
    if p.returncode==0 and p.stdout.strip():
        state,reason=p.stdout.strip().split('\n')[0].split('|',1); return state,'',reason
    p=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True)
    if p.returncode==0:
        for line in p.stdout.splitlines():
            fields=line.split('|')
            if len(fields)>=3 and fields[0]==job: return fields[1].split()[0],fields[2],''
    return 'UNKNOWN','','Accounting not yet available'
def artifacts(run,stage):
    directory=run/('data' if stage=='prepare' else 'results')
    try:
        if stage=='prepare':
            marker=read(directory/'PREPARED.json'); manifest=read(directory/'manifest.json')
            if marker['manifest_sha256']!=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(): return False
            return all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected for name,expected in manifest['hashes'].items())
        marker=read(directory/'SUCCEEDED.json')
        return marker['summary_sha256']==hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest() and read(directory/'config.json')['manifest_hash']==hashlib.sha256((run/'data'/'manifest.json').read_bytes()).hexdigest() and artifacts(run,'prepare')
    except (OSError,KeyError,ValueError): return False
def dependency(run,stage):
    p=receipt(run,stage)
    if not p.exists(): return None
    r=read(p); job=r.get('job_id')
    if not job: raise RuntimeError(f'{stage}: unresolved submission; inspect {p}; do not blindly retry')
    state,code,reason=scheduler(job)
    if state=='COMPLETED' and code=='0:0':
        if r.get('application_state')!='SUCCEEDED' or not artifacts(run,stage): raise RuntimeError(f'{stage}: scheduler completed but application/artifacts unverified')
        return 'complete'
    if state in {'PENDING','RUNNING','CONFIGURING','COMPLETING'}: return job
    raise RuntimeError(f'{stage}: dependency blocked: {state}|{code} {reason}')
def submit(run,stage,cfg,dep=None):
    if stage=='evaluate':
        predecessor=dependency(run,'prepare')
        if predecessor is None: raise RuntimeError('Preparation submission missing')
        dep=None if predecessor=='complete' else predecessor
    ts=stamp(); token=uuid.uuid4().hex; out=run/'logs'/f'{stage}-{ts}-%j.out'; err=run/'logs'/f'{stage}-{ts}-%j.err'
    args=['sbatch','--parsable',f'--chdir={ROOT}',f'--job-name=evidence-{stage}-{token[:10]}',f'--output={out}',f'--error={err}',f'--mem={cfg["memory"]}',f'--cpus-per-task={cfg["cpus"]}',f'--time={cfg[stage+"_time"]}']
    for key in ('account','partition','qos','reservation'):
        if cfg[key]: args.append(f'--{key}={cfg[key]}')
    if stage=='evaluate': args.append(f'--gres={cfg["gres"]}')
    if dep: args.append(f'--dependency=afterok:{dep}')
    args += [str(ROOT/'deploy/scripts'/f'{stage}.sbatch'),str(run)]
    update(run,stage,application_state='SUBMITTING',job_id=None,submission_token=token,submission_time=ts,command=args,run_dir=str(run),stdout=str(out),stderr=str(err))
    p=subprocess.run(args,capture_output=True,text=True)
    (run/'logs'/f'{stage}-{ts}-submission.stdout').write_text(p.stdout)
    (run/'logs'/f'{stage}-{ts}-submission.stderr').write_text(p.stderr)
    if p.returncode:
        update(run,stage,application_state='SUBMISSION_REJECTED',exit_code=p.returncode,reason=p.stderr); raise RuntimeError(p.stderr)
    if not re.fullmatch(r'\d+(?:;[^\s;]+)?',p.stdout.strip()):
        update(run,stage,application_state='UNKNOWN',reason='Ambiguous sbatch response; reconcile submission token before retry'); raise RuntimeError('Ambiguous submission; do not resubmit blindly')
    job=p.stdout.strip().split(';')[0]
    update(run,stage,job_id=job,application_state='SUBMITTED',stdout=str(out).replace('%j',job),stderr=str(err).replace('%j',job))
    return job

def monitor(run,stage,seconds):
    job=read(receipt(run,stage))['job_id']; deadline=time.monotonic()+seconds
    while True:
        state,code,reason=scheduler(job); update(run,stage,scheduler_state=state,scheduler_exit=code,scheduler_reason=reason)
        r=read(receipt(run,stage))
        if r.get('application_state') in {'STARTED','SUCCEEDED'} and (state in {'RUNNING','COMPLETING'} or (state=='COMPLETED' and code=='0:0' and r.get('application_state')=='SUCCEEDED' and artifacts(run,stage))):
            print(f'{stage}: application {r["application_state"]}; scheduler {state}|{code}; receipt {receipt(run,stage)}'); return
        if state in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'} or (state=='COMPLETED' and (code!='0:0' or r.get('application_state')!='SUCCEEDED')):
            print(json.dumps(r,indent=2))
            log=pathlib.Path(r['stderr'])
            if log.exists(): print('\n'.join(log.read_text(errors='replace').splitlines()[-40:]))
            raise RuntimeError(f'{stage}: terminal {state}|{code}')
        if time.monotonic()>=deadline:
            print(f'{stage}: {state}: {reason}; startup not verified. Job {job}; receipt {receipt(run,stage)}. Keep queued job; do not blindly resubmit.'); return
        time.sleep(min(2,max(0,deadline-time.monotonic())))

def worker(stage,run):
    cfg=read(run/'manifest.json'); job=os.environ.get('SLURM_JOB_ID')
    if not job: raise RuntimeError('No Slurm allocation')
    for attempt in range(20):
        if read(receipt(run,stage)).get('job_id')==job: break
        time.sleep(0.1)
    else: raise RuntimeError('Allocation job ID does not match submission receipt')
    details=command(['scontrol','show','job',job],capture_output=True).stdout
    if 'JobState=RUNNING' not in details: raise RuntimeError('Active allocation not verified')
    if pathlib.Path(cfg['project'])!=ROOT: raise RuntimeError('Project location changed since submission')
    def fail(signum,frame): raise RuntimeError(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,fail); signal.signal(signal.SIGINT,fail)
    try:
        env=os.environ.copy(); env.update(PYTHONUNBUFFERED='1',HF_HOME=cfg['cache_dir'])
        if cfg['modules']:
            # Module names validated by submit CLI; no user shell code is accepted.
            module_cmd='set -Eeuo pipefail; module load '+ ' '.join(cfg['modules'])+'; exec "$@"'
            def execute(args): return command(['bash','-lc',module_cmd,'bash',*args],cwd=ROOT,env=env)
        else:
            def execute(args): return command(args,cwd=ROOT,env=env)
        if stage=='prepare': execute(['uv','sync','--frozen'])
        else:
            if dependency(run,'prepare')!='complete': raise RuntimeError('Preparation must be verified COMPLETED|0:0 with artifacts')
            execute(['uv','run','--no-sync','python','-c','import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(p); assert p.total_memory >= '+str(cfg['min_vram_gb'])+'*1024**3; assert not '+str(cfg['require_bf16'])+' or torch.cuda.is_bf16_supported()'])
        marker={'job_id':job,'host':socket.gethostname(),'timestamp':stamp(),'run_dir':str(run),'allocation':details,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
        atomic(run/'receipts'/f'{stage}-ALLOCATED_READY.json',marker); update(run,stage,application_state='ALLOCATED_READY',allocation=marker)
        env.update(EVIDENCE_RUN_DIR=str(run),EVIDENCE_STAGE=stage,EVIDENCE_ALLOCATION_JSON=json.dumps(marker))
        args=['uv','run','--no-sync','python','-m','one_shot_evidence_retrieval.benchmark',stage]
        if stage=='prepare': args += ['--dataset','hotpotqa','--output',str(run/'data'),'--cache-dir',cfg['cache_dir'],'--model',cfg['model'],'--limit',str(cfg['limit'])]
        else:
            args += ['--data',str(run/'data'),'--output',str(run/'results'),'--model',cfg['model'],'--device','cuda']
            if (run/'results'/'config.json').is_file(): args.append('--resume')
            for key in ('top_k','bundle_size','max_bundles','max_tokens','limit'): args += ['--'+key.replace('_','-'),str(cfg[key])]
        update(run,stage,application_command=args); execute(args)
        if not artifacts(run,stage): raise RuntimeError('Expected output artifacts missing')
        update(run,stage,application_state='SUCCEEDED',exit_code=0)
    except BaseException as exc:
        update(run,stage,application_state='FAILED',exit_code=1,reason=str(exc)); raise

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('submit'); s.add_argument('--output',type=required_path); s.add_argument('--resume',type=required_path)
    for name,default in [('model','Qwen/Qwen3-4B'),('cache-dir',None),('account',None),('partition',None),('qos',None),('reservation',None),('gres','gpu:h100:1'),('memory','128G'),('prepare-time','02:00:00'),('evaluate-time','04:00:00')]: s.add_argument('--'+name,default=default)
    for name,default in [('cpus',8),('top-k',5),('bundle-size',2),('max-bundles',30),('max-tokens',2048),('limit',20),('min-vram-gb',16),('min-free-gb',30)]: s.add_argument('--'+name,type=int,default=default)
    s.add_argument('--module',action='append',default=[]); s.add_argument('--require-bf16',action='store_true'); s.add_argument('--monitor-seconds',type=int,default=10)
    for name in ('status','results'):
        x=sub.add_parser(name); x.add_argument('run',type=required_path)
    w=sub.add_parser('worker'); w.add_argument('stage',choices=['prepare','evaluate']); w.add_argument('run',type=required_path)
    a=p.parse_args()
    if a.action=='worker': worker(a.stage,a.run.resolve()); return
    if a.action in ('status','results'):
        run=a.run.resolve(); read(run/'manifest.json')
        if a.action=='status':
            for stage in ('prepare','evaluate'):
                p=receipt(run,stage)
                if not p.exists(): print(stage, 'not submitted'); continue
                r=read(p); print(stage,json.dumps(r,indent=2)); print('scheduler:',scheduler(r['job_id']) if r.get('job_id') else 'unresolved')
        else:
            if dependency(run,'evaluate')!='complete': raise RuntimeError('Results not verified complete')
            print('\n'.join(str(x) for x in (run/'results').rglob('*') if x.is_file()))
        return
    if bool(a.output)==bool(a.resume): p.error('Specify exactly one of --output or --resume')
    if not 0<=a.monitor_seconds<=60: p.error('--monitor-seconds must be 0..60')
    for key in ('cpus','top_k','bundle_size','max_bundles','max_tokens','limit','min_vram_gb','min_free_gb'):
        if getattr(a,key)<=0: p.error(key+' must be positive')
    if a.bundle_size>a.top_k: p.error('bundle-size cannot exceed top-k')
    if math.comb(a.top_k,a.bundle_size)>a.max_bundles: p.error('max-bundles must cover all fixed-size bundles')
    for stage in ('prepare','evaluate'):
        if not (ROOT/'deploy/scripts'/f'{stage}.sbatch').is_file(): p.error('Missing stage script '+stage)
    if not (ROOT/'uv.lock').is_file(): p.error('Missing lock file')
    if not re.fullmatch(r'[1-9][0-9]*[KMGTP]?',a.memory): p.error('Invalid memory resource')
    if not re.fullmatch(r'(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]',a.prepare_time) or not re.fullmatch(r'(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]',a.evaluate_time): p.error('Invalid time resource')
    if not re.fullmatch(r'gpu(?::[A-Za-z0-9_.-]+)?:[1-9][0-9]*',a.gres): p.error('Invalid GPU resource')
    for module in a.module:
        if not re.fullmatch(r'[A-Za-z0-9_.+/-]+',module): p.error('Invalid module name')
    for exe in ('sbatch','squeue','sacct','scontrol','bash'):
        if not shutil.which(exe): p.error('Missing executable: '+exe)
    if a.resume:
        run=a.resume.resolve(); cfg=read(run/'manifest.json')
        if cfg['project']!=str(ROOT): p.error('Resume project mismatch')
    else:
        if not a.account: p.error('--account is required for a fresh submission')
        if not a.cache_dir: p.error('--cache-dir must point to site-approved shared storage')
        cache=pathlib.Path(a.cache_dir).expanduser().resolve(); cache.mkdir(parents=True,exist_ok=True)
        if not os.access(cache,os.W_OK) or shutil.disk_usage(cache).free<a.min_free_gb*1024**3: p.error('Cache not writable or insufficient free cache space')
        base=a.output.expanduser().resolve(); base.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(base.parent).free<a.min_free_gb*1024**3: p.error('Insufficient free output space')
        for n in range(10000):
            run=base if n==0 else base.with_name(base.name+f'-{n:03}')
            try: run.mkdir(); break
            except FileExistsError: continue
        else: p.error('Could not claim run directory')
        cfg=vars(a).copy(); cfg.pop('action'); cfg.pop('output'); cfg.pop('resume'); cfg['modules']=cfg.pop('module'); cfg['project']=str(ROOT); cfg['cache_dir']=str(pathlib.Path(a.cache_dir).expanduser().resolve()); cfg['created']=stamp(); atomic(run/'manifest.json',cfg)
    if not cfg['modules'] and not shutil.which('uv'): p.error('uv must be on PATH or supplied through a module')
    (run/'logs').mkdir(exist_ok=True); print('RUN_DIR='+str(run),flush=True)
    submit_lock=(run/'submit.lock').open('a')
    try: fcntl.flock(submit_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another launcher owns this run; no duplicate submission attempted')
    prep=dependency(run,'prepare')
    if prep is None: prep=submit(run,'prepare',cfg)
    try:
        evaluation=dependency(run,'evaluate')
    except RuntimeError:
        if not a.resume or prep!='complete': raise
        old=read(receipt(run,'evaluate')); job=old.get('job_id')
        if not job: raise
        state,code,reason=scheduler(job)
        if state not in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}: raise
        # Explicit resume is authorized; archive the failed receipt and startup marker.
        archive=run/'receipts'/'attempts'; archive.mkdir(exist_ok=True)
        atomic(archive/f'evaluate-{job}.json',old)
        started=run/'receipts'/'evaluate-STARTED.json'
        if started.exists(): os.replace(started,archive/f'evaluate-{job}-STARTED.json')
        evaluation=None
    if evaluation is None: submit(run,'evaluate',cfg,None if prep=='complete' else prep)
    for stage in ('prepare','evaluate'): monitor(run,stage,a.monitor_seconds)

if __name__=='__main__':
    try: main()
    except Exception as exc: print('ERROR:',exc,file=sys.stderr); sys.exit(1)

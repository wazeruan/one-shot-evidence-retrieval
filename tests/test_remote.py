"""Scheduler simulations only: these tests never execute Slurm or SSH."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('pipeline',Path(__file__).resolve().parents[1]/'deploy/scripts/pipeline.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

@pytest.fixture
def run(tmp_path):
    (tmp_path/'logs').mkdir()
    return tmp_path

CFG={'memory':'128G','cpus':8,'prepare_time':'01:00:00','evaluate_time':'01:00:00',
     'account':'mock','partition':None,'qos':None,'reservation':None,'gres':'gpu:h100:1'}

def test_submission_stdout_separate_diagnostics(run,monkeypatch):
    calls=[]
    monkeypatch.setattr(p,'dependency',lambda *a:'999')
    def simulated(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='12345;mock\n',stderr='site memory notice\n')
    monkeypatch.setattr(p.subprocess,'run',simulated)
    assert p.submit(run,'evaluate',CFG,'999')=='12345'
    r=p.read(p.receipt(run,'evaluate'))
    assert r['job_id']=='12345' and r['application_state']=='SUBMITTED'
    assert '--dependency=afterok:999' in calls[0]
    assert '--gres=gpu:h100:1' in calls[0]
    assert list((run/'logs').glob('*submission.stderr'))[0].read_text()=='site memory notice\n'

@pytest.mark.parametrize('returncode,stdout,state',[(1,'','SUBMISSION_REJECTED'),(0,'notice\n123','UNKNOWN')])
def test_rejected_or_ambiguous_submission_preserved(run,monkeypatch,returncode,stdout,state):
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=returncode,stdout=stdout,stderr='rejected'))
    with pytest.raises(RuntimeError): p.submit(run,'prepare',CFG)
    assert p.read(p.receipt(run,'prepare'))['application_state']==state
    assert not p.read(p.receipt(run,'prepare')).get('job_id')

def test_scheduler_falls_back_to_accounting(monkeypatch):
    results=iter([SimpleNamespace(returncode=0,stdout=''),SimpleNamespace(returncode=0,stdout='99.batch|COMPLETED|0:0\n99|COMPLETED|0:0\n')])
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:next(results))
    assert p.scheduler('99')==('COMPLETED','0:0','')

def test_dependency_completed_requires_artifacts(run,monkeypatch):
    p.update(run,'prepare',job_id='99',application_state='SUCCEEDED')
    monkeypatch.setattr(p,'scheduler',lambda job:('COMPLETED','0:0',''))
    monkeypatch.setattr(p,'artifacts',lambda *a:False)
    with pytest.raises(RuntimeError,match='unverified'): p.dependency(run,'prepare')
    monkeypatch.setattr(p,'artifacts',lambda *a:True)
    assert p.dependency(run,'prepare')=='complete'
    monkeypatch.setattr(p,'scheduler',lambda job:('RUNNING','',''))
    assert p.dependency(run,'prepare')=='99'
    monkeypatch.setattr(p,'scheduler',lambda job:('FAILED','1:0',''))
    with pytest.raises(RuntimeError,match='blocked'): p.dependency(run,'prepare')

def test_monitor_terminal_nonzero_even_if_started(run,monkeypatch):
    p.update(run,'evaluate',job_id='99',application_state='STARTED',stderr=str(run/'missing'))
    monkeypatch.setattr(p,'scheduler',lambda job:('COMPLETED','1:0',''))
    with pytest.raises(RuntimeError,match='terminal'): p.monitor(run,'evaluate',0)

def test_cpu_submission_has_no_gpu_request(run,monkeypatch):
    calls=[]
    def simulate(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
    monkeypatch.setattr(p.subprocess,'run',simulate)
    p.submit(run,'prepare',CFG)
    assert not any(a.startswith('--gres') for a in calls[0])
    assert not any(a.startswith('--dependency') for a in calls[0])

@pytest.mark.parametrize('resume',[False,True])
@pytest.mark.parametrize('judge_options',[{}, {'judge_batch_size':3,'no_prefix_cache':True}])
def test_evaluate_worker_resume_only_when_config_exists(run,monkeypatch,resume,judge_options):
    import os
    cfg={**CFG,'project':str(p.ROOT),'cache_dir':str(run/'cache'),'modules':[], 'min_vram_gb':16,
        'require_bf16':False,'model':'fake','top_k':5,'bundle_size':2,'max_bundles':30,'max_tokens':32,'limit':20}
    cfg.update(judge_options)
    p.atomic(run/'manifest.json',cfg)
    p.update(run,'evaluate',job_id='123',application_state='SUBMITTED')
    if resume: p.atomic(run/'results/config.json',{})
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(p,'dependency',lambda *a:'complete')
    monkeypatch.setattr(p,'artifacts',lambda *a:True)
    calls=[]
    def fake_command(args,**kw):
        calls.append((args,kw))
        return SimpleNamespace(stdout='JobState=RUNNING',returncode=0)
    monkeypatch.setattr(p,'command',fake_command)
    p.worker('evaluate',run)
    args,kwargs=calls[-1]
    assert ('--resume' in args)==resume
    assert args[args.index('--judge-batch-size')+1] == str(judge_options.get('judge_batch_size',8))
    assert ('--no-prefix-cache' in args) == judge_options.get('no_prefix_cache',False)
    assert kwargs['env']['UV_CACHE_DIR']==str(p.ROOT/'cache'/'uv')
    assert kwargs['env']['EVIDENCE_STAGE']=='evaluate'
    assert kwargs['env']['EVIDENCE_RUN_DIR']==str(run)
    assert p.read(p.receipt(run,'evaluate'))['application_state']=='SUCCEEDED'

def test_submit_reconciles_completed_predecessor_omits_stale_dependency(run,monkeypatch):
    calls=[]
    monkeypatch.setattr(p,'dependency',lambda *a:'complete')
    def simulate(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='234\n',stderr='')
    monkeypatch.setattr(p.subprocess,'run',simulate)
    p.submit(run,'evaluate',CFG,'old-job')
    assert not any(a.startswith('--dependency') for a in calls[0])

def test_submit_blocks_unverified_predecessor_without_sbatch(run,monkeypatch):
    def blocked(*args): raise RuntimeError('unverified')
    monkeypatch.setattr(p,'dependency',blocked)
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:pytest.fail('Cannot submit with unverified predecessor'))
    with pytest.raises(RuntimeError,match='unverified'): p.submit(run,'evaluate',CFG,'old-job')

def test_batch_entrypoints_resolve_chdir_and_cpu_no_gpu():
    for stage in ('prepare','evaluate'):
        script=(p.ROOT/'deploy/scripts'/f'{stage}.sbatch').read_text()
        assert 'set -Eeuo pipefail' in script
        assert 'SLURM_SUBMIT_DIR' not in script
        assert './deploy/scripts/pipeline.py' in script
        if stage=='prepare': assert '--gres' not in script

def test_command_terminates_isolated_child_group_on_interrupt(monkeypatch):
    class FakeProcess:
        pid=999
        returncode=0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def communicate(self): raise KeyboardInterrupt()
        def wait(self,timeout=None): return 0
    options=[]
    monkeypatch.setattr(p.subprocess,'Popen',lambda *a,**kw:options.append(kw) or FakeProcess())
    killed=[]
    monkeypatch.setattr(p.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt): p.command(['mock'])
    assert options[0]['start_new_session'] is True
    assert killed==[(999,p.signal.SIGTERM)]

def test_empty_path_rejected():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError): p.required_path('  ')

@pytest.fixture
def mocked_launcher(tmp_path, monkeypatch):
    """Exercise real CLI parsing/filesystem defaults without scheduler or downloads."""
    root = tmp_path / 'project'
    scripts = root / 'deploy' / 'scripts'
    scripts.mkdir(parents=True)
    for stage in ('prepare', 'evaluate'):
        (scripts / f'{stage}.sbatch').write_text('# mock')
    (root / 'uv.lock').write_text('# mock')
    monkeypatch.setattr(p, 'ROOT', root)
    monkeypatch.setattr(p, 'stamp', lambda: '20261005T120000Z')
    monkeypatch.setattr(p.shutil, 'which', lambda name: '/mock/' + name)
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(p, 'dependency', lambda *args: None)
    monkeypatch.setattr(p, 'monitor', lambda *args: None)
    calls = []
    monkeypatch.setattr(p, 'submit', lambda run, stage, cfg, dep=None: calls.append((run, stage, cfg, dep)) or '123')
    return root, calls


def test_submit_account_only_uses_project_defaults(mocked_launcher, monkeypatch):
    root, calls = mocked_launcher
    monkeypatch.setattr(p.sys, 'argv', ['pipeline.py', 'submit', '--account', 'mock'])
    p.main()
    run = root / 'runs' / 'run-20261005T120000Z'
    cfg = p.read(run / 'manifest.json')
    assert cfg['account'] == 'mock'
    assert cfg['partition'] is None
    assert cfg['cache_dir'] == str(root / 'cache' / 'huggingface')
    assert (root / 'cache' / 'huggingface').is_dir()
    assert [call[1] for call in calls] == ['prepare', 'evaluate']
    assert all(call[0] == run for call in calls)


def test_submit_default_run_collision_preserves_existing(mocked_launcher, monkeypatch):
    root, calls = mocked_launcher
    existing = root / 'runs' / 'run-20261005T120000Z'
    existing.mkdir(parents=True)
    (existing / 'preserved').write_text('untouched')
    monkeypatch.setattr(p.sys, 'argv', ['pipeline.py', 'submit', '--account', 'mock'])
    p.main()
    assert calls[0][0] == existing.with_name(existing.name + '-001')
    assert (existing / 'preserved').read_text() == 'untouched'


def test_submit_resume_has_no_implicit_output_conflict(mocked_launcher, monkeypatch):
    root, calls = mocked_launcher
    run = root / 'old-run'
    cfg = {**CFG, 'project': str(root), 'modules': [], 'cache_dir': str(root / 'old-cache')}
    p.atomic(run / 'manifest.json', cfg)
    monkeypatch.setattr(p.sys, 'argv', ['pipeline.py', 'submit', '--resume', str(run)])
    p.main()
    assert all(call[0] == run for call in calls)
    assert calls[0][2]['cache_dir'] == str(root / 'old-cache')
    assert not (root / 'runs').exists()
    assert not (root / 'cache').exists()


def test_submit_explicit_output_and_resume_rejected(mocked_launcher, monkeypatch, capsys):
    root, calls = mocked_launcher
    monkeypatch.setattr(p.sys, 'argv', ['pipeline.py', 'submit', '--output', str(root / 'new'), '--resume', str(root / 'old')])
    with pytest.raises(SystemExit) as exc:
        p.main()
    assert exc.value.code == 2
    assert '--output and --resume cannot be used together' in capsys.readouterr().err
    assert calls == []


def test_submit_without_partition_emits_no_partition_flag(run, monkeypatch):
    calls = []
    monkeypatch.setattr(p.subprocess, 'run', lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0, stdout='123\n', stderr=''))
    p.submit(run, 'prepare', CFG)
    assert not any(arg.startswith('--partition') for arg in calls[0])

def test_submit_judge_flags_persist_to_manifest(mocked_launcher, monkeypatch):
    root, calls = mocked_launcher
    monkeypatch.setattr(p.sys, 'argv', ['pipeline.py','submit','--account','mock',
        '--judge-batch-size','3','--no-prefix-cache'])
    p.main()
    cfg = p.read(root/'runs'/'run-20261005T120000Z'/'manifest.json')
    assert cfg['judge_batch_size'] == 3
    assert cfg['no_prefix_cache'] is True
    assert all(call[2]['judge_batch_size'] == 3 and call[2]['no_prefix_cache'] for call in calls)

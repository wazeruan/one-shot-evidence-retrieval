import hashlib
import json
from types import SimpleNamespace
import pytest
import torch
from one_shot_evidence_retrieval import benchmark as b

def rows():
    return [dict(id='q1', question='Which university?', answer='GOLD SECRET',
        context={'title':['Alice','University'], 'sentences':[['Alice wrote a book.'], ['Alice attended U.']]},
        supporting_facts={'title':['Alice','University','Alice'], 'sent_id':[0,0,0]}),
        dict(id='q2', question='Who wrote?', answer='Alice',
        context=[['Alice',['Alice wrote a book.']], ['Other',['Another document.']]],
        supporting_facts=[['Alice',0]])]

def prepared(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    docs, queries = b.convert_rows(rows())
    for name, records in [('documents',docs), ('queries',queries)]:
        (data/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    embedding_path=tmp_path/'embedding-snapshot'
    embedding_path.mkdir()
    (embedding_path/'config.json').write_text('{"hidden_size":2}')
    manifest = {'embedding_model':'embedding-fake','embedding_revision':'embedding-pin',
        'embedding_path':str(embedding_path),'embedding_config_hash':b.digest(embedding_path/'config.json'), 'model':'fake', 'model_revision':'pin', 'model_path':'unused',
        'hashes':{name:b.digest(data/name) for name in ('documents.jsonl','queries.jsonl')}}
    b.atomic_json(data/'manifest.json',manifest)
    b.atomic_json(data/'PREPARED.json',{'manifest_sha256':b.digest(data/'manifest.json')})
    return data, docs, queries

def test_convert_real_hf_dict_and_legacy_schema_dedup():
    docs, queries = b.convert_rows(rows())
    assert len(docs)==3
    assert len(queries[0]['gold_ids'])==2
    alice = next(d for d in docs if d['title']=='Alice')
    assert alice['id']==hashlib.sha256(alice['text'].encode()).hexdigest()
    assert queries[1]['gold_ids']==[alice['id']]
    assert set(queries[0]['gold_ids']) <= {d['id'] for d in docs}

def test_convert_rejects_missing_support_duplicates_empty():
    bad = rows()[0]
    bad['supporting_facts']['title']=['Absent']
    with pytest.raises(ValueError, match='missing'): b.convert_rows([bad])
    with pytest.raises(ValueError, match='Duplicate'): b.convert_rows([rows()[0],rows()[0]])
    with pytest.raises(ValueError, match='empty'): b.convert_rows([])

def test_hashes_detect_modified_data_and_manifest(tmp_path):
    data, _, _ = prepared(tmp_path)
    b.validate_prepared(data)
    original = (data/'documents.jsonl').read_text()
    (data/'documents.jsonl').write_text(original+'\n')
    with pytest.raises(ValueError, match='Data hash'): b.validate_prepared(data)
    (data/'documents.jsonl').write_text(original)
    (data/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest hash'): b.validate_prepared(data)

def test_metrics_and_summary():
    m=b.evidence_metrics(['a','x','a'],['a','b'])
    assert m=={'support_recall':.5,'support_precision':.5,'complete_support':0.}
    assert b.evidence_metrics(['a','b','x'],['a','b'])['complete_support']==1
    assert b.evidence_metrics([],['a'])['support_precision']==0
    with pytest.raises(ValueError): b.evidence_metrics([],[])
    records=[{'metrics':{'baseline':m,'judged':m}, 'timing_ms':{'baseline_online':v,'judge_extra':2*v},
        'shortlist_complete_support':1.} for v in (1.,3.)]
    summary=b.summarize(records)
    assert summary['metrics']['baseline']['support_recall']==.5
    assert summary['timing_ms']['baseline_online']['p50']==2
    assert summary['timing_ms']['judge_extra']['p95']==pytest.approx(5.8)
    assert summary['query_count']==2

def test_evaluate_resume_and_no_gold_leakage(tmp_path, monkeypatch):
    import transformers
    data, docs, queries = prepared(tmp_path)
    calls=[]
    loads=[]
    class FakeModel:
        config=SimpleNamespace(hidden_size=2)
        def eval(self): return self
        def get_input_embeddings(self): return SimpleNamespace(weight=torch.zeros(64,2))
        def to(self, device): return self
        def parameters(self): return iter([torch.zeros(1)])
    def load(kind,path,**kwargs):
        loads.append((kind,path,kwargs))
        return object() if kind=='tokenizer' else FakeModel()
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda path, **kw: load('tokenizer',path,**kw))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda path, **kw: load('judge',path,**kw))
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', lambda path, **kw: load('embedding',path,**kw))
    class FakeSearch:
        def __init__(self,*a): pass
        def sync(self): pass
        def encode(self,text):
            assert 'GOLD SECRET' not in text
            return torch.ones(2),False
        def run(self, query, documents, *args, **kwargs):
            calls.append(query)
            assert set(documents[0])=={'id','title','text'}
            assert kwargs['index'][0].shape==(3,2)
            return {'query':query, 'baseline_ids':[docs[0]['id']], 'judge_ids':[docs[0]['id']],
                'shortlist':[{'id':d['id']} for d in docs],
                'bundles':[{'combined':.8, 'coverage':{'score':.7,'yes_no_mass':.8},
                    'correctness':{'score':.9,'yes_no_mass':.8}}],
                'timing_ms':{'baseline_online':1.,'judge_extra':2.}}
    monkeypatch.setattr(b,'Search',FakeSearch)
    args=SimpleNamespace(embedding_model='embedding-fake',data=str(data),output=str(tmp_path/'results'),model='fake',device='cpu',
        top_k=3,bundle_size=1,max_tokens=32,limit=2,weights=[1,1,1],contextual=False,max_bundles=5,resume=False,judge_batch_size=8,no_prefix_cache=False)
    b.evaluate(args)
    assert calls==[q['text'] for q in queries]
    assert [kind for kind,_,_ in loads]==['tokenizer','judge','tokenizer','embedding']
    assert all(kwargs['local_files_only'] for _,_,kwargs in loads)
    manifest=json.loads((data/'manifest.json').read_text())
    assert loads[-1][1]==manifest['embedding_path']
    assert loads[1][1]==manifest['model_path']
    out=tmp_path/'results'
    saved_config=json.loads((out/'config.json').read_text())
    assert saved_config['embedding_model']=='embedding-fake'
    assert saved_config['embedding_revision']=='embedding-pin'
    assert saved_config['embedding_config_hash']==manifest['embedding_config_hash']
    receipt=json.loads((out/'SUCCEEDED.json').read_text())
    assert receipt['summary_sha256']==b.digest(out/'summary.json')
    assert json.loads((out/'progress.json').read_text())['query_ids']==['q1','q2']
    assert 'Completed 2 / 2 queries' in (out/'progress.html').read_text()
    args.resume=True
    calls.clear()
    b.evaluate(args)
    assert calls==[]
    assert json.loads((out/'progress.json').read_text())['completed']==2
    assert 'Completed 2 / 2 queries' in (out/'progress.html').read_text()
    args.top_k=2
    with pytest.raises(ValueError,match='configuration mismatch'): b.evaluate(args)

def test_started_marker_preserves_submission_metadata(tmp_path,monkeypatch):
    run=tmp_path/'run'
    b.atomic_json(run/'receipts/evaluate.json',{'job_id':'123','submission_token':'secret-id','application_state':'SUBMITTED'})
    monkeypatch.setenv('EVIDENCE_RUN_DIR',str(run))
    monkeypatch.setenv('EVIDENCE_STAGE','evaluate')
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('EVIDENCE_ALLOCATION_JSON',json.dumps({'job_id':'123','cuda_visible_devices':'0'}))
    b.notify_started()
    receipt=json.loads((run/'receipts/evaluate.json').read_text())
    assert receipt['submission_token']=='secret-id'
    assert receipt['application_state']=='STARTED'
    marker=json.loads((run/'receipts/evaluate-STARTED.json').read_text())
    assert marker['job_id']=='123'
    assert marker['run_dir']==str(run.resolve())

def test_prepare_pins_both_model_snapshots(tmp_path,monkeypatch):
    import datasets
    import huggingface_hub
    model_calls=[]
    snapshots=[]
    class Api:
        def dataset_info(self,name,revision):return SimpleNamespace(sha='dataset-sha')
        def model_info(self,name,revision):
            model_calls.append((name,revision))
            return SimpleNamespace(sha=name+'-sha')
    def snapshot(name,**kwargs):
        snapshots.append((name,kwargs))
        path=tmp_path/name
        path.mkdir(parents=True)
        if kwargs.get('repo_type')=='dataset':
            (path/'distractor').mkdir()
            (path/'distractor/validation-00000.parquet').write_text('fake')
        else:(path/'config.json').write_text('{}')
        return str(path)
    class Dataset(list):
        def shuffle(self,seed):return self
        def select(self,indices):return [self[i] for i in indices]
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**kw:Dataset(rows()))
    args=SimpleNamespace(output=str(tmp_path/'data'),cache_dir=str(tmp_path/'cache'),
        model='judge',model_revision='judge-requested',embedding_model='embedding',
        embedding_revision='embedding-requested',dataset_revision='dataset-requested',seed=17,limit=2)
    b.prepare(args)
    manifest=b.validate_prepared(tmp_path/'data')
    assert model_calls==[('judge','judge-requested'),('embedding','embedding-requested')]
    assert [(name,kw['revision']) for name,kw in snapshots]==[
        ('hotpotqa/hotpot_qa','dataset-sha'),('judge','judge-sha'),('embedding','embedding-sha')]
    assert manifest['model_revision']=='judge-sha'
    assert manifest['embedding_revision']=='embedding-sha'
    assert manifest['embedding_config_hash']==b.digest(tmp_path/'embedding/config.json')

@pytest.mark.parametrize('change',['wrong-model','modified-config','legacy-manifest'])
def test_evaluate_rejects_incompatible_embedding_before_loading(tmp_path,monkeypatch,change):
    import transformers
    data,_,_=prepared(tmp_path)
    args=SimpleNamespace(data=str(data),model='fake',embedding_model='embedding-fake')
    manifest=json.loads((data/'manifest.json').read_text())
    if change=='wrong-model':args.embedding_model='other'
    if change=='modified-config':(tmp_path/'embedding-snapshot/config.json').write_text('changed')
    if change=='legacy-manifest':
        manifest.pop('embedding_model')
        b.atomic_json(data/'manifest.json',manifest)
        b.atomic_json(data/'PREPARED.json',{'manifest_sha256':b.digest(data/'manifest.json')})
    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',lambda *a,**kw:pytest.fail('Must reject before loading'))
    with pytest.raises(ValueError,match='Embedding|embedding'):
        b.evaluate(args)

@pytest.mark.parametrize('command',['prepare','evaluate'])
def test_embedding_cli_defaults(monkeypatch,tmp_path,command):
    import sys
    captured=[]
    monkeypatch.setattr(b,command,lambda args:captured.append(args))
    args=['benchmark',command,'--output',str(tmp_path/'output')]
    args+=['--cache-dir',str(tmp_path/'cache')] if command=='prepare' else ['--data',str(tmp_path/'data')]
    monkeypatch.setattr(sys,'argv',args)
    b.main()
    assert captured[0].embedding_model=='Qwen/Qwen3-Embedding-4B'
    assert captured[0].model=='Qwen/Qwen3-4B'
    if command=='prepare':assert captured[0].embedding_revision=='main'

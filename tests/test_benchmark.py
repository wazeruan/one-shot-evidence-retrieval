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
    manifest = {'model':'fake', 'model_revision':'pin', 'model_path':'unused',
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
    class FakeModel:
        def to(self, device): return self
        def parameters(self): return iter([torch.zeros(1)])
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: FakeModel())
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
                'timing_ms':{'baseline_online':1.,'judge_extra':2.}}
    monkeypatch.setattr(b,'Search',FakeSearch)
    args=SimpleNamespace(data=str(data),output=str(tmp_path/'results'),model='fake',device='cpu',
        top_k=3,bundle_size=1,max_tokens=32,limit=2,weights=[1,1,1],contextual=False,max_bundles=5,resume=False)
    b.evaluate(args)
    assert calls==[q['text'] for q in queries]
    out=tmp_path/'results'
    receipt=json.loads((out/'SUCCEEDED.json').read_text())
    assert receipt['summary_sha256']==b.digest(out/'summary.json')
    args.resume=True
    calls.clear()
    b.evaluate(args)
    assert calls==[]
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

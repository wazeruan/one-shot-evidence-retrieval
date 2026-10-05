import math
import json
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from one_shot_evidence_retrieval import Search, candidate_bundles, combined_score, read_jsonl

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == ' yes': return [1]
        if text == ' no': return [2]
        return [3 + ord(c) % 60 for c in text]
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(m['content'] for m in messages) + '\nAssistant:'

@pytest.fixture
def runner():
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=2048))
    model.generate = lambda *a, **k: pytest.fail('Decoding must never be called')
    return Search(model, Tokenizer(), 'cpu', max_tokens=32)

DOCS = [{'id': str(i), 'text': text} for i, text in enumerate(['alpha author', 'beta university', 'unrelated gamma', 'delta'])]

def test_literal_pooling_and_frozen(runner):
    vector, truncated = runner.encode('alpha')
    ids = torch.tensor([runner.tokenizer.encode('alpha')])
    expected = torch.nn.functional.normalize(runner.layer(ids).float().mean(1), dim=-1)[0]
    torch.testing.assert_close(vector, expected)
    assert not truncated
    assert not vector.requires_grad
    assert all(not p.requires_grad for p in runner.model.parameters())
    assert not runner.model.training
    assert runner.encode('a' * 100)[1]

def test_judge_injects_pooled_vectors_and_no_decode(runner):
    q, _ = runner.encode('query')
    d, _ = runner.encode('document')
    seen = []
    handle = runner.model.register_forward_pre_hook(lambda model, args, kwargs: seen.append(kwargs), with_kwargs=True)
    out = runner.judge(q, [d], 'coverage')
    handle.remove()
    assert len(seen) == 1
    inputs = seen[0]['inputs_embeds'][0]
    for vector in (q, d):
        target = vector * runner.scale
        assert torch.isclose(inputs, target).all(-1).any()
    assert seen[0]['use_cache'] is False
    assert 0 <= out['score'] <= 1
    assert 0 <= out['yes_no_mass'] <= 1
    assert out['input_tokens'] == inputs.shape[0]

def test_cascade_baseline_fairness_and_candidate_bounds(runner):
    result = runner.run('author university', DOCS, 3, 2, [1, 1, 1], 10)
    baseline = runner.run('author university', DOCS, 3, 2, [1, 1, 1], 10, True)
    assert result['baseline_ids'] == baseline['baseline_ids']
    assert result['shortlist'] == baseline['shortlist']
    assert result['judge_calls'] == 6
    assert baseline['judge_calls'] == 0
    assert baseline['judge_ids'] is None
    assert len(result['judge_ids']) == len(result['baseline_ids']) == 2
    shortlist = {d['id'] for d in result['shortlist']}
    assert all(set(b['ids']) <= shortlist for b in result['bundles'])
    assert all(math.isfinite(b['combined']) for b in result['bundles'])
    assert all(v >= 0 for v in result['timing_ms'].values())

def test_contextual_encoder(runner):
    runner.contextual = True
    vector, _ = runner.encode('test contextual')
    assert vector.shape == (16,)
    assert torch.isfinite(vector).all()
    torch.testing.assert_close(vector.norm(), torch.tensor(1.))

@pytest.mark.parametrize('n,size,limit', [(3,0,10),(3,4,10),(5,2,9)])
def test_bundle_errors(n,size,limit):
    with pytest.raises(ValueError): candidate_bundles(n,size,limit)

@pytest.mark.parametrize('weights', [[0,0,0],[-1,1,1],[float('nan'),1,1],[1,2]])
def test_weight_errors(weights):
    with pytest.raises(ValueError): combined_score(.5,.5,.5,weights)

def test_input_errors(tmp_path, runner):
    with pytest.raises(ValueError): runner.encode('')
    path = tmp_path/'docs.jsonl'
    path.write_text('')
    with pytest.raises(ValueError): read_jsonl(path)
    path.write_text('\n'.join(json.dumps(d) for d in [DOCS[0],DOCS[0]]))
    with pytest.raises(ValueError): read_jsonl(path)
    path.write_text(json.dumps({'id':'x','text':' '}))
    with pytest.raises(ValueError): read_jsonl(path)

def test_label_readout_conditional_probability(runner, monkeypatch):
    from types import SimpleNamespace
    q, _ = runner.encode('query')
    logits = torch.zeros(1, 1, 64)
    logits[0, 0, 1] = 2
    logits[0, 0, 2] = -1
    monkeypatch.setattr(runner.model, 'forward', lambda **kwargs: SimpleNamespace(logits=logits))
    result = runner.judge(q, [q], 'correctness')
    expected = torch.softmax(torch.tensor([2., -1.]), 0)[0].item()
    mass = torch.softmax(logits[0,0], 0)[[1,2]].sum().item()
    assert result['score'] == pytest.approx(expected)
    assert result['yes_no_mass'] == pytest.approx(mass)

def test_baseline_skips_judging_bundle_limit(runner, monkeypatch):
    monkeypatch.setattr(runner, 'judge', lambda *args: pytest.fail('Baseline cannot judge'))
    result = runner.run('query', DOCS, 4, 2, [1,1,1], 1, True)
    assert result['judge_calls'] == 0
    assert len(result['baseline_ids']) == 2
    with pytest.raises(ValueError): runner.run('query', DOCS, 4, 5, [1,1,1], 1, True)

def test_judge_context_guard_before_model_execution(runner, monkeypatch):
    q, _ = runner.encode('query')
    runner.model.config.max_position_embeddings = 5
    monkeypatch.setattr(runner.model, 'forward', lambda **kwargs: pytest.fail('Oversized input must not run'))
    with pytest.raises(ValueError, match='context|limit|tokens'):
        runner.judge(q, [q], 'coverage')

def test_cached_index_baseline_identical_no_document_reencoding(runner, monkeypatch):
    fresh = runner.run('query', DOCS, 3, 2, [1,1,1], 10, True)
    vectors, truncations = zip(*(runner.encode(d['text']) for d in DOCS))
    index = (torch.stack(vectors), truncations)
    original = runner.encode
    calls = []
    def tracked(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(runner, 'encode', tracked)
    cached = runner.run('query', DOCS, 3, 2, [1,1,1], 10, True, index=index)
    assert calls == ['query']
    assert cached['baseline_ids'] == fresh['baseline_ids']
    assert cached['shortlist'] == fresh['shortlist']
    with pytest.raises(ValueError, match='count'):
        runner.run('query', DOCS, 3, 2, [1,1,1], 10, True, index=(index[0][:-1], truncations))

@pytest.mark.parametrize('value',[0.,float('nan'),float('inf')])
def test_encode_rejects_invalid_vectors(runner,value):
    with torch.no_grad(): runner.layer.weight.fill_(value)
    with pytest.raises(ValueError,match='non-finite|zero'): runner.encode('query')

def test_judge_rejects_nonfinite_logits(runner,monkeypatch):
    from types import SimpleNamespace
    q,_=runner.encode('query')
    monkeypatch.setattr(runner.model,'forward',lambda **kwargs:SimpleNamespace(logits=torch.full((1,1,64),float('nan'))))
    with pytest.raises(ValueError,match='non-finite'): runner.judge(q,[q],'coverage')

def test_cached_index_rejects_nonfinite(runner):
    with pytest.raises(ValueError,match='non-finite'):
        runner.run('query',DOCS,3,2,[1,1,1],10,True,index=(torch.full((4,16),float('inf')),[False]*4))

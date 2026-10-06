"""Numerical regression tests using real random-weight Qwen3, without downloads."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from one_shot_evidence_retrieval import Search
from test_pipeline import Tokenizer, DOCS

@pytest.fixture
def runner():
    torch.manual_seed(19)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=2048))
    model.generate = lambda *a, **k: pytest.fail('Generation/decoding is forbidden')
    return Search(model, Tokenizer(), 'cpu', max_tokens=32)

def requests_for(runner):
    vectors = [runner.encode(d['text'])[0] for d in DOCS]
    return [(vectors[:2], 'coverage'), (vectors[1:3], 'correctness'),
            (vectors[2:], 'coverage'), (vectors[:1], 'correctness'),
            (vectors[:2], 'correctness'), (vectors[1:3], 'coverage')]

@pytest.mark.parametrize('batch_size', [1, 2, 8])
@pytest.mark.parametrize('cached', [False, True])
def test_real_qwen_batch_scores_match_sequential(runner, batch_size, cached):
    q = runner.encode('author university')[0]
    requests = requests_for(runner)
    sequential = [runner.judge(q, v, c) for v,c in requests]
    runner.judge_batch_size = batch_size
    runner.prefix_cache = cached
    batched = runner.judge_many(q, requests)
    for expected, actual in zip(sequential, batched):
        assert actual['input_tokens'] == expected['input_tokens']
        assert actual['score'] == pytest.approx(expected['score'], abs=2e-6)
        assert actual['yes_no_mass'] == pytest.approx(expected['yes_no_mass'], abs=2e-6)
    assert runner.last_judge_stats['requests'] == len(requests)
    assert (runner.last_judge_stats['prefix_tokens'] > 0) == cached

def test_cache_isolation_and_causal_prefix(runner):
    requests = requests_for(runner)
    q1 = runner.encode('first query')[0]
    q2 = runner.encode('different query')[0]
    first = runner.judge_many(q1, requests)
    original = runner.model.model.forward
    seen = []
    def tracked(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)
    runner.model.model.forward = tracked
    second = runner.judge_many(q2, requests)
    # The first call is a fresh prefix with no inherited KV cache.
    assert seen[0].get('past_key_values') is None
    prefix = runner.last_judge_stats['prefix_tokens']
    assert prefix > 0
    for call in seen[1:]:
        assert call['attention_mask'].shape[-1] == prefix + call['inputs_embeds'].shape[1]
        assert call['position_ids'][0,0].item() == prefix
    runner.model.model.forward = original
    sequential = [runner.judge(q2,v,c) for v,c in requests]
    assert [x['score'] for x in second] == pytest.approx([x['score'] for x in sequential], abs=2e-6)
    repeated = runner.judge_many(q1, requests)
    assert [x['score'] for x in repeated] == pytest.approx([x['score'] for x in first], abs=2e-6)

def test_run_reports_actual_batch_and_prefix_calls_and_baseline_skips(runner, monkeypatch):
    runner.judge_batch_size = 2
    result = runner.run('query', DOCS, 3, 2, [1,1,1], 10)
    assert result['judge_calls'] == 6
    # Three coverage prompts and three longer correctness prompts: two batches each.
    assert result['judge_forward_calls'] == 5  # one prefix + four suffix batches
    assert result['judge_prefix_tokens'] > 0
    assert result['judge_batch_size'] == 2
    monkeypatch.setattr(runner, 'judge_many', lambda *a: pytest.fail('Baseline must not judge'))
    baseline = runner.run('query', DOCS, 3, 2, [1,1,1], 10, True)
    assert baseline['judge_calls'] == baseline['judge_forward_calls'] == baseline['judge_prefix_tokens'] == 0

def test_empty_single_and_invalid_batch(runner):
    q = runner.encode('query')[0]
    assert runner.judge_many(q, []) == []
    assert runner.last_judge_stats == {'requests':0, 'forward_calls':0, 'prefix_tokens':0}
    single = runner.judge_many(q, [([q], 'coverage')])
    assert len(single) == 1
    assert runner.last_judge_stats['forward_calls'] == 1
    assert runner.last_judge_stats['prefix_tokens'] == 0
    with pytest.raises(ValueError, match='positive'):
        Search(runner.model, runner.tokenizer, 'cpu', judge_batch_size=0)

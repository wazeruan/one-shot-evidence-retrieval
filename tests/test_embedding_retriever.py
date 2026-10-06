"""Dedicated embedding encoder checks using actual tiny Qwen3 forward passes."""
import pytest
import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3Model, Qwen3ForCausalLM
from one_shot_evidence_retrieval import EmbeddingRetriever, Search
from test_pipeline import Tokenizer, DOCS

class EmbeddingTokenizer:
    def __init__(self, padding=None):
        self.calls=[]
        self.padding=padding
    def __call__(self,text,**kwargs):
        self.calls.append((text,kwargs.copy()))
        ids=[3+ord(c)%60 for c in text]+[1]
        if kwargs.get('truncation'): ids=ids[:kwargs['max_length']]
        mask=[1]*len(ids)
        if kwargs.get('return_tensors'):
            if self.padding=='left': ids=[0,0]+ids;mask=[0,0]+mask
            if self.padding=='right': ids=ids+[0,0];mask=mask+[0,0]
            return BatchEncoding({'input_ids':torch.tensor([ids]),'attention_mask':torch.tensor([mask])})
        return {'input_ids':ids,'attention_mask':mask}

@pytest.fixture
def retriever():
    torch.manual_seed(43)
    model=Qwen3Model(Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=2048))
    return EmbeddingRetriever(model,EmbeddingTokenizer(),'cpu',max_tokens=256)

@pytest.mark.parametrize('padding',[None,'left','right'])
def test_real_qwen_contextual_last_token_not_literal_pooling(retriever,padding):
    retriever.tokenizer.padding=padding
    vector,clipped=retriever.encode('Alice attended U')
    inputs=retriever.tokenizer('Alice attended U',add_special_tokens=True,truncation=True,max_length=256,return_tensors='pt')
    with torch.inference_mode(): hidden=retriever.model(**inputs,use_cache=False).last_hidden_state
    position=torch.where(inputs['attention_mask'][0]>0)[0][-1]
    expected=torch.nn.functional.normalize(hidden[0,position].float(),dim=-1)
    literal=torch.nn.functional.normalize(retriever.model.get_input_embeddings()(inputs['input_ids']).float().mean(1)[0],dim=-1)
    torch.testing.assert_close(vector,expected)
    assert not torch.allclose(vector,literal)
    assert vector.dtype==torch.float32
    assert torch.isfinite(vector).all()
    assert vector.norm().item()==pytest.approx(1.)
    assert not vector.requires_grad and not clipped
    assert not retriever.model.training
    assert all(not p.requires_grad for p in retriever.model.parameters())

def test_instruction_only_on_query_and_truncation(retriever):
    retriever.encode('question',is_query=True)
    assert all(text==f'Instruct: {retriever.instruction}\nQuery:question' for text,_ in retriever.tokenizer.calls)
    retriever.tokenizer.calls.clear()
    retriever.encode('document')
    assert all(text=='document' for text,_ in retriever.tokenizer.calls)
    retriever.max_tokens=4
    assert retriever.encode('long document')[1]
    with pytest.raises(ValueError,match='nonempty'):retriever.encode(' ')

@pytest.mark.parametrize('value',[0.,float('nan'),float('inf')])
def test_nonfinite_or_zero_embeddings_rejected(retriever,monkeypatch,value):
    from types import SimpleNamespace
    def bad(**kwargs):
        return SimpleNamespace(last_hidden_state=torch.full((*kwargs['input_ids'].shape,16),value))
    monkeypatch.setattr(retriever.model,'forward',bad)
    with pytest.raises(ValueError,match='non-finite|zero'):retriever.encode('text')

def test_search_routes_documents_and_query_and_freezes_judge(retriever,monkeypatch):
    judge=Qwen3ForCausalLM(retriever.model.config)
    search=Search(judge,Tokenizer(),'cpu',retriever=retriever)
    calls=[]
    original=retriever.encode
    def tracked(text,is_query=False):
        calls.append((text,is_query))
        return original(text,is_query)
    monkeypatch.setattr(retriever,'encode',tracked)
    result=search.run('question',DOCS,3,2,[1,1,1],10,True)
    assert calls==[(d['text'],False) for d in DOCS]+[('question',True)]
    assert result['judge_calls']==0
    assert all(not p.requires_grad for p in judge.parameters())
    assert not judge.training

def test_dimension_mismatch_fails_before_judge(retriever,monkeypatch):
    judge=Qwen3ForCausalLM(retriever.model.config)
    search=Search(judge,Tokenizer(),'cpu',retriever=retriever)
    monkeypatch.setattr(judge,'forward',lambda **kwargs:pytest.fail('Mismatched vectors must not enter judge'))
    with pytest.raises(ValueError,match='width|adapter'):
        search.judge(torch.ones(8),[torch.ones(8)],'coverage')

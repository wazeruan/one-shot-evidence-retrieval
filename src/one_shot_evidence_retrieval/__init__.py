"""One-shot cosine retrieval with experimental frozen soft-vector judgment."""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('IDs must be unique')
    if not rows or any(not isinstance(row.get('text'), str) or not row['text'].strip() for row in rows):
        raise ValueError('Provide nonempty text rows')
    return rows


def candidate_bundles(n, size, limit):
    if not 1 <= size <= n:
        raise ValueError('bundle size must be between 1 and shortlist size')
    count = math.comb(n, size)
    if count > limit:
        raise ValueError(f'{count} bundles exceeds limit {limit}; reduce top-k or bundle-size')
    return list(itertools.combinations(range(n), size))


def combined_score(relevance, coverage, correctness, weights):
    if len(weights) != 3 or any(w < 0 or not math.isfinite(w) for w in weights) or sum(weights) <= 0:
        raise ValueError('Use three finite nonnegative weights with positive sum')
    return sum(w * s for w, s in zip(weights, (relevance, coverage, correctness))) / sum(weights)


class EmbeddingRetriever:
    """Frozen Qwen3-Embedding contextual last-token retriever."""
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def __init__(self, model, tokenizer, device, max_tokens=512):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, text, is_query=False):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Embedding input must be nonempty text')
        if is_query:
            text = f'Instruct: {self.instruction}\nQuery:{text}'
        full = self.tokenizer(text, add_special_tokens=True, truncation=False)
        clipped = len(full['input_ids']) > self.max_tokens
        inputs = self.tokenizer(text, add_special_tokens=True, truncation=True,
                                max_length=self.max_tokens, return_tensors='pt').to(self.device)
        hidden = self.model(**inputs, use_cache=False).last_hidden_state
        mask = inputs['attention_mask']
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(mask == 0, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
        if not torch.isfinite(pooled).all() or (pooled.norm(dim=-1) == 0).any():
            raise ValueError('Embedding retriever produced non-finite or zero embeddings')
        return F.normalize(pooled, dim=-1)[0], clipped


class Search:
    def __init__(self, model, tokenizer, device, max_tokens=512, contextual=False, judge_batch_size=8, prefix_cache=True, retriever=None):
        if judge_batch_size < 1:
            raise ValueError('judge_batch_size must be positive')
        self.judge_batch_size = judge_batch_size
        self.prefix_cache = prefix_cache
        self.last_judge_stats = {'requests': 0, 'forward_calls': 0, 'prefix_tokens': 0}
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.contextual = contextual
        self.retriever = retriever
        for p in model.parameters():
            p.requires_grad_(False)
        self.layer = model.get_input_embeddings()
        self.scale = float(self.layer.weight.detach().float().norm(dim=-1).median().item())
        self.labels = [tokenizer.encode(label, add_special_tokens=False) for label in (' yes', ' no')]
        if any(len(label) != 1 for label in self.labels):
            raise ValueError('yes/no labels must each tokenize to one token for prefill-only scoring')

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
        elif str(self.device) == 'mps':
            torch.mps.synchronize()

    @torch.inference_mode()
    def encode(self, text):
        if self.retriever is not None:
            return self.retriever.encode(text)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError('Empty token sequence')
        clipped = len(ids) > self.max_tokens
        inputs = torch.tensor([ids[:self.max_tokens]], device=self.device)
        if self.contextual:
            hidden = self.model.model(input_ids=inputs, use_cache=False).last_hidden_state
        else:
            hidden = self.layer(inputs)
        pooled = hidden.float().mean(dim=1)
        if not torch.isfinite(pooled).all() or (pooled.norm(dim=-1) == 0).any():
            raise ValueError('Encoder produced non-finite or zero embeddings')
        vector = F.normalize(pooled, dim=-1)[0]
        return vector, clipped

    def encode_query(self, text):
        return self.retriever.encode(text, is_query=True) if self.retriever is not None else self.encode(text)

    @torch.inference_mode()
    def _judge_inputs(self, query_vector, vectors, criterion):
        # Experimental soft-token interface: unit vectors are rescaled to the
        # input embedding table's median norm. No trained adapter is claimed.
        expected = self.layer.weight.shape[1]
        if any(vector.ndim != 1 or vector.shape[0] != expected for vector in [query_vector, *vectors]):
            raise ValueError(f'Retriever vectors must match judge input width {expected}; no trained adapter is available')
        question = {
            'coverage': 'Does the supplied evidence jointly cover EVERY information requirement of the query?',
            'correctness': 'Are the claims in the supplied evidence factually correct and applicable to the query?',
        }[criterion]
        parts = ['Query: <<V0>>\nEvidence:\n']
        parts += [f'Document {i}: <<V{i + 1}>>\n' for i in range(len(vectors))]
        parts += [question + ' Judge only the supplied evidence. Answer yes or no.']
        messages = [
            {'role': 'system', 'content': 'You judge evidence. Answer only yes or no.'},
            {'role': 'user', 'content': ''.join(parts)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        chunks = []
        remaining = prompt
        for i, vector in enumerate([query_vector, *vectors]):
            before, remaining = remaining.split(f'<<V{i}>>', 1)
            ids = self.tokenizer.encode(before, add_special_tokens=False)
            if ids:
                chunks.append(self.layer(torch.tensor([ids], device=self.device)))
            chunks.append((vector * self.scale).to(self.layer.weight.dtype).reshape(1, 1, -1))
        ids = self.tokenizer.encode(remaining, add_special_tokens=False)
        if ids:
            chunks.append(self.layer(torch.tensor([ids], device=self.device)))
        embedded = torch.cat(chunks, dim=1)
        context_limit = getattr(self.model.config, 'max_position_embeddings', None)
        if context_limit is not None and embedded.shape[1] > context_limit:
            raise ValueError('Judge prompt exceeds model context limit; reduce bundle size')
        return embedded

    def _read_judgment(self, logits, input_tokens):
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError('Judge produced non-finite logits')
        label_ids = [label[0] for label in self.labels]
        conditional = logits[label_ids].softmax(dim=0)
        label_mass = logits.softmax(dim=0)[label_ids].sum()
        return {'score': float(conditional[0]), 'yes_no_mass': float(label_mass), 'input_tokens': input_tokens}


    @torch.inference_mode()
    def judge(self, query_vector, vectors, criterion):
        embedded = self._judge_inputs(query_vector, vectors, criterion)
        logits = self.model(inputs_embeds=embedded,
            attention_mask=torch.ones(embedded.shape[:2], device=self.device, dtype=torch.long),
            use_cache=False).logits[0, -1]
        return self._read_judgment(logits, embedded.shape[1])

    @torch.inference_mode()
    def judge_many(self, query_vector, requests):
        """Batch independent prompts; reuse ONLY their exact common causal prefix.

        Grouping by length avoids padding and lets us request final-position
        logits only. The cache is scoped to this query and cloned before each
        batch, so suffixes cannot contaminate another judgment.
        """
        import copy
        prompts = [self._judge_inputs(query_vector, vectors, criterion)
                   for vectors, criterion in requests]
        stats = {'requests': len(prompts), 'forward_calls': 0, 'prefix_tokens': 0}
        self.last_judge_stats = stats
        if not prompts:
            return []
        prefix_length = 0
        cache = None
        if self.prefix_cache and len(prompts) > 1:
            minimum = min(p.shape[1] for p in prompts) - 1
            shared = torch.stack([p[0, :minimum] for p in prompts])
            equal = (shared == shared[:1]).all(dim=0).all(dim=-1)
            prefix_length = int(equal.to(torch.int64).cumprod(0).sum().item())
            if prefix_length:
                prefix = prompts[0][:, :prefix_length]
                cache = self.model.model(inputs_embeds=prefix,
                    attention_mask=torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
                    use_cache=True).past_key_values
                if cache is None or not hasattr(cache, 'batch_repeat_interleave'):
                    raise ValueError('Model must provide a repeatable Transformers KV cache; disable prefix caching for unsupported models')
                stats['forward_calls'] += 1
                stats['prefix_tokens'] = prefix_length
        groups = {}
        for i, prompt in enumerate(prompts):
            groups.setdefault(prompt.shape[1], []).append(i)
        results = [None] * len(prompts)
        for length, indices in groups.items():
            for offset in range(0, len(indices), self.judge_batch_size):
                selected = indices[offset:offset+self.judge_batch_size]
                embedded = torch.cat([prompts[i][:, prefix_length:] for i in selected], dim=0)
                kwargs = {}
                if cache is not None:
                    batch_cache = copy.deepcopy(cache)
                    batch_cache.batch_repeat_interleave(len(selected))
                    kwargs['past_key_values'] = batch_cache
                positions = torch.arange(prefix_length, length, device=self.device)[None, :].expand(len(selected), -1)
                logits = self.model(inputs_embeds=embedded,
                    attention_mask=torch.ones((len(selected), length), device=self.device, dtype=torch.long),
                    position_ids=positions, use_cache=cache is not None,
                    logits_to_keep=1, **kwargs).logits[:, -1]
                stats['forward_calls'] += 1
                for i, row in zip(selected, logits):
                    results[i] = self._read_judgment(row, length)
        return results

    def run(self, query, documents, top_k, size, weights, max_bundles, baseline_only=False, index=None):
        self.sync()
        start = time.perf_counter()
        if index is None:
            vectors, truncations = zip(*(self.encode(d['text']) for d in documents))
            matrix = torch.stack(vectors)
        else:
            matrix, truncations = index
            if matrix.shape[0] != len(documents) or len(truncations) != len(documents):
                raise ValueError('Index must match document count')
        self.sync()
        indexed = time.perf_counter()
        q, qtruncated = self.encode_query(query)
        if not torch.isfinite(matrix).all():
            raise ValueError('Index contains non-finite embeddings')
        scores = matrix @ q
        count = min(top_k, len(documents))
        if not 1 <= size <= count:
            raise ValueError('bundle size must be between 1 and shortlist size')
        bundles = [] if baseline_only else candidate_bundles(count, size, max_bundles)
        indices = torch.argsort(scores, descending=True, stable=True)[:count].tolist()
        self.sync()
        retrieved = time.perf_counter()
        baseline = [documents[i]['id'] for i in indices[:size]]
        evaluations = []
        self.last_judge_stats = {'requests': 0, 'forward_calls': 0, 'prefix_tokens': 0}
        if not baseline_only:
            requests = []
            for bundle in bundles:
                evidence = [matrix[indices[j]] for j in bundle]
                requests.extend([(evidence, 'coverage'), (evidence, 'correctness')])
            judgments = self.judge_many(q, requests)
            for bundle_index, bundle in enumerate(bundles):
                chosen = [indices[j] for j in bundle]
                evidence = [matrix[i] for i in chosen]
                coverage, correctness = judgments[2*bundle_index:2*bundle_index+2]
                relevance = float(scores[chosen].mean())
                value = combined_score((relevance + 1) / 2, coverage['score'], correctness['score'], weights)
                evaluations.append({'ids': [documents[i]['id'] for i in chosen], 'cosine': relevance,
                                    'coverage': coverage, 'correctness': correctness, 'combined': value})
        self.sync()
        end = time.perf_counter()
        winner = max(evaluations, key=lambda e: e['combined']) if evaluations else None
        return {'query': query, 'baseline_ids': baseline, 'judge_ids': winner['ids'] if winner else None,
                'shortlist': [{'id': documents[i]['id'], 'cosine': float(scores[i])} for i in indices],
                'bundles': evaluations, 'truncated_document_ids': [d['id'] for d, t in zip(documents, truncations) if t],
                'query_truncated': qtruncated, 'timing_ms': {'index': 1000*(indexed-start),
                'baseline_online': 1000*(retrieved-indexed), 'judge_extra': 1000*(end-retrieved)},
                'judge_calls': 2*len(evaluations),
                'judge_forward_calls': self.last_judge_stats['forward_calls'],
                'judge_prefix_tokens': self.last_judge_stats['prefix_tokens'],
                'judge_batch_size': self.judge_batch_size,
                'warning': 'Untrained pooled-vector interface. Scores are NOT calibrated and correctness has no external verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-4B')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--bundle-size', type=int, default=2)
    parser.add_argument('--max-bundles', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--weights', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--judge-batch-size', type=int, default=8)
    parser.add_argument('--no-prefix-cache', action='store_true')
    parser.add_argument('--contextual', action='store_true', help='Use final contextual hidden states rather than the input embedding layer')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_tokens < 1 or args.max_bundles < 1 or args.judge_batch_size < 1:
        parser.error('top-k, max-tokens, max-bundles and judge-batch-size must be positive')
    combined_score(0, 0, 0, args.weights)
    documents = read_jsonl(args.documents)
    count = min(args.top_k, len(documents))
    if not 1 <= args.bundle_size <= count:
        parser.error('bundle size must be between 1 and shortlist size')
    if not args.baseline_only:
        candidate_bundles(count, args.bundle_size, args.max_bundles)
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=args.local_files_only).to(args.device)
    if sum(p.numel() for p in model.parameters()) >= 9_000_000_000:
        raise ValueError('This experiment requires a model below 9B parameters')
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=args.local_files_only)
    embedding_model = AutoModel.from_pretrained(args.embedding_model, dtype=dtype, local_files_only=args.local_files_only).to(args.device)
    if sum(p.numel() for p in embedding_model.parameters()) >= 9_000_000_000:
        raise ValueError('Embedding model must have fewer than 9B parameters')
    retriever = EmbeddingRetriever(embedding_model, embedding_tokenizer, args.device, args.max_tokens)
    runner = Search(model, tokenizer, args.device, args.max_tokens, False, args.judge_batch_size, not args.no_prefix_cache, retriever)
    result = runner.run(args.query, documents, args.top_k, args.bundle_size, args.weights, args.max_bundles, args.baseline_only)
    result['config'] = vars(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(path.resolve()), 'baseline_ids': result['baseline_ids'], 'judge_ids': result['judge_ids'], 'timing_ms': result['timing_ms']}, indent=2))

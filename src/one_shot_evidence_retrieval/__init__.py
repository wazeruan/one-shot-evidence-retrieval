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


class Search:
    def __init__(self, model, tokenizer, device, max_tokens=512, contextual=False):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.contextual = contextual
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
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError('Empty token sequence')
        clipped = len(ids) > self.max_tokens
        inputs = torch.tensor([ids[:self.max_tokens]], device=self.device)
        if self.contextual:
            hidden = self.model.model(input_ids=inputs, use_cache=False).last_hidden_state
        else:
            hidden = self.layer(inputs)
        vector = F.normalize(hidden.float().mean(dim=1), dim=-1)[0]
        return vector, clipped

    @torch.inference_mode()
    def judge(self, query_vector, vectors, criterion):
        # Experimental soft-token interface: unit vectors are rescaled to the
        # input embedding table's median norm. No trained adapter is claimed.
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
        logits = self.model(inputs_embeds=embedded, attention_mask=torch.ones(embedded.shape[:2], device=self.device, dtype=torch.long), use_cache=False).logits[0, -1].float()
        label_ids = [label[0] for label in self.labels]
        conditional = logits[label_ids].softmax(dim=0)
        label_mass = logits.softmax(dim=0)[label_ids].sum()
        return {'score': float(conditional[0]), 'yes_no_mass': float(label_mass), 'input_tokens': embedded.shape[1]}

    def run(self, query, documents, top_k, size, weights, max_bundles, baseline_only=False):
        self.sync()
        start = time.perf_counter()
        vectors, truncations = zip(*(self.encode(d['text']) for d in documents))
        matrix = torch.stack(vectors)
        self.sync()
        indexed = time.perf_counter()
        q, qtruncated = self.encode(query)
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
        if not baseline_only:
            for bundle in bundles:
                chosen = [indices[j] for j in bundle]
                evidence = [matrix[i] for i in chosen]
                coverage = self.judge(q, evidence, 'coverage')
                correctness = self.judge(q, evidence, 'correctness')
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
                'warning': 'Untrained pooled-vector interface. Scores are NOT calibrated and correctness has no external verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--bundle-size', type=int, default=2)
    parser.add_argument('--max-bundles', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--weights', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--contextual', action='store_true', help='Use final contextual hidden states rather than the input embedding layer')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_tokens < 1 or args.max_bundles < 1:
        parser.error('top-k, max-tokens and max-bundles must be positive')
    combined_score(0, 0, 0, args.weights)
    documents = read_jsonl(args.documents)
    count = min(args.top_k, len(documents))
    if not 1 <= args.bundle_size <= count:
        parser.error('bundle size must be between 1 and shortlist size')
    if not args.baseline_only:
        candidate_bundles(count, args.bundle_size, args.max_bundles)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=args.local_files_only).to(args.device)
    if sum(p.numel() for p in model.parameters()) >= 9_000_000_000:
        raise ValueError('This experiment requires a model below 9B parameters')
    runner = Search(model, tokenizer, args.device, args.max_tokens, args.contextual)
    result = runner.run(args.query, documents, args.top_k, args.bundle_size, args.weights, args.max_bundles, args.baseline_only)
    result['config'] = vars(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(path.resolve()), 'baseline_ids': result['baseline_ids'], 'judge_ids': result['judge_ids'], 'timing_ms': result['timing_ms']}, indent=2))

"""Headless HotpotQA preparation and resumable one-shot retrieval evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from . import EmbeddingRetriever, Search, read_jsonl, candidate_bundles, combined_score


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def notify_started():
    """Only worker-launched allocated applications publish startup evidence."""
    run = os.environ.get('EVIDENCE_RUN_DIR')
    if not run:
        return
    import datetime
    import socket
    stage = os.environ['EVIDENCE_STAGE']
    job = os.environ.get('SLURM_JOB_ID')
    if not job or stage not in ('prepare', 'evaluate'):
        raise ValueError('Invalid application startup allocation metadata')
    run = Path(run).resolve()
    allocation = json.loads(os.environ['EVIDENCE_ALLOCATION_JSON'])
    marker = {'job_id': job, 'host': socket.gethostname(),
              'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'run_dir': str(run), 'allocation': allocation}
    atomic_json(run / 'receipts' / f'{stage}-STARTED.json', marker)
    receipt = run / 'receipts' / f'{stage}.json'
    import fcntl
    with (run / 'receipts' / f'{stage}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(receipt.read_text())
        state.update(application_state='STARTED', allocation=marker)
        atomic_json(receipt, state)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_rows(rows):
    documents, queries = {}, []
    for row in rows:
        context = row['context']
        pairs = zip(context['title'], context['sentences']) if isinstance(context, dict) else context
        local = {}
        for title, sentences in pairs:
            text = title + '\n' + ' '.join(sentences)
            ident = hashlib.sha256(text.encode()).hexdigest()
            documents[ident] = {'id': ident, 'title': title, 'text': text}
            local[title] = ident
        supporting = row['supporting_facts']
        titles = supporting['title'] if isinstance(supporting, dict) else [p[0] for p in supporting]
        required = sorted(set(titles))
        if not required or any(title not in local for title in required):
            raise ValueError('Supporting evidence is missing from supplied contexts')
        queries.append({'id': row['id'], 'text': row['question'], 'answer': row['answer'],
                        'gold_ids': sorted({local[title] for title in required}),
                        'supporting_titles': required})
    if len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Duplicate query IDs')
    if not queries:
        raise ValueError('Dataset sample is empty')
    return list(documents.values()), queries


def prepare(args):
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Preparation output must be empty; preserve existing artifacts')
    notify_started()
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revision = api.dataset_info('hotpotqa/hotpot_qa', revision=args.dataset_revision).sha
    model_revision = api.model_info(args.model, revision=args.model_revision).sha
    embedding_revision = api.model_info(args.embedding_model, revision=args.embedding_revision).sha
    ds_snapshot = snapshot_download('hotpotqa/hotpot_qa', repo_type='dataset', revision=revision,
        allow_patterns=['distractor/validation-*.parquet'], cache_dir=str(cache / 'hub'))
    parquet = sorted(str(p) for p in Path(ds_snapshot).glob('distractor/validation-*.parquet'))
    if not parquet:
        raise ValueError('Pinned dataset has no validation parquet files')
    ds = load_dataset('parquet', data_files={'validation': parquet}, split='validation', cache_dir=str(cache / 'datasets'))
    ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))
    documents, queries = convert_rows(ds)
    for name, rows in [('documents', documents), ('queries', queries)]:
        path = output / f'{name}.jsonl'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        temporary.replace(path)
    model_path = snapshot_download(args.model, revision=model_revision, cache_dir=str(cache / 'hub'),
                                   ignore_patterns=['*.gguf', '*.bin', '*.onnx', '*.h5', '*.msgpack'])
    embedding_path = snapshot_download(args.embedding_model, revision=embedding_revision, cache_dir=str(cache / 'hub'),
        ignore_patterns=['*.gguf', '*.bin', '*.onnx', '*.h5', '*.msgpack'])
    manifest = {'embedding_model': args.embedding_model, 'embedding_revision': embedding_revision,
        'embedding_path': embedding_path, 'embedding_config_hash': digest(Path(embedding_path) / 'config.json'), 'dataset': 'hotpotqa/hotpot_qa', 'config': 'distractor', 'split': 'validation',
        'dataset_revision': revision, 'model': args.model, 'model_revision': model_revision,
        'model_path': model_path, 'seed': args.seed, 'query_count': len(queries), 'document_count': len(documents),
        'corpus_scope': 'union of contexts from sampled validation queries; not full Wikipedia',
        'hashes': {name: digest(output / name) for name in ['documents.jsonl', 'queries.jsonl']}}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'PREPARED.json', {'manifest_sha256': digest(output / 'manifest.json')})
    print(json.dumps(manifest, indent=2), flush=True)


def validate_prepared(data):
    data = Path(data).resolve()
    ready = json.loads((data / 'PREPARED.json').read_text())
    if ready['manifest_sha256'] != digest(data / 'manifest.json'):
        raise ValueError('Preparation manifest hash mismatch')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name, expected in manifest['hashes'].items():
        if digest(data / name) != expected:
            raise ValueError(f'Data hash mismatch: {name}')
    return manifest


def evidence_metrics(selected, gold):
    selected, gold = set(selected), set(gold)
    if not gold:
        raise ValueError('Gold evidence must be nonempty')
    hits = len(selected & gold)
    return {'support_recall': hits / len(gold), 'support_precision': hits / len(selected) if selected else 0.0,
            'complete_support': float(gold <= selected)}


def summarize(records):
    import numpy as np
    metrics = {}
    for method in ('baseline', 'judged'):
        metrics[method] = {key: float(np.mean([r['metrics'][method][key] for r in records]))
            for key in ('support_recall', 'support_precision', 'complete_support')}
    timings = {}
    for label in ('baseline_online', 'judge_extra'):
        values = [r['timing_ms'][label] for r in records]
        timings[label] = {f'p{p}': float(np.percentile(values, p)) for p in (50, 95, 99)}
    return {'query_count': len(records), 'metrics': metrics, 'timing_ms': timings,
        'shortlist_complete_support': float(np.mean([r['shortlist_complete_support'] for r in records])),
        'scope': 'Sequential single-client retrieval benchmark; percentiles are NOT production SLAs.',
        'limitations': ['No answer generation or answer-accuracy measurement.',
            'Gold evidence membership is not a factual-correctness label.',
            'Coverage measured at paragraph/document level, not supporting-sentence level.',
            'Frozen soft-vector judge has no trained adapter; predictions are uncalibrated.',
            'Shared sampled corpus is not the official fullwiki or per-question distractor evaluation.']}


def evaluate(args):
    data = Path(args.data).resolve()
    manifest = validate_prepared(data)
    if args.model != manifest['model']:
        raise ValueError('Model must match prepared snapshot')
    if args.embedding_model != manifest.get('embedding_model'):
        raise ValueError('Embedding model must match prepared snapshot; prepare a fresh run for the new retriever')
    if digest(Path(manifest['embedding_path']) / 'config.json') != manifest['embedding_config_hash']:
        raise ValueError('Embedding model config hash mismatch')
    documents = read_jsonl(data / 'documents.jsonl')
    queries = read_jsonl(data / 'queries.jsonl')[:args.limit]
    if args.judge_batch_size < 1:
        raise ValueError('judge batch size must be positive')
    candidate_bundles(min(args.top_k, len(documents)), args.bundle_size, args.max_bundles)
    combined_score(0, 0, 0, args.weights)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {'manifest_hash': digest(data / 'manifest.json'), 'model_revision': manifest['model_revision'],
        'embedding_model': args.embedding_model, 'embedding_revision': manifest['embedding_revision'],
        'embedding_config_hash': manifest['embedding_config_hash'], 'model': args.model, 'device': args.device, 'top_k': args.top_k, 'bundle_size': args.bundle_size,
        'max_tokens': args.max_tokens, 'limit': args.limit, 'weights': args.weights, 'contextual': args.contextual,
        'max_bundles': args.max_bundles, 'judge_batch_size': args.judge_batch_size,
        'prefix_cache': not args.no_prefix_cache, 'code_hash': digest(Path(__file__)) + digest(Path(__file__).with_name('__init__.py'))}
    if args.resume:
        if json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration mismatch')
    else:
        if any(out.iterdir()):
            raise ValueError('Nonempty output; use --resume only for compatible results')
        atomic_json(out / 'config.json', config)
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    path = manifest['model_path']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=dtype).to(args.device)
    if sum(p.numel() for p in model.parameters()) >= 9_000_000_000:
        raise ValueError('Model must have fewer than 9B parameters')
    embedding_tokenizer = AutoTokenizer.from_pretrained(manifest['embedding_path'], local_files_only=True)
    embedding_model = AutoModel.from_pretrained(manifest['embedding_path'], local_files_only=True, dtype=dtype).to(args.device)
    if sum(p.numel() for p in embedding_model.parameters()) >= 9_000_000_000:
        raise ValueError('Embedding model must have fewer than 9B parameters')
    if embedding_model.config.hidden_size != model.get_input_embeddings().weight.shape[1]:
        raise ValueError('Embedding model width must match judge input width; no trained adapter is available')
    retriever = EmbeddingRetriever(embedding_model, embedding_tokenizer, args.device, args.max_tokens)
    runner = Search(model, tokenizer, args.device, args.max_tokens, False, args.judge_batch_size, not args.no_prefix_cache, retriever)
    notify_started()
    runner.sync()
    t0 = time.perf_counter()
    vectors, clipped = zip(*(runner.encode(d['text']) for d in documents))
    index = (torch.stack(vectors), clipped)
    runner.sync()
    indexing_ms = 1000*(time.perf_counter()-t0)
    atomic_json(out / 'index_metadata.json', {'indexing_ms': indexing_ms, 'documents': len(documents),
        'truncated_documents': sum(clipped), 'recomputed_on_resume': args.resume})
    records = []
    for query in queries:
        file = out / 'queries' / (hashlib.sha256(query['id'].encode()).hexdigest() + '.json')
        if args.resume and file.exists():
            result = json.loads(file.read_text())
            if result['query_id'] != query['id'] or result['query'] != query['text']:
                raise ValueError('Checkpoint query mismatch')
        else:
            # Gold answers/support labels are NOT passed into encoder/judge.
            result = runner.run(query['text'], documents, args.top_k, args.bundle_size,
                                args.weights, args.max_bundles, index=index)
            result['query_id'] = query['id']
            result['metrics'] = {'baseline': evidence_metrics(result['baseline_ids'], query['gold_ids']),
                                 'judged': evidence_metrics(result['judge_ids'], query['gold_ids'])}
            result['shortlist_complete_support'] = evidence_metrics([r['id'] for r in result['shortlist']], query['gold_ids'])['complete_support']
            atomic_json(file, result)
        records.append(result)
        from .graphs import write_graphs
        write_graphs(out, records, len(queries))
        print(json.dumps({'completed': len(records), 'total': len(queries), 'query_id': query['id']}), flush=True)
    summary = summarize(records)
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'SUCCEEDED.json', {'queries': len(records), 'summary_sha256': digest(out / 'summary.json')})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', choices=['hotpotqa'], default='hotpotqa')
    prep.add_argument('--output', required=True)
    prep.add_argument('--cache-dir', required=True)
    prep.add_argument('--model', default='Qwen/Qwen3-4B')
    prep.add_argument('--dataset-revision', default='main')
    prep.add_argument('--model-revision', default='main')
    prep.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-4B')
    prep.add_argument('--embedding-revision', default='main')
    prep.add_argument('--limit', type=int, default=200)
    prep.add_argument('--seed', type=int, default=17)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--data', required=True)
    ev.add_argument('--output', required=True)
    ev.add_argument('--model', default='Qwen/Qwen3-4B')
    ev.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-4B')
    ev.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    ev.add_argument('--top-k', type=int, default=5)
    ev.add_argument('--bundle-size', type=int, default=2)
    ev.add_argument('--max-bundles', type=int, default=30)
    ev.add_argument('--max-tokens', type=int, default=512)
    ev.add_argument('--limit', type=int, default=20)
    ev.add_argument('--weights', type=float, nargs=3, default=[1,1,1])
    ev.add_argument('--judge-batch-size', type=int, default=8)
    ev.add_argument('--no-prefix-cache', action='store_true')
    ev.add_argument('--contextual', action='store_true')
    ev.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('limit must be positive')
    if args.command == 'evaluate' and min(args.top_k, args.bundle_size, args.max_bundles, args.max_tokens) <= 0:
        parser.error('evaluation sizes must be positive')
    (prepare if args.command == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()

# One-shot evidence retrieval

Research prototype: frozen Qwen3-Embedding-4B contextual retriever → last-token pooling and L2 normalization → cosine shortlist → frozen Qwen3-4B prefill-only judge → final evidence bundle. No retry, generation or training loop. Two separate frozen models are resident. `--embedding-model` selects the retriever; `--model` selects the judge. Each must have fewer than 9B parameters, and their representation widths must match for the experimental soft-token interface.

## Remote execution

The primary workflow is now SSH/Slurm: see [remote workflow](docs/remote.md). Start with `python3 deploy/scripts/pipeline.py submit --account YOUR_ACCOUNT`. Partition is optional; runs and caches default to `runs/` and `cache/huggingface/` under the project root. CPU preparation installs dependencies and downloads pinned HotpotQA validation/model snapshots; GPU evaluation runs offline. This revised two-model configuration has not been verified on allocated hardware.

## Small local example (optional)

```
uv sync --locked
uv run one-shot-evidence-retrieval --documents examples/documents.jsonl --query "Who wrote Silver Orchard and which university did the author attend?" --device mps --output outputs/comparison.json
```

First use downloads model weights; allow storage for both model snapshots. Two 4B models use roughly 16 GB for FP16 weights alone, excluding activations, KV caches and runtime overhead; actual memory is unmeasured. Use `--local-files-only` for an already cached model. CPU defaults to FP32; MPS/CUDA use FP16. Use the remote workflow for cluster execution. Model loading is excluded from timing.

The comparison returns both cosine-only top-m and judged selection from the SAME top-k pool. `--baseline-only` skips judging. `--top-k 5 --bundle-size 2` evaluates all ten pairs with two logical judgments per pair, processed in batches. This is one-shot search with multiple bundle evaluations, not a single judge call. `--max-bundles` bounds cost. Coverage is set-level. The shortlist is never expanded. The baseline and judged method select the same number of documents, not necessarily equal text-token budgets.

## Exact experimental interface

The retriever uses the dedicated `Qwen/Qwen3-Embedding-4B` model via `AutoModel`, taking the last attended token of its final contextual hidden states and L2-normalizing it. Queries receive the instruction `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}`; documents receive no instruction. Vectors are indexed once per invocation; the index is not persisted. The legacy `--contextual` flag does not change the dedicated retriever, which is always contextual.

The judge does NOT receive original document text. Each normalized query/document vector becomes one soft token in a textual instruction prompt via `inputs_embeds`. A fixed scale matches the median input-token norm; there is no trained adapter. This is an intentionally unvalidated compressed-vector interface, not a claim that pretrained LLMs understand pooled vectors. Dedicated retrieval training does not fix the frozen judge interface mismatch automatically. No vector truncation or arbitrary projection is used; incompatible widths fail clearly. Empty or malformed data is rejected; token truncation is recorded.

Coverage asks whether the bundle addresses every query requirement. Correctness asks whether its claims are factually correct and applicable. Without reference evidence this measures the model's judgment, NOT verified factual truth. Both use conditional yes/no logits after prefill, with no decoding. Reports include total vocabulary probability mass on those labels: low mass indicates poor adherence to the readout. Conditional probabilities are not calibrated confidence.

Cosine is mapped from [-1,1] to [0,1]. Bundle rank is the weighted average of mean relevance, coverage and correctness (`--weights 1 1 1`). These heuristic weights require validation. The winner is selected among size-m subsets of the cosine shortlist. Stored timings separate indexing, baseline query/ranking, and additional judging; synchronize accelerators. They are single-run observations, not P95/P99 results.

## Research boundaries

The code only implements forward inference; it does not train an encoder, adapter or judge. Both branches currently use frozen weights. Validate the judge interface on labeled bundles before drawing quality conclusions. Full pretrained execution and useful retrieval quality require separate measurement. Fixtures are synthetic and contain no evaluation labels; a toy example is not an accuracy benchmark.

## Checks

Independent tests use tiny random Transformers to verify plumbing, frozen weights, no decoding, finite scores and candidate boundaries. They do not validate Qwen's semantic judgments or hardware latency.

Sources: https://huggingface.co/Qwen/Qwen3-Embedding-4B and https://huggingface.co/Qwen/Qwen3-4B

## Real-data benchmark

`python -m one_shot_evidence_retrieval.benchmark prepare` creates a shared corpus from the contexts of a seeded HotpotQA validation sample. The dataset, judge and embedding-model revisions, embedding config hash and data hashes are recorded. Both model snapshots are prepared; GPU loading is offline. `evaluate` builds embeddings once, searches the shared corpus for every query, and compares baseline with judged bundles. Gold answers and supporting IDs are never passed into the judge. Each completed query is saved atomically; explicit compatible resume reuses those results.

Metrics include gold supporting-document recall, precision, complete coverage, shortlist coverage ceiling, and latency percentiles. These are document-level evidence metrics, not factual truth or generated-answer accuracy. The corpus is a sampled context union, not full Wikipedia and not the official per-query distractor protocol. The frozen judge remains unvalidated. Dataset: https://hotpotqa.github.io/ ; hosted data: https://huggingface.co/datasets/hotpotqa/hotpot_qa .

## Progress graphs

GPU evaluation automatically writes `results/progress.html` and `results/progress.json` after every completed query, including resumed queries. Download the standalone HTML and open it in a browser; reload after fetching updates. No server, plotting dependency or external assets are needed. Curves show running evidence recall/complete coverage for both methods, per-query cosine and judging latency, selected-bundle coverage/correctness predictions, and yes/no vocabulary mass. These are evaluation curves: the current project has no training loop or losses. They do not establish factual correctness or calibrated judgment probabilities.

## Batched prefill-only judging

The judge never calls `generate`, samples an output token, or executes a decoding loop: it reads final-position yes/no logits. Default judging now batches up to 8 independent prompts (`--judge-batch-size 8`) and reuses the exact common instruction/query prefix KV cache once per query. Equal-length prompts are grouped to avoid padding. Each suffix batch receives its own cache copy; cache state is not reused across queries or persisted. Prefix caching is a two-part prefill, not token generation. Final vocabulary logits are computed only at the last suffix position.

Ten document pairs still imply 20 logical judgments, but fewer model forwards. Reports distinguish `judge_calls` (logical judgments), `judge_forward_calls` (including prefix prefill), and `judge_prefix_tokens`. Use `--judge-batch-size 1 --no-prefix-cache` to compare with sequential execution, and `--no-prefix-cache` alone for batching without KV reuse. GPU memory grows with batch size. Speedup is not guaranteed; benchmark on allocated hardware. The judge's compressed-vector semantics are unchanged.

Start a fresh run after updating code; prior run configuration and checkpoint hashes prevent silently mixing implementations on resume. Preparation currently runs again for a fresh run, using existing cached downloads.

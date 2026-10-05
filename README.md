# One-shot evidence retrieval

Research prototype: shared LLM encoder → masked-by-construction mean pooling and L2 normalization → cosine shortlist → frozen prefill-only judge → final evidence bundle. No retry, no generation, no training loop. Default model: Qwen/Qwen3-4B (under 9B); also accepts compatible Qwen3 sizes or local model paths. One frozen model instance is reused sequentially for encoding and judging.

## Remote execution

The primary workflow is now SSH/Slurm: see [remote workflow](docs/remote.md). Supply the account in the submission command. CPU preparation installs dependencies and downloads pinned HotpotQA validation/model snapshots; GPU evaluation runs offline. No remote host or workload has been executed yet.

## Small local example (optional)

```
uv sync --locked
uv run one-shot-evidence-retrieval --documents examples/documents.jsonl --query "Who wrote Silver Orchard and which university did the author attend?" --device mps --output outputs/comparison.json
```

First use downloads model weights; allow several GB of storage. Use `--local-files-only` for an already cached model. CPU defaults to FP32; MPS/CUDA use FP16. Use the remote workflow for cluster execution. Model loading is excluded from timing.

The comparison returns both cosine-only top-m and judged selection from the SAME top-k pool. `--baseline-only` skips judging. `--top-k 5 --bundle-size 2` evaluates all ten pairs with two judge forwards per pair. This is one-shot search with multiple bundle evaluations, not a single judge call. `--max-bundles` bounds cost. Coverage is set-level. The shortlist is never expanded. The baseline and judged method select the same number of documents, not necessarily equal text-token budgets.

## Exact experimental interface

By default the encoder uses the LLM's literal INPUT embedding table (not an embedding-trained retrieval model). Token vectors are mean-pooled then normalized. `--contextual` instead pools final contextual hidden states. Full document vectors are indexed once per invocation; this initial implementation does not persist an index.

The judge does NOT receive original document text. Each normalized query/document vector becomes one soft token in a textual instruction prompt via `inputs_embeds`. A fixed scale matches the median input-token norm; there is no trained adapter. This is an intentionally unvalidated compressed-vector interface, not a claim that pretrained LLMs understand pooled vectors. Contextual mode does not fix that mismatch automatically. Empty or malformed data is rejected; token truncation is recorded.

Coverage asks whether the bundle addresses every query requirement. Correctness asks whether its claims are factually correct and applicable. Without reference evidence this measures the model's judgment, NOT verified factual truth. Both use conditional yes/no logits after prefill, with no decoding. Reports include total vocabulary probability mass on those labels: low mass indicates poor adherence to the readout. Conditional probabilities are not calibrated confidence.

Cosine is mapped from [-1,1] to [0,1]. Bundle rank is the weighted average of mean relevance, coverage and correctness (`--weights 1 1 1`). These heuristic weights require validation. The winner is selected among size-m subsets of the cosine shortlist. Stored timings separate indexing, baseline query/ranking, and additional judging; synchronize accelerators. They are single-run observations, not P95/P99 results.

## Research boundaries

The code only implements forward inference; it does not train an encoder, adapter or judge. Both branches currently use frozen weights. Literal embedding-layer mean pooling discards word order; treat it as an architectural baseline, not a strong semantic retriever. Validate the judge interface on labeled bundles before drawing quality conclusions. Full pretrained execution and useful retrieval quality require separate measurement. Fixtures are synthetic and contain no evaluation labels; a toy example is not an accuracy benchmark.

## Checks

Independent tests use tiny random Transformers to verify plumbing, frozen weights, no decoding, finite scores and candidate boundaries. They do not validate Qwen's semantic judgments or hardware latency.

Sources: https://huggingface.co/Qwen/Qwen3-4B

## Real-data benchmark

`python -m one_shot_evidence_retrieval.benchmark prepare` creates a shared corpus from the contexts of a seeded HotpotQA validation sample. The dataset/model revisions and data hashes are recorded. `evaluate` builds embeddings once, searches the shared corpus for every query, and compares baseline with judged bundles. Gold answers and supporting IDs are never passed into the judge. Each completed query is saved atomically; explicit compatible resume reuses those results.

Metrics include gold supporting-document recall, precision, complete coverage, shortlist coverage ceiling, and latency percentiles. These are document-level evidence metrics, not factual truth or generated-answer accuracy. The corpus is a sampled context union, not full Wikipedia and not the official per-query distractor protocol. The frozen judge remains unvalidated. Dataset: https://hotpotqa.github.io/ ; hosted data: https://huggingface.co/datasets/hotpotqa/hotpot_qa .

## Progress graphs

GPU evaluation automatically writes `results/progress.html` and `results/progress.json` after every completed query, including resumed queries. Download the standalone HTML and open it in a browser; reload after fetching updates. No server, plotting dependency or external assets are needed. Curves show running evidence recall/complete coverage for both methods, per-query cosine and judging latency, selected-bundle coverage/correctness predictions, and yes/no vocabulary mass. These are evaluation curves: the current project has no training loop or losses. They do not establish factual correctness or calibrated judgment probabilities.

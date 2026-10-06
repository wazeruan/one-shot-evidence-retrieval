# SSH / Slurm benchmark workflow

This project is ready to copy to an SSH-accessible Slurm cluster. The prior one-model run was reported by the user; this revised two-model configuration and its hardware performance remain unverified. Before submission, verify the site's account, partition, modules, GPU resource names, network/download policy, and storage policy. `gpu:h100:1` and 128 GB host RAM are configurable defaults, not claims about a particular cluster. Host RAM is separate from GPU VRAM.

Copy the checkout, including its lock file but excluding `.venv`, caches, and outputs, to an approved shared project directory. Keep the project on storage accessible to both CPU and GPU nodes. By default runs go under `PROJECT_ROOT/runs/run-TIMESTAMP/` and model/data caches under `PROJECT_ROOT/cache/huggingface/`; both are Git-ignored. The dependency cache also stays under `PROJECT_ROOT/cache/uv/`. Optional `--output` and `--cache-dir` override these locations. The scripts never change scheduler-assigned GPU visibility.

From the remote checkout, one command submits CPU installation/data/model preparation followed by GPU evaluation with an `afterok` dependency:

```bash
python3 deploy/scripts/pipeline.py submit --account YOUR_ACCOUNT
```

Replace `YOUR_ACCOUNT` with your Slurm account. No partition is passed by default; the scheduler uses its configured default. Optional `--partition` remains available. `uv` must already be available on compute nodes, either on PATH or through repeated `--module NAME` arguments. CPU preparation performs `uv sync --frozen`, downloads real HotpotQA data and both Qwen3-Embedding-4B retriever and Qwen3-4B judge weights, and saves the pinned preparation manifest. GPU evaluation uses offline cached weights and compares cosine-only retrieval with frozen pooled-vector judging. Installation/downloads happen inside the CPU allocation. No workload runs on the login node.

`--gres`, `--memory`, `--cpus`, `--prepare-time`, `--evaluate-time`, `--qos`, and `--reservation` configure scheduler resources. `--min-vram-gb` defaults to 24; `--require-bf16` optionally checks BF16 support. Two frozen 4B models reside on the GPU; FP16 weights alone are roughly 16 GB before activation/cache/runtime overhead. The 24 GB preflight threshold is a conservative configuration, not measured memory use. Actual memory use and throughput still require allocated-node measurements. The wrapper checks at least `--min-free-gb` free space (default 30 GB); allow more for large datasets and caches.

Fresh output names are atomically claimed and receive numbered suffixes on collision. Save the printed `RUN_DIR`. Submission stdout and stderr remain separate; job IDs, commands, timestamps, application states, allocation evidence, and timestamped logs are persisted under that run directory. Successful submission is not successful execution. Startup monitoring is bounded and reports pending or unknown jobs without canceling or resubmitting them.

```bash
python3 deploy/scripts/pipeline.py status /absolute/RUN_DIR
python3 deploy/scripts/pipeline.py results /absolute/RUN_DIR
python3 deploy/scripts/pipeline.py submit --resume /absolute/RUN_DIR
```

Resume preserves the saved configuration and active jobs. Successful completed stages are reused only after `sacct` confirms `COMPLETED|0:0` and the expected application marker and artifact hashes validate. Active dependencies are retained. Explicit resume may retry a scheduler-confirmed failed evaluation after verified successful preparation, preserving the prior receipt and per-query checkpoints. Failed preparation, rejected or unverifiable submissions block resume: inspect receipts and create a fresh run after diagnosis. There is no automatic retry of a scientific workload. The evaluation command uses compatible per-query checkpoints through `--resume`.

For scheduler diagnosis, take job IDs and log paths from receipts:

```bash
squeue -j JOB_ID --format='%.18i %.12T %.30R %.10M'
scontrol show job JOB_ID
sacct -X -j JOB_ID --format=JobID,State,ExitCode,Elapsed,Start,End
tail -n 100 /absolute/RUN_DIR/logs/evaluate-TIMESTAMP-JOB_ID.err
```

The CPU marker is `data/PREPARED.json`; the GPU marker is `results/SUCCEEDED.json`. `results` reports completion only with both application evidence and successful scheduler accounting. Each stage has a separate atomic receipt. Preserve receipts, logs, dataset manifests, and results for reproducibility.

The frozen pooled-vector judge interface remains experimental: successful scheduler execution does not establish that coverage/correctness scores are calibrated or scientifically valid. This workflow measures one-shot retrieval, not training or adaptive retries.

## Progress graphs

GPU evaluation automatically writes `results/progress.html` and `results/progress.json` after every completed query, including resumed queries. Download the standalone HTML and open it in a browser; reload after fetching updates. No server, plotting dependency or external assets are needed. Curves show running evidence recall/complete coverage for both methods, per-query cosine and judging latency, selected-bundle coverage/correctness predictions, and yes/no vocabulary mass. These are evaluation curves: the current project has no training loop or losses. They do not establish factual correctness or calibrated judgment probabilities.

## Batched prefill-only judging

The judge never calls `generate`, samples an output token, or executes a decoding loop: it reads final-position yes/no logits. Default judging now batches up to 8 independent prompts (`--judge-batch-size 8`) and reuses the exact common instruction/query prefix KV cache once per query. Equal-length prompts are grouped to avoid padding. Each suffix batch receives its own cache copy; cache state is not reused across queries or persisted. Prefix caching is a two-part prefill, not token generation. Final vocabulary logits are computed only at the last suffix position.

Ten document pairs still imply 20 logical judgments, but fewer model forwards. Reports distinguish `judge_calls` (logical judgments), `judge_forward_calls` (including prefix prefill), and `judge_prefix_tokens`. Use `--judge-batch-size 1 --no-prefix-cache` to compare with sequential execution, and `--no-prefix-cache` alone for batching without KV reuse. GPU memory grows with batch size. Speedup is not guaranteed; benchmark on allocated hardware. The judge's compressed-vector semantics are unchanged.

Start a fresh run after updating code; prior run configuration and checkpoint hashes prevent silently mixing implementations on resume. Preparation currently runs again for a fresh run, using existing cached downloads.

## Dedicated embedding retriever

Fresh runs now use `Qwen/Qwen3-Embedding-4B` for contextual last-token embeddings, with query instructions and L2 normalization. The frozen `Qwen/Qwen3-4B` judge still consumes compressed vectors and remains unvalidated. Use `--embedding-model` and `--embedding-revision` to configure preparation; judge selection remains `--model`. Both snapshots are pinned in the prepared manifest and loaded offline in evaluation. Start a fresh run: older preparation manifests and result checkpoints cannot be resumed with the new retriever. Cached downloads are reused.

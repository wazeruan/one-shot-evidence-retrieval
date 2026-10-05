# SSH / Slurm benchmark workflow

This project is ready to copy to an SSH-accessible Slurm cluster. No host, allocation, site policy, or actual GPU run has yet been verified. Before submission, verify the site's account, partition, modules, GPU resource names, network/download policy, and storage policy. `gpu:h100:1` and 128 GB host RAM are configurable defaults, not claims about a particular cluster. Host RAM is separate from GPU VRAM.

Copy the checkout, including its lock file but excluding `.venv`, caches, and outputs, to an approved shared project directory. Supply a site-approved cache directory accessible to both CPU and GPU nodes. The scripts never change scheduler-assigned GPU visibility.

From the remote checkout, one command submits CPU installation/data/model preparation followed by GPU evaluation with an `afterok` dependency:

```bash
python3 deploy/scripts/pipeline.py submit \
  --output /approved/scratch/experiment-001 \
  --cache-dir /approved/scratch/hf-cache \
  --account YOUR_ACCOUNT --partition YOUR_PARTITION \
  --model Qwen/Qwen3-4B --limit 20 \
  --top-k 5 --bundle-size 2 --max-bundles 30 --max-tokens 2048
```

The paths and site flags are placeholders: replace them with verified values. `uv` must already be available on compute nodes, either on PATH or through repeated `--module NAME` arguments. CPU preparation performs `uv sync --frozen`, downloads real HotpotQA data and model weights, and saves the pinned preparation manifest. GPU evaluation uses offline cached weights and compares cosine-only retrieval with frozen pooled-vector judging. Installation/downloads happen inside the CPU allocation. No workload runs on the login node.

`--gres`, `--memory`, `--cpus`, `--prepare-time`, `--evaluate-time`, `--qos`, and `--reservation` configure scheduler resources. `--min-vram-gb` defaults to 16; `--require-bf16` optionally checks BF16 support. Actual memory use and throughput still require allocated-node measurements. The wrapper checks at least `--min-free-gb` free space (default 30 GB); allow more for large datasets and caches.

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

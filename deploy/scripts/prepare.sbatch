#!/usr/bin/env bash
#SBATCH --job-name=evidence-prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=prepare-%j.out
#SBATCH --error=prepare-%j.err
set -Eeuo pipefail
exec python3 "./deploy/scripts/pipeline.py" worker prepare "$1"

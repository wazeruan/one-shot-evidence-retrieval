#!/usr/bin/env bash
#SBATCH --job-name=evidence-evaluate
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=evaluate-%j.out
#SBATCH --error=evaluate-%j.err
set -Eeuo pipefail
exec python3 "./deploy/scripts/pipeline.py" worker evaluate "$1"

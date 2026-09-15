#!/bin/bash

#SBATCH --job-name=lm_eval_zeroshot_llama-3.1-8b_gsm8k
#SBATCH --output=.log/lm_eval_zeroshot_llama-3.1-8b_gsm8k/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_zeroshot_llama-3.1-8b_gsm8k/slurm_backup/job_%j.err
#SBATCH --time=12:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Plain zeroshot GSM8K eval of B = meta-llama/Llama-3.1-8B, no rebasing, no steering,
# no dependency on lm_eval/rebase or any rebase config -- a single ordinary "hf" model
# run through lm_eval. num_fewshot comes from gsm8k.yaml's own default (5-shot).
#
# Memory (calculated, not measured): a single 8B model in bf16 is ~16 GB of VRAM
# weights, no sources loaded alongside it, no CPU-side covariances/deltas -- much
# lighter than the rebased runs. --mem=32G is a generous rounding, not a measurement;
# check `seff <jobid>` after the run to size the next one tighter.
#
# Before submitting: the HF token used here must have accepted the meta-llama license
# for Llama-3.1-8B (HF_HOME=${WORK_ROOT}/hf_cache huggingface-cli login, or export
# HF_TOKEN below).

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_zeroshot_llama-3.1-8b_gsm8k"
mkdir -p "$LOG_DIR/slurm_backup"

CUSTOM_LOG="${LOG_DIR}/job_${SLURM_JOB_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${CUSTOM_LOG}"

export PYTHONUNBUFFERED=1
# Silence transformers' per-tensor "Loading weights" progress bars.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="${PROJECT_ROOT}"
export HF_HOME="${WORK_ROOT}/hf_cache"
# export HF_TOKEN=...  # alternative to huggingface-cli login for the gated meta-llama repo

OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/zeroshot/llama-3.1-8b_gsm8k"

echo "--- zeroshot: B=Llama-3.1-8B, no rebasing ---"
python -m lm_eval run \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16 \
  --tasks gsm8k \
  --batch_size 8 \
  --output_path "$OUTPUT_PATH" \
  --log_samples

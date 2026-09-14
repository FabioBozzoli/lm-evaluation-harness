#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_smoke_test_math_theseus
#SBATCH --output=.log/lm_eval_rebase_smoke_test_math_theseus/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_rebase_smoke_test_math_theseus/slurm_backup/job_%j.err
#SBATCH --time=04:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Smoke test of theseus on the math pair:
#   A = Llama-3.2-3B-Instruct -> Llama-3.2-3B_math, B = meta-llama/Llama-3.2-3B,
#   calibration on dart-math-hard, evaluation on 5 GSM8K problems.
# Scores on 5 problems mean nothing; check that the run completes and that the log
# prints "[theseus] prepare: computed transforms = N" with N > 0.
#
# Step 1: unit tests on tiny random models (fast, CPU only).
# Step 2: the math theseus config end to end with --limit 5.
#
# Resources: theseus on 3B models needs ~110 GB of CPU RAM and ~32 GB of GPU
# (24 GB cards are excluded above).
#
# Before submitting: source_finetuned in the math theseus config must point to the
# Llama-3.2-3B_math directory, and the HF token must have access to the gated
# meta-llama repos (HF_HOME=${WORK_ROOT}/hf_cache huggingface-cli login, or export HF_TOKEN).

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" pytest from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
cd "$PROJECT_ROOT"

RUN="theseus"

LOG_DIR=".log/lm_eval_rebase_smoke_test_math_theseus"
mkdir -p "$LOG_DIR/slurm_backup"

CUSTOM_LOG="${LOG_DIR}/job_${SLURM_JOB_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Run: ${RUN}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${CUSTOM_LOG}"

export PYTHONUNBUFFERED=1
# Silence transformers' per-tensor "Loading weights" progress bars.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="${PROJECT_ROOT}"
export HF_HOME="${WORK_ROOT}/hf_cache"
# export HF_TOKEN=...  # alternative to huggingface-cli login for the gated meta-llama repos

CONFIG="configs/rebase/math_llama-3.2-3b-instruct-math_to_llama-3.2-3b_${RUN}.yaml"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/smoke_test_math/${SLURM_JOB_ID}/${RUN}"

echo "--- Step 1: unit tests ---"
python -m pytest tests/models/test_hf_rebased.py -v || exit 1

echo "--- Step 2: ${RUN} on 5 GSM8K problems ---"
python -m lm_eval run \
  --config "$CONFIG" \
  --tasks gsm8k \
  --limit 5 \
  --output_path "$OUTPUT_PATH" || exit 1

echo "--- math ${RUN} smoke test passed ---"

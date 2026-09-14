#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_smoke_test
#SBATCH --output=.log/lm_eval_rebase_smoke_test/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_rebase_smoke_test/slurm_backup/job_%j.err
#SBATCH --time=04:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Smoke test of the rebase methods ported into lm-eval (lm_eval/rebase + the "rebased"
# model in lm_eval/models/hf_rebased.py). Run this first: it is short and fails fast.
#
# Step 1: unit tests on tiny random Qwen2 models (theseus across widths, steer_text
#         global_ridge and block_ridge on lm_head, hooks restored on exit). CPU only.
# Step 2: the three code configs end to end on 5 HumanEval problems each
#         (A=Qwen2.5-0.5B -> Qwen2.5-Coder-0.5B, B=Qwen2.5-1.5B, calibration on
#         Magicoder-OSS-Instruct). Scores on 5 problems mean nothing; what matters is
#         that every run completes and the steer_text logs print stage0/1/2 accuracies.

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" pytest from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_rebase_smoke_test"
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
# humaneval executes the generated code to score it.
export HF_ALLOW_CODE_EVAL=1

CONFIG_PREFIX="configs/rebase/code_qwen2.5-0.5b-coder_to_qwen2.5-1.5b"
OUTPUT_ROOT="${WORK_ROOT}/lm_eval_results/smoke_test/${SLURM_JOB_ID}"

echo "--- Step 1: unit tests ---"
python -m pytest tests/models/test_hf_rebased.py -v || exit 1

for RUN in theseus steer_global_ridge steer_block_ridge; do
  echo "--- Step 2: ${RUN} on 5 HumanEval problems ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_${RUN}.yaml" \
    --tasks humaneval \
    --limit 5 \
    --output_path "${OUTPUT_ROOT}/${RUN}" || exit 1
done

echo "--- smoke test passed ---"

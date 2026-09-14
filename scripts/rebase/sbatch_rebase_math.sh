#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_math
#SBATCH --output=.log/lm_eval_rebase_math/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/lm_eval_rebase_math/slurm_backup/job_%A_%a.err
#SBATCH --time=24:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod
#SBATCH --array=0-2

# Job array comparing, on GSM8K, the target B=Qwen2.5-1.5B:
#   0 baseline            B as is (plain "hf" model, no rebasing)
#   1 theseus             + A's math task vector transported into B's weights
#   2 steer_block_ridge   + steer_text correction (linear regime, block_ridge)
# with A = Qwen2.5-0.5B -> your DART-Math finetune, calibration on dart-math-hard.
#
# TODO before submitting: set source_finetuned in both math configs under
# configs/rebase/ to your DART-Math finetune (Hugging Face model directory).

RUNS=(baseline theseus steer_block_ridge)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/tesi_bcalderara/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_rebase_math/${RUN}"
mkdir -p "$LOG_DIR"
mkdir -p ".log/lm_eval_rebase_math/slurm_backup"

CUSTOM_LOG="${LOG_DIR}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Array job ID: ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
echo "Run: ${RUN}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${CUSTOM_LOG}"

export PYTHONUNBUFFERED=1
# Silence transformers' per-tensor "Loading weights" progress bars.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="${PROJECT_ROOT}"
export HF_HOME="${WORK_ROOT}/hf_cache"

CONFIG_PREFIX="configs/rebase/math_qwen2.5-0.5b-dartmath_to_qwen2.5-1.5b"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/math/${RUN}"

if [ "$RUN" = "baseline" ]; then
  echo "--- baseline: Qwen2.5-1.5B without rebasing ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_theseus.yaml" \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-1.5B,dtype=bfloat16 \
    --output_path "$OUTPUT_PATH"
else
  echo "--- ${RUN}: A=Qwen2.5-0.5B -> DART-Math finetune, B=Qwen2.5-1.5B ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_${RUN}.yaml" \
    --output_path "$OUTPUT_PATH"
fi

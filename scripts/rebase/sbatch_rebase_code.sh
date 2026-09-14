#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_code
#SBATCH --output=.log/lm_eval_rebase_code/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/lm_eval_rebase_code/slurm_backup/job_%A_%a.err
#SBATCH --time=24:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod
#SBATCH --array=0-3

# Job array comparing, on HumanEval + MBPP, the target B=Qwen2.5-1.5B:
#   0 baseline            B as is (plain "hf" model, no rebasing)
#   1 theseus             + A's code task vector transported into B's weights
#   2 steer_global_ridge  + steer_text correction (standard regime, global_ridge)
#   3 steer_block_ridge   + steer_text correction (linear regime, block_ridge)
# with A = Qwen2.5-0.5B -> Qwen2.5-Coder-0.5B and calibration on Magicoder-OSS-Instruct.
# Rebasing helps when 1-3 beat 0. All runs share the tasks and generation settings of
# the configs under configs/rebase/; the baseline reuses the theseus config with the
# model swapped on the command line (--model/--model_args override the YAML wholesale).
#
# To use your own Magicoder finetune, change source_finetuned in the three configs.

RUNS=(baseline theseus steer_global_ridge steer_block_ridge)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/tesi_bcalderara/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_rebase_code/${RUN}"
mkdir -p "$LOG_DIR"
mkdir -p ".log/lm_eval_rebase_code/slurm_backup"

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
# humaneval/mbpp execute the generated code to score it.
export HF_ALLOW_CODE_EVAL=1

CONFIG_PREFIX="configs/rebase/code_qwen2.5-0.5b-coder_to_qwen2.5-1.5b"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/code/${RUN}"

if [ "$RUN" = "baseline" ]; then
  echo "--- baseline: Qwen2.5-1.5B without rebasing ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_theseus.yaml" \
    --model hf \
    --model_args pretrained=Qwen/Qwen2.5-1.5B,dtype=bfloat16 \
    --output_path "$OUTPUT_PATH"
else
  echo "--- ${RUN}: A=Qwen2.5-0.5B -> Qwen2.5-Coder-0.5B, B=Qwen2.5-1.5B ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_${RUN}.yaml" \
    --output_path "$OUTPUT_PATH"
fi

#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_math_3b_to_8b_steer_global_ridge
#SBATCH --output=.log/lm_eval_rebase_math_3b_to_8b_steer_global_ridge/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_rebase_math_3b_to_8b_steer_global_ridge/slurm_backup/job_%j.err
#SBATCH --time=24:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# steer_text (standard regime + global_ridge) on GSM8K: A = Llama-3.2-3B-Instruct ->
# Llama-3.2-3B_math (your finetune), B = meta-llama/Llama-3.1-8B. The affordable
# steer_text path -- see the matching config for the BLOCKING PREREQUISITE (a
# vocab-width mismatch between A's finetuned head and B's, same issue as the
# block_ridge variant) that must be resolved before this run will complete.
#
# Memory (calculated, not measured -- see the config file for the full breakdown):
# - VRAM  ~32-35 GB (B bf16 ~16 GB + two 3B-family sources bf16 ~6.4 GB each + a
#   normal forward pass's activation memory -- no jvp doubling, unlike block_ridge).
#   --constraint above targets 45-96 GB of headroom.
# - RAM   ~30-40 GB estimated (cached calibration features + Stage 1's pinv(B's
#   [128256, 4096] lm_head), transiently heavier during the SVD itself).
#   --mem=64G above is a generous rounding, not a measurement.
# After a run, check `seff <jobid>` (peak RSS, GPU util) to size the next one tighter.
#
# Before submitting:
# - set source_finetuned in the matching config to the Llama-3.2-3B_math directory;
# - resolve the vocab-mismatch prerequisite above;
# - the HF token used here must have accepted the meta-llama license for both
#   Llama-3.2-3B-Instruct and Llama-3.1-8B (HF_HOME=${WORK_ROOT}/hf_cache
#   huggingface-cli login, or export HF_TOKEN below).

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_rebase_math_3b_to_8b_steer_global_ridge"
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
# export HF_TOKEN=...  # alternative to huggingface-cli login for the gated meta-llama repos

CONFIG="configs/rebase/math_llama-3.2-3b-instruct-math_to_llama-3.1-8b_steer_global_ridge.yaml"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/math_llama-3.2-3b_to_llama-3.1-8b/steer_global_ridge"

echo "--- steer_text global_ridge: A=Llama-3.2-3B-Instruct -> Llama-3.2-3B_math, B=Llama-3.1-8B ---"
python -m lm_eval run \
  --config "$CONFIG" \
  --output_path "$OUTPUT_PATH"

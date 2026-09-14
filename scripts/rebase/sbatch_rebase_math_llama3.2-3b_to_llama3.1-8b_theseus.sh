#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_math_3b_to_8b_theseus
#SBATCH --output=.log/lm_eval_rebase_math_3b_to_8b_theseus/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_rebase_math_3b_to_8b_theseus/slurm_backup/job_%j.err
#SBATCH --time=24:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# theseus on GSM8K: A = Llama-3.2-3B-Instruct -> Llama-3.2-3B_math (your finetune),
# B = meta-llama/Llama-3.1-8B, aligned data-free (weight SVDs, no calibration forward
# passes -- see the config's comment on why "activations" mode is not the default here).
#
# Memory (calculated, not measured -- see the config file for the full breakdown):
# - VRAM  ~29 GB base (B bf16 ~16 GB + two 3B-family sources bf16 ~6.4 GB each).
#   --constraint above targets 45 GB cards. The 96 GB gpu_RTXPro6000B card (Blackwell,
#   sm_120) is NOT usable on this cluster's current PyTorch (2.1.2+cu121, compiled up
#   to sm_90 only -- jobs landing on it fail with "no kernel image is available for
#   execution on the device"); it is deliberately left out of the constraint above.
#   If a 45 GB card OOMs here, there is no larger *compatible* card to fall back to --
#   lower calib_batch_size/calib_max_length, or ask about upgrading PyTorch to a build
#   with sm_120 support (CUDA 12.4+/12.6+), which would reopen that card as an option.
# - RAM   ~70 GB estimated (three float32 CPU backbone copies + the task-vector delta).
#   --mem=128G above is a generous rounding, not a measurement.
# After a run, check `seff <jobid>` (peak RSS, GPU util) to size the next one tighter.
#
# Before submitting:
# - set source_finetuned in the matching config to the Llama-3.2-3B_math directory;
# - the HF token used here must have accepted the meta-llama license for both
#   Llama-3.2-3B-Instruct and Llama-3.1-8B (HF_HOME=${WORK_ROOT}/hf_cache
#   huggingface-cli login, or export HF_TOKEN below).

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_rebase_math_3b_to_8b_theseus"
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

CONFIG="configs/rebase/math_llama-3.2-3b-instruct-math_to_llama-3.1-8b_theseus.yaml"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/math_llama-3.2-3b_to_llama-3.1-8b/theseus"

echo "--- theseus: A=Llama-3.2-3B-Instruct -> Llama-3.2-3B_math, B=Llama-3.1-8B ---"
python -m lm_eval run \
  --config "$CONFIG" \
  --output_path "$OUTPUT_PATH"

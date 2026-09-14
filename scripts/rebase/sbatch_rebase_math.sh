#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_math
#SBATCH --output=.log/lm_eval_rebase_math/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/lm_eval_rebase_math/slurm_backup/job_%A_%a.err
#SBATCH --time=24:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod
#SBATCH --array=0-2

# Job array comparing, on GSM8K, the target B=meta-llama/Llama-3.2-3B:
#   0 baseline            B as is (plain "hf" model, no rebasing)
#   1 theseus             + A's math task vector transported into B's weights
#   2 steer_block_ridge   + steer_text correction (linear regime, block_ridge)
# with A = Llama-3.2-3B-Instruct -> Llama-3.2-3B_math (your finetune of Instruct),
# calibration on dart-math-hard. A and B share the architecture (28 layers, hidden 3072)
# and the tokenizer, so no block grouping is needed.
#
# Resources (3B models):
# - GPU: the two source models load in the target's own dtype (bf16 by default, ~6.4 GB
#   each) next to B in bf16, ~29 GB before evaluation -> 24 GB cards are excluded above.
# - steer_block_ridge also runs forward-mode AD (jvp) on the source model, extra GPU
#   overhead beyond that base. The 96 GB gpu_RTXPro6000B card (Blackwell, sm_120) is NOT
#   usable on this cluster's current PyTorch (2.1.2+cu121, compiled up to sm_90 only --
#   jobs on it fail with "no kernel image is available for execution on the device"),
#   so that task now runs on the same 45 GB cards as everything else, with less memory
#   headroom than originally intended for it -- if --array=2 OOMs, lower
#   method_params.calib_batch_size/calib_max_length in the steer_block_ridge config, or
#   ask about upgrading PyTorch to a build with sm_120 support (CUDA 12.4+/12.6+).
# - CPU: theseus holds four float32 copies of the backbone state (A pre/finetuned,
#   B base, delta) plus float64 covariances for every Linear (~110 GB) -> --mem=192G.
#
# TODO before submitting:
# - set source_finetuned in both math configs under configs/rebase/ to the
#   Llama-3.2-3B_math directory on the cluster;
# - meta-llama repos are gated: the account whose token is used must have accepted the
#   Llama 3.2 license. Log in once with this same HF_HOME
#   (HF_HOME=${WORK_ROOT}/hf_cache huggingface-cli login) or export HF_TOKEN below.

RUNS=(baseline theseus steer_block_ridge)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
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
# export HF_TOKEN=...  # alternative to huggingface-cli login for the gated meta-llama repos

CONFIG_PREFIX="configs/rebase/math_llama-3.2-3b-instruct-math_to_llama-3.2-3b"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/math_llama-3.2-3b/${RUN}"

if [ "$RUN" = "baseline" ]; then
  echo "--- baseline: Llama-3.2-3B without rebasing ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_theseus.yaml" \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B,dtype=bfloat16 \
    --output_path "$OUTPUT_PATH"
else
  echo "--- ${RUN}: A=Llama-3.2-3B-Instruct -> Llama-3.2-3B_math, B=Llama-3.2-3B ---"
  python -m lm_eval run \
    --config "${CONFIG_PREFIX}_${RUN}.yaml" \
    --output_path "$OUTPUT_PATH"
fi

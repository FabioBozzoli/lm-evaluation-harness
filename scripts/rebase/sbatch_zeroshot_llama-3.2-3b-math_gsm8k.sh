#!/bin/bash

#SBATCH --job-name=lm_eval_zeroshot_llama-3.2-3b-math_gsm8k
#SBATCH --output=.log/lm_eval_zeroshot_llama-3.2-3b-math_gsm8k/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_zeroshot_llama-3.2-3b-math_gsm8k/slurm_backup/job_%j.err
#SBATCH --time=12:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Plain zeroshot GSM8K eval of A = your Llama-3.2-3B_math finetune (of
# Llama-3.2-3B-Instruct), no rebasing, no steering, no dependency on lm_eval/rebase or
# any rebase config -- a single ordinary "hf" model run through lm_eval, its own
# checkpoint's actual math performance before any transport to B. num_fewshot comes
# from gsm8k.yaml's own default (5-shot).
#
# Memory (calculated, not measured): a single 3.2B model in bf16 is ~6.4 GB of VRAM
# weights -- lighter than the 8B baseline script. --mem=32G is a generous rounding,
# not a measurement; check `seff <jobid>` after the run to size the next one tighter.
#
# Before submitting: set MODEL_PATH below to the Llama-3.2-3B_math directory on the
# cluster (Hugging Face format: config.json + model.safetensors.index.json + shards).

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
MODEL_PATH="/work/intesasanpaolo_phd/lm-eval/checkpoints/Llama-3.2-3B_math"  # TODO
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_zeroshot_llama-3.2-3b-math_gsm8k"
mkdir -p "$LOG_DIR/slurm_backup"

CUSTOM_LOG="${LOG_DIR}/job_${SLURM_JOB_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${CUSTOM_LOG}"
echo "Model path: ${MODEL_PATH}"

export PYTHONUNBUFFERED=1
# Silence transformers' per-tensor "Loading weights" progress bars.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="${PROJECT_ROOT}"
export HF_HOME="${WORK_ROOT}/hf_cache"

OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/zeroshot/llama-3.2-3b-math_gsm8k"

echo "--- zeroshot: A=Llama-3.2-3B_math, no rebasing ---"
python -m lm_eval run \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=bfloat16" \
  --tasks gsm8k \
  --batch_size 16 \
  --output_path "$OUTPUT_PATH" \
  --log_samples

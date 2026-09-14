#!/bin/bash

#SBATCH --job-name=lm_eval_rebase_smoke_test_global_mlp
#SBATCH --output=.log/lm_eval_rebase_smoke_test_global_mlp/slurm_backup/job_%j.out
#SBATCH --error=.log/lm_eval_rebase_smoke_test_global_mlp/slurm_backup/job_%j.err
#SBATCH --time=02:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Smoke test of steer_text with stage_2_strategy=global_mlp (standard regime), the
# lm_head-as-head port in lm_eval/models/hf_rebased.py. Short and fails fast.
#
# Step 1: unit tests on tiny random Qwen2 models restricted to global_mlp: the MLP is
#         fit on lm_head's next-token rows, its correction changes the logits inside the
#         context and the model is restored exactly on exit. CPU only.
# Step 2: the global_mlp code config end to end on 5 HumanEval problems
#         (A=Qwen2.5-0.5B -> Qwen2.5-Coder-0.5B, B=Qwen2.5-1.5B, calibration on
#         Magicoder-OSS-Instruct). Scores on 5 problems mean nothing; check that the run
#         completes and that the log prints "stage2 (global_mlp) cached test acc".
#
# No theseus here, so 64G of RAM is plenty (steer_text's largest CPU tensors are
# pinv(lm_head) [151936, 1536] and the test diagnostics, a few GB in float64).

# TODO: verify these paths against your cluster layout. The environment must have
# lm-eval's dependencies installed (pip install -e ".[hf]" pytest from PROJECT_ROOT).
PROJECT_ROOT="/homes/fbozzoli/lm-evaluation-harness"
WORK_ROOT="/work/intesasanpaolo_phd/merge-and-rebase"
cd "$PROJECT_ROOT"

LOG_DIR=".log/lm_eval_rebase_smoke_test_global_mlp"
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

CONFIG="configs/rebase/code_qwen2.5-0.5b-coder_to_qwen2.5-1.5b_steer_global_mlp.yaml"
OUTPUT_PATH="${WORK_ROOT}/lm_eval_results/smoke_test_global_mlp/${SLURM_JOB_ID}"

echo "--- Step 1: unit tests (global_mlp) ---"
python -m pytest tests/models/test_hf_rebased.py -v -k global_mlp || exit 1

echo "--- Step 2: steer_text global_mlp on 5 HumanEval problems ---"
python -m lm_eval run \
  --config "$CONFIG" \
  --tasks humaneval \
  --limit 5 \
  --output_path "$OUTPUT_PATH" || exit 1

echo "--- global_mlp smoke test passed ---"

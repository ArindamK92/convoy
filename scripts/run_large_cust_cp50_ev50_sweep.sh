#!/usr/bin/env bash
set -euo pipefail

# Large-customer sweep with fixed CP/EV and pretrained RL checkpoints.
#
# Flow:
# 1) Pretrain + save convoy_hybrid model at 200 customers.
# 2) Pretrain + save convoy_rl_partial_ch model at 500 customers.
# 3) Sweep customer-num from 200 to 1000 (step 200):
#    - iterations=5 for 200/400/600/800
#    - iterations=1 for 1000
#    - fixed CP=50, EV=50
#    - reuse saved checkpoints (do not clear/delete checkpoints)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/akkcm/myenv/bin/python}"

# CUDA allocator tuning to reduce fragmentation-related OOM.
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

COMBINED_DETAILS_CSV="${COMBINED_DETAILS_CSV:-data/combined_data_jd1000_2.csv}"
COMBINED_DIST_MATRIX_CSV="${COMBINED_DIST_MATRIX_CSV:-data/distance_matrix_jd1000_2.csv}"
COMBINED_TIME_MATRIX_CSV="${COMBINED_TIME_MATRIX_CSV:-data/time_matrix_jd1000_2.csv}"

PRETRAIN_TEST_CSV="${PRETRAIN_TEST_CSV:-data/test_instance_500c_50cp.csv}"
PRETRAIN_CUSTOMER_NUM="${PRETRAIN_CUSTOMER_NUM:-200}"
CP_NUM="${CP_NUM:-50}"
EV_NUM="${EV_NUM:-50}"

SEED="${SEED:-111}"
EPOCHS="${EPOCHS:-100}"
FIXED_EVAL_EVERY="${FIXED_EVAL_EVERY:-5}"
COST_WEIGHT="${COST_WEIGHT:-1.0}"

# Large instances are memory heavy; use safer defaults than parser defaults.
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

HYBRID_CHECKPOINT_DIR="${HYBRID_CHECKPOINT_DIR:-checkpoints_vrptw/hybrid_c500_cp50_ev50_e200}"
RL_CHECKPOINT_DIR="${RL_CHECKPOINT_DIR:-checkpoints_vrptw/rl_c500_cp50_ev50_e200}"

CUST_START="${CUST_START:-200}"
CUST_END="${CUST_END:-1000}"
CUST_STEP="${CUST_STEP:-200}"
ITERATIONS_DEFAULT="${ITERATIONS_DEFAULT:-5}"
ITERATIONS_CUST_1000="${ITERATIONS_CUST_1000:-1}"

RESULT_PREFIX="${RESULT_PREFIX:-large}"

RL_EXTRA="--print-solution --save-model --seed ${SEED} --epochs ${EPOCHS} --fixed-eval-every ${FIXED_EVAL_EVERY} --batch-size ${BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE}"
OPT_HEU_EXTRA="--random-seed ${SEED} --skip-optimal"

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Combined details: ${COMBINED_DETAILS_CSV}"
echo "Combined distance matrix: ${COMBINED_DIST_MATRIX_CSV}"
echo "Combined time matrix: ${COMBINED_TIME_MATRIX_CSV}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Batch sizes: train=${BATCH_SIZE}, eval=${EVAL_BATCH_SIZE}"

echo
echo "=== Step 1: Pretrain + save convoy_hybrid checkpoint ==="
hybrid_best_ckpt="${HYBRID_CHECKPOINT_DIR}/best_model_hybrid.ckpt"
if [[ -f "${hybrid_best_ckpt}" ]]; then
  echo "[SKIP] Found existing hybrid checkpoint: ${hybrid_best_ckpt}"
else
  cmd_hybrid=(
    "${PYTHON_BIN}" "tests/test_convoy_hybrid.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--test-csv" "${PRETRAIN_TEST_CSV}"
    "--customer-num" "${PRETRAIN_CUSTOMER_NUM}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${EV_NUM}"
    "--epochs" "${EPOCHS}"
    "--fixed-eval-every" "${FIXED_EVAL_EVERY}"
    "--batch-size" "${BATCH_SIZE}"
    "--eval-batch-size" "${EVAL_BATCH_SIZE}"
    "--seed" "${SEED}"
    "--print-solution"
    "--save-model"
    "--checkpoint-dir" "${HYBRID_CHECKPOINT_DIR}"
    "--verbose"
  )
  printf '[CMD]'; printf ' %q' "${cmd_hybrid[@]}"; printf '\n'
  "${cmd_hybrid[@]}"
fi

echo
echo "=== Step 2: Pretrain + save convoy_rl_partial_ch checkpoint ==="
rl_best_ckpt="${RL_CHECKPOINT_DIR}/best_model.ckpt"
if [[ -f "${rl_best_ckpt}" ]]; then
  echo "[SKIP] Found existing RL checkpoint: ${rl_best_ckpt}"
else
  cmd_rl=(
    "${PYTHON_BIN}" "tests/test_convoy_CPs1.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--test-csv" "${PRETRAIN_TEST_CSV}"
    "--customer-num" "${PRETRAIN_CUSTOMER_NUM}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${EV_NUM}"
    "--epochs" "${EPOCHS}"
    "--fixed-eval-every" "${FIXED_EVAL_EVERY}"
    "--batch-size" "${BATCH_SIZE}"
    "--eval-batch-size" "${EVAL_BATCH_SIZE}"
    "--seed" "${SEED}"
    "--print-solution"
    "--save-model"
    "--checkpoint-dir" "${RL_CHECKPOINT_DIR}"
  )
  printf '[CMD]'; printf ' %q' "${cmd_rl[@]}"; printf '\n'
  "${cmd_rl[@]}"
fi

echo
echo "=== Step 3: Sweep customer-num ${CUST_START}..${CUST_END} (step ${CUST_STEP}) ==="
echo "Fixed: cp=${CP_NUM}, ev=${EV_NUM}"

for cust_num in $(seq "${CUST_START}" "${CUST_STEP}" "${CUST_END}"); do
  iterations="${ITERATIONS_DEFAULT}"
  if (( cust_num == 1000 )); then
    iterations="${ITERATIONS_CUST_1000}"
  fi

  results_file="${RESULT_PREFIX}_cust${cust_num}_cp${CP_NUM}_ev${EV_NUM}_itr${iterations}.csv"

  cmd_sweep=(
    "${PYTHON_BIN}" "convoy_main.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--customer-num" "${cust_num}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${EV_NUM}"
    "--cost-weight" "${COST_WEIGHT}"
    "--iterations" "${iterations}"
    "--results-file" "${results_file}"
    "--opt-rl-extra" "${RL_EXTRA}"
    "--opt-heu-extra" "${OPT_HEU_EXTRA}"
    "--hybrid-checkpoint-dir" "${HYBRID_CHECKPOINT_DIR}"
    "--rl-checkpoint-dir" "${RL_CHECKPOINT_DIR}"
  )

  echo
  echo "[RUN] customer=${cust_num} cp=${CP_NUM} ev=${EV_NUM} iterations=${iterations} results=${results_file}"
  printf '[CMD]'; printf ' %q' "${cmd_sweep[@]}"; printf '\n'
  "${cmd_sweep[@]}"
done

echo
echo "Large sweep completed."

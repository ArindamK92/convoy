#!/usr/bin/env bash
set -euo pipefail

# Runs small-case sweep:
# 1) Pretrain convoy_rl_partial_ch2 (POMO) and store best checkpoint.
# 2) Generate fixed test instances (one per customer size) with seed control.
# 2) For customer-num = 5,10:
#    - Optimal (MILP) + Heuristic (no EDF/NDF)
#    - convoy_rl_partial_ch2 inference
# 3) For customer-num = 15:
#    - Heuristic only (skip Optimal; no EDF/NDF)
#    - convoy_rl_partial_ch2 inference

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

COMBINED_DETAILS_CSV="${COMBINED_DETAILS_CSV:-data/combined_data_jd200_1.csv}"
COMBINED_DIST_MATRIX_CSV="${COMBINED_DIST_MATRIX_CSV:-data/distance_matrix_jd200_1.csv}"
COMBINED_TIME_MATRIX_CSV="${COMBINED_TIME_MATRIX_CSV:-data/time_matrix_jd200_1.csv}"

CP_NUM="${CP_NUM:-3}"
EV_NUM="${EV_NUM:-2}"
ITERATIONS="${ITERATIONS:-1}"
SEED="${SEED:-111}"
EPOCHS="${EPOCHS:-100}"
FIXED_EVAL_EVERY="${FIXED_EVAL_EVERY:-5}"
COST_WEIGHT="${COST_WEIGHT:-1.0}"
EV_ENERGY_RATE_KWH_PER_DISTANCE="${EV_ENERGY_RATE_KWH_PER_DISTANCE:-0.00025}"
FIXED_INSTANCE_DIR="${FIXED_INSTANCE_DIR:-data/fixed_small_instances}"

RL_V2_CHECKPOINT_DIR="${RL_V2_CHECKPOINT_DIR:-checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100_pomo}"
PRETRAIN_TEST_CSV="${PRETRAIN_TEST_CSV:-data/test_instance_50c_10cp.csv}"
PRETRAIN_CUSTOMER_NUM="${PRETRAIN_CUSTOMER_NUM:-50}"
PRETRAIN_CP_NUM="${PRETRAIN_CP_NUM:-10}"
PRETRAIN_EV_NUM="${PRETRAIN_EV_NUM:-10}"

RL_V2_EXTRA="--seed ${SEED} --epochs ${EPOCHS} --fixed-eval-every ${FIXED_EVAL_EVERY} --save-model --rl-algo pomo --baseline shared --pomo-num-starts 10 --pomo-num-augment 8 --decode-type beam_search --decode-beam-width 10"

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Checkpoint dir: ${RL_V2_CHECKPOINT_DIR}"
echo "Small sweep cases: customer-num=5,10,15 cp=${CP_NUM} ev=${EV_NUM}"
echo "Iterations per run: ${ITERATIONS}"

mkdir -p "${FIXED_INSTANCE_DIR}"

generate_fixed_instance() {
  local cust_num="$1"
  local fixed_csv="$2"
  if [[ -f "${fixed_csv}" ]]; then
    echo "[SKIP] Fixed instance exists: ${fixed_csv}"
    return
  fi

  local nS=$(( (cust_num + EV_NUM - 1) / EV_NUM ))
  echo "[GEN] Creating fixed instance: ${fixed_csv}"

  CONVOY_COMBINED_DETAILS="${COMBINED_DETAILS_CSV}" \
  CONVOY_COMBINED_DIST="${COMBINED_DIST_MATRIX_CSV}" \
  CONVOY_COMBINED_TIME="${COMBINED_TIME_MATRIX_CSV}" \
  CONVOY_CUST_NUM="${cust_num}" \
  CONVOY_CP_NUM="${CP_NUM}" \
  CONVOY_NS="${nS}" \
  CONVOY_EV_NUM="${EV_NUM}" \
  CONVOY_SEED="${SEED}" \
  CONVOY_ENERGY_RATE="${EV_ENERGY_RATE_KWH_PER_DISTANCE}" \
  CONVOY_FIXED_OUT="${fixed_csv}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import shutil

from src.convoy_opt_and_heu.helper import preProcess

combined_details = os.environ["CONVOY_COMBINED_DETAILS"]
combined_dist = os.environ["CONVOY_COMBINED_DIST"]
combined_time = os.environ["CONVOY_COMBINED_TIME"]
cust_num = int(os.environ["CONVOY_CUST_NUM"])
cp_num = int(os.environ["CONVOY_CP_NUM"])
nS = int(os.environ["CONVOY_NS"])
ev_num = int(os.environ["CONVOY_EV_NUM"])
seed = int(os.environ["CONVOY_SEED"])
energy_rate = float(os.environ["CONVOY_ENERGY_RATE"])
out_csv = os.environ["CONVOY_FIXED_OUT"]

preProcess(
    cust_num,
    cp_num,
    nS,
    ev_num,
    combined_details,
    combined_dist,
    combined_time,
    ev_energy_rate_kwh_per_distance=energy_rate,
    random_seed=seed,
    use_full_instance=False,
)
generated_csv = os.path.join(os.path.dirname(os.path.abspath(combined_details)), "test_instance.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
shutil.copyfile(generated_csv, out_csv)
print(f"Fixed instance saved: {out_csv}")
PY
}

echo
echo "=== Step 1: Pretrain + save convoy_rl_partial_ch2 checkpoint (POMO + beam-search) ==="
rl_v2_best_ckpt="${RL_V2_CHECKPOINT_DIR}/best_model_pomo.ckpt"
if [[ -f "${rl_v2_best_ckpt}" ]]; then
  echo "[SKIP] Found existing RL-v2 checkpoint: ${rl_v2_best_ckpt}"
else
  cmd_pretrain=(
    "${PYTHON_BIN}" "-m" "src.convoy_rl_partial_ch2.convoy_rl_main"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--test-csv" "${PRETRAIN_TEST_CSV}"
    "--test-distance-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--test-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--customer-num" "${PRETRAIN_CUSTOMER_NUM}"
    "--charging-stations-num" "${PRETRAIN_CP_NUM}"
    "--ev-num" "${PRETRAIN_EV_NUM}"
    "--epochs" "${EPOCHS}"
    "--fixed-eval-every" "${FIXED_EVAL_EVERY}"
    "--seed" "${SEED}"
    "--print-solution"
    "--save-model"
    "--checkpoint-dir" "${RL_V2_CHECKPOINT_DIR}"
    "--rl-algo" "pomo"
    "--baseline" "shared"
    "--pomo-num-starts" "10"
    "--pomo-num-augment" "8"
    "--decode-type" "beam_search"
    "--decode-beam-width" "10"
  )
  printf '[CMD]'; printf ' %q' "${cmd_pretrain[@]}"; printf '\n'
  "${cmd_pretrain[@]}"
fi

run_case() {
  local cust_num="$1"
  local skip_optimal="$2"
  local fixed_case_csv="${FIXED_INSTANCE_DIR}/test_instance_${cust_num}c_${CP_NUM}cp_seed${SEED}.csv"
  local opt_heu_results_file="small_methods_opt_heu_cust${cust_num}_cp${CP_NUM}_ev${EV_NUM}.csv"
  local rlv2_results_file="small_methods_rlv2_cust${cust_num}_cp${CP_NUM}_ev${EV_NUM}.csv"

  generate_fixed_instance "${cust_num}" "${fixed_case_csv}"

  local opt_heu_extra="--random-seed ${SEED}"
  if [[ "${skip_optimal}" == "1" ]]; then
    opt_heu_extra="${opt_heu_extra} --skip-optimal"
  fi

  local -a cmd_opt_heu=(
    "${PYTHON_BIN}" "convoy_main.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--test-for-opt-heu" "${fixed_case_csv}"
    "--customer-num" "${cust_num}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${EV_NUM}"
    "--cost-weight" "${COST_WEIGHT}"
    "--iterations" "${ITERATIONS}"
    "--results-file" "${opt_heu_results_file}"
    "--only-opt-heu"
    "--no-edf-ndf"
    "--opt-heu-extra" "${opt_heu_extra}"
  )

  echo
  if [[ "${skip_optimal}" == "1" ]]; then
    echo "[RUN] customer=${cust_num} cp=${CP_NUM} ev=${EV_NUM} methods=Heuristic"
  else
    echo "[RUN] customer=${cust_num} cp=${CP_NUM} ev=${EV_NUM} methods=Optimal+Heuristic"
  fi
  printf '[CMD]'; printf ' %q' "${cmd_opt_heu[@]}"; printf '\n'
  "${cmd_opt_heu[@]}"

  local rl_v2_extra="${RL_V2_EXTRA} --checkpoint-dir ${RL_V2_CHECKPOINT_DIR} --test-csv ${fixed_case_csv} --test-distance-matrix-csv ${COMBINED_DIST_MATRIX_CSV} --test-time-matrix-csv ${COMBINED_TIME_MATRIX_CSV}"
  local -a cmd=(
    "${PYTHON_BIN}" "convoy_main.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--test-for-opt-heu" "${fixed_case_csv}"
    "--customer-num" "${cust_num}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${EV_NUM}"
    "--cost-weight" "${COST_WEIGHT}"
    "--iterations" "${ITERATIONS}"
    "--results-file" "${rlv2_results_file}"
    "--only-rl-v2"
    "--opt-rl-extra" "${rl_v2_extra}"
    "--opt-heu-extra" "--random-seed ${SEED}"
    "--rl-v2-checkpoint-dir" "${RL_V2_CHECKPOINT_DIR}"
  )

  echo
  echo "[RUN] customer=${cust_num} cp=${CP_NUM} ev=${EV_NUM} methods=RL-v2"
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
}

# Same small customer cases as run_small_methods_sweep.sh
run_case 5 0
run_case 10 0
run_case 15 1

echo
echo "Completed small RL-v2 methods sweep."

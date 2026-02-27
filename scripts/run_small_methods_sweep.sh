#!/usr/bin/env bash
set -euo pipefail

# Runs:
# 1) customer-num = 5,10 with:
#    - Optimal (MILP) + Heuristic (no EDF/NDF),
#    - convoy_hybrid.
# 2) customer-num = 15 with:
#    - Heuristic only (skip Optimal; no EDF/NDF),
#    - convoy_hybrid.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

COMBINED_DETAILS_CSV="${COMBINED_DETAILS_CSV:-data/combined_data_jd200_1.csv}"
COMBINED_DIST_MATRIX_CSV="${COMBINED_DIST_MATRIX_CSV:-data/distance_matrix_jd200_1.csv}"
COMBINED_TIME_MATRIX_CSV="${COMBINED_TIME_MATRIX_CSV:-data/time_matrix_jd200_1.csv}"

CP_NUM="${CP_NUM:-3}"
EV_NUM="${EV_NUM:-2}"
ITERATIONS="${ITERATIONS:-5}"
SEED="${SEED:-111}"
EPOCHS="${EPOCHS:-100}"
COST_WEIGHT="${COST_WEIGHT:-1.0}"

RL_EXTRA_BASE="--seed ${SEED} --epochs ${EPOCHS}"
HYBRID_CHECKPOINT_DIR="${HYBRID_CHECKPOINT_DIR:-checkpoints_vrptw/hybrid_c50_cp10_ev10_e100}"

run_case() {
  local cust_num="$1"
  local skip_optimal="$2"
  local results_file="small_methods_cust${cust_num}_cp${CP_NUM}_ev${EV_NUM}.csv"

  local opt_heu_extra="--random-seed ${SEED}"
  if [[ "${skip_optimal}" == "1" ]]; then
    opt_heu_extra="${opt_heu_extra} --skip-optimal"
  fi
  local rl_extra="${RL_EXTRA_BASE} --save-model --checkpoint-dir ${HYBRID_CHECKPOINT_DIR}"

  local -a cmd=(
    "${PYTHON_BIN}" "convoy_main.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--customer-num" "${cust_num}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${EV_NUM}"
    "--cost-weight" "${COST_WEIGHT}"
    "--iterations" "${ITERATIONS}"
    "--results-file" "${results_file}"
    "--no-edf-ndf"
    "--skip-convoy-rl"
    "--opt-rl-extra" "${rl_extra}"
    "--opt-heu-extra" "${opt_heu_extra}"
  )

  echo
  if [[ "${skip_optimal}" == "1" ]]; then
    echo "[RUN] customer=${cust_num} cp=${CP_NUM} ev=${EV_NUM} methods=Heuristic+Hybrid"
  else
    echo "[RUN] customer=${cust_num} cp=${CP_NUM} ev=${EV_NUM} methods=Optimal+Heuristic+Hybrid"
  fi
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
}

# customer 5 and 10: include Optimal + Heuristic
run_case 5 0
run_case 10 0

# customer 15: Heuristic only (skip Optimal)
run_case 15 1

echo
echo "Completed small methods sweep."

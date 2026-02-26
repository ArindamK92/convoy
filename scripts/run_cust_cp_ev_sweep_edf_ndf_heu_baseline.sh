#!/usr/bin/env bash
set -euo pipefail

# Sweep settings:
# - customer-num: 10,20,...,100
# - charging-stations-num: 20 (fixed)
# - ev-num: customer-num / 5
#
# Methods run via convoy_main:
# - Heuristic
# - NDF
# - EDF
# - Baseline
# (Optimal, convoy_hybrid, convoy_rl_partial_ch are skipped.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

COMBINED_DETAILS_CSV="${COMBINED_DETAILS_CSV:-data/combined_data_jd200_1.csv}"
COMBINED_DIST_MATRIX_CSV="${COMBINED_DIST_MATRIX_CSV:-data/distance_matrix_jd200_1.csv}"
COMBINED_TIME_MATRIX_CSV="${COMBINED_TIME_MATRIX_CSV:-data/time_matrix_jd200_1.csv}"

CUST_START="${CUST_START:-10}"
CUST_END="${CUST_END:-100}"
CUST_STEP="${CUST_STEP:-10}"
CP_NUM="${CP_NUM:-20}"
DELIVERY_TO_EV_RATIO="${DELIVERY_TO_EV_RATIO:-5}"

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-15}"
COST_WEIGHT="${COST_WEIGHT:-1.0}"

BASELINE_TIME="${BASELINE_TIME:-630}"
BASELINE_RUNS="${BASELINE_RUNS:-2}"

OPT_HEU_EXTRA="--random-seed ${SEED} --skip-optimal"
BASELINE_EXTRA="--g_1 20 --pop_size 4 --init rcrs --cross_repair regret --parent_selection circle --replacement one_on_one --O_1_eval --two_opt --two_opt_star --or_opt 2 --two_exchange 2 --elo 1 --removal_lower 0.1 --removal_upper 0.2 --individual_search --population_search --parallel_insertion --conservative_local_search --aggressive_local_search --station_range 0.5 --subproblem_range 1"

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Sweep: customers=${CUST_START}..${CUST_END} step=${CUST_STEP}, cp=${CP_NUM}, ratio=${DELIVERY_TO_EV_RATIO}"
echo "Methods: Heuristic + NDF + EDF + Baseline"

for cust_num in $(seq "${CUST_START}" "${CUST_STEP}" "${CUST_END}"); do
  if (( cust_num % DELIVERY_TO_EV_RATIO != 0 )); then
    echo "[SKIP] customer-num=${cust_num} is not divisible by ratio=${DELIVERY_TO_EV_RATIO}"
    continue
  fi

  ev_num=$((cust_num / DELIVERY_TO_EV_RATIO))
  results_file="result_edf_ndf_heu_baseline_seed${SEED}_cust${cust_num}_it${ITERATIONS}.csv"

  cmd=(
    "${PYTHON_BIN}" "convoy_main.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--customer-num" "${cust_num}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${ev_num}"
    "--cost-weight" "${COST_WEIGHT}"
    "--iterations" "${ITERATIONS}"
    "--results-file" "${results_file}"
    "--only-opt-heu"
    "--run-baseline"
    "--baseline-print-charging-events"
    "--baseline-time" "${BASELINE_TIME}"
    "--baseline-runs" "${BASELINE_RUNS}"
    "--baseline-no-quiet"
    "--baseline-extra" "${BASELINE_EXTRA}"
    "--opt-heu-extra" "${OPT_HEU_EXTRA}"
  )

  echo
  echo "[RUN] customer-num=${cust_num} cp-num=${CP_NUM} ev-num=${ev_num} results-file=${results_file}"
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
done

echo
echo "Sweep completed."

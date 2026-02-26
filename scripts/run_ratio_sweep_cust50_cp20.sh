#!/usr/bin/env bash
set -euo pipefail

# Sweep settings:
# - customer-num: fixed 50
# - charging-stations-num: fixed 20
# - target ratio customer/ev: 5..10 (inclusive)
# - ev-num computed as integer division: ev = floor(50 / ratio)
#
# Methods run via convoy_main:
# - Heuristic + NDF + EDF (Optimal skipped)
# - convoy_hybrid
# - convoy_rl_partial_ch
# - baseline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

COMBINED_DETAILS_CSV="${COMBINED_DETAILS_CSV:-data/combined_data_jd200_1.csv}"
COMBINED_DIST_MATRIX_CSV="${COMBINED_DIST_MATRIX_CSV:-data/distance_matrix_jd200_1.csv}"
COMBINED_TIME_MATRIX_CSV="${COMBINED_TIME_MATRIX_CSV:-data/time_matrix_jd200_1.csv}"

TOTAL_CUSTOMERS="${TOTAL_CUSTOMERS:-50}"
CP_NUM="${CP_NUM:-20}"
RATIO_START="${RATIO_START:-5}"
RATIO_END="${RATIO_END:-10}"

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
ITERATIONS="${ITERATIONS:-15}"
COST_WEIGHT="${COST_WEIGHT:-1.0}"

BASELINE_TIME="${BASELINE_TIME:-630}"
BASELINE_RUNS="${BASELINE_RUNS:-2}"

RL_EXTRA="--print-solution --save-model --seed ${SEED} --epochs ${EPOCHS} --rl-algo pomo --baseline shared --pomo-num-starts 10 --pomo-num-augment 8 --decode-type beam_search --decode-beam-width 10"
OPT_HEU_EXTRA="--random-seed ${SEED} --skip-optimal"
BASELINE_EXTRA="--g_1 20 --pop_size 4 --init rcrs --cross_repair regret --parent_selection circle --replacement one_on_one --O_1_eval --two_opt --two_opt_star --or_opt 2 --two_exchange 2 --elo 1 --removal_lower 0.1 --removal_upper 0.2 --individual_search --population_search --parallel_insertion --conservative_local_search --aggressive_local_search --station_range 0.5 --subproblem_range 1"

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Sweep: customer-num=${TOTAL_CUSTOMERS}, cp-num=${CP_NUM}, ratio=${RATIO_START}..${RATIO_END}"

for ratio in $(seq "${RATIO_START}" "${RATIO_END}"); do
  if (( ratio <= 0 )); then
    echo "[SKIP] invalid ratio=${ratio}"
    continue
  fi

  ev_num=$((TOTAL_CUSTOMERS / ratio))
  if (( ev_num <= 0 )); then
    echo "[SKIP] ratio=${ratio} produced ev-num=${ev_num}"
    continue
  fi

  results_file="ratio_sweep_c${TOTAL_CUSTOMERS}_cp${CP_NUM}_r${ratio}_ev${ev_num}_seed${SEED}_e${EPOCHS}.csv"

  cmd=(
    "${PYTHON_BIN}" "convoy_main.py"
    "--combined-details-csv" "${COMBINED_DETAILS_CSV}"
    "--combined-dist-matrix-csv" "${COMBINED_DIST_MATRIX_CSV}"
    "--combined-time-matrix-csv" "${COMBINED_TIME_MATRIX_CSV}"
    "--customer-num" "${TOTAL_CUSTOMERS}"
    "--charging-stations-num" "${CP_NUM}"
    "--ev-num" "${ev_num}"
    "--cost-weight" "${COST_WEIGHT}"
    "--iterations" "${ITERATIONS}"
    "--results-file" "${results_file}"
    "--run-baseline"
    "--baseline-print-charging-events"
    "--baseline-time" "${BASELINE_TIME}"
    "--baseline-runs" "${BASELINE_RUNS}"
    "--baseline-no-quiet"
    "--baseline-extra" "${BASELINE_EXTRA}"
    "--opt-rl-extra" "${RL_EXTRA}"
    "--opt-heu-extra" "${OPT_HEU_EXTRA}"
    "--clear-rl-checkpoints"
  )

  echo
  echo "[RUN] ratio=${ratio} customer=${TOTAL_CUSTOMERS} cp=${CP_NUM} ev=${ev_num} results-file=${results_file}"
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
done

echo
echo "Ratio sweep completed."


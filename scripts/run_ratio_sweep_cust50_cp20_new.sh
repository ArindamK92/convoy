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
# (convoy_rl_partial_ch and baseline are skipped)

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

HYBRID_CHECKPOINT_DIR="${HYBRID_CHECKPOINT_DIR:-checkpoints_vrptw/hybrid_c50_cp10_ev10_e100}"
RL_EXTRA="--print-solution --save-model --checkpoint-dir ${HYBRID_CHECKPOINT_DIR} --seed ${SEED} --epochs ${EPOCHS} --rl-algo pomo --baseline shared --pomo-num-starts 10 --pomo-num-augment 8 --decode-type beam_search --decode-beam-width 10"
OPT_HEU_EXTRA="--random-seed ${SEED} --skip-optimal"

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
    "--skip-convoy-rl"
    "--opt-rl-extra" "${RL_EXTRA}"
    "--opt-heu-extra" "${OPT_HEU_EXTRA}"
  )

  echo
  echo "[RUN] ratio=${ratio} customer=${TOTAL_CUSTOMERS} cp=${CP_NUM} ev=${ev_num} results-file=${results_file}"
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  "${cmd[@]}"
done

echo
echo "Ratio sweep completed."

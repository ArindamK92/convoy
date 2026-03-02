# Combined Runner CLI (`convoy_main.py`)

Top-level parser is defined in `convoy_parser.py` (`build_main_parser`).

## Purpose
- Run Opt+Heu, `m_VRPTW`, and `convoy_rl_partial_ch2` from one command.
- Share one dataset/sample configuration across stages.
- Write one consolidated results CSV.

## Execution Behavior
- Default stage order per iteration:
  1. Opt+Heu
  2. Hybrid RL
  3. CONVOY RL-v2 partial charging
  4. Baseline (only if `--run-baseline` is set)
- `--only-opt-heu` runs only Opt+Heu.
- `--only-rl` skips Opt+Heu and runs RL stages.
- `--clear-rl-checkpoints` clears resolved checkpoint directories once before iterations start.

## Full Argument Reference

### Required Arguments

| Argument | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `--combined-details-csv` | `str` | Yes | - | Combined node details CSV (depot/customers/CPs). |
| `--combined-dist-matrix-csv` | `str` | Yes | - | Distance matrix CSV. |
| `--combined-time-matrix-csv` | `str` | Yes | - | Time matrix CSV. |
| `--customer-num` | `int` | Yes | - | Number of customers to sample/use. |
| `--charging-stations-num` | `int` | Yes | - | Number of charging stations to sample/use. |
| `--ev-num` | `int` | Yes | - | Number of EVs. |

### Core Optional Arguments

| Argument | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `--test-for-opt-heu` | optional `str` | No | `None` | Use existing test CSV for Opt+Heu instead of sampling from combined details. If passed without value: `data/test_instance.csv`. |
| `--ev-energy-rate-kwh-per-distance` | `float` | No | `0.00025` | Shared energy usage per distance unit. |
| `--reserve-battery` | `float` | No | `0.0` | Shared reserve battery in kWh. |
| `--no-EDF-NDF` / `--no-edf-ndf` | flag | No | `False` | Skip EDF/NDF in Opt+Heu runner. |
| `--cost-weight` | `float` | No | `1.0` | Shared into RL parser path (reward cost weight in RL stage). |
| `--iterations` | `int` | No | `1` | Number of repeated iterations. |
| `--results-file` | `str` | No | `None` | Output CSV path/name. If omitted: `results/results3_<combined-details-stem>.csv`. |
| `--clear-rl-checkpoints` | flag | No | `False` | Delete resolved RL checkpoint directories before run. |

### Runner Selection + Pass-Through Arguments

| Argument | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `--only-rl` | flag | No | `False` | Skip Opt+Heu and run RL stages only. |
| `--only-opt-heu` | flag | No | `False` | Skip RL stages and run Opt+Heu only. |
| `--only-rl-v2` | flag | No | `False` | Run only `convoy_rl_partial_ch2` and skip Opt+Heu + hybrid. |
| `--hybrid-checkpoint-dir` | `str` | No | `None` | Override `--checkpoint-dir` only for hybrid stage. |
| `--rl-v2-checkpoint-dir` | `str` | No | `None` | Override `--checkpoint-dir` only for CONVOY RL-v2 stage. |
| `--opt-rl-extra` | append string | No | `[]` | Quoted pass-through flags for RL/hybrid parser. Can be repeated. |
| `--opt-heu-extra` | append string | No | `[]` | Quoted pass-through flags for Opt+Heu parser. Can be repeated. |

### Baseline Pipeline Arguments

Used when `--run-baseline` is enabled.

| Argument | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `--run-baseline` | flag | No | `False` | Run EVRP baseline and append baseline row(s). |
| `--baseline-bin` | `str` | No | `baseline/bin/evrp-tw-spd` | Baseline solver binary path. |
| `--baseline-time` | `int` | No | `10` | Baseline solver time limit (seconds). |
| `--baseline-runs` | `int` | No | `5` | Number of baseline solver runs. |
| `--baseline-instance-output-path` | `str` | No | `baseline/data/test_instance_evrp.txt` | Converted baseline instance path. |
| `--baseline-output-file` | `str` | No | `baseline/data/latest_baseline_output.txt` | Stable copied baseline output file. |
| `--baseline-quiet` | flag | No | `True` | Redirect baseline solver logs to file. |
| `--baseline-no-quiet` | flag | No | `False` | Print baseline solver output to console. |
| `--baseline-solver-log` | `str` | No | `baseline/data/baseline_solver.log` | Solver log path when quiet mode is on. |
| `--baseline-print-charging-events` | flag | No | `False` | Print charging-event breakdown from computed baseline metrics. |
| `--baseline-print-route-trace` | flag | No | `False` | Print per-node baseline route trace. |
| `--baseline-extra` | append string | No | `[]` | Quoted extra solver flags, repeatable. |

## Run Examples

### 1) Combined Run (Opt+Heu + Hybrid + CONVOY RL)

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --iterations 1 \
  --results-file combined_run.csv \
  --opt-heu-extra "--random-seed 111 --skip-optimal" \
  --opt-rl-extra "--save-model --seed 111 --epochs 100 --fixed-eval-every 5" \
  --hybrid-checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100 \
  --rl-v2-checkpoint-dir checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100
```

### 2) RL Stages Only (Hybrid + CONVOY RL)

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-rl \
  --opt-rl-extra "--test-csv data/test_instance_50c_10cp.csv --test-distance-matrix-csv data/distance_matrix_jd200_1.csv --test-time-matrix-csv data/time_matrix_jd200_1.csv --save-model --seed 111"
```

### 3) RL-v2 Only

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-rl-v2 \
  --opt-rl-extra "--test-csv data/test_instance_50c_10cp.csv --test-distance-matrix-csv data/distance_matrix_jd200_1.csv --test-time-matrix-csv data/time_matrix_jd200_1.csv --save-model --seed 111 --epochs 100 --fixed-eval-every 5" \
  --rl-v2-checkpoint-dir checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100
```

### 4) Heuristics Only (Opt+Heu without MILP)

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-opt-heu \
  --iterations 1 \
  --results-file heuristics_only.csv \
  --opt-heu-extra "--random-seed 111 --skip-optimal"
```

### 5) Combined Run With Baseline Enabled

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --run-baseline \
  --baseline-time 10 \
  --baseline-runs 5 \
  --baseline-quiet \
  --baseline-extra "--g_1 20 --pop_size 4"
```

## Important Notes

- Keep pass-through blocks quoted:
  - `--opt-rl-extra "--seed 111 --save-model"`
  - `--opt-heu-extra "--skip-optimal --random-seed 111"`
  - `--baseline-extra "--g_1 20 --pop_size 4"`
- In `convoy_main`, `--print-solution` is removed for the hybrid stage timing path so hybrid elapsed time excludes printing overhead.
- For full RL/hybrid/opt+heu inner argument lists, use:
  - `m_VRPTW/README.md`
  - `src/convoy_rl_partial_ch2/README.md`
  - `tests/README.md`

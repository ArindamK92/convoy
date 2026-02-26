# Combined Runner CLI (`convoy_main.py`)

Top-level parser is defined in `convoy_parser.py` (`build_main_parser`).

## Purpose
- Run Opt+Heu, RL, and optional baseline from one command
- Share common inputs and sampling controls
- Append all selected method metrics into one results CSV

## Full Argument Reference

### Required Arguments

| Argument | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--combined-details-csv` | `str` | Yes | - | Combined node details CSV (depot/customers/CPs). |
| `--combined-dist-matrix-csv` | `str` | Yes | - | Distance matrix CSV (IDs on row/column). |
| `--combined-time-matrix-csv` | `str` | Yes | - | Time matrix CSV (IDs on row/column). |
| `--customer-num` | `int` | Yes | - | Number of customers sampled/used in each run. |
| `--charging-stations-num` | `int` | Yes | - | Number of charging stations sampled/used in each run. |
| `--ev-num` | `int` | Yes | - | Number of EVs. |

### Core Optional Arguments

| Argument | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--ev-energy-rate-kwh-per-distance` | `float` | No | `0.00025` | Energy usage per matrix distance unit. For meter-based matrices: `0.00025` = `4 km/kWh`. |
| `--reserve-battery` | `float` | No | `0.0` | Reserve battery in kWh. Effective usable battery is `(full_battery - reserve_battery)`. |
| `--iterations` | `int` | No | `1` | Number of repeated iterations to run. |
| `--results-file` | `str` | No | `None` | Output results CSV. Relative paths are created under `CONVOY2/results/`. If omitted: `results3_<combined-details-stem>.csv`. |
| `--clear-rl-checkpoints` | flag | No | `False` | Delete RL checkpoints before running (uses RL `--checkpoint-dir` if forwarded; otherwise `checkpoints_vrptw`). |

### Baseline Pipeline Arguments

These are used when `--run-baseline` is enabled.

| Argument | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--run-baseline` | flag | No | `False` | Run EVRP baseline and append baseline metrics in results CSV. |
| `--baseline-bin` | `str` | No | `baseline/bin/evrp-tw-spd` | Baseline solver binary path. |
| `--baseline-time` | `int` | No | `10` | Baseline time limit in seconds per run. |
| `--baseline-runs` | `int` | No | `5` | Number of baseline solver runs. |
| `--baseline-instance-output-path` | `str` | No | `baseline/data/test_instance_evrp.txt` | Generated EVRP instance path. |
| `--baseline-output-file` | `str` | No | `baseline/data/latest_baseline_output.txt` | Stable copied baseline output file used for metric computation. |
| `--baseline-quiet` | flag | No | `True` | Redirect baseline solver output to log file. |
| `--baseline-no-quiet` | flag | No | `False` | Print baseline solver output to console. |
| `--baseline-solver-log` | `str` | No | `baseline/data/baseline_solver.log` | Baseline solver log path when quiet mode is on. |
| `--baseline-print-charging-events` | flag | No | `False` | Print charging-event breakdown from computed baseline metrics. |
| `--baseline-extra` | append string | No | `[]` | Quoted extra args forwarded to baseline solver; can be passed multiple times. |

### Runner Selection And Pass-Through Arguments

| Argument | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `--only-rl` | flag | No | `False` | Run RL only (skip Opt+Heu). |
| `--only-opt-heu` | flag | No | `False` | Run Opt+Heu only (skip RL). |
| `--opt-rl-extra` | append string | No | `[]` | Quoted extra args forwarded to RL runner parser; can be passed multiple times. |
| `--opt-heu-extra` | append string | No | `[]` | Quoted extra args forwarded to Opt+Heu runner parser; can be passed multiple times. |

## Usage Patterns

### Base Combined Run (Opt+Heu + RL)

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --opt-rl-extra "--print-solution --save-model"
```

### Iterations + Baseline + Explicit Seeds

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 10 \
  --run-baseline \
  --baseline-time 10 \
  --baseline-runs 5 \
  --results-file jd200_itr10.csv \
  --opt-rl-extra "--print-solution --save-model --seed 42" \
  --opt-heu-extra "--random-seed 123"
```

### RL-Only Mode

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --only-rl \
  --opt-rl-extra "--test-csv data/test_instance.csv --print-solution"
```

## Important Notes

- If neither `--only-rl` nor `--only-opt-heu` is set, `convoy_main` runs Opt+Heu first, then RL, then baseline (if enabled).
- When Opt+Heu runs, it regenerates `test_instance.csv`; RL and baseline use this generated file unless you explicitly override test CSV in forwarded args.
- Keep pass-through argument blocks quoted:
  - `--opt-rl-extra "--print-solution --save-model --seed 42"`
  - `--opt-heu-extra "--skip-optimal --random-seed 123"`
  - `--baseline-extra "--g_1 20 --pop_size 9"`

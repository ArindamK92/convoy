# Tests Runners

This folder contains runnable entrypoint scripts for the main CONVOY2 methods.
These are not unit tests; they are CLI launchers for training/evaluation pipelines.

## Setup

```bash
source /home/akkcm/myenv/bin/activate
cd /home/akkcm/CONVOY2
```

## Available Runners

1. `tests/test_convoy_CPs1.py` -> convoy RL partial charging pipeline (`src/convoy_rl_partial_ch/convoy_rl_main.py`)
2. `tests/test_convoy_hybrid.py` -> hybrid RL pipeline (`convoy_hybrid/convoy_hybrid_main.py`)
3. `tests/test_convoy_opt_and_heu.py` -> optimization + heuristic pipeline (`src/convoy_opt_and_heu/opt_and_hue.py`)

## 1) `test_convoy_CPs1.py`

Run:

```bash
python tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10
```

Required arguments:
- `--combined-details-csv`
- `--combined-dist-matrix-csv`

Arguments:
- Core training: `--epochs`, `--batch-size`, `--eval-batch-size`, `--train-data-size`, `--val-data-size`, `--test-data-size`, `--lr`, `--max-time`, `--seed`, `--accelerator`
- RL algorithm: `--rl-algo`, `--baseline`, `--pomo-num-starts`, `--pomo-num-augment`
- Decoder: `--decode-type`, `--decode-num-samples`, `--decode-num-starts`, `--decode-select-best`, `--decode-temperature`, `--decode-top-p`, `--decode-top-k`, `--decode-beam-width`
- Eval/checkpoint: `--fixed-eval-size`, `--fixed-eval-every`, `--checkpoint-dir`, `--save-model`
- Data/model settings: `--test-csv`, `--csv-vehicle-capacity`, `--ev-battery-capacity-kwh`, `--ev-energy-rate-kwh-per-distance`, `--cost-weight`, `--ev-charge-rate-kwh-per-hour`, `--reserve-battery` (alias `--ev-reserve-soc-kwh`), `--ev-num`, `--charging-stations-num`, `--combined-time-matrix-csv`, `--customer-num`, `--pool-vehicle-capacity`, `--test-distance-matrix-csv`, `--test-time-matrix-csv`
- Output: `--print-solution`

## 2) `test_convoy_hybrid.py`

Run:

```bash
python tests/test_convoy_hybrid.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance_50c_10cp.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --epochs 100 \
  --seed 111 \
  --save-model \
  --checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100
```

Required arguments:
- `--combined-details-csv`
- `--combined-dist-matrix-csv`

Arguments:
- All shared RL arguments from `test_convoy_CPs1.py` are accepted (same base parser).
- Hybrid-specific additions:
  - `--fixed-instance-csv`
  - `--fixed-instance-seed`
  - `--verbose`

Hybrid semantics notes:
- `--charging-stations-num` is kept for CLI compatibility and ignored in hybrid mode.
- In hybrid mode, CP rows in `--test-csv` are ignored.

## 3) `test_convoy_opt_and_heu.py`

Run:

```bash
python tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 20 \
  --charging-stations-num 10 \
  --ev-num 8
```

Required arguments:
- `--combined-details-csv`
- `--combined-dist-matrix-csv`
- `--combined-time-matrix-csv`
- `--customer-num`
- `--charging-stations-num`
- `--ev-num`

Arguments:
- `--test-for-opt-heu`
- `--ev-energy-rate-kwh-per-distance`
- `--reserve-battery`
- `--skip-optimal`
- `--only-milp`
- `--random-seed`
- `--no-EDF-NDF` (alias `--no-edf-ndf`)

## Show Full Help

Use these to print full argument documentation directly from code:

```bash
python tests/test_convoy_CPs1.py --help
python tests/test_convoy_hybrid.py --help
python tests/test_convoy_opt_and_heu.py --help
```

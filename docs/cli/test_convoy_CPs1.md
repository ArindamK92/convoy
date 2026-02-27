# RL Test Runner (`tests/test_convoy_CPs1.py`)

Shim for `src/convoy_rl_partial_ch/convoy_rl_main.py`.

## Purpose
- Train RL4CO CVRPTW with CONVOY partial-charging reward.
- Evaluate on generated data or a provided `--test-csv`.
- Optionally save/reuse best checkpoints.

## Required Arguments

| Argument | Type | Required | Default |
| --- | --- | --- | --- |
| `--combined-details-csv` | `str` | Yes | - |
| `--combined-dist-matrix-csv` | `str` | Yes | - |

## Common Arguments

| Group | Arguments |
| --- | --- |
| Training | `--epochs`, `--batch-size`, `--eval-batch-size`, `--train-data-size`, `--val-data-size`, `--test-data-size`, `--lr`, `--seed`, `--accelerator` |
| RL Algo | `--rl-algo {am,pomo}`, `--baseline`, `--pomo-num-starts`, `--pomo-num-augment` |
| Decoder | `--decode-type`, `--decode-num-samples`, `--decode-num-starts`, `--decode-select-best`, `--decode-temperature`, `--decode-top-p`, `--decode-top-k`, `--decode-beam-width` |
| Data | `--combined-time-matrix-csv`, `--customer-num`, `--charging-stations-num`, `--ev-num`, `--pool-vehicle-capacity`, `--test-csv`, `--test-distance-matrix-csv`, `--test-time-matrix-csv`, `--csv-vehicle-capacity` |
| EV / Reward | `--ev-battery-capacity-kwh`, `--ev-energy-rate-kwh-per-distance`, `--ev-charge-rate-kwh-per-hour`, `--reserve-battery` (`--ev-reserve-soc-kwh`), `--cost-weight` |
| Eval / Checkpoint | `--fixed-eval-size`, `--fixed-eval-every`, `--checkpoint-dir`, `--save-model`, `--print-solution` |

## Run Examples

### 1) Basic Run

```bash
python tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10
```

### 2) Run With Test CSV + Checkpoint Save

```bash
python tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance_50c_10cp.csv \
  --test-distance-matrix-csv data/distance_matrix_jd200_1.csv \
  --test-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --epochs 100 \
  --fixed-eval-every 5 \
  --seed 111 \
  --save-model \
  --checkpoint-dir checkpoints_vrptw/rl_partial_c50_cp10_ev10_e100 \
  --print-solution
```

### 3) POMO Run

```bash
python tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --rl-algo pomo \
  --baseline shared \
  --pomo-num-starts 16 \
  --pomo-num-augment 8
```

## Notes

- For `--rl-algo pomo`, set `--baseline shared`.
- Checkpoint names when `--save-model`:
  - `best_model.ckpt` for `am`
  - `best_model_pomo.ckpt` for `pomo`
- Matrix fallback with `--test-csv`:
  - distance: `--test-distance-matrix-csv` -> `--combined-dist-matrix-csv`
  - time: `--test-time-matrix-csv` -> `--combined-time-matrix-csv` -> `--combined-dist-matrix-csv`

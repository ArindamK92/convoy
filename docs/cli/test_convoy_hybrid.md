# Hybrid Test Runner (`tests/test_m_VRPTW.py`)

Shim for `m_VRPTW/convoy_hybrid_main.py`.

## Purpose
- Train/evaluate hybrid RL4CO CVRPTW model.
- Decode customer routes, then run CP augmentation for report metrics.
- Report both full-charging and partial-charging objective components.

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
| Data | `--combined-time-matrix-csv`, `--customer-num`, `--ev-num`, `--pool-vehicle-capacity`, `--test-csv`, `--test-distance-matrix-csv`, `--test-time-matrix-csv`, `--fixed-instance-csv`, `--fixed-instance-seed`, `--csv-vehicle-capacity` |
| EV / Trace | `--ev-battery-capacity-kwh`, `--ev-energy-rate-kwh-per-distance`, `--ev-charge-rate-kwh-per-hour`, `--reserve-battery` |
| Eval / Checkpoint | `--fixed-eval-size`, `--fixed-eval-every`, `--checkpoint-dir`, `--save-model`, `--print-solution`, `--verbose` |

## Run Examples

### 1) Basic Hybrid Run

```bash
python tests/test_m_VRPTW.py \
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
  --checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100
```

### 2) POMO + Beam Search

```bash
python tests/test_m_VRPTW.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance_50c_10cp.csv \
  --test-distance-matrix-csv data/distance_matrix_jd200_1.csv \
  --test-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --ev-num 10 \
  --rl-algo pomo \
  --baseline shared \
  --pomo-num-starts 16 \
  --pomo-num-augment 8 \
  --decode-type beam_search \
  --decode-beam-width 8 \
  --save-model
```

### 3) Compact vs Verbose Output

Default compact output:

```bash
python tests/test_m_VRPTW.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --customer-num 50 \
  --ev-num 10
```

Verbose output:

```bash
python tests/test_m_VRPTW.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --customer-num 50 \
  --ev-num 10 \
  --verbose true
```

## Hybrid-Specific Notes

- `--charging-stations-num` is accepted for compatibility but ignored by hybrid training.
- `--cost-weight` is accepted for compatibility; hybrid RL training reward remains RL4CO CVRPTW reward.
- In hybrid mode, CP rows in `--test-csv` are ignored during RL decode and used later during CP-augmentation reporting.

# RL Test Runner (`tests/test_convoy_CPs1.py`)

Wrapper around `src/convoy_rl/convoy_rl_main.py`.

## Base Command
```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --print-solution \
  --save-model
```

## With Explicit Test CSV + Matrices
```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_delivery10_jd200.csv \
  --test-distance-matrix-csv data/distance_matrix_jd200_1.csv \
  --test-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2
```

## Important Parameters
- `--ev-energy-rate-kwh-per-distance` default `0.00025` (4 km/kWh on meter-based matrices)
- `--ev-battery-capacity-kwh` default `30.0`
- `--charging-stations-num` controls sampled charging stations in generated instances

# Opt+Heu Test Runner (`tests/test_convoy_opt_and_heu.py`)

Wrapper around `src/convoy_opt_and_heu/opt_and_hue.py`.

## Base Command
```bash
python3 tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2
```

## Heuristic Only
```bash
python3 tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --skip-optimal
```

## Important Parameters
- `--alpha1` and `--alpha2` affect MILP objective weighting
- `--ev-energy-rate-kwh-per-distance` is used to derive mileage conversion
- `--random-seed` controls preprocessing sampling (customers/CPs and EV charge-acceptance rates)

## Gurobi Config
- MILP license/options are loaded from `config/gurobi_wls.json` if present.
- Start from template:

```bash
cp config/gurobi_wls.example.json config/gurobi_wls.json
```

- You can also set `CONVOY_GUROBI_CONFIG` to another JSON file path.

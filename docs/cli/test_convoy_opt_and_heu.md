# Opt+Heu Test Runner (`tests/test_convoy_opt_and_heu.py`)

Shim for `src/convoy_opt_and_heu/opt_and_hue.py`.

## Purpose
- Run MILP optimization and heuristic methods on the same sampled instance.
- Optionally run only heuristics or only MILP.
- Write method rows to the standard results output flow.

## Required Arguments

| Argument | Type | Required | Default |
| --- | --- | --- | --- |
| `--combined-details-csv` | `str` | Yes | - |
| `--combined-dist-matrix-csv` | `str` | Yes | - |
| `--combined-time-matrix-csv` | `str` | Yes | - |
| `--customer-num` | `int` | Yes | - |
| `--charging-stations-num` | `int` | Yes | - |
| `--ev-num` | `int` | Yes | - |

## Optional Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--test-for-opt-heu [CSV]` | `None` | Use an existing test instance CSV instead of sampling from combined details (if flag given without value, uses `data/test_instance.csv`). |
| `--ev-energy-rate-kwh-per-distance` | `0.00025` | Energy-per-distance conversion used by pipeline logic. |
| `--reserve-battery` | `0.0` | Reserve battery in kWh. |
| `--alpha1` | `1.0` | MILP objective weight for successful deliveries. |
| `--alpha2` | `1.0` | MILP objective weight for energy cost. |
| `--skip-optimal` | `False` | Skip MILP; run heuristic methods only. |
| `--only-milp` | `False` | Run MILP only (skip heuristic/EDF/NDF). |
| `--random-seed` | `None` | Seed for preprocessing sampling. |
| `--no-EDF-NDF` / `--no-edf-ndf` | `False` | Skip EDF and NDF baseline heuristics. |

## Run Examples

### 1) Full Opt+Heu Run

```bash
python tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 20 \
  --charging-stations-num 10 \
  --ev-num 8
```

### 2) Heuristics Only

```bash
python tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 20 \
  --charging-stations-num 10 \
  --ev-num 8 \
  --skip-optimal
```

### 3) MILP Only

```bash
python tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 20 \
  --charging-stations-num 10 \
  --ev-num 8 \
  --only-milp
```

### 4) Run On Existing Test Instance CSV

```bash
python tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-for-opt-heu data/test_instance_10c_3cp.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --skip-optimal \
  --no-edf-ndf
```

## Gurobi Config

- MILP license/options are loaded from `config/gurobi_wls.json` if present.
- Start from template:

```bash
cp config/gurobi_wls.example.json config/gurobi_wls.json
```

- You can override config path via `CONVOY_GUROBI_CONFIG`.

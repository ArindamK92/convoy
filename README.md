# CONVOY VRPTW

## 1. Setup

```bash
python3 -m venv myenv
myenv/bin/pip install -U pip
myenv/bin/pip install -r requirements.txt
# If needed:
myenv/bin/pip install rl4co lightning torch tensordict torchrl
```

## 2. Base Script (`test_vrptw.py`)

### Train + test on synthetic data

```bash
myenv/bin/python test_vrptw.py
```

### Print one decoded solution

```bash
myenv/bin/python test_vrptw.py --print-solution
```

### Longer training

```bash
myenv/bin/python test_vrptw.py --epochs 100 --train-data-size 20000 --val-data-size 5000 --test-data-size 5000
```

### Evaluate additional custom CSV instance

```bash
myenv/bin/python test_vrptw.py --test-csv vrptw_data.csv --csv-vehicle-capacity 30 --print-solution
```

## 3. Advanced Script (`test_vrptw_v2.py`)

### Distance mode options

```bash
myenv/bin/python test_vrptw_v2.py --distance-mode euclidean
myenv/bin/python test_vrptw_v2.py --distance-mode manhattan
myenv/bin/python test_vrptw_v2.py --distance-mode linear_sum
```

### Train using customer pool CSV (sample 30 from 200)

```bash
myenv/bin/python test_vrptw_v2.py --train-pool-csv vrptw_pool_200.csv --pool-sample-size 30 --epochs 100
```

### Train using distance matrix

```bash
myenv/bin/python test_vrptw_v2.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv dist_201x201.csv
```

### Train using separate distance + travel-time matrices

```bash
myenv/bin/python test_vrptw_v2.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv dist_201x201.csv \
  --time-matrix-csv time_201x201.csv
```

### Test a custom CSV using large matrix slicing via `customer_id`

```bash
myenv/bin/python test_vrptw_v2.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv dist_201x201.csv \
  --time-matrix-csv time_201x201.csv \
  --test-csv vrptw_data.csv \
  --test-distance-matrix-csv dist_201x201.csv \
  --test-time-matrix-csv time_201x201.csv \
  --print-solution
```

### Fixed quality tracking during training

```bash
myenv/bin/python test_vrptw_v2.py --fixed-eval-every 5 --fixed-eval-size 1000
```

## 4. Data Files in Repo

- `vrptw_pool_200.csv`: 1 depot + 200 customers (`customer_id` included)
- `vrptw_data.csv`: sample test instance (`customer_id` included)
- `dist_201x201.csv`: distance matrix
- `time_201x201.csv`: travel-time matrix

## 5. Notes

- In `test_vrptw_v2.py`, reward uses distance source:
  - `dist_matrix` if provided, else `--distance-mode`.
- Time-window feasibility uses time source:
  - `travel_time_matrix` if provided, else distance source.
- For large matrix slicing on test CSV, include `customer_id` in `--test-csv`.

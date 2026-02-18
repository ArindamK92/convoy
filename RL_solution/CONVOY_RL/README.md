# CONVOY2 Combined-CSV Run Commands

## Setup
```bash
source ~/myenv/bin/activate
cd ~/CONVOY2
```

## Required Inputs
- `--combined-details-csv` : combined depot/customer/CP details CSV
- `--combined-dist-matrix-csv` : combined distance matrix CSV

`--combined-time-matrix-csv` is optional. If omitted, distance matrix is reused as travel-time matrix.

## Accepted Arguments

Required:
- `--combined-details-csv PATH`
- `--combined-dist-matrix-csv PATH`

Optional:
- `-h, --help`
- `--epochs INT` (default: `100`)
- `--batch-size INT` (default: `256`)
- `--eval-batch-size INT` (default: `512`)
- `--train-data-size INT` (default: `4096`)
- `--val-data-size INT` (default: `1024`)
- `--test-data-size INT` (default: `1024`)
- `--lr FLOAT` (default: `1e-4`)
- `--max-time FLOAT` (default: `480.0`)
- `--seed INT` (default: `42`)
- `--baseline {exponential,rollout,shared,mean,no,critic}` (default: `exponential`)
- `--accelerator {auto,cpu,gpu}` (default: `auto`)
- `--print-solution` (flag)
- `--fixed-eval-size INT` (default: `512`)
- `--fixed-eval-every INT` (default: `5`)
- `--checkpoint-dir PATH` (default: `checkpoints_vrptw`)
- `--save-model` (flag)
- `--test-csv PATH`
- `--csv-vehicle-capacity FLOAT` (default: `30.0`)
- `--ev-battery-capacity-kwh FLOAT` (default: `60.0`)
- `--ev-energy-rate-kwh-per-distance FLOAT` (default: `0.5`)
- `--ev-charge-rate-kwh-per-hour FLOAT` (default: `120.0`)
- `--ev-reserve-soc-kwh FLOAT` (default: `0.0`)
- `--ev-num-vehicles INT` (default: `1`)
- `--charging-pool-sample-size INT` (default: `5`)
- `--combined-time-matrix-csv PATH` (default: use `--combined-dist-matrix-csv`)
- `--pool-sample-size INT` (default: `30`)
- `--pool-vehicle-capacity FLOAT` (default: `30.0`)
- `--test-distance-matrix-csv PATH` (optional with `--test-csv`; default: use `--combined-dist-matrix-csv`)
- `--test-time-matrix-csv PATH` (optional with `--test-csv`; default: use `--combined-time-matrix-csv`, else `--combined-dist-matrix-csv`)

## Basic Train/Test Run
```bash
python3 test_convoy_CPs1.py \
  --combined-details-csv data/combined_cust_CP_details.csv \
  --combined-dist-matrix-csv data/combined_dist.csv
```

## Train + Print One Decoded Solution
```bash
python3 test_convoy_CPs1.py \
  --combined-details-csv data/combined_cust_CP_details.csv \
  --combined-dist-matrix-csv data/combined_dist.csv \
  --combined-time-matrix-csv data/combined_time.csv \
  --print-solution
```

## Train + CSV Test Instance
`--test-distance-matrix-csv` and `--test-time-matrix-csv` are optional when using `--test-csv`.  
If omitted, testing falls back to combined matrices.

```bash
python3 test_convoy_CPs1.py \
  --combined-details-csv data/combined_cust_CP_details.csv \
  --combined-dist-matrix-csv data/combined_dist.csv \
  --combined-time-matrix-csv data/combined_time.csv \
  --test-csv data/test_delivery30.csv \
  --test-distance-matrix-csv data/combined_dist.csv \
  --test-time-matrix-csv data/combined_time.csv \
  --print-solution
```

## Save Best Model and Reuse
```bash
python3 test_convoy_CPs1.py \
  --combined-details-csv data/combined_cust_CP_details.csv \
  --combined-dist-matrix-csv data/combined_dist.csv \
  --combined-time-matrix-csv data/combined_time.csv \
  --test-csv data/test_delivery30.csv \
  --charging-pool-sample-size 10 \
  --ev-num-vehicles 5 \
  --print-solution \
  --save-model
```

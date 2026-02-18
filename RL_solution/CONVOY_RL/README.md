# CONVOY2 Run Commands

## Setup
```bash
source ~/myenv/bin/activate
cd ~/CONVOY2
```

## Basic Run
```bash
python3 test_convoy_CPs1.py
```

## Print Solution Paths
```bash
python3 test_convoy_CPs1.py --print-solution
```

## Fixed-Set Quality Check
```bash
python3 test_convoy_CPs1.py --epochs 100 --fixed-eval-every 5 --fixed-eval-size 1000
```

## Test Using CSV Instance
```bash
python3 test_convoy_CPs1.py --test-csv data/vrptw_data.csv --csv-vehicle-capacity 30
```

## Distance Mode Options
```bash
python3 test_convoy_CPs1.py --distance-mode euclidean
python3 test_convoy_CPs1.py --distance-mode linear_sum
python3 test_convoy_CPs1.py --distance-mode manhattan --test-csv data/vrptw_data.csv --print-solution
```

## Train From 200-Customer Pool
```bash
python3 test_convoy_CPs1.py --train-pool-csv data/customers200.csv --pool-sample-size 30
```

## Train Pool + External Distance Matrix
```bash
python3 test_convoy_CPs1.py --train-pool-csv data/customers200.csv --pool-sample-size 30 --distance-matrix-csv data/dist_201x201.csv
```

## Train + CSV Test
```bash
python3 test_convoy_CPs1.py \
  --train-pool-csv data/customers200.csv \
  --pool-sample-size 30 \
  --epochs 100 \
  --test-csv data/vrptw_data.csv \
  --print-solution
```

## Train + Distance Matrix + CSV Test
```bash
python3 test_convoy_CPs1.py \
  --train-pool-csv data/customers200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv data/dist_201x201.csv \
  --test-csv data/vrptw_data.csv \
  --print-solution
```

## Train + Distance Matrix for Train and Test
```bash
python3 test_convoy_CPs1.py \
  --train-pool-csv data/customers200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv data/dist_201x201.csv \
  --test-csv data/vrptw_data.csv \
  --test-distance-matrix-csv data/dist_201x201.csv \
  --print-solution
```

## Train + Distance and Time Matrices
```bash
python3 test_convoy_CPs1.py \
  --train-pool-csv data/customers200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv data/dist_201x201.csv \
  --time-matrix-csv data/time_201x201.csv \
  --test-csv data/vrptw_data.csv \
  --test-distance-matrix-csv data/dist_201x201.csv \
  --test-time-matrix-csv data/time_201x201.csv \
  --print-solution
```

## Custom Charging Points
```bash
python3 test_convoy_CPs1.py \
  --train-pool-csv data/customers200.csv \
  --distance-matrix-csv data/dist_201x201.csv \
  --time-matrix-csv data/time_201x201.csv \
  --test-csv data/vrptw_data.csv \
  --test-distance-matrix-csv data/dist_201x201.csv \
  --test-time-matrix-csv data/time_201x201.csv \
  --charging-pool-csv data/CP_details.csv \
  --charging-pool-sample-size 10 \
  --ev-num-vehicles 5 \
  --print-solution
```

## Save Best Model and Reuse
```bash
python3 test_convoy_CPs1.py \
  --train-pool-csv data/customers200.csv \
  --distance-matrix-csv data/dist_201x201.csv \
  --time-matrix-csv data/time_201x201.csv \
  --test-csv data/test_delivery30.csv \
  --test-distance-matrix-csv data/dist_201x201.csv \
  --test-time-matrix-csv data/time_201x201.csv \
  --charging-pool-csv data/CP_details.csv \
  --charging-pool-sample-size 10 \
  --ev-num-vehicles 5 \
  --print-solution \
  --save-model
```

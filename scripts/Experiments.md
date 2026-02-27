## Common For All Experiments
Run this first to train and store the model checkpoint. Then use it in the scripts below.

### Step 1: Save checkpoint in a directory and reuse later
`convoy_hybrid`: decoder type `greedy`, RL algo `am`

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
  --fixed-eval-every 5 \
  --seed 111 \
  --print-solution \
  --save-model \
  --checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100
```

## Comparison Experiments (using POMO + beam search)
```bash
./run_cust_cp_ev_sweep_new.sh
```

## Delivery-To-EV Ratio Experiments (using POMO with beam search)
```bash
./run_ratio_sweep_cust50_cp20_new.sh
```

## Small Experiments (using RL algo Attention Model (am) with greedy)
```bash
./run_small_methods_sweep.sh
```

## Common For All Experiments
Run this first to train and store the model checkpoint. Then use it in the scripts below.

### Step 1: Save checkpoint in a directory and reuse later
`convoy_hybrid`: decoder type `greedy`, RL algo `am`

```bash
/home/akkcm/myenv/bin/python tests/test_convoy_hybrid.py \
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

```
 /home/akkcm/myenv/bin/python -m src.convoy_rl_partial_ch2.convoy_rl_main \
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
  --print-solution \
  --rl-algo pomo \
  --baseline shared \
  --pomo-num-starts 10 \
  --pomo-num-augment 8 \
  --save-model \
  --checkpoint-dir checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100_pomo
  ```

## Comparison Experiments (using POMO + beam search)
```bash
./run_cust_cp_ev_sweep_new_pomo.sh
./run_cust_cp_ev_sweep_new_pomo_rlv2.sh
```
old: ./run_cust_cp_ev_sweep_new.sh

## Delivery-To-EV Ratio Experiments (using POMO with beam search)
```bash
./run_ratio_sweep_cust50_cp20_new.sh
./run_ratio_sweep_cust50_cp20_new_rlv2.sh
```

## Small Experiments (using RL algo Attention Model (am) with greedy)
```bash
./scripts/run_small_methods_sweep_rlv2.sh
```

# More Running Examples

Use `python` below from repo root (`~/CONVOY`).
Activate your virtual environment first if needed (for example, `source ~/myenv/bin/activate`).

## 1) Run `convoy_hybrid` only

Use this to run hybrid directly on a 50-customer / 10-CP test setup.

```bash
python tests/test_convoy_hybrid.py \
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
  --print-solution
```

## 2) Save checkpoint for hybrid (AM + greedy)

Use this to train/evaluate hybrid and save checkpoint for later reuse.

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
  --checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100 \
  --verbose true
```

## 3) Hybrid with POMO + beam search

Use this to switch RL algorithm to POMO and decode with beam search.

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
  --rl-algo pomo \
  --baseline shared \
  --decode-type beam_search \
  --decode-beam-width 10 \
  --save-model \
  --checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100_POMO
```

## 4) Run `convoy_rl_partial_ch` only

Use this to train/evaluate only partial-charging RL and save checkpoint.

```bash
python -m src.convoy_rl_partial_ch.convoy_rl_main \
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
  --save-model \
  --checkpoint-dir checkpoints_vrptw/rl_partial_c50_cp10_ev10_e100
```

## 5) `convoy_main` with separate checkpoint dirs (hybrid vs RL partial)

Use this to run both RL stages while keeping their checkpoints in different directories.
Note: stage-specific options like `--epochs`, `--fixed-eval-every`, and `--seed` must go inside `--opt-rl-extra`.

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --opt-rl-extra "--save-model --epochs 100 --fixed-eval-every 5 --seed 111" \
  --rl-checkpoint-dir checkpoints_vrptw/rl_partial_c50_cp10_ev10_e100 \
  --hybrid-checkpoint-dir checkpoints_vrptw/hybrid_c50_cp10_ev10_e100
```

## 6) Run heuristic CSA + EDF + NDF only

Use this to run only opt+heu pipeline (with MILP skipped).

```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-for-opt-heu data/test_instance_50c_10cp.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-opt-heu \
  --iterations 1 \
  --results-file heu_ndf_edf_test50c10cp.csv \
  --opt-heu-extra "--skip-optimal --random-seed 111"
```

## 7) Test multiple methods on the same file (`test_instance_5c_3cp.csv`)

Use these to compare methods on one fixed small test instance.

### 7.1 Hybrid on the same test file

```bash
python tests/test_convoy_hybrid.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance_5c_3cp.csv \
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

### 7.2 Opt+Heu heuristics on the same test file

```bash
python -m src.convoy_opt_and_heu.opt_and_hue \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-for-opt-heu data/test_instance_5c_3cp.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --skip-optimal \
  --no-edf-ndf
```

### 7.3 Opt+Heu MILP-only on the same test file

```bash
python -m src.convoy_opt_and_heu.opt_and_hue \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-for-opt-heu data/test_instance_5c_3cp.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --only-milp
```

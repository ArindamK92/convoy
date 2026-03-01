# Convoy RL Partial Charging (`convoy_rl_partial_ch`)

`convoy_rl_partial_ch` trains a modified RL4CO CVRPTW model and evaluates on a test instance with reward decomposition:
- before partial charging (full-charge style),
- after partial charging.

## Reward Function Used In RL

Let:
- `E_t = distance_t * ev_energy_rate_kwh_per_distance` (energy used on step `t`)
- `B_eff` = effective battery (`ev_battery_capacity_kwh - reserve_battery`)
- `c_j` = charge cost per kWh at node `j`
- `lambda` = `cost_weight`

Per-step reward in `_get_reward` is:

1. Customer node `j` (non-charging node):
- if first-visit and on-time (Assumption: max unit charging cost is $0.6 per kWh):
  - `R_t = customer_reward_j - 0.6 * E_t`
- otherwise:
  - `R_t = 0 - 0.6 * E_t`

2. Charging node `j` (depot or charging station):
- `charge_needed_t = max(B_eff - max(B_prev - E_t, 0), 0)`
- `charging_penalty_t = charge_needed_t * (0.6 - c_j)`
- `R_t = 0.6 * E_t + lambda * charging_penalty_t`

If rollout ends away from depot, one implicit final return-to-depot term is added with the same charging-node rule:
- `R_final = 0.6 * E_back + lambda * (charge_needed_back * (0.6 - c_depot))`

Total RL reward is:
- `R_total = sum_t R_t (+ R_final when needed)`

## Recommended Run Method

Run from repo root (`/home/akkcm/CONVOY2`) using the test shim:

```bash
python tests/test_convoy_CPs1.py \
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
  --checkpoint-dir checkpoints_vrptw/rl_partial_c50_cp10_ev10_e100
```

Alternative (module form):

```bash
python -m src.convoy_rl_partial_ch.convoy_rl_main --help
```

## Required Arguments

- `--combined-details-csv`
- `--combined-dist-matrix-csv`

## Full Argument Reference

### Training / Optimization

- `--epochs`
- `--batch-size`
- `--eval-batch-size`
- `--train-data-size`
- `--val-data-size`
- `--test-data-size`
- `--lr`
- `--max-time`
- `--seed`
- `--accelerator {auto,cpu,gpu}`

### RL Algorithm

- `--rl-algo {am,pomo}`
- `--baseline {exponential,rollout,shared,mean,no,critic}`
- `--pomo-num-starts`
- `--pomo-num-augment`

Notes:
- For `--rl-algo pomo`, use `--baseline shared`.

### Decoder

- `--decode-type {greedy,sampling,beam_search,multistart_greedy,multistart_sampling}`
- `--decode-num-samples`
- `--decode-num-starts`
- `--decode-select-best`
- `--decode-temperature`
- `--decode-top-p`
- `--decode-top-k`
- `--decode-beam-width`

### Evaluation / Checkpoint

- `--fixed-eval-size`
- `--fixed-eval-every`
- `--checkpoint-dir`
- `--save-model`
- `--print-solution`

Checkpoint names used when `--save-model`:
- `best_model.ckpt` for `--rl-algo am`
- `best_model_pomo.ckpt` for `--rl-algo pomo`

### Data / Instance Inputs

- `--combined-details-csv`
- `--combined-dist-matrix-csv`
- `--combined-time-matrix-csv`
- `--customer-num`
- `--charging-stations-num`
- `--ev-num`
- `--pool-vehicle-capacity`
- `--test-csv`
- `--test-distance-matrix-csv`
- `--test-time-matrix-csv`
- `--csv-vehicle-capacity`

Matrix fallback for `--test-csv`:
- distance matrix: `--test-distance-matrix-csv` -> `--combined-dist-matrix-csv`
- time matrix: `--test-time-matrix-csv` -> `--combined-time-matrix-csv` -> `--combined-dist-matrix-csv`

### EV / Cost / Energy

- `--ev-battery-capacity-kwh`
- `--ev-energy-rate-kwh-per-distance`
- `--ev-charge-rate-kwh-per-hour`
- `--reserve-battery` (alias `--ev-reserve-soc-kwh`)
- `--cost-weight`

## Help Command

```bash
python tests/test_convoy_CPs1.py --help
```

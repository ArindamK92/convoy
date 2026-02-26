# CONVOY2 Combined-CSV Run Commands

## Setup
```bash
source ~/myenv/bin/activate
cd ~/CONVOY2
pip install -r requirements.txt
```

## Gurobi License Config
MILP now reads Gurobi options from a separate config file (no hardcoded credentials in code).

1. Create local config from template:
```bash
cp config/gurobi_wls.example.json config/gurobi_wls.json
```
2. Fill `config/gurobi_wls.json` with your `WLSACCESSID`, `WLSSECRET`, and `LICENSEID`.

Optional overrides:
- Use a different config path: set `CONVOY_GUROBI_CONFIG=/path/to/gurobi_wls.json`
- Override directly by env vars:
  - `GUROBI_WLSACCESSID`
  - `GUROBI_WLSSECRET`
  - `GUROBI_LICENSEID`
  - `GUROBI_OUTPUTFLAG`
  - `GUROBI_LOGTOCONSOLE`

## Documentation Site
Install docs dependencies:

```bash
make docs-install
```

Serve locally:

```bash
make docs-serve
```

Build static site:

```bash
make docs-build
```

## CLI Architecture
- `convoy_parser.py` is the parser entry point for this repo.
- `convoy_main.py` parses at top level, then passes parsed parameters to:
- `src/convoy_opt_and_heu/opt_and_hue.py` via `run_opt_heu_with_params(...)`
- `convoy_hybrid/convoy_hybrid_main.py` via `run_rl(args)`
- `src/convoy_rl_partial_ch/convoy_rl_main.py` via `run_rl(args)`
- Dedicated launchers still work:
- `tests/test_convoy_CPs1.py` calls RL main wrapper.
- `tests/test_convoy_opt_and_heu.py` calls opt+heu main wrapper.
- These wrappers also parse through `convoy_parser.py` (not separate parser definitions in runner files).

## Required Inputs
- `--combined-details-csv` : combined depot/customer/CP details CSV
- `--combined-dist-matrix-csv` : combined distance matrix CSV

`--combined-time-matrix-csv` is optional. If omitted, distance matrix is reused as travel-time matrix.

## Argument Reference (convoy_rl / `tests/test_convoy_CPs1.py`)

Required:
- `--combined-details-csv PATH`: Combined depot/customer/charging-point details CSV used for sampling training customers and charging stations.
- `--combined-dist-matrix-csv PATH`: Combined distance matrix CSV used for travel distance lookup by global node IDs.

Optional:
- `-h, --help`: Print command help and exit.
- `--epochs INT` (default: `100`): Number of training epochs.
- `--batch-size INT` (default: `256`): Training batch size.
- `--eval-batch-size INT` (default: `512`): Batch size for validation and testing.
- `--train-data-size INT` (default: `4096`): Number of training instances generated per epoch.
- `--val-data-size INT` (default: `1024`): Number of validation instances.
- `--test-data-size INT` (default: `1024`): Number of test instances.
- `--lr FLOAT` (default: `1e-4`): Learning rate.
- `--max-time FLOAT` (default: `480.0`): Time-window horizon used by generator-mode data.
- `--seed INT` (default: `42`): Random seed.
- `--rl-algo {am,pomo}` (default: `am`): RL algorithm. `am` uses AttentionModel (REINFORCE), `pomo` uses POMO.
- `--baseline {exponential,rollout,shared,mean,no,critic}` (default: `exponential`): REINFORCE baseline type. For `--rl-algo pomo`, this must be `shared`.
- `--pomo-num-starts INT` (default: `0`): POMO starts per instance (`0` means auto).
- `--pomo-num-augment INT` (default: `8`): POMO augmentation count (`1` disables augmentation).
- `--accelerator {auto,cpu,gpu}` (default: `auto`): Lightning accelerator selection.
- `--print-solution` (flag): Print one decoded route solution after testing.
- `--decode-type {greedy,sampling,beam_search,multistart_greedy,multistart_sampling}` (default: `greedy`): Decoder strategy for fixed-eval/test/printed solution.
- `--decode-num-samples INT` (default: `1`): Number of sampled rollouts for sampling decode.
- `--decode-num-starts INT` (default: `0`): Number of starts for multistart decode.
- `--decode-select-best` (flag): Select best rollout among sampled/multistart candidates.
- `--decode-temperature FLOAT` (default: `1.0`): Decoder sampling temperature.
- `--decode-top-p FLOAT` (default: `0.0`): Top-p nucleus sampling threshold (`0` disables).
- `--decode-top-k INT` (default: `0`): Top-k sampling truncation (`0` disables).
- `--decode-beam-width INT` (default: `8`): Beam width when `--decode-type beam_search`.
- `--fixed-eval-size INT` (default: `512`): Fixed evaluation set size tracked across epochs.
- `--fixed-eval-every INT` (default: `5`): Evaluate on fixed set every N epochs.
- `--checkpoint-dir PATH` (default: `checkpoints_vrptw`): Directory where checkpoints are saved/loaded.
- `--save-model` (flag): Save best checkpoint and reuse it in future runs. File name is `best_model.ckpt` for `--rl-algo am` and `best_model_pomo.ckpt` for `--rl-algo pomo`.
- `--test-csv PATH` (default: none): Optional custom single test-instance CSV for final decoding/evaluation. If this CSV includes CP rows, those exact CPs are used directly during testing.
- `--csv-vehicle-capacity FLOAT` (default: `30.0`): Vehicle capacity used to normalize demand values in `--test-csv`.
- `--ev-battery-capacity-kwh FLOAT` (default: `30.0`): EV battery capacity in kWh (full SOC).
- `--ev-energy-rate-kwh-per-distance FLOAT` (default: `0.00025`): Energy use per distance unit. For meter-based matrices this default means 4 km/kWh mileage (`4 km/kWh => 0.25 kWh/km => 0.00025 kWh/m`).
- `--ev-charge-rate-kwh-per-hour FLOAT` (default: `120.0`): Depot charging rate (kWh/hour).
- `--reserve-battery FLOAT` (default: `0.0`): Reserve battery in kWh. Effective usable battery is `ev-battery-capacity-kwh - reserve-battery`. Alias: `--ev-reserve-soc-kwh`.
- `--ev-num INT` (default: `1`): Fleet size (number of EVs).
- `--charging-stations-num INT` (default: `5`): Number of charging stations sampled into each generated instance.
- `--combined-time-matrix-csv PATH` (default: none): Optional combined travel-time matrix CSV. If omitted, distance matrix is reused as travel-time matrix.
- `--customer-num INT` (default: `30`): Number of customers sampled per generated training/validation/test pool instance.
- `--pool-vehicle-capacity FLOAT` (default: `30.0`): Vehicle capacity used for demand normalization in pool-generated instances.
- `--test-distance-matrix-csv PATH` (default: none): Optional distance-matrix override for `--test-csv`; if omitted, uses `--combined-dist-matrix-csv`.
- `--test-time-matrix-csv PATH` (default: none): Optional travel-time-matrix override for `--test-csv`; if omitted, uses `--combined-time-matrix-csv`, else falls back to `--combined-dist-matrix-csv`.

## Argument Scope By Runner
Primary shared parser location:
- `convoy_parser.py` (main combined parser + opt/heu parser + RL forwarding helpers).

Shared in both `convoy_rl` (`tests/test_convoy_CPs1.py`) and `convoy_opt_and_heu` (`tests/test_convoy_opt_and_heu.py`):
- `--combined-details-csv`
- `--combined-dist-matrix-csv`
- `--combined-time-matrix-csv`
- `--customer-num`
- `--charging-stations-num`
- `--ev-num`
- `--ev-energy-rate-kwh-per-distance`
- `--reserve-battery`

`convoy_rl` (`tests/test_convoy_CPs1.py`) only:
- `--epochs`
- `--batch-size`
- `--eval-batch-size`
- `--train-data-size`
- `--val-data-size`
- `--test-data-size`
- `--lr`
- `--max-time`
- `--seed`
- `--rl-algo`
- `--baseline`
- `--pomo-num-starts`
- `--pomo-num-augment`
- `--accelerator`
- `--print-solution`
- `--decode-type`
- `--decode-num-samples`
- `--decode-num-starts`
- `--decode-select-best`
- `--decode-temperature`
- `--decode-top-p`
- `--decode-top-k`
- `--decode-beam-width`
- `--fixed-eval-size`
- `--fixed-eval-every`
- `--checkpoint-dir`
- `--save-model`
- `--test-csv`
- `--csv-vehicle-capacity`
- `--ev-battery-capacity-kwh`
- `--ev-charge-rate-kwh-per-hour`
- `--reserve-battery` (alias: `--ev-reserve-soc-kwh`)
- `--pool-vehicle-capacity`
- `--test-distance-matrix-csv`
- `--test-time-matrix-csv`

`convoy_opt_and_heu` (`tests/test_convoy_opt_and_heu.py`) only:
- `--alpha1`
- `--alpha2`
- `--skip-optimal`
- `--random-seed`

Combined wrapper `convoy_main.py` only:
- `--only-rl`
- `--only-opt-heu`
- `--skip-convoy-rl`
- `--opt-rl-extra`
- `--opt-heu-extra`
- `--iterations`
- `--results-file`
- `--clear-rl-checkpoints`
- `--run-baseline`
- `--baseline-bin`
- `--baseline-time`
- `--baseline-runs`
- `--baseline-instance-output-path`
- `--baseline-output-file`
- `--baseline-quiet`
- `--baseline-solver-log`
- `--baseline-print-charging-events`
- `--baseline-extra`

### Meaning Of `--opt-rl-extra` And `--opt-heu-extra`
- `--opt-rl-extra` is a pass-through string for arguments used by both hybrid and RL runners (`convoy_hybrid/convoy_hybrid_main.py` and `src/convoy_rl_partial_ch/convoy_rl_main.py`).
- `--opt-heu-extra` is a pass-through string for arguments that belong only to the opt+heu runner (`tests/test_convoy_opt_and_heu.py` / `src/convoy_opt_and_heu/opt_and_hue.py`).
- `convoy_main.py` does not interpret these inner flags itself; it forwards them to the corresponding runner parser.
- Use quoted strings so multiple forwarded flags stay grouped.

Examples:
- `--opt-rl-extra "--print-solution --save-model"`
- `--opt-rl-extra "--seed 77 --test-csv data/test_instance.csv"`
- `--opt-rl-extra "--rl-algo pomo --baseline shared --pomo-num-starts 16 --pomo-num-augment 8"`
- `--opt-heu-extra "--skip-optimal"`
- `--opt-heu-extra "--random-seed 123 --alpha1 1.0 --alpha2 1.0"`

`--clear-rl-checkpoints` behavior:
- Off by default.
- When enabled, it deletes the RL checkpoint folder once at the start of `convoy_main.py`.
- If `--opt-rl-extra` includes `--checkpoint-dir ...`, that directory is deleted.
- Otherwise, default `checkpoints_vrptw` under `CONVOY2` is deleted.

`--run-baseline` behavior:
- When enabled, `convoy_main.py` runs baseline conversion + solver + metric computation in each iteration.
- Baseline row is appended to the same results CSV with:
  - `Total reward` from visited-customer rewards,
  - `Total cost` from charging cost (including final depot recharge-to-full),
  - `Objective val = Total reward - Total cost`,
  - `Total successful delivery` from visited customer count.
- Baseline `Elapsed time (ms)` uses solver-reported time when available; otherwise wall-clock pipeline time.
- Baseline defaults in `convoy_main.py`:
  - `--baseline-runs` defaults to `5`.
  - Baseline quiet mode is enabled by default (logs are written to `baseline/data/baseline_solver.log`).
  - Use `--baseline-no-quiet` to print baseline solver logs to console.
- Use `--baseline-extra "..."` to pass additional baseline solver flags.


## Test Commands

### Train + Print One Decoded Solution
```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_cust_CP_details.csv \
  --combined-dist-matrix-csv data/combined_dist.csv \
  --combined-time-matrix-csv data/combined_time.csv \
  --print-solution
```

### Train + CSV Test Instance
`--test-distance-matrix-csv` and `--test-time-matrix-csv` are optional when using `--test-csv`.  
If omitted, testing falls back to combined matrices.

`--test-csv` supports two schemas:
- Legacy schema: `customer_id,is_depot,x,y,demand,tw_start,tw_end,service_time,reward,...`
- Combined schema: `ID,type,lng,lat,first_receive_tm,last_receive_tm,service_time,reward,...`

To include charging points in `--test-csv`:
- Legacy schema: mark CP rows with `is_charging_station=1` (or `is_cp=1` / `node_type=f`) and provide `cp_id`.
- Combined schema: include rows with `type=f` and `ID` as CP id.

When CP rows are present in `--test-csv`, they are used as fixed test CPs (no random CP sampling for that test instance).

```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_cust_CP_details.csv \
  --combined-dist-matrix-csv data/combined_dist.csv \
  --combined-time-matrix-csv data/combined_time.csv \
  --test-csv data/test_delivery30.csv \
  --test-distance-matrix-csv data/combined_dist.csv \
  --test-time-matrix-csv data/combined_time.csv \
  --print-solution
```

### Save Best Model and Reuse
```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_delivery10_jd200.csv \
  --charging-stations-num 3 \
  --ev-num 2 \
  --customer-num 10 \
  --print-solution \
  --save-model
```

### RL Decode Alternatives (Better Than Plain Greedy)
Best-of-sampling on one test instance:

```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance.csv \
  --test-distance-matrix-csv data/distance_matrix_jd200_1.csv \
  --test-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --decode-type sampling \
  --decode-num-samples 128 \
  --decode-select-best \
  --decode-temperature 1.0 \
  --print-solution
```

Beam search decode:

```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --decode-type beam_search \
  --decode-beam-width 16 \
  --print-solution
```

### Run RL With POMO (runtime-selectable)
```bash
python3 tests/test_convoy_CPs1.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --rl-algo pomo \
  --baseline shared \
  --pomo-num-starts 16 \
  --pomo-num-augment 8 \
  --decode-type multistart_greedy \
  --decode-num-starts 16 \
  --decode-select-best \
  --print-solution
```

### Run Optimal + Heuristic
Use the dedicated launcher:

```bash
python3 tests/test_convoy_opt_and_heu.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2
```

Notes:
- `convoy_opt_and_heu` now takes file paths directly (no base-file name argument).
- `--customer-num`, `--charging-stations-num`, and `--ev-num` are required and control deliveries, charging points (excluding depot), and EV count.
- Optional: `--ev-energy-rate-kwh-per-distance` controls mileage conversion in opt/heu via `mj = 1 / (rate * 1000)`.
- Optional: `--reserve-battery` sets reserve SOC margin. Heuristic uses effective battery as `full_battery - reserve_battery`.
- Optional: `--skip-optimal` runs only heuristic and skips MILP.
- Optional: `--random-seed` controls preprocessing sampling for customers/CPs and EV charge-acceptance rates.
- By default, opt+heu also runs `NDF` and `EDF` baselines and appends them to results.
- Optional: `--no-EDF-NDF` skips `NDF` and `EDF`.
- Result CSV is written to `results/` under `CONVOY2` (for example: `results/results3_combined_data_jd200_1.csv`).

## Run RL + Optimal/Heuristic Together
Use a single launcher with shared arguments:

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --opt-rl-extra "--print-solution --save-model"
```

Notes:
- The command above runs optimal+heuristic first, then `convoy_hybrid`, then `convoy_rl_partial_ch`.
- In the combined run, optimal+heuristic writes `test_instance.csv` in the same folder as `--combined-details-csv`, and both hybrid and RL automatically use it as `--test-csv`.
- `convoy_main.py` also accepts `--ev-energy-rate-kwh-per-distance` and forwards it to both runners.
- `convoy_main.py` writes a consolidated CSV after all enabled runners finish, including `Optimal`, `CSA`, `NDF`, `EDF`, `Hybrid_RL4CO`, `Hybrid_RL4CO_partial_charging`, `RL`, `RL_partial_charging`, and (optionally) `Baseline` rows.
- `--iterations` (default `1`) repeats the full run N times and logs an `itr` column in results.
- `--results-file` lets you choose the output CSV file. If relative, it is created under `CONVOY2/results/`.
- For iterations, opt/heu preprocessing sampling uses an iteration-specific seed so customer/CP subsets can vary across runs.
- You can set opt/heu base seed via `--opt-heu-extra="--random-seed <INT>"`; with iterations, it increments by iteration index.
- For RL, if `--seed` is not provided in `--opt-rl-extra`, `convoy_main.py` offsets RL seed per iteration (`seed + itr - 1`).
- Use `--opt-rl-extra "..."` to pass hybrid/RL flags.
- Use `--opt-heu-extra "..."` to pass optimal/heuristic-only flags (for example `--opt-heu-extra "--skip-optimal --no-EDF-NDF"`).
- Use `--only-rl` or `--only-opt-heu` to run just one side.
- Use `--skip-convoy-rl` to skip only `convoy_rl_partial_ch` while still running `convoy_hybrid`.
- For `--opt-rl-extra` and `--opt-heu-extra`, pass one quoted string per flag group (or use `--flag="..."` form), otherwise argparse may report `expected one argument`.

### Run Only Hybrid In `convoy_main` (Skip `convoy_rl_partial_ch`)
```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --only-rl \
  --skip-convoy-rl \
  --opt-rl-extra "--print-solution --seed 42"
```

### Run `convoy_main` With Sampling Decoder
```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 1 \
  --results-file rl_sampling_vs_opt_heu.csv \
  --opt-rl-extra "--print-solution --save-model --seed 42 --decode-type sampling --decode-num-samples 128 --decode-select-best --decode-temperature 1.0" \
  --opt-heu-extra "--random-seed 123"
```

### Run `convoy_main` With POMO
```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 1 \
  --only-rl \
  --clear-rl-checkpoints \
  --opt-rl-extra "--print-solution --seed 111 --rl-algo pomo --baseline shared --pomo-num-starts 16 --pomo-num-augment 8 --decode-type multistart_greedy --decode-num-starts 16 --decode-select-best"
```

### Run Combined Launcher For 10 Iterations
```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 10 \
  --results-file jd200_itr10_2.csv \
  --opt-rl-extra "--print-solution --save-model --seed 42" \
  --opt-heu-extra "--random-seed 42"
```

### Run Combined Launcher For 10 Iterations (Cust=5, Clear RL Checkpoints)
```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 10 \
  --results-file jd200_itr10_cust5_4.csv \
  --reserve-battery 0 \
  --clear-rl-checkpoints \
  --opt-rl-extra "--print-solution --save-model --seed 42" \
  --opt-heu-extra "--random-seed 123"
```

### Run Combined Launcher + Baseline (same results CSV)
```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 1 \
  --results-file with_baseline.csv \
  --run-baseline \
  --baseline-time 10 \
  --opt-rl-extra "--print-solution --save-model --seed 42" \
  --opt-heu-extra "--random-seed 123"
```

## Run Only RL On `test_instance.csv`
When using `--only-rl`, pass `--test-csv` explicitly in `--opt-rl-extra`:

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 35 \
  --charging-stations-num 5 \
  --ev-num 5 \
  --only-rl \
  --opt-rl-extra="--test-csv data/test_instance.csv --test-distance-matrix-csv data/distance_matrix_jd200_1.csv --test-time-matrix-csv data/time_matrix_jd200_1.csv --print-solution --save-model"
```

## Baseline (EVRP-TW-SPD-HMA)

Baseline repository:
- https://github.com/0SliverBullet/EVRP-TW-SPD-HMA.git

This project supports running the baseline solver binary from `CONVOY2/baseline/bin/evrp-tw-spd` on instances converted from CONVOY2 CSVs.

### One-shot pipeline (recommended)
```bash
cd ~/CONVOY2
/home/akkcm/myenv/bin/python tools/run_baseline_pipeline.py \
  --test-csv data/test_instance.csv \
  --dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --time-matrix-csv data/time_matrix_jd200_1.csv \
  --vehicles 5 \
  --instance-output-path baseline/data/test_instance_evrp.txt \
  --latest-baseline-output baseline/data/latest_baseline_output.txt \
  --baseline-time 10 \
  --ev-energy-rate-kwh-per-distance 0.00025 \
  --print-charging-events
```

Defaults for one-shot pipeline:
- `--baseline-runs` defaults to `5`.
- Quiet mode is on by default and overwrites `baseline/data/baseline_solver.log`.
- Use `--baseline-no-quiet` to print solver logs to console.

### 1) Convert `test_instance.csv` to EVRP-TW-SPD-HMA format
```bash
cd ~/CONVOY2
/home/akkcm/myenv/bin/python tools/convert_to_evrp_instance.py \
  --test-csv data/test_instance.csv \
  --dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --time-matrix-csv data/time_matrix_jd200_1.csv \
  --vehicles 5 \
  --output-path baseline/data/test_instance_evrp.txt
```

Generated files:
- `baseline/data/test_instance_evrp.txt`
- `baseline/data/test_instance_evrp.id_map.csv` (mapped contiguous IDs to original IDs)

### 2) Run baseline solver on converted instance
```bash
cd ~/CONVOY2/baseline
./bin/evrp-tw-spd \
  --problem ./data/test_instance_evrp.txt \
  --pruning \
  --time 10 \
  --runs 1 \
  --related_removal \
  --regret_insertion

# Copy the latest solver output to a stable filename used by metric computation.
cp ~/CONVOY2/baseline/" test_instance_evrp_timelimit=10_subproblem=1.txt" \
  ~/CONVOY2/baseline/data/latest_baseline_output.txt
```

### 3) Compute CONVOY metrics from baseline output
```bash
/home/akkcm/myenv/bin/python ~/CONVOY2/tools/compute_baseline_metrics.py \
  --baseline-output-file ~/CONVOY2/baseline/data/latest_baseline_output.txt \
  --test-instance-csv ~/CONVOY2/data/test_instance.csv \
  --id-map-csv ~/CONVOY2/baseline/data/test_instance_evrp.id_map.csv \
  --ev-energy-rate-kwh-per-distance 0.00025 \
  --print-charging-events
```

Notes:
- `--related_removal` + `--regret_insertion` are recommended to avoid empty destroy/repair operator sets.
- If needed, set executable permission once: `chmod +x baseline/bin/evrp-tw-spd`.
- Baseline metric computation always includes final recharge-to-full at depot (to match CONVOY MILP/heuristic cost accounting).

## Tools

### Convert EVRP-TW-SPD-HMA JD instance to CONVOY2 CSVs
```bash
python3 tools/convert_data.py \
  --input-txt /home/akkcm/EVRP-TW-SPD-HMA/data/jd_instances/jd200_4.txt \
  --output-dir data \
  --seed 123
```

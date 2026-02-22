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

## CLI Architecture (Refactored)
- `convoy_parser.py` is the parser entry point for this repo.
- `convoy_main.py` parses at top level, then passes parsed parameters to:
- `src/convoy_opt_and_heu/opt_and_hue.py` via `run_opt_heu_with_params(...)`
- `src/convoy_rl/convoy_rl_main.py` via `run_rl(args)`
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
- `--baseline {exponential,rollout,shared,mean,no,critic}` (default: `exponential`): REINFORCE baseline type.
- `--accelerator {auto,cpu,gpu}` (default: `auto`): Lightning accelerator selection.
- `--print-solution` (flag): Print one decoded route solution after testing.
- `--fixed-eval-size INT` (default: `512`): Fixed evaluation set size tracked across epochs.
- `--fixed-eval-every INT` (default: `5`): Evaluate on fixed set every N epochs.
- `--checkpoint-dir PATH` (default: `checkpoints_vrptw`): Directory where checkpoints are saved/loaded.
- `--save-model` (flag): Save best checkpoint to `best_model.ckpt` and reuse it in future runs.
- `--test-csv PATH` (default: none): Optional custom single test-instance CSV for final decoding/evaluation. If this CSV includes CP rows, those exact CPs are used directly during testing.
- `--csv-vehicle-capacity FLOAT` (default: `30.0`): Vehicle capacity used to normalize demand values in `--test-csv`.
- `--ev-battery-capacity-kwh FLOAT` (default: `30.0`): EV battery capacity in kWh (full SOC).
- `--ev-energy-rate-kwh-per-distance FLOAT` (default: `0.00025`): Energy use per distance unit. For meter-based matrices this default means 4 km/kWh mileage (`4 km/kWh => 0.25 kWh/km => 0.00025 kWh/m`).
- `--ev-charge-rate-kwh-per-hour FLOAT` (default: `120.0`): Depot charging rate (kWh/hour).
- `--ev-reserve-soc-kwh FLOAT` (default: `0.0`): Minimum battery reserve that must remain.
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
- `--baseline`
- `--accelerator`
- `--print-solution`
- `--fixed-eval-size`
- `--fixed-eval-every`
- `--checkpoint-dir`
- `--save-model`
- `--test-csv`
- `--csv-vehicle-capacity`
- `--ev-battery-capacity-kwh`
- `--ev-charge-rate-kwh-per-hour`
- `--ev-reserve-soc-kwh`
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
- `--opt-rl-extra`
- `--opt-heu-extra`
- `--iterations`
- `--results-file`

### Meaning Of `--opt-rl-extra` And `--opt-heu-extra`
- `--opt-rl-extra` is a pass-through string for arguments that belong only to the RL runner (`tests/test_convoy_CPs1.py` / `src/convoy_rl/convoy_rl_main.py`).
- `--opt-heu-extra` is a pass-through string for arguments that belong only to the opt+heu runner (`tests/test_convoy_opt_and_heu.py` / `src/convoy_opt_and_heu/opt_and_hue.py`).
- `convoy_main.py` does not interpret these inner flags itself; it forwards them to the corresponding runner parser.
- Use quoted strings so multiple forwarded flags stay grouped.

Examples:
- `--opt-rl-extra "--print-solution --save-model"`
- `--opt-rl-extra "--seed 77 --test-csv data/test_instance.csv"`
- `--opt-heu-extra "--skip-optimal"`
- `--opt-heu-extra "--random-seed 123 --alpha1 1.0 --alpha2 1.0"`


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
- Optional: `--skip-optimal` runs only heuristic and skips MILP.
- Optional: `--random-seed` controls preprocessing sampling for customers/CPs and EV charge-acceptance rates.
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
- The command above runs optimal+heuristic first, then RL.
- In the combined run, optimal+heuristic writes `test_instance.csv` in the same folder as `--combined-details-csv`, and RL automatically uses it as `--test-csv`.
- `convoy_main.py` also accepts `--ev-energy-rate-kwh-per-distance` and forwards it to both runners.
- `convoy_main.py` writes a consolidated CSV after all enabled runners finish, including an `RL` row.
- `--iterations` (default `1`) repeats the full run N times and logs an `itr` column in results.
- `--results-file` lets you choose the output CSV file. If relative, it is created under `CONVOY2/results/`.
- For iterations, opt/heu preprocessing sampling uses an iteration-specific seed so customer/CP subsets can vary across runs.
- You can set opt/heu base seed via `--opt-heu-extra="--random-seed <INT>"`; with iterations, it increments by iteration index.
- For RL, if `--seed` is not provided in `--opt-rl-extra`, `convoy_main.py` offsets RL seed per iteration (`seed + itr - 1`).
- Use `--opt-rl-extra "..."` to pass RL-only flags.
- Use `--opt-heu-extra "..."` to pass optimal/heuristic-only flags (for example `--opt-heu-extra "--skip-optimal"`).
- Use `--only-rl` or `--only-opt-heu` to run just one side.
- For `--opt-rl-extra` and `--opt-heu-extra`, pass one quoted string per flag group (or use `--flag="..."` form), otherwise argparse may report `expected one argument`.

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

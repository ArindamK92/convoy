# CONVOY

## Setup



### Environment setup
```bash
python3 -m venv ~/myenv
source ~/myenv/bin/activate
cd ~/CONVOY2
pip install -r requirements.txt
```

### Gurobi License Config
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

### Documentation Site
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



## Required Inputs
- `--combined-details-csv` : combined depot/customer/CP details CSV
- `--combined-dist-matrix-csv` : combined distance matrix CSV

`--combined-time-matrix-csv` is optional. If omitted, distance matrix is reused as travel-time matrix.



## `convoy_main.py` Wrapper Arguments

Wrapper-only arguments in `convoy_main.py`:

| Argument | Default | Purpose / Behavior |
| --- | --- | --- |
| `--only-rl` | `False` | Skip opt+heu stage and run RL stages only. |
| `--only-opt-heu` | `False` | Skip RL stages and run opt+heu only. |
| `--only-rl-v2` | `False` | Run only `convoy_rl_partial_ch2` stage. |
| `--rl-v2-checkpoint-dir` | `None` | Stage-specific checkpoint directory override for `convoy_rl_partial_ch2`. |
| `--hybrid-checkpoint-dir` | `None` | Stage-specific checkpoint directory override for `m_VRPTW`. |
| `--opt-rl-extra` | `[]` | Repeatable quoted pass-through args for RL/hybrid parsers (`action=append`). |
| `--opt-heu-extra` | `[]` | Repeatable quoted pass-through args for opt+heu parser (`action=append`). |
| `--iterations` | `1` | Number of repeated runs (`itr`) written to results CSV. |
| `--results-file` | `None` | Output CSV path/name. When omitted, uses `results/results3_<combined-details-stem>.csv`. |
| `--clear-rl-checkpoints` | `False` | If set, delete resolved RL checkpoint directories once before execution. |

### Important Arguments
- `--opt-rl-extra` is a pass-through string for arguments used by both hybrid and RL runners (`m_VRPTW/convoy_hybrid_main.py` and `src/convoy_rl_partial_ch2/convoy_rl_main.py`).
- `--opt-heu-extra` is a pass-through string for arguments that belong only to the opt+heu runner (`tests/test_convoy_opt_and_heu.py` / `src/convoy_opt_and_heu/opt_and_hue.py`).
- `convoy_main.py` does not interpret these inner flags itself; it forwards them to the corresponding runner parser.
- Use quoted strings so multiple forwarded flags stay grouped.
- `--rl-v2-checkpoint-dir` overrides `--checkpoint-dir` only for `convoy_rl_partial_ch2`.
- `--hybrid-checkpoint-dir` overrides `--checkpoint-dir` only for `m_VRPTW`.

Examples:
- `--opt-rl-extra "--print-solution --save-model"`
- `--opt-rl-extra "--seed 77 --test-csv data/test_instance.csv"`
- `--opt-rl-extra "--rl-algo pomo --baseline shared --pomo-num-starts 16 --pomo-num-augment 8"`
- `--opt-heu-extra "--skip-optimal"`
- `--opt-heu-extra "--random-seed 123"`

`--clear-rl-checkpoints` behavior:
- Off by default.
- When enabled, it deletes checkpoint folders once at the start of `convoy_main.py`.
- If both hybrid and RL-v2 stages run, both stage checkpoint directories are cleared.
- Directory resolution order per stage:
  - stage-specific override (`--hybrid-checkpoint-dir` or `--rl-v2-checkpoint-dir`),
  - then `--checkpoint-dir` from `--opt-rl-extra`,
  - otherwise default `checkpoints_vrptw` under `CONVOY2`.




## Runner CLI Docs

For full method-specific argument details, use the dedicated READMEs:

- RL partial charging: `src/convoy_rl_partial_ch2/README.md`
- m_VRPTW RL: `m_VRPTW/README.md`
- Test entrypoint summary (all test launchers): `tests/README.md`




## Example Commands

### 1) Combined Run (Opt + CSA + m_VRPTW + RL)
```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 1 \
  --results-file combined_run.csv \
  --opt-heu-extra "--random-seed 111 --skip-optimal" \
  --opt-rl-extra "--save-model --seed 111 --epochs 100 --fixed-eval-every 5" \
  --rl-v2-checkpoint-dir checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100 \
  --hybrid-checkpoint-dir checkpoints_vrptw/m_VRPTW_c50_cp10_ev10_e100
```

### 2) Run both RL models: m_VRPTW and convoy_rl_partial_ch2 (named convoy_hybrid in paper)
```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-rl \
  --opt-rl-extra "--test-csv data/test_instance_50c_10cp.csv --test-distance-matrix-csv data/distance_matrix_jd200_1.csv --test-time-matrix-csv data/time_matrix_jd200_1.csv --print-solution --save-model --seed 111"
```

### 3) convoy_rl_partial_ch2 Only (AM)
```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-rl-v2 \
  --rl-v2-checkpoint-dir checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100 \
  --opt-rl-extra "--test-csv data/test_instance_50c_10cp.csv --test-distance-matrix-csv data/distance_matrix_jd200_1.csv --test-time-matrix-csv data/time_matrix_jd200_1.csv --print-solution --save-model --seed 111 --epochs 100 --fixed-eval-every 5"
```

### 4) convoy_rl_partial_ch2 Only (POMO)
```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-rl-v2 \
  --rl-v2-checkpoint-dir checkpoints_vrptw/rl_partial_ch2_c50_cp10_ev10_e100_pomo \
  --opt-rl-extra "--test-csv data/test_instance_50c_10cp.csv --test-distance-matrix-csv data/distance_matrix_jd200_1.csv --test-time-matrix-csv data/time_matrix_jd200_1.csv --print-solution --save-model --seed 111 --epochs 100 --fixed-eval-every 5 --rl-algo pomo --baseline shared --pomo-num-starts 10 --pomo-num-augment 8"
```

### 5) m_VRPTW Only
```bash
python tests/test_m_VRPTW.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance_50c_10cp.csv \
  --test-distance-matrix-csv data/distance_matrix_jd200_1.csv \
  --test-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --save-model \
  --seed 111 \
  --epochs 100 \
  --fixed-eval-every 5 \
  --checkpoint-dir checkpoints_vrptw/m_VRPTW_c50_cp10_ev10_e100
```

### 6) Heuristics Only
```bash
python convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 50 \
  --charging-stations-num 10 \
  --ev-num 10 \
  --only-opt-heu \
  --iterations 1 \
  --results-file heuristics_only.csv \
  --opt-heu-extra "--random-seed 111 --skip-optimal"
```

## CLI Architecture
- `convoy_parser.py` is the parser entry point for this repo.
- `convoy_main.py` parses at top level, then passes parsed parameters to:
- `src/convoy_opt_and_heu/opt_and_hue.py` via `run_opt_heu_with_params(...)`
- `m_VRPTW/convoy_hybrid_main.py` via `run_rl(args)`
- `src/convoy_rl_partial_ch2/convoy_rl_main.py` via `run_rl(args)`




## More
For additional variants (sampling/beam/POMO sweeps, experiment script automation), see:
- `tests/README.md`
- `scripts/README.md`


For tool-based commands (pipeline, conversion, metrics, and data utilities), use:
- `tools/README.md`

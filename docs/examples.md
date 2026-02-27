# Examples

## Combined Run With Baseline

Command:

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 5 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --iterations 10 \
  --results-file jd200_itr10_cust5_1.csv \
  --run-baseline \
  --opt-rl-extra "--print-solution --save-model --seed 42" \
  --opt-heu-extra "--random-seed 123"
```

### Argument Explanation

| Argument | Value | Meaning |
|---|---|---|
| `--combined-details-csv` | `data/combined_data_jd200_1.csv` | Combined node details input (depot, customers, charging stations, rewards, time windows, costs). |
| `--combined-dist-matrix-csv` | `data/distance_matrix_jd200_1.csv` | Full distance matrix used by Opt+Heu, RL, and baseline conversion. |
| `--combined-time-matrix-csv` | `data/time_matrix_jd200_1.csv` | Full travel-time matrix used by Opt+Heu, RL, and baseline conversion. |
| `--customer-num` | `5` | Number of customers to sample/use in each iteration. |
| `--charging-stations-num` | `3` | Number of charging stations to sample/use in each iteration. |
| `--ev-num` | `2` | Number of EVs used by all selected methods. |
| `--iterations` | `10` | Run the full pipeline 10 times and append each iteration to the results CSV (`itr` column). |
| `--results-file` | `jd200_itr10_cust5_1.csv` | Name of output results CSV under `CONVOY/results/` (relative path behavior). |
| `--run-baseline` | enabled | Also run EVRP-TW-SPD-HMA baseline and append its metrics to results. |
| `--opt-rl-extra` | `"--print-solution --save-model --seed 42"` | Extra args forwarded to RL runner: print solution, save model/checkpoint, and use RL seed `42`. |
| `--opt-heu-extra` | `"--random-seed 123"` | Extra args forwarded to Opt+Heu runner: set preprocessing random seed to `123`. |

### What This Run Produces

- One consolidated CSV in `results/jd200_itr10_cust5_1.csv`.
- Up to four method rows per iteration depending on enabled methods:
  - `Optimal` (if not skipped and problem size allows MILP),
  - `CSA` heuristic,
  - `RL`,
  - `Baseline` (because `--run-baseline` is enabled).


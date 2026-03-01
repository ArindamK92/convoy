# Convoy Hybrid (RL4CO CVRPTW + CP Postprocessing)

`convoy_hybrid` trains RL4CO CVRPTW on depot + customer nodes, then (for test CSV reporting) augments decoded routes with charging-point (CP) insertion and prints both:

- Full-charging reward components
- Partial-charging reward components

## Run Command

Run from `/home/akkcm/CONVOY2`:

```bash
python tests/test_convoy_hybrid.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --test-csv data/test_instance.csv \
  --test-distance-matrix-csv data/distance_matrix_jd200_1.csv \
  --test-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 20 \
  --ev-num 8 \
  --epochs 10 \
  --fixed-eval-every 5 \
  --seed 111 \
  --print-solution
```

## Arguments Used In The Command

| Argument | Meaning |
| --- | --- |
| `--combined-details-csv` | Combined node table CSV (`ID,type,lng,lat,...`) used as the customer pool source. |
| `--combined-dist-matrix-csv` | Distance matrix CSV used for training/evaluation distance lookup. |
| `--combined-time-matrix-csv` | Time matrix CSV used for time-window feasibility and arrival/departure timing. |
| `--test-csv` | Test instance CSV for final decoded solution print and CP augmentation report. |
| `--test-distance-matrix-csv` | Distance matrix override for test CSV rollout/trace. |
| `--test-time-matrix-csv` | Time matrix override for test CSV rollout/trace. |
| `--customer-num` | Number of customers sampled per generated training/validation/test instance. |
| `--ev-num` | Max number of EV routes that can start from depot (enforced). |
| `--epochs` | Number of RL training epochs. |
| `--fixed-eval-every` | Evaluate quality on fixed set every N epochs. |
| `--seed` | Random seed for reproducibility. |
| `--print-solution` | Print decoded action sequence, routes, traces, and reward components. |

## Full CLI Argument Reference

### Training / Optimization

| Argument | Meaning |
| --- | --- |
| `--epochs` | Training epochs. |
| `--batch-size` | Training batch size. |
| `--eval-batch-size` | Validation/test batch size. |
| `--train-data-size` | Number of generated train samples per epoch. |
| `--val-data-size` | Validation sample count. |
| `--test-data-size` | Test sample count. |
| `--lr` | Learning rate. |
| `--seed` | Global random seed. |
| `--accelerator {auto,cpu,gpu}` | Lightning accelerator selection. |
| `--checkpoint-dir` | Directory to save/load checkpoints. |
| `--save-model` | Save and reuse best checkpoint. |

### RL Model / Decoder

| Argument | Meaning |
| --- | --- |
| `--rl-algo {am,pomo}` | RL algorithm (`am` = REINFORCE AttentionModel, `pomo` = POMO). |
| `--baseline {exponential,rollout,shared,mean,no,critic}` | REINFORCE baseline type. For POMO, use `shared`. |
| `--pomo-num-starts` | POMO starts per instance (`0` lets RL4CO decide). |
| `--pomo-num-augment` | POMO augmentation count (`1` disables augmentation). |
| `--decode-type {greedy,sampling,beam_search,multistart_greedy,multistart_sampling}` | Decode strategy for eval/test/printing. |
| `--decode-num-samples` | Number of sampled rollouts for sampling decode. |
| `--decode-num-starts` | Number of starts for multistart decode. |
| `--decode-select-best` | Pick best candidate among sampled/multistart rollouts. |
| `--decode-temperature` | Sampling temperature. |
| `--decode-top-p` | Top-p (nucleus) sampling threshold. |
| `--decode-top-k` | Top-k truncation for sampling. |
| `--decode-beam-width` | Beam width for `beam_search`. |
| `--print-solution` | Print decoded route/trace details. |

### Data Inputs

| Argument | Meaning |
| --- | --- |
| `--combined-details-csv` | Required combined details CSV with depot/customers/CP rows. |
| `--combined-dist-matrix-csv` | Required combined distance matrix CSV. |
| `--combined-time-matrix-csv` | Optional combined time matrix CSV. If omitted, distance matrix is reused as time. |
| `--customer-num` | Number of customers sampled per generated instance. |
| `--pool-vehicle-capacity` | Capacity for demand normalization in pool generation. |
| `--max-time` | Time-window horizon for generated samples. |
| `--test-csv` | Optional custom test instance CSV. |
| `--test-distance-matrix-csv` | Optional test distance matrix override. |
| `--test-time-matrix-csv` | Optional test time matrix override. |
| `--fixed-instance-csv` | Optional fixed train/val instance CSV. |
| `--fixed-instance-seed` | Seed for creating fixed train/val instance. |
| `--csv-vehicle-capacity` | Capacity used when normalizing demand from `--test-csv` / fixed CSV. |

### EV / Trace Settings

| Argument | Meaning |
| --- | --- |
| `--ev-num` | Max number of EVs (max routes started from depot). |
| `--ev-battery-capacity-kwh` | Battery capacity used in printed SoC/charge trace. |
| `--reserve-battery` (`--ev-reserve-soc-kwh`) | Reserve SoC used in printed SoC/charge trace. |
| `--ev-energy-rate-kwh-per-distance` | Energy-per-distance used in printed SoC/charge trace and CP postprocessing. |
| `--ev-charge-rate-kwh-per-hour` | Default charge rate used in printed SoC/charge trace when needed. |

### Fixed Evaluation

| Argument | Meaning |
| --- | --- |
| `--fixed-eval-size` | Fixed-set size for quality trend tracking. |
| `--fixed-eval-every` | Evaluate fixed set every N epochs. |

### Compatibility-Only Arguments In Hybrid

| Argument | Behavior In `convoy_hybrid` |
| --- | --- |
| `--charging-stations-num` | Accepted for CLI compatibility, ignored in hybrid training. |
| `--cost-weight` | Accepted for compatibility; hybrid RL training reward remains RL4CO CVRPTW reward. |

## What Gets Printed For Test CSV

- RL4CO decode reward (`CSV instance RL4CO reward`)
- Decoded routes and node IDs
- Augmented route trace with CP insertion
- Skipped late customers (if any)
- `Reward components (CP-augmented full charging)`
- `Partial charging summary`
- `Reward components (CP-augmented partial charging)`

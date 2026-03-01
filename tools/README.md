# Tools

Utility scripts under `CONVOY2/tools` for data conversion, test-instance generation, and baseline evaluation.

Run from repo root (`/home/akkcm/CONVOY2`) unless noted.

## File Summary

| File | Main Function(s) | Purpose |
| --- | --- | --- |
| `tools/generate_test_instance.py` | `generate_test_instance_from_combined` | Build `test_instance.csv` by sampling depot/customers/CPs from combined details CSV. |
| `tools/convert_data.py` | `_parse_instance_file`, `_build_combined_rows`, `_build_matrices`, `main` | Convert EVRP-TW-SPD-HMA `.txt` instance to `combined_data_*.csv`, `distance_matrix_*.csv`, `time_matrix_*.csv`. |
| `tools/convert_to_evrp_instance.py` | `convert_test_csv_to_evrp_instance` | Convert CONVOY test CSV + matrix CSVs to EVRP baseline input text format. |
| `tools/compute_baseline_metrics.py` | `compute_metrics` | Compute CONVOY reward/cost/objective from baseline solver output. |
| `tools/run_baseline_pipeline.py` | `run_baseline_pipeline` | End-to-end: convert -> run baseline solver -> parse metrics. |
| `tools/clear_cuda_memory.py` | `clear_cuda_memory` | Release Python references and trigger CUDA cache cleanup (`gc.collect`, `torch.cuda.empty_cache`, `torch.cuda.ipc_collect`). |

## Commands

### 1) Generate `test_instance.csv` from combined details

```bash
python tools/generate_test_instance.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --customer-num 20 \
  --charging-stations-num 10 \
  --seed 111 \
  --output-path data/test_instance.csv
```

Notes:
- Overwrites output file if it exists.
- Output order is always: depot first, then sampled customers, then sampled CPs.

### 2) Convert baseline TXT instance to CONVOY CSV datasets

```bash
python tools/convert_data.py \
  --input-txt /home/akkcm/EVRP-TW-SPD-HMA/data/jd_instances/jd200_4.txt \
  --output-dir data \
  --seed 123
```

### 3) Convert `test_instance.csv` to EVRP baseline input file

```bash
python tools/convert_to_evrp_instance.py \
  --test-csv data/test_instance.csv \
  --dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --time-matrix-csv data/time_matrix_jd200_1.csv \
  --vehicles 5 \
  --output-path baseline/data/test_instance_evrp.txt
```

### 4) Compute metrics from baseline output

```bash
python tools/compute_baseline_metrics.py \
  --baseline-output-file baseline/data/latest_baseline_output.txt \
  --test-instance-csv data/test_instance.csv \
  --id-map-csv baseline/data/test_instance_evrp.id_map.csv \
  --ev-energy-rate-kwh-per-distance 0.00025 \
  --print-charging-events \
  --print-route-trace
```

### 5) Run full baseline pipeline in one command

```bash
python tools/run_baseline_pipeline.py \
  --test-csv data/test_instance.csv \
  --dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --time-matrix-csv data/time_matrix_jd200_1.csv \
  --vehicles 5 \
  --baseline-time 10 \
  --baseline-runs 5 \
  --baseline-quiet \
  --solver-log-path baseline/data/baseline_solver.log \
  --print-charging-events
```

### 6) Clear CUDA memory cache from Python

Run standalone:

```bash
python tools/clear_cuda_memory.py
```

Use in code:

```python
from tools.clear_cuda_memory import clear_cuda_memory

clear_cuda_memory(model, trainer, env, batch, out)
```

## Inspect CLI Help

```bash
python tools/generate_test_instance.py --help
python tools/convert_data.py --help
python tools/convert_to_evrp_instance.py --help
python tools/compute_baseline_metrics.py --help
python tools/run_baseline_pipeline.py --help
python tools/clear_cuda_memory.py
```

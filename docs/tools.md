# Tools

This page summarizes utility scripts in `CONVOY/tools`.

For full commands and flags, use:
- `tools/README.md`

## Tool Scripts

| Script | Main Function | Purpose | Typical Use |
| --- | --- | --- | --- |
| `tools/generate_test_instance.py` | `generate_test_instance_from_combined` | Sample depot/customers/CPs from combined details CSV and write `test_instance.csv`. | Create fixed/smaller test instances for repeatable runs. |
| `tools/convert_data.py` | `_parse_instance_file`, `_build_combined_rows`, `_build_matrices`, `main` | Convert EVRP-TW-SPD-HMA `.txt` dataset format into CONVOY CSV files. | Import external EVRP instances into CONVOY format. |
| `tools/convert_to_evrp_instance.py` | `convert_test_csv_to_evrp_instance` | Convert CONVOY test CSV + matrices to EVRP baseline input text format. | Prepare baseline solver input. |
| `tools/compute_baseline_metrics.py` | `compute_metrics` | Compute CONVOY-style reward/cost/objective from baseline solver output. | Post-process baseline results for fair comparison. |
| `tools/run_baseline_pipeline.py` | `run_baseline_pipeline` | End-to-end baseline flow: convert -> run solver -> compute metrics. | One-command baseline benchmarking from CONVOY data. |
| `tools/clear_cuda_memory.py` | `clear_cuda_memory` | Force cleanup of Python refs and CUDA cached memory. | Recover from CUDA memory pressure between runs. |

## How These Tools Fit In

1. Data prep:
   - `convert_data.py`
   - `generate_test_instance.py`
2. Baseline flow:
   - `convert_to_evrp_instance.py`
   - `run_baseline_pipeline.py`
   - `compute_baseline_metrics.py`
3. Runtime maintenance:
   - `clear_cuda_memory.py`

## Quick Help

Run tool help directly:

```bash
python tools/generate_test_instance.py --help
python tools/convert_data.py --help
python tools/convert_to_evrp_instance.py --help
python tools/compute_baseline_metrics.py --help
python tools/run_baseline_pipeline.py --help
python tools/clear_cuda_memory.py
```

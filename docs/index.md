# CONVOY2 Documentation

This site documents the combined RL + optimal/heuristic pipeline in `CONVOY2`.

## What Is Covered
- How to run the project quickly
- CLI arguments and run patterns
- CSV and matrix format requirements
- High-level architecture
- Common runtime issues and fixes

## Main Entry Points
- Combined runner: `convoy_main.py`
- RL-only test launcher: `tests/test_convoy_CPs1_ch2.py`
- Hybrid test launcher: `tests/test_m_VRPTW.py`
- Opt+Heu-only test launcher: `tests/test_convoy_opt_and_heu.py`

## Start Here
- `Quickstart` for setup and docs commands
- `More Running Examples` for command cookbook style runs
- `Tools` for utility scripts (data conversion, baseline pipeline, CUDA cleanup)
- `CLI` for executable command templates
- `CLI -> Argument Groups By Module` for where each argument family is documented

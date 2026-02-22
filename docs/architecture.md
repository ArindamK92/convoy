# Architecture

## Top-Level Flow
`convoy_main.py` orchestrates:
1. Parse top-level args from `convoy_parser.py`
2. Run opt+heu first (unless `--only-rl`)
3. Run RL (unless `--only-opt-heu`)
4. Write consolidated results CSV

## Parsing Strategy
- All top-level and launcher-facing parsing is centralized in `convoy_parser.py`.
- Runners receive parsed namespaces/parameters.

## Components
- `src/convoy_opt_and_heu/`
  - preprocessing (`helper.py`)
  - exact MILP solver (`MILP_fn.py`)
  - heuristic solver (`heuristic_partial_charging_fn.py`)
- `src/convoy_rl/`
  - RL environment, training, and test evaluation pipeline

## Result Aggregation
`convoy_main.py` writes rows with:
- method (`Optimal`, `CSA`, `RL`)
- objective and cost fields
- successful delivery fields
- iteration index (`itr`) when repeated runs are enabled

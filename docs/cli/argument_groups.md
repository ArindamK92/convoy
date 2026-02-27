# Argument Groups By Module

This page organizes CLI arguments by the module that owns them, matching the current split across README files and docs pages.

## Argument Ownership Map

| Group | Main Entry Point | Typical Arguments | Full Details |
| --- | --- | --- | --- |
| Combined runner wrapper args | `convoy_main.py` | `--only-rl`, `--only-opt-heu`, `--skip-convoy-rl`, `--rl-checkpoint-dir`, `--hybrid-checkpoint-dir`, `--opt-rl-extra`, `--opt-heu-extra`, `--iterations`, `--results-file`, `--clear-rl-checkpoints` | `README.md` (repo root), `docs/cli/convoy_main.md` |
| Shared dataset/input args (used by multiple runners) | `convoy_main.py`, test runners | `--combined-details-csv`, `--combined-dist-matrix-csv`, `--combined-time-matrix-csv`, `--customer-num`, `--charging-stations-num`, `--ev-num`, `--ev-energy-rate-kwh-per-distance`, `--reserve-battery` | `docs/cli/convoy_main.md`, `tests/README.md` |
| Convoy RL partial charging args | `tests/test_convoy_CPs1.py` -> `src/convoy_rl_partial_ch/convoy_rl_main.py` | RL algo/decoder/train/checkpoint flags, EV/cost/reward flags | `src/convoy_rl_partial_ch/README.md`, `tests/README.md`, `docs/cli/test_convoy_CPs1.md` |
| Convoy hybrid args | `tests/test_convoy_hybrid.py` -> `convoy_hybrid/convoy_hybrid_main.py` | RL algo/decoder/train/checkpoint flags, `--verbose`, fixed-instance flags | `convoy_hybrid/README.md`, `tests/README.md`, `docs/cli/test_convoy_hybrid.md` |
| Opt+Heu args (MILP + heuristics) | `tests/test_convoy_opt_and_heu.py` -> `src/convoy_opt_and_heu/opt_and_hue.py` | `--skip-optimal`, `--only-milp`, `--random-seed`, `--no-edf-ndf`, objective/energy flags | `tests/README.md`, `docs/cli/test_convoy_opt_and_heu.md` |
| Tools and conversion/baseline helper args | scripts under `tools/` | conversion, instance generation, baseline metrics/pipeline args | `tools/README.md` |
| Experiment script controls | scripts under `scripts/` | sweep dimensions, iterations, checkpoint reuse choices | `scripts/README.md` |

## How To Choose The Right Argument Set

1. Running everything via one command: start with `convoy_main.py` arguments in `README.md` and `docs/cli/convoy_main.md`.
2. Running a single method directly: use `tests/README.md` to pick the right test launcher.
3. Method-specific tuning (RL/hybrid): use that method folder README:
   - `convoy_hybrid/README.md`
   - `src/convoy_rl_partial_ch/README.md`
4. Data conversion or baseline utilities: use `tools/README.md`.
5. Reproducible sweeps: use `scripts/README.md`.

## Pass-Through Reminder (`convoy_main.py`)

- `--opt-rl-extra "..."` forwards quoted flags to both RL/hybrid parsers.
- `--opt-heu-extra "..."` forwards quoted flags to the Opt+Heu parser.
- Stage-specific checkpoint overrides:
  - `--hybrid-checkpoint-dir` for `convoy_hybrid`
  - `--rl-checkpoint-dir` for `convoy_rl_partial_ch`

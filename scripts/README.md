# Scripts Overview

This folder contains reproducible sweep/experiment launchers for `convoy_main.py` and standalone runners.

## Quick Start
From repo root:

```bash
cd ~/CONVOY
./scripts/<script_name>.sh
```

Most scripts allow overriding settings via environment variables (for example `SEED`, `EPOCHS`, `ITERATIONS`, `CP_NUM`).

## Script Functions

### `run_cust_cp_ev_sweep.sh`
Function:
- Customer-count sweep (`10..100`, step `10`)
- Fixed `CP_NUM=20`, `EV = customer / 5`
- Runs `convoy_main` with:
  - heuristic (`--skip-optimal`),
  - `convoy_hybrid`,
  - `convoy_rl_partial_ch`,
  - baseline.
- Uses POMO + beam-search style RL args.
- Clears checkpoint directory at start of each run (`--clear-rl-checkpoints`).

Use when:
- You want full multi-method comparison and retraining from scratch per run.

### `run_cust_cp_ev_sweep_new.sh`
Function:
- Same customer/CP/EV sweep structure as above.
- Runs heuristic + hybrid + RL + baseline.
- Uses separate stage checkpoint dirs:
  - `--hybrid-checkpoint-dir`
  - `--rl-checkpoint-dir`
- Does **not** clear checkpoints.

Use when:
- You want checkpoint reuse for hybrid/RL across runs.

### `run_cust_cp_ev_sweep_edf_ndf_heu_baseline.sh`
Function:
- Customer-count sweep (`10..100`, step `10`)
- Fixed `CP_NUM=20`, `EV = customer / 5`
- Runs only:
  - Heuristic,
  - NDF,
  - EDF,
  - baseline.
- Skips hybrid and RL (`--only-opt-heu` + baseline enabled).

Use when:
- You want non-RL method comparison only.

### `run_ratio_sweep_cust50_cp20.sh`
Function:
- Fixed `customer=50`, `CP=20`
- Sweeps delivery-to-EV ratio from `5..10` (`EV = floor(50/ratio)`).
- Runs heuristic + NDF + EDF + hybrid + RL + baseline.
- Uses POMO + beam-search style RL args.
- Clears RL checkpoints each run (`--clear-rl-checkpoints`).

Use when:
- You want full ratio-sweep comparison with retraining.

### `run_ratio_sweep_cust50_cp20_new.sh`
Function:
- Same ratio sweep (`customer=50`, `CP=20`, ratio `5..10`).
- Runs:
  - heuristic + NDF + EDF,
  - hybrid only.
- Skips convoy RL stage (`--skip-convoy-rl`) and baseline.
- Uses saved hybrid checkpoint dir via `--opt-rl-extra "--checkpoint-dir ..."` and POMO+beam params.

Use when:
- You want faster ratio sweeps focused on heuristic+hybrid only.

### `run_small_methods_sweep.sh`
Function:
- Small-size cases:
  - customers `5,10`: Optimal + Heuristic + Hybrid
  - customer `15`: Heuristic + Hybrid (skip Optimal)
- Uses fixed `CP_NUM=3`, `EV_NUM=2`.
- Skips convoy RL (`--skip-convoy-rl`).
- Disables NDF/EDF (`--no-edf-ndf`).

Use when:
- You want quick sanity checks on small instances.

### `run_large_cust_cp50_ev50_sweep.sh`
Function:
- Large-data workflow on `jd1000_2` files.
- Step 1: pretrain/save `convoy_hybrid` checkpoint (`hybrid_c500_cp50_ev50_e200`).
- Step 2: pretrain/save `convoy_rl_partial_ch` checkpoint (`rl_c500_cp50_ev50_e200`).
- Step 3: sweep customers `200..1000` (step `200`), fixed `CP=50`, `EV=50`:
  - iterations `5` for `200/400/600/800`
  - iteration `1` for `1000`
- Result files start with `large_`.
- Does **not** delete checkpoints; reuses saved hybrid/RL models via stage-specific checkpoint dirs.

Use when:
- You want large-scale experiments with pretrained model reuse.

### `Experiments.md`
Function:
- Markdown runbook with curated command examples for common experiment flows.


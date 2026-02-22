# Troubleshooting

## `--opt-heu-extra: expected one argument`
Pass extra args as one quoted value:

```bash
--opt-heu-extra "--skip-optimal"
```

or:

```bash
--opt-heu-extra="--skip-optimal"
```

## RL Reward Looks Unchanged After Editing a CSV
If you run with `--save-model`, an existing checkpoint can be reused.  
Ensure `--test-csv` points to the intended file and verify matrix overrides.

## `ModuleNotFoundError: torch` or `pandas`
Install runtime dependencies in your environment before running RL/opt-heu pipelines.

## Gurobi License Errors
- Configure MILP license via `config/gurobi_wls.json` (use `config/gurobi_wls.example.json` as template), or
- Set `CONVOY_GUROBI_CONFIG` to a custom JSON config path, or
- Provide WLS fields using env vars:
  - `GUROBI_WLSACCESSID`, `GUROBI_WLSSECRET`, `GUROBI_LICENSEID`.

## No `test_instance.csv` Found In Combined Run
`test_instance.csv` is produced by opt+heu preprocessing.  
If running `--only-rl`, pass `--test-csv` explicitly via `--opt-rl-extra`.

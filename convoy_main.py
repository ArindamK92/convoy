"""Top-level orchestrator for CONVOY2 experiments.

This runner combines:
- `convoy_opt_and_heu` (Optimal + CSA heuristic),
- `convoy_rl_partial_ch` (RL training/testing + partial-charge evaluation),
- optional EVRP baseline pipeline,

and writes one consolidated results CSV across selected methods and iterations.
"""

import csv
import math
import os
import shlex
import shutil

from convoy_parser import (
    parse_args,
    parse_extra_args,
    parse_hybrid_args,
    parse_opt_heu_args,
    parse_rl_args,
)


def _print_cmd(runner_name, args):
    """Print a shell-like command preview for reproducibility/debugging."""
    print("$", runner_name, " ".join(shlex.quote(a) for a in args))


def _contains_flag(args, flag_name):
    """Return True if `flag_name` appears in tokenized CLI args."""
    for token in args:
        if token == flag_name or token.startswith(flag_name + "="):
            return True
    return False


def _get_flag_value(args, flag_name):
    """Extract the last value for a flag from tokenized args.

    Supports both forms:
    - `--flag value`
    - `--flag=value`
    """
    value = None
    for idx, token in enumerate(args):
        if token == flag_name and idx + 1 < len(args):
            value = args[idx + 1]
        elif token.startswith(flag_name + "="):
            value = token.split("=", 1)[1]
    return value


def _safe_avg(total_cost, total_deliveries_completed):
    """Return average cost per successful delivery, or empty string if undefined."""
    if total_deliveries_completed <= 0:
        return ""
    return total_cost / total_deliveries_completed


def _remove_boolean_flag(args, flag_name):
    """Return CLI tokens with a boolean flag removed (`--flag`/`--flag=...`)."""
    out = []
    for token in args:
        if token == flag_name or token.startswith(flag_name + "="):
            continue
        out.append(token)
    return out


def _resolve_iteration_seed(base_seed, itr, total_iterations):
    """Resolve iteration-aware seed with deterministic offset behavior."""
    if total_iterations > 1:
        if base_seed is None:
            return itr
        return base_seed + (itr - 1)
    return base_seed


def _parse_opt_heu_base_seed(opt_extra_raw):
    """Read optional `--random-seed` forwarded via `--opt-heu-extra`."""
    seed_txt = _get_flag_value(opt_extra_raw, "--random-seed")
    if seed_txt is None:
        return None
    try:
        return int(seed_txt)
    except ValueError as exc:
        raise ValueError(
            "Invalid --random-seed value in --opt-heu-extra: {}".format(seed_txt)
        ) from exc


def _generate_test_instance_csv(args, opt_extra_raw, itr):
    """Generate `test_instance.csv` from shared inputs for RL/Baseline-only runs."""
    from src.convoy_opt_and_heu.helper import preProcess

    use_full_instance = bool(getattr(args, "test_for_opt_heu", None))
    details_csv_for_preprocess = (
        args.test_for_opt_heu if use_full_instance else args.combined_details_csv
    )

    if use_full_instance:
        import csv as _csv

        nD = 0
        nC = 0
        with open(details_csv_for_preprocess, mode="r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                t = str(row.get("type", "")).strip().lower()
                if t == "c":
                    nD += 1
                elif t == "f":
                    nC += 1
    else:
        nD = int(args.customer_num)
        nC = int(args.charging_stations_num)

    nE = int(args.ev_num)
    nS = math.ceil(nD / nE)
    base_seed = _parse_opt_heu_base_seed(opt_extra_raw)
    random_seed = _resolve_iteration_seed(base_seed, itr, args.iterations)

    # We invoke preprocessing for its deterministic sampling + CSV export side-effect.
    preProcess(
        nD,
        nC,
        nS,
        nE,
        details_csv_for_preprocess,
        args.combined_dist_matrix_csv,
        args.combined_time_matrix_csv,
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        random_seed=random_seed,
        use_full_instance=use_full_instance,
    )


def _resolve_rl_checkpoint_dir(repo_root, rl_extra_args):
    """Resolve RL checkpoint directory considering `--checkpoint-dir` override."""
    ckpt_dir = _get_flag_value(rl_extra_args, "--checkpoint-dir")
    if not ckpt_dir:
        ckpt_dir = "checkpoints_vrptw"
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(repo_root, ckpt_dir)
    return os.path.abspath(ckpt_dir)


def _results_output_path(repo_root, combined_details_csv, results_file=None):
    """Resolve final results CSV path and ensure parent directories exist."""
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    if results_file:
        if os.path.isabs(results_file):
            output_file = results_file
        else:
            output_file = os.path.join(results_dir, results_file)
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_file
    details_stem = os.path.splitext(os.path.basename(combined_details_csv))[0]
    return os.path.join(results_dir, "results3_{}.csv".format(details_stem))


def _write_results_csv(output_file, rows):
    """Write consolidated experiment rows to CSV with fixed column order."""
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "itr",
                "Methods",
                "Total delivery",
                "Total CP",
                "Total EV",
                "delivery2ev_ratio",
                "Elapsed time (ms)",
                "Total reward",
                "Total cost",
                "Objective val",
                "Total successful delivery",
                "Avg. cost per successful delivery",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("itr", ""),
                    row.get("method", ""),
                    row.get("total_delivery", ""),
                    row.get("total_cp", ""),
                    row.get("total_ev", ""),
                    row.get("delivery2ev_ratio", ""),
                    row.get("elapsed_time_ms", ""),
                    row.get("total_reward", ""),
                    row.get("total_cost", ""),
                    row.get("objective_val", ""),
                    row.get("total_successful_delivery", ""),
                    row.get("avg_cost_per_successful_delivery", ""),
                ]
            )


def main():
    """Run selected methods for N iterations and persist merged results.

    Execution order per iteration:
    1) optional Optimal/Heuristic (also generates `test_instance.csv`),
    2) optional Hybrid RL4CO stage,
    3) optional convoy RL stage,
    4) optional Baseline pipeline (same instance + EV count).
    """
    args = parse_args()
    if args.only_rl and args.only_opt_heu:
        raise ValueError("Use at most one of --only-rl or --only-opt-heu.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0.")

    run_hybrid = not args.only_opt_heu
    run_rl = run_hybrid and (not args.skip_convoy_rl)
    run_opt_heu = not args.only_rl
    run_baseline = bool(getattr(args, "run_baseline", False))

    repo_root = os.path.dirname(os.path.abspath(__file__))
    combined_details_abs = os.path.abspath(args.combined_details_csv)
    output_file = _results_output_path(
        repo_root, combined_details_abs, results_file=args.results_file
    )
    generated_test_csv = os.path.join(os.path.dirname(combined_details_abs), "test_instance.csv")
    all_results_rows = []
    opt_extra_raw = parse_extra_args(args.opt_heu_extra)
    rl_extra_args = parse_extra_args(args.opt_rl_extra)
    # Hybrid stage uses same RL-style args but omits print-only overhead for timing.
    hybrid_extra_args = _remove_boolean_flag(rl_extra_args, "--print-solution")
    baseline_extra_args = parse_extra_args(getattr(args, "baseline_extra", []))
    if run_hybrid and _contains_flag(rl_extra_args, "--print-solution"):
        print(
            "Hybrid timing note: --print-solution is ignored for convoy_hybrid "
            "inside convoy_main so elapsed time excludes printing overhead."
        )

    # Optional cleanup so RL always retrains from scratch.
    if args.clear_rl_checkpoints:
        rl_checkpoint_dir = _resolve_rl_checkpoint_dir(repo_root, rl_extra_args)
        if os.path.isdir(rl_checkpoint_dir):
            shutil.rmtree(rl_checkpoint_dir)
            print("Deleted RL checkpoint directory:", rl_checkpoint_dir)
        else:
            print("RL checkpoint directory not found; skip delete:", rl_checkpoint_dir)

    for itr in range(1, args.iterations + 1):
        print("=== Iteration {}/{} ===".format(itr, args.iterations))
        iter_rows = []

        # Run optimal+heuristic first so helper preprocessing can write test_instance.csv.
        if run_opt_heu:
            from src.convoy_opt_and_heu.opt_and_hue import run_opt_heu_with_params

            opt_namespace, opt_cli_args = parse_opt_heu_args(args, opt_extra_raw)
            _print_cmd("convoy_opt_and_heu", opt_cli_args)
            opt_random_seed = _resolve_iteration_seed(
                getattr(opt_namespace, "random_seed", None), itr, args.iterations
            )
            opt_rows = run_opt_heu_with_params(
                combined_details_csv=opt_namespace.combined_details_csv,
                combined_dist_matrix_csv=opt_namespace.combined_dist_matrix_csv,
                combined_time_matrix_csv=opt_namespace.combined_time_matrix_csv,
                customer_num=opt_namespace.customer_num,
                charging_stations_num=opt_namespace.charging_stations_num,
                ev_num=opt_namespace.ev_num,
                ev_energy_rate_kwh_per_distance=opt_namespace.ev_energy_rate_kwh_per_distance,
                reserve_battery=opt_namespace.reserve_battery,
                alpha1_override=opt_namespace.alpha1,
                alpha2_override=opt_namespace.alpha2,
                skip_optimal=opt_namespace.skip_optimal,
                random_seed=opt_random_seed,
                no_edf_ndf=getattr(opt_namespace, "no_edf_ndf", False),
                test_for_opt_heu=getattr(opt_namespace, "test_for_opt_heu", None),
            )
            for row in opt_rows:
                row["itr"] = itr
            iter_rows.extend(opt_rows)
        elif run_hybrid or run_baseline:
            # Ensure RL-only / Baseline-only runs still regenerate test_instance.csv
            # according to shared sampling arguments.
            _generate_test_instance_csv(args, opt_extra_raw, itr)
            if not os.path.isfile(generated_test_csv):
                raise FileNotFoundError(
                    "Expected generated test instance not found: {}".format(
                        generated_test_csv
                    )
                )

        if run_hybrid and not run_rl:
            from convoy_hybrid.convoy_hybrid_main import run_rl as run_hybrid

            auto_test_csv = None
            if os.path.isfile(generated_test_csv):
                auto_test_csv = generated_test_csv

            hybrid_namespace, hybrid_cli_args = parse_hybrid_args(
                args, hybrid_extra_args, auto_test_csv=auto_test_csv
            )
            if args.iterations > 1 and not _contains_flag(hybrid_extra_args, "--seed"):
                hybrid_namespace.seed = int(hybrid_namespace.seed) + (itr - 1)
                hybrid_cli_args = [*hybrid_cli_args, "--seed", str(hybrid_namespace.seed)]
            _print_cmd("convoy_hybrid", hybrid_cli_args)
            hybrid_summary = run_hybrid(hybrid_namespace)

            hybrid_elapsed_ms = None
            hybrid_full_total_reward = None
            hybrid_full_total_cost = None
            hybrid_full_objective_val = None
            hybrid_full_total_successful_delivery = None
            hybrid_partial_total_reward = None
            hybrid_partial_total_cost = None
            hybrid_partial_objective_val = None
            hybrid_partial_total_successful_delivery = None
            if isinstance(hybrid_summary, dict):
                hybrid_elapsed_ms = hybrid_summary.get("csv_inference_time_ms")
                hybrid_full_total_reward = hybrid_summary.get(
                    "csv_augmented_full_total_reward"
                )
                hybrid_full_total_cost = hybrid_summary.get("csv_augmented_full_total_cost")
                hybrid_full_objective_val = hybrid_summary.get(
                    "csv_augmented_full_objective_val"
                )
                hybrid_full_total_successful_delivery = hybrid_summary.get(
                    "csv_augmented_full_total_successful_delivery"
                )
                hybrid_partial_total_reward = hybrid_summary.get(
                    "csv_augmented_partial_total_reward"
                )
                hybrid_partial_total_cost = hybrid_summary.get(
                    "csv_augmented_partial_total_cost"
                )
                hybrid_partial_objective_val = hybrid_summary.get(
                    "csv_augmented_partial_objective_val"
                )
                hybrid_partial_total_successful_delivery = hybrid_summary.get(
                    "csv_augmented_partial_total_successful_delivery"
                )

                if hybrid_full_total_reward is None:
                    hybrid_full_total_reward = hybrid_summary.get("csv_augmented_total_reward")
                if hybrid_full_total_cost is None:
                    hybrid_full_total_cost = hybrid_summary.get("csv_augmented_total_cost")
                if hybrid_full_objective_val is None:
                    hybrid_full_objective_val = hybrid_summary.get("csv_augmented_objective_val")
                if hybrid_full_total_successful_delivery is None:
                    hybrid_full_total_successful_delivery = hybrid_summary.get(
                        "csv_augmented_total_successful_delivery"
                    )
                if hybrid_full_objective_val is None:
                    hybrid_full_objective_val = hybrid_summary.get("csv_instance_reward")

            hybrid_full_avg_cost = ""
            if (
                hybrid_full_total_cost is not None
                and hybrid_full_total_successful_delivery is not None
            ):
                hybrid_full_avg_cost = _safe_avg(
                    hybrid_full_total_cost, hybrid_full_total_successful_delivery
                )
            iter_rows.append(
                {
                    "itr": itr,
                    "method": "Hybrid_RL4CO",
                    "total_delivery": args.customer_num,
                    "total_cp": args.charging_stations_num,
                    "total_ev": args.ev_num,
                    "delivery2ev_ratio": args.customer_num / args.ev_num,
                    "elapsed_time_ms": (
                        hybrid_elapsed_ms if hybrid_elapsed_ms is not None else ""
                    ),
                    "total_reward": (
                        hybrid_full_total_reward
                        if hybrid_full_total_reward is not None
                        else ""
                    ),
                    "total_cost": (
                        hybrid_full_total_cost if hybrid_full_total_cost is not None else ""
                    ),
                    "objective_val": (
                        hybrid_full_objective_val
                        if hybrid_full_objective_val is not None
                        else ""
                    ),
                    "total_successful_delivery": (
                        hybrid_full_total_successful_delivery
                        if hybrid_full_total_successful_delivery is not None
                        else ""
                    ),
                    "avg_cost_per_successful_delivery": hybrid_full_avg_cost,
                }
            )

            hybrid_partial_avg_cost = ""
            if (
                hybrid_partial_total_cost is not None
                and hybrid_partial_total_successful_delivery is not None
            ):
                hybrid_partial_avg_cost = _safe_avg(
                    hybrid_partial_total_cost, hybrid_partial_total_successful_delivery
                )
            iter_rows.append(
                {
                    "itr": itr,
                    "method": "Hybrid_RL4CO_partial_charging",
                    "total_delivery": args.customer_num,
                    "total_cp": args.charging_stations_num,
                    "total_ev": args.ev_num,
                    "delivery2ev_ratio": args.customer_num / args.ev_num,
                    "elapsed_time_ms": (
                        hybrid_elapsed_ms if hybrid_elapsed_ms is not None else ""
                    ),
                    "total_reward": (
                        hybrid_partial_total_reward
                        if hybrid_partial_total_reward is not None
                        else (
                            hybrid_full_total_reward
                            if hybrid_full_total_reward is not None
                            else ""
                        )
                    ),
                    "total_cost": (
                        hybrid_partial_total_cost
                        if hybrid_partial_total_cost is not None
                        else (
                            hybrid_full_total_cost
                            if hybrid_full_total_cost is not None
                            else ""
                        )
                    ),
                    "objective_val": (
                        hybrid_partial_objective_val
                        if hybrid_partial_objective_val is not None
                        else (
                            hybrid_full_objective_val
                            if hybrid_full_objective_val is not None
                            else ""
                        )
                    ),
                    "total_successful_delivery": (
                        hybrid_partial_total_successful_delivery
                        if hybrid_partial_total_successful_delivery is not None
                        else (
                            hybrid_full_total_successful_delivery
                            if hybrid_full_total_successful_delivery is not None
                            else ""
                        )
                    ),
                    "avg_cost_per_successful_delivery": (
                        hybrid_partial_avg_cost
                        if hybrid_partial_avg_cost != ""
                        else hybrid_full_avg_cost
                    ),
                }
            )

        if run_rl:
            from convoy_hybrid.convoy_hybrid_main import run_rl as run_hybrid
            from src.convoy_rl_partial_ch.convoy_rl_main import run_rl

            auto_test_csv = None
            if os.path.isfile(generated_test_csv):
                auto_test_csv = generated_test_csv

            # Run hybrid first (same argument style as RL), then RL.
            hybrid_namespace, hybrid_cli_args = parse_hybrid_args(
                args, hybrid_extra_args, auto_test_csv=auto_test_csv
            )
            if args.iterations > 1 and not _contains_flag(hybrid_extra_args, "--seed"):
                hybrid_namespace.seed = int(hybrid_namespace.seed) + (itr - 1)
                hybrid_cli_args = [*hybrid_cli_args, "--seed", str(hybrid_namespace.seed)]
            _print_cmd("convoy_hybrid", hybrid_cli_args)
            hybrid_summary = run_hybrid(hybrid_namespace)

            hybrid_elapsed_ms = None
            hybrid_full_total_reward = None
            hybrid_full_total_cost = None
            hybrid_full_objective_val = None
            hybrid_full_total_successful_delivery = None
            hybrid_partial_total_reward = None
            hybrid_partial_total_cost = None
            hybrid_partial_objective_val = None
            hybrid_partial_total_successful_delivery = None
            if isinstance(hybrid_summary, dict):
                hybrid_elapsed_ms = hybrid_summary.get("csv_inference_time_ms")
                hybrid_full_total_reward = hybrid_summary.get(
                    "csv_augmented_full_total_reward"
                )
                hybrid_full_total_cost = hybrid_summary.get("csv_augmented_full_total_cost")
                hybrid_full_objective_val = hybrid_summary.get(
                    "csv_augmented_full_objective_val"
                )
                hybrid_full_total_successful_delivery = hybrid_summary.get(
                    "csv_augmented_full_total_successful_delivery"
                )
                hybrid_partial_total_reward = hybrid_summary.get(
                    "csv_augmented_partial_total_reward"
                )
                hybrid_partial_total_cost = hybrid_summary.get(
                    "csv_augmented_partial_total_cost"
                )
                hybrid_partial_objective_val = hybrid_summary.get(
                    "csv_augmented_partial_objective_val"
                )
                hybrid_partial_total_successful_delivery = hybrid_summary.get(
                    "csv_augmented_partial_total_successful_delivery"
                )

                if hybrid_full_total_reward is None:
                    hybrid_full_total_reward = hybrid_summary.get("csv_augmented_total_reward")
                if hybrid_full_total_cost is None:
                    hybrid_full_total_cost = hybrid_summary.get("csv_augmented_total_cost")
                if hybrid_full_objective_val is None:
                    hybrid_full_objective_val = hybrid_summary.get("csv_augmented_objective_val")
                if hybrid_full_total_successful_delivery is None:
                    hybrid_full_total_successful_delivery = hybrid_summary.get(
                        "csv_augmented_total_successful_delivery"
                    )
                if hybrid_full_objective_val is None:
                    hybrid_full_objective_val = hybrid_summary.get("csv_instance_reward")

            hybrid_full_avg_cost = ""
            if (
                hybrid_full_total_cost is not None
                and hybrid_full_total_successful_delivery is not None
            ):
                hybrid_full_avg_cost = _safe_avg(
                    hybrid_full_total_cost, hybrid_full_total_successful_delivery
                )
            iter_rows.append(
                {
                    "itr": itr,
                    "method": "Hybrid_RL4CO",
                    "total_delivery": args.customer_num,
                    "total_cp": args.charging_stations_num,
                    "total_ev": args.ev_num,
                    "delivery2ev_ratio": args.customer_num / args.ev_num,
                    "elapsed_time_ms": (
                        hybrid_elapsed_ms if hybrid_elapsed_ms is not None else ""
                    ),
                    "total_reward": (
                        hybrid_full_total_reward
                        if hybrid_full_total_reward is not None
                        else ""
                    ),
                    "total_cost": (
                        hybrid_full_total_cost if hybrid_full_total_cost is not None else ""
                    ),
                    "objective_val": (
                        hybrid_full_objective_val
                        if hybrid_full_objective_val is not None
                        else ""
                    ),
                    "total_successful_delivery": (
                        hybrid_full_total_successful_delivery
                        if hybrid_full_total_successful_delivery is not None
                        else ""
                    ),
                    "avg_cost_per_successful_delivery": hybrid_full_avg_cost,
                }
            )

            hybrid_partial_avg_cost = ""
            if (
                hybrid_partial_total_cost is not None
                and hybrid_partial_total_successful_delivery is not None
            ):
                hybrid_partial_avg_cost = _safe_avg(
                    hybrid_partial_total_cost, hybrid_partial_total_successful_delivery
                )
            iter_rows.append(
                {
                    "itr": itr,
                    "method": "Hybrid_RL4CO_partial_charging",
                    "total_delivery": args.customer_num,
                    "total_cp": args.charging_stations_num,
                    "total_ev": args.ev_num,
                    "delivery2ev_ratio": args.customer_num / args.ev_num,
                    "elapsed_time_ms": (
                        hybrid_elapsed_ms if hybrid_elapsed_ms is not None else ""
                    ),
                    "total_reward": (
                        hybrid_partial_total_reward
                        if hybrid_partial_total_reward is not None
                        else (
                            hybrid_full_total_reward
                            if hybrid_full_total_reward is not None
                            else ""
                        )
                    ),
                    "total_cost": (
                        hybrid_partial_total_cost
                        if hybrid_partial_total_cost is not None
                        else (
                            hybrid_full_total_cost
                            if hybrid_full_total_cost is not None
                            else ""
                        )
                    ),
                    "objective_val": (
                        hybrid_partial_objective_val
                        if hybrid_partial_objective_val is not None
                        else (
                            hybrid_full_objective_val
                            if hybrid_full_objective_val is not None
                            else ""
                        )
                    ),
                    "total_successful_delivery": (
                        hybrid_partial_total_successful_delivery
                        if hybrid_partial_total_successful_delivery is not None
                        else (
                            hybrid_full_total_successful_delivery
                            if hybrid_full_total_successful_delivery is not None
                            else ""
                        )
                    ),
                    "avg_cost_per_successful_delivery": (
                        hybrid_partial_avg_cost
                        if hybrid_partial_avg_cost != ""
                        else hybrid_full_avg_cost
                    ),
                }
            )

            rl_namespace, rl_cli_args = parse_rl_args(
                args, rl_extra_args, auto_test_csv=auto_test_csv
            )
            if args.iterations > 1 and not _contains_flag(rl_extra_args, "--seed"):
                rl_namespace.seed = int(rl_namespace.seed) + (itr - 1)
                rl_cli_args = [*rl_cli_args, "--seed", str(rl_namespace.seed)]
            _print_cmd("convoy_rl_partial_ch", rl_cli_args)
            rl_summary = run_rl(rl_namespace)

            # RL metrics with partial-charging post-processing
            rl_total_reward = None
            rl_total_cost = None
            rl_objective_val = None
            rl_total_successful_delivery = None
            rl_inference_time_ms = None
            # RL metrics before partial charging (full-charge trace decomposition)
            rl_full_total_reward = None
            rl_full_total_cost = None
            rl_full_objective_val = None
            rl_full_total_successful_delivery = None
            if isinstance(rl_summary, dict):
                rl_total_reward = rl_summary.get("csv_total_reward")
                rl_total_cost = rl_summary.get("csv_total_cost")
                rl_objective_val = rl_summary.get("csv_objective_val")
                rl_total_successful_delivery = rl_summary.get(
                    "csv_total_successful_delivery"
                )
                if rl_objective_val is None:
                    rl_objective_val = rl_summary.get("csv_instance_reward")
                if rl_objective_val is None:
                    rl_objective_val = rl_summary.get("test_reward")
                rl_inference_time_ms = rl_summary.get("csv_inference_time_ms")
                rl_full_total_reward = rl_summary.get("csv_full_charge_total_reward")
                rl_full_total_cost = rl_summary.get("csv_full_charge_total_cost")
                rl_full_objective_val = rl_summary.get("csv_full_charge_objective_val")
                rl_full_total_successful_delivery = rl_summary.get(
                    "csv_full_charge_total_successful_delivery"
                )

            rl_avg_cost = ""
            if rl_total_cost is not None and rl_total_successful_delivery is not None:
                rl_avg_cost = _safe_avg(rl_total_cost, rl_total_successful_delivery)

            # Row 1: RL reward components before partial charging.
            if (
                rl_full_total_reward is not None
                or rl_full_total_cost is not None
                or rl_full_objective_val is not None
            ):
                rl_full_avg_cost = ""
                if (
                    rl_full_total_cost is not None
                    and rl_full_total_successful_delivery is not None
                ):
                    rl_full_avg_cost = _safe_avg(
                        rl_full_total_cost, rl_full_total_successful_delivery
                    )
                iter_rows.append(
                    {
                        "itr": itr,
                        "method": "RL",
                        "total_delivery": args.customer_num,
                        "total_cp": args.charging_stations_num,
                        "total_ev": args.ev_num,
                        "delivery2ev_ratio": args.customer_num / args.ev_num,
                        "elapsed_time_ms": (
                            rl_inference_time_ms if rl_inference_time_ms is not None else ""
                        ),
                        "total_reward": (
                            rl_full_total_reward if rl_full_total_reward is not None else ""
                        ),
                        "total_cost": (
                            rl_full_total_cost if rl_full_total_cost is not None else ""
                        ),
                        "objective_val": (
                            rl_full_objective_val
                            if rl_full_objective_val is not None
                            else ""
                        ),
                        "total_successful_delivery": (
                            rl_full_total_successful_delivery
                            if rl_full_total_successful_delivery is not None
                            else ""
                        ),
                        "avg_cost_per_successful_delivery": rl_full_avg_cost,
                    }
                )

            # Row 2: RL reward components after partial charging.
            iter_rows.append(
                {
                    "itr": itr,
                    "method": "RL_partial_charging",
                    "total_delivery": args.customer_num,
                    "total_cp": args.charging_stations_num,
                    "total_ev": args.ev_num,
                    "delivery2ev_ratio": args.customer_num / args.ev_num,
                    "elapsed_time_ms": (
                        rl_inference_time_ms if rl_inference_time_ms is not None else ""
                    ),
                    "total_reward": rl_total_reward if rl_total_reward is not None else "",
                    "total_cost": rl_total_cost if rl_total_cost is not None else "",
                    "objective_val": rl_objective_val if rl_objective_val is not None else "",
                    "total_successful_delivery": (
                        rl_total_successful_delivery
                        if rl_total_successful_delivery is not None
                        else ""
                    ),
                    "avg_cost_per_successful_delivery": rl_avg_cost,
                }
            )

        if run_baseline:
            from tools.run_baseline_pipeline import run_baseline_pipeline

            baseline_test_csv = None
            baseline_test_csv = _get_flag_value(rl_extra_args, "--test-csv")
            if baseline_test_csv is None:
                if os.path.isfile(generated_test_csv):
                    baseline_test_csv = generated_test_csv
                else:
                    baseline_test_csv = os.path.join(repo_root, "data", "test_instance.csv")

            baseline_summary = None
            baseline_error = None
            try:
                baseline_summary = run_baseline_pipeline(
                    repo_root=repo_root,
                    test_csv=baseline_test_csv,
                    dist_matrix_csv=args.combined_dist_matrix_csv,
                    time_matrix_csv=args.combined_time_matrix_csv,
                    vehicles=args.ev_num,
                    instance_output_path=args.baseline_instance_output_path,
                    latest_baseline_output=args.baseline_output_file,
                    baseline_bin=args.baseline_bin,
                    baseline_time=args.baseline_time,
                    baseline_runs=args.baseline_runs,
                    baseline_extra=baseline_extra_args,
                    ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
                    print_solver_cmd=True,
                    quiet_solver=args.baseline_quiet,
                    solver_log_path=args.baseline_solver_log,
                )
            except Exception as exc:
                baseline_error = exc
                print(f"Baseline failed in iteration {itr}: {exc}")

            if baseline_error is not None:
                iter_rows.append(
                    {
                        "itr": itr,
                        "method": "Baseline_failed",
                        "total_delivery": args.customer_num,
                        "total_cp": args.charging_stations_num,
                        "total_ev": args.ev_num,
                        "delivery2ev_ratio": args.customer_num / args.ev_num,
                        "elapsed_time_ms": "",
                        "total_reward": "",
                        "total_cost": "",
                        "objective_val": "FAILED",
                        "total_successful_delivery": "",
                        "avg_cost_per_successful_delivery": "",
                    }
                )
            else:
                baseline_total_reward = baseline_summary.get("total_reward")
                baseline_total_cost = baseline_summary.get("total_charging_cost")
                baseline_objective_val = baseline_summary.get("objective_score")
                baseline_total_successful_delivery = baseline_summary.get(
                    "visited_customer_count"
                )
                baseline_elapsed_time_ms = baseline_summary.get("solver_time_ms")
                if baseline_elapsed_time_ms is None:
                    baseline_elapsed_time_ms = baseline_summary.get("elapsed_time_ms")

                baseline_avg_cost = ""
                if (
                    baseline_total_cost is not None
                    and baseline_total_successful_delivery is not None
                ):
                    baseline_avg_cost = _safe_avg(
                        baseline_total_cost, baseline_total_successful_delivery
                    )

                if (
                    baseline_total_reward is not None
                    and baseline_total_cost is not None
                    and baseline_objective_val is not None
                    and baseline_total_successful_delivery is not None
                ):
                    print(
                        "Reward components: "
                        f"total_reward={float(baseline_total_reward):.6f}, "
                        f"total_cost={float(baseline_total_cost):.6f}, "
                        f"successful_delivery={int(baseline_total_successful_delivery)}, "
                        f"objective={float(baseline_objective_val):.6f}"
                    )

                iter_rows.append(
                    {
                        "itr": itr,
                        "method": "Baseline",
                        "total_delivery": args.customer_num,
                        "total_cp": args.charging_stations_num,
                        "total_ev": args.ev_num,
                        "delivery2ev_ratio": args.customer_num / args.ev_num,
                        "elapsed_time_ms": (
                            baseline_elapsed_time_ms
                            if baseline_elapsed_time_ms is not None
                            else ""
                        ),
                        "total_reward": (
                            baseline_total_reward if baseline_total_reward is not None else ""
                        ),
                        "total_cost": (
                            baseline_total_cost if baseline_total_cost is not None else ""
                        ),
                        "objective_val": (
                            baseline_objective_val
                            if baseline_objective_val is not None
                            else ""
                        ),
                        "total_successful_delivery": (
                            baseline_total_successful_delivery
                            if baseline_total_successful_delivery is not None
                            else ""
                        ),
                        "avg_cost_per_successful_delivery": baseline_avg_cost,
                    }
                )

                if args.baseline_print_charging_events:
                    print("Baseline charging events:")
                    events = baseline_summary.get("charging_events", [])
                    if not events:
                        print("  (none)")
                    for event in events:
                        print(
                            "  route={route} mapped_id={mapped_id} original_id={original_id} "
                            "type={node_type} arr_rd={arr_rd:.2f} dep_rd={dep_rd:.2f} "
                            "charged_dist={charged_distance_units:.2f} charged_kwh={charged_energy_kwh:.6f} "
                            "unit_cost={unit_charging_cost:.6f} cost={charging_cost:.6f}".format(
                                **event
                            )
                        )

                if args.baseline_print_route_trace:
                    print("Baseline route trace:")
                    route_traces = baseline_summary.get("route_traces", [])
                    if not route_traces:
                        print("  (none)")
                    for ridx, trace in enumerate(route_traces):
                        print(f"  Route {ridx}:")
                        for rec in trace:
                            travel_d = rec.get("travel_distance_from_prev")
                            travel_d_txt = "None" if travel_d is None else f"{float(travel_d):.2f}"
                            energy = rec.get("energy_used_kwh_from_prev")
                            energy_txt = "None" if energy is None else f"{float(energy):.6f}"
                            soc_arr = rec.get("soc_arrival_kwh")
                            soc_arr_txt = "None" if soc_arr is None else f"{float(soc_arr):.2f}"
                            soc_dep = rec.get("soc_departure_kwh")
                            soc_dep_txt = "None" if soc_dep is None else f"{float(soc_dep):.2f}"
                            print(
                                "    step={step:>2d} mapped={mapped_id:>3d} "
                                "original={original_id:>3d} type={node_type} "
                                "arr={arrival_time:.2f} dep={departure_time:.2f} "
                                "travel_t={travel_time_from_prev:.2f} "
                                "travel_d={travel_d} "
                                "energy={energy} "
                                "soc_arr={soc_arr} soc_dep={soc_dep} "
                                "wait={wait_time:.2f} service={service_time:.2f} "
                                "charge_t={charge_time:.2f} charge_kwh={charge_energy_kwh:.6f} "
                                "arr_rd={arr_rd} dep_rd={dep_rd}".format(
                                    **rec,
                                    travel_d=travel_d_txt,
                                    energy=energy_txt,
                                    soc_arr=soc_arr_txt,
                                    soc_dep=soc_dep_txt,
                                )
                            )

        # Keep all method rows for this iteration together in final CSV.
        all_results_rows.extend(iter_rows)

    if all_results_rows:
        _write_results_csv(output_file, all_results_rows)
        print("Results saved:", output_file)


if __name__ == "__main__":
    main()

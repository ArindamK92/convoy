import csv
import os
import shlex

from convoy_parser import (
    parse_args,
    parse_extra_args,
    parse_opt_heu_args,
    parse_rl_args,
)


def _print_cmd(runner_name, args):
    print("$", runner_name, " ".join(shlex.quote(a) for a in args))


def _contains_flag(args, flag_name):
    for token in args:
        if token == flag_name or token.startswith(flag_name + "="):
            return True
    return False


def _safe_avg(total_cost, total_deliveries_completed):
    if total_deliveries_completed <= 0:
        return ""
    return total_cost / total_deliveries_completed


def _results_output_path(repo_root, combined_details_csv, results_file=None):
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
    args = parse_args()
    if args.only_rl and args.only_opt_heu:
        raise ValueError("Use at most one of --only-rl or --only-opt-heu.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0.")

    run_rl = not args.only_opt_heu
    run_opt_heu = not args.only_rl

    repo_root = os.path.dirname(os.path.abspath(__file__))
    combined_details_abs = os.path.abspath(args.combined_details_csv)
    output_file = _results_output_path(
        repo_root, combined_details_abs, results_file=args.results_file
    )
    generated_test_csv = os.path.join(os.path.dirname(combined_details_abs), "test_instance.csv")
    all_results_rows = []
    opt_extra_raw = parse_extra_args(args.opt_heu_extra)
    rl_extra_args = parse_extra_args(args.opt_rl_extra)

    for itr in range(1, args.iterations + 1):
        print("=== Iteration {}/{} ===".format(itr, args.iterations))
        iter_rows = []

        # Run optimal+heuristic first so helper preprocessing can write test_instance.csv.
        if run_opt_heu:
            from src.convoy_opt_and_heu.opt_and_hue import run_opt_heu_with_params

            opt_namespace, opt_cli_args = parse_opt_heu_args(args, opt_extra_raw)
            _print_cmd("convoy_opt_and_heu", opt_cli_args)
            opt_random_seed = getattr(opt_namespace, "random_seed", None)
            if args.iterations > 1:
                if opt_random_seed is None:
                    opt_random_seed = itr
                else:
                    opt_random_seed = opt_random_seed + (itr - 1)
            opt_rows = run_opt_heu_with_params(
                combined_details_csv=opt_namespace.combined_details_csv,
                combined_dist_matrix_csv=opt_namespace.combined_dist_matrix_csv,
                combined_time_matrix_csv=opt_namespace.combined_time_matrix_csv,
                customer_num=opt_namespace.customer_num,
                charging_stations_num=opt_namespace.charging_stations_num,
                ev_num=opt_namespace.ev_num,
                ev_energy_rate_kwh_per_distance=opt_namespace.ev_energy_rate_kwh_per_distance,
                alpha1_override=opt_namespace.alpha1,
                alpha2_override=opt_namespace.alpha2,
                skip_optimal=opt_namespace.skip_optimal,
                random_seed=opt_random_seed,
            )
            for row in opt_rows:
                row["itr"] = itr
            iter_rows.extend(opt_rows)

        if run_rl:
            from src.convoy_rl.convoy_rl_main import run_rl

            auto_test_csv = None
            if run_opt_heu:
                if not os.path.isfile(generated_test_csv):
                    raise FileNotFoundError(
                        "Expected generated test instance not found: {}".format(
                            generated_test_csv
                        )
                    )
                auto_test_csv = generated_test_csv
            rl_namespace, rl_cli_args = parse_rl_args(
                args, rl_extra_args, auto_test_csv=auto_test_csv
            )
            if args.iterations > 1 and not _contains_flag(rl_extra_args, "--seed"):
                rl_namespace.seed = int(rl_namespace.seed) + (itr - 1)
                rl_cli_args = [*rl_cli_args, "--seed", str(rl_namespace.seed)]
            _print_cmd("convoy_rl", rl_cli_args)
            rl_summary = run_rl(rl_namespace)

            rl_total_reward = None
            rl_total_cost = None
            rl_objective_val = None
            rl_total_successful_delivery = None
            rl_inference_time_ms = None
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

            rl_avg_cost = ""
            if rl_total_cost is not None and rl_total_successful_delivery is not None:
                rl_avg_cost = _safe_avg(rl_total_cost, rl_total_successful_delivery)

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

        all_results_rows.extend(iter_rows)

    if all_results_rows:
        _write_results_csv(output_file, all_results_rows)
        print("Results saved:", output_file)


if __name__ == "__main__":
    main()

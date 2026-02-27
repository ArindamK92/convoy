"""Main orchestration entry for Optimal and Heuristic CONVOY runs."""

import copy
import csv
import math
import os
from types import SimpleNamespace

try:
    from .helper import preProcess
    from .ev import EV
    from .heuristic_partial_charging_fn import heuristic
    from .MILP_fn import MILP
    from .baselines import EDF, NDF
except ImportError:
    from helper import preProcess
    from ev import EV
    from heuristic_partial_charging_fn import heuristic
    from MILP_fn import MILP
    from baselines import EDF, NDF

DEBUG = False


def printDebug(string, val):
    if DEBUG:
        print(string, val)


def _safe_avg(total_cost, total_deliveries_completed):
    if total_deliveries_completed <= 0:
        return ""
    return total_cost / total_deliveries_completed


def _validate_common_inputs(
    combined_details_csv,
    combined_dist_matrix_csv,
    combined_time_matrix_csv,
    customer_num,
    charging_stations_num,
    ev_num,
    ev_energy_rate_kwh_per_distance,
    reserve_battery,
):
    for p in [combined_details_csv, combined_dist_matrix_csv, combined_time_matrix_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError("Input file not found: {}".format(p))

    if customer_num <= 0:
        raise ValueError("--customer-num must be > 0.")
    if charging_stations_num < 0:
        raise ValueError("--charging-stations-num must be >= 0.")
    if ev_num <= 0:
        raise ValueError("--ev-num must be > 0.")
    if ev_energy_rate_kwh_per_distance <= 0:
        raise ValueError("--ev-energy-rate-kwh-per-distance must be > 0.")
    if reserve_battery < 0:
        raise ValueError("--reserve-battery must be >= 0.")


def _derive_counts_from_test_csv(test_csv_path):
    """Return (customer_count, charging_station_count) from a test-instance CSV."""
    customer_count = 0
    charging_count = 0
    depot_count = 0
    with open(test_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "type" not in reader.fieldnames:
            raise ValueError(
                "test-for-opt-heu CSV must contain a 'type' column: {}".format(
                    test_csv_path
                )
            )
        for row in reader:
            t = str(row.get("type", "")).strip().lower()
            if t == "c":
                customer_count += 1
            elif t == "f":
                charging_count += 1
            elif t == "d":
                depot_count += 1
    if depot_count != 1:
        raise ValueError(
            "Expected exactly one depot row (type=d) in {}, found {}.".format(
                test_csv_path, depot_count
            )
        )
    if customer_count <= 0:
        raise ValueError(
            "No customer rows (type=c) found in test-for-opt-heu CSV: {}".format(
                test_csv_path
            )
        )
    return customer_count, charging_count


def run_opt_heu(args):
    combined_details_csv = os.path.abspath(args.combined_details_csv)
    combined_dist_matrix_csv = os.path.abspath(args.combined_dist_matrix_csv)
    combined_time_matrix_csv = os.path.abspath(args.combined_time_matrix_csv)

    test_for_opt_heu = getattr(args, "test_for_opt_heu", None)
    use_full_instance = bool(test_for_opt_heu)
    if use_full_instance:
        details_csv_for_opt_heu = os.path.abspath(test_for_opt_heu)
        nD, nC = _derive_counts_from_test_csv(details_csv_for_opt_heu)
    else:
        details_csv_for_opt_heu = combined_details_csv
        nD = int(args.customer_num)
        nC = int(args.charging_stations_num)

    _validate_common_inputs(
        combined_details_csv=details_csv_for_opt_heu,
        combined_dist_matrix_csv=combined_dist_matrix_csv,
        combined_time_matrix_csv=combined_time_matrix_csv,
        customer_num=nD,
        charging_stations_num=nC,
        ev_num=args.ev_num,
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        reserve_battery=args.reserve_battery,
    )

    nE = int(args.ev_num)
    reserve_battery = float(args.reserve_battery)
    nS = math.ceil(nD / nE)
    delivery2ev_ratio = nD / nE
    only_milp = bool(getattr(args, "only_milp", False))
    if only_milp and bool(getattr(args, "skip_optimal", False)):
        raise ValueError("Use at most one of --skip-optimal or --only-milp.")

    (
        cp,
        deliveries,
        theta,
        reward,
        C,
        D,
        E,
        C1,
        S,
        tau_start,
        tau_end,
        nS,
        EVs,
        gamma_DD,
        psi_DD,
        gamma_DC,
        psi_DC,
        beta_f,
        rateE,
        rateC,
        alpha1,
        alpha2,
        gamma_C0,
        psi_C0,
    ) = preProcess(
        nD,
        nC,
        nS,
        nE,
        details_csv_for_opt_heu,
        combined_dist_matrix_csv,
        combined_time_matrix_csv,
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        random_seed=getattr(args, "random_seed", None),
        use_full_instance=use_full_instance,
    )

    rows = []
    run_milp = (not args.skip_optimal) and (only_milp or (nD < 20 and nC < 20))
    if run_milp:
        (
            elapsed_time,
            total_cost,
            total_deliveries_completed,
            objective_val,
        ) = MILP(
            cp,
            deliveries,
            theta,
            reward,
            C,
            D,
            E,
            C1,
            S,
            tau_start,
            tau_end,
            nS,
            EVs,
            gamma_DD,
            psi_DD,
            gamma_DC,
            psi_DC,
            beta_f,
            rateE,
            rateC,
            alpha1,
            alpha2,
            reserve_battery=reserve_battery,
            gamma_C0=gamma_C0,
            psi_C0=psi_C0,
            return_objective=True,
        )
        rows.append(
            {
                "method": "Optimal",
                "total_delivery": nD,
                "total_cp": nC,
                "total_ev": nE,
                "delivery2ev_ratio": delivery2ev_ratio,
                "elapsed_time_ms": elapsed_time,
                "total_reward": (args.alpha2 * total_cost) + objective_val,
                "total_cost": total_cost,
                "objective_val": objective_val,
                "total_successful_delivery": total_deliveries_completed,
                "avg_cost_per_successful_delivery": _safe_avg(
                    total_cost, total_deliveries_completed
                ),
            }
        )
    elif only_milp:
        raise ValueError(
            "MILP-only run requested but MILP is disabled by flags. "
            "Remove --skip-optimal."
        )

    if only_milp:
        return rows

    # Reserve SOC is modeled by shrinking usable battery to (full - reserve).
    effective_battery_kwh = beta_f - reserve_battery
    if effective_battery_kwh <= 0:
        raise ValueError(
            "--reserve-battery must be smaller than full battery ({} kWh).".format(
                beta_f
            )
        )
    EVs = [EV(j, cp[0], effective_battery_kwh) for j in E]
    (
        elapsed_time,
        total_cost,
        total_reward,
        total_deliveries_completed,
    ) = heuristic(
        cp,
        deliveries,
        theta,
        reward,
        C,
        D,
        E,
        tau_start,
        tau_end,
        nS,
        EVs,
        gamma_DD,
        psi_DD,
        gamma_DC,
        psi_DC,
        beta_f,
        effective_battery_kwh,
        rateE,
        rateC,
    )
    rows.append(
        {
            "method": "CSA",
            "total_delivery": nD,
            "total_cp": nC,
            "total_ev": nE,
            "delivery2ev_ratio": delivery2ev_ratio,
            "elapsed_time_ms": elapsed_time,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "objective_val": total_reward - total_cost,
            "total_successful_delivery": total_deliveries_completed,
            "avg_cost_per_successful_delivery": _safe_avg(
                total_cost, total_deliveries_completed
            ),
        }
    )

    if not getattr(args, "no_edf_ndf", False):
        # Run NDF and EDF on fresh copies so policy state does not leak across methods.
        baseline_methods = [("NDF", NDF), ("EDF", EDF)]
        for method_name, method_fn in baseline_methods:
            baseline_deliveries = copy.deepcopy(deliveries)
            baseline_evs = [EV(j, cp[0], effective_battery_kwh) for j in E]
            (
                elapsed_time_baseline,
                total_cost_baseline,
                total_reward_baseline,
                total_deliveries_completed_baseline,
            ) = method_fn(
                cp,
                baseline_deliveries,
                theta,
                C,
                D,
                E,
                tau_start,
                tau_end,
                nS,
                baseline_evs,
                gamma_DD,
                psi_DD,
                gamma_DC,
                psi_DC,
                effective_battery_kwh,
                rateE,
                rateC,
            )
            rows.append(
                {
                    "method": method_name,
                    "total_delivery": nD,
                    "total_cp": nC,
                    "total_ev": nE,
                    "delivery2ev_ratio": delivery2ev_ratio,
                    "elapsed_time_ms": elapsed_time_baseline,
                    "total_reward": total_reward_baseline,
                    "total_cost": total_cost_baseline,
                    "objective_val": total_reward_baseline - total_cost_baseline,
                    "total_successful_delivery": total_deliveries_completed_baseline,
                    "avg_cost_per_successful_delivery": _safe_avg(
                        total_cost_baseline, total_deliveries_completed_baseline
                    ),
                }
            )
    return rows


def run_opt_heu_with_params(
    combined_details_csv,
    combined_dist_matrix_csv,
    combined_time_matrix_csv,
    customer_num,
    charging_stations_num,
    ev_num,
    ev_energy_rate_kwh_per_distance=0.00025,
    reserve_battery=0.0,
    alpha1_override=1.0,
    alpha2_override=1.0,
    skip_optimal=False,
    only_milp=False,
    random_seed=None,
    no_edf_ndf=False,
    test_for_opt_heu=None,
):
    args = SimpleNamespace(
        combined_details_csv=combined_details_csv,
        combined_dist_matrix_csv=combined_dist_matrix_csv,
        combined_time_matrix_csv=combined_time_matrix_csv,
        customer_num=customer_num,
        charging_stations_num=charging_stations_num,
        ev_num=ev_num,
        ev_energy_rate_kwh_per_distance=ev_energy_rate_kwh_per_distance,
        reserve_battery=reserve_battery,
        alpha1=alpha1_override,
        alpha2=alpha2_override,
        skip_optimal=skip_optimal,
        only_milp=only_milp,
        random_seed=random_seed,
        no_edf_ndf=no_edf_ndf,
        test_for_opt_heu=test_for_opt_heu,
    )
    return run_opt_heu(args)


def _results_output_path(combined_details_csv):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    details_stem = os.path.splitext(os.path.basename(os.path.abspath(combined_details_csv)))[
        0
    ]
    return os.path.join(results_dir, "results3_{}.csv".format(details_stem))


def _write_results_csv(output_file, rows):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
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


def opt_heu_main(args=None):
    if args is None:
        from convoy_parser import parse_opt_heu_direct_args

        args = parse_opt_heu_direct_args()
    rows = run_opt_heu(args)
    output_file = _results_output_path(args.combined_details_csv)
    _write_results_csv(output_file, rows)
    print("Results saved:", output_file)
    return rows


def main():
    """Backward-compatible wrapper for legacy callers."""
    return opt_heu_main()


if __name__ == "__main__":
    opt_heu_main()

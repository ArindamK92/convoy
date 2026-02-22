import csv
import math
import os
from types import SimpleNamespace

try:
    from .helper import preProcess
    from .ev import EV
    from .heuristic_partial_charging_fn import heuristic
    from .MILP_fn import MILP
except ImportError:
    from helper import preProcess
    from ev import EV
    from heuristic_partial_charging_fn import heuristic
    from MILP_fn import MILP

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


def run_opt_heu(args):
    combined_details_csv = os.path.abspath(args.combined_details_csv)
    combined_dist_matrix_csv = os.path.abspath(args.combined_dist_matrix_csv)
    combined_time_matrix_csv = os.path.abspath(args.combined_time_matrix_csv)

    _validate_common_inputs(
        combined_details_csv=combined_details_csv,
        combined_dist_matrix_csv=combined_dist_matrix_csv,
        combined_time_matrix_csv=combined_time_matrix_csv,
        customer_num=args.customer_num,
        charging_stations_num=args.charging_stations_num,
        ev_num=args.ev_num,
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
    )

    nD = int(args.customer_num)
    nC = int(args.charging_stations_num)
    nE = int(args.ev_num)
    nS = math.ceil(nD / nE)
    delivery2ev_ratio = nD / nE

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
    ) = preProcess(
        nD,
        nC,
        nS,
        nE,
        combined_details_csv,
        combined_dist_matrix_csv,
        combined_time_matrix_csv,
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        random_seed=getattr(args, "random_seed", None),
    )

    rows = []
    if not args.skip_optimal and nD < 20 and nC < 20:
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

    EVs = [EV(j, cp[0], beta_f) for j in E]
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
    return rows


def run_opt_heu_with_params(
    combined_details_csv,
    combined_dist_matrix_csv,
    combined_time_matrix_csv,
    customer_num,
    charging_stations_num,
    ev_num,
    ev_energy_rate_kwh_per_distance=0.00025,
    alpha1_override=1.0,
    alpha2_override=1.0,
    skip_optimal=False,
    random_seed=None,
):
    args = SimpleNamespace(
        combined_details_csv=combined_details_csv,
        combined_dist_matrix_csv=combined_dist_matrix_csv,
        combined_time_matrix_csv=combined_time_matrix_csv,
        customer_num=customer_num,
        charging_stations_num=charging_stations_num,
        ev_num=ev_num,
        ev_energy_rate_kwh_per_distance=ev_energy_rate_kwh_per_distance,
        alpha1=alpha1_override,
        alpha2=alpha2_override,
        skip_optimal=skip_optimal,
        random_seed=random_seed,
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

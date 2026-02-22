import argparse
import shlex

def parse_extra_args(extra_chunks):
    args = []
    for chunk in extra_chunks:
        args.extend(shlex.split(chunk))
    return args


def _contains_flag(args, flag_name):
    for token in args:
        if token == flag_name or token.startswith(flag_name + "="):
            return True
    return False


def _build_rl_parser():
    from src.convoy_rl.myparser import build_parser as build_rl_parser

    return build_rl_parser()


def build_rl_cli_args(main_args, rl_extra_args, auto_test_csv=None):
    rl_args = [
        "--combined-details-csv",
        main_args.combined_details_csv,
        "--combined-dist-matrix-csv",
        main_args.combined_dist_matrix_csv,
        "--combined-time-matrix-csv",
        main_args.combined_time_matrix_csv,
        "--customer-num",
        str(main_args.customer_num),
        "--charging-stations-num",
        str(main_args.charging_stations_num),
        "--ev-num",
        str(main_args.ev_num),
        "--ev-energy-rate-kwh-per-distance",
        str(main_args.ev_energy_rate_kwh_per_distance),
    ]
    if auto_test_csv and not _contains_flag(rl_extra_args, "--test-csv"):
        rl_args.extend(["--test-csv", auto_test_csv])
    rl_args.extend(rl_extra_args)
    return rl_args


def parse_rl_direct_args(cli_args=None):
    return _build_rl_parser().parse_args(cli_args)


def parse_rl_args(main_args, rl_extra_args, auto_test_csv=None):
    rl_cli_args = build_rl_cli_args(main_args, rl_extra_args, auto_test_csv=auto_test_csv)
    rl_parser = _build_rl_parser()
    return rl_parser.parse_args(rl_cli_args), rl_cli_args


def build_opt_heu_parser():
    parser = argparse.ArgumentParser(
        description="Run optimal + heuristic EV-delivery pipeline using explicit CSV paths."
    )
    parser.add_argument(
        "--combined-details-csv",
        type=str,
        required=True,
        help="Path to combined details CSV (ID,type,lng,lat,...).",
    )
    parser.add_argument(
        "--combined-dist-matrix-csv",
        type=str,
        required=True,
        help="Path to combined distance matrix CSV.",
    )
    parser.add_argument(
        "--combined-time-matrix-csv",
        type=str,
        required=True,
        help="Path to combined time matrix CSV.",
    )
    parser.add_argument(
        "--customer-num",
        type=int,
        required=True,
        help="Number of delivery customers to use.",
    )
    parser.add_argument(
        "--charging-stations-num",
        type=int,
        required=True,
        help="Number of charging points to use (excluding depot).",
    )
    parser.add_argument(
        "--ev-num",
        type=int,
        required=True,
        help="Number of EVs to use.",
    )
    parser.add_argument(
        "--ev-energy-rate-kwh-per-distance",
        type=float,
        default=0.00025,
        help=(
            "Energy use per distance unit (kWh / distance-unit). "
            "Default 0.00025 corresponds to mileage 4 km/kWh for meter-based distances."
        ),
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default=1.0,
        help="Objective weight for successful deliveries (MILP).",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default=1.0,
        help="Objective weight for energy cost (MILP).",
    )
    parser.add_argument(
        "--skip-optimal",
        action="store_true",
        help="Skip MILP and run only heuristic.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "Optional base seed for opt/heu preprocessing sampling "
            "(customer/CP selection and EV charge-acceptance rates)."
        ),
    )
    return parser


def parse_opt_heu_direct_args(cli_args=None):
    return build_opt_heu_parser().parse_args(cli_args)


def build_opt_heu_cli_args(main_args, opt_extra_args):
    return [
        "--combined-details-csv",
        main_args.combined_details_csv,
        "--combined-dist-matrix-csv",
        main_args.combined_dist_matrix_csv,
        "--combined-time-matrix-csv",
        main_args.combined_time_matrix_csv,
        "--customer-num",
        str(main_args.customer_num),
        "--charging-stations-num",
        str(main_args.charging_stations_num),
        "--ev-num",
        str(main_args.ev_num),
        "--ev-energy-rate-kwh-per-distance",
        str(main_args.ev_energy_rate_kwh_per_distance),
        *opt_extra_args,
    ]


def parse_opt_heu_args(main_args, opt_extra_args):
    opt_heu_cli_args = build_opt_heu_cli_args(main_args, opt_extra_args)
    parser = build_opt_heu_parser()
    return parser.parse_args(opt_heu_cli_args), opt_heu_cli_args


def build_main_parser():
    parser = argparse.ArgumentParser(
        description="Run RL and/or optimal+heuristic with one shared argument set."
    )
    parser.add_argument(
        "--combined-details-csv",
        type=str,
        required=True,
        help="Path to combined details CSV (ID,type,lng,lat,...).",
    )
    parser.add_argument(
        "--combined-dist-matrix-csv",
        type=str,
        required=True,
        help="Path to combined distance matrix CSV.",
    )
    parser.add_argument(
        "--combined-time-matrix-csv",
        type=str,
        required=True,
        help="Path to combined time matrix CSV.",
    )
    parser.add_argument(
        "--customer-num",
        type=int,
        required=True,
        help="Number of customers to sample/use in both runners.",
    )
    parser.add_argument(
        "--charging-stations-num",
        type=int,
        required=True,
        help="Number of charging stations to sample/use in both runners.",
    )
    parser.add_argument(
        "--ev-num",
        type=int,
        required=True,
        help="Number of EVs to use in both runners.",
    )
    parser.add_argument(
        "--ev-energy-rate-kwh-per-distance",
        type=float,
        default=0.00025,
        help=(
            "Energy use per distance unit (kWh / distance-unit) shared by RL and opt/heu. "
            "For meter-based distances, 0.00025 corresponds to 4 km/kWh."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of repeated runs to execute and log in results CSV.",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help=(
            "Output results CSV filename/path. "
            "If relative, it is created under CONVOY2/results/. "
            "If omitted, default is results3_<combined-details-stem>.csv."
        ),
    )
    parser.add_argument(
        "--only-rl",
        action="store_true",
        help="Run only RL launcher (skip optimal+heuristic).",
    )
    parser.add_argument(
        "--only-opt-heu",
        action="store_true",
        help="Run only optimal+heuristic launcher (skip RL).",
    )
    parser.add_argument(
        "--opt-rl-extra",
        action="append",
        default=[],
        help=(
            "Optional quoted extra flags passed to RL launcher. "
            "Example: --opt-rl-extra \"--test-csv data/test_delivery10_jd200.csv --print-solution\""
        ),
    )
    parser.add_argument(
        "--opt-heu-extra",
        action="append",
        default=[],
        help=(
            "Quoted extra flags passed to optimal+heuristic launcher. "
            "Example: --opt-heu-extra \"--skip-optimal\""
        ),
    )
    return parser


def parse_args():
    return build_main_parser().parse_args()

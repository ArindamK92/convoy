"""Central CLI parser utilities for CONVOY2 runners.

This module provides:
- top-level `convoy_main` argument parsing,
- helper parsers/builders for RL and Opt+Heu sub-runners,
- pass-through handling for quoted extra argument groups.
"""

import argparse
import shlex

def parse_extra_args(extra_chunks):
    """Split repeated quoted extra-arg chunks into one token list."""
    args = []
    for chunk in extra_chunks:
        args.extend(shlex.split(chunk))
    return args


def _contains_flag(args, flag_name):
    """Return True when `flag_name` exists in token list.

    Supports both:
    - `--flag value`
    - `--flag=value`
    """
    for token in args:
        if token == flag_name or token.startswith(flag_name + "="):
            return True
    return False


def _build_rl_parser():
    """Load and return the RL parser from `src.convoy_rl_partial_ch.myparser`."""
    from src.convoy_rl_partial_ch.myparser import build_parser as build_rl_parser

    return build_rl_parser()


def _build_hybrid_parser():
    """Load and return the hybrid parser from `convoy_hybrid`."""
    from convoy_hybrid.convoy_hybrid_parser import build_parser as build_hybrid_parser

    return build_hybrid_parser()


def build_rl_cli_args(main_args, rl_extra_args, auto_test_csv=None):
    """Build RL CLI args from shared main args plus RL-specific extras."""
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
        "--cost-weight",
        str(main_args.cost_weight),
        "--reserve-battery",
        str(main_args.reserve_battery),
    ]
    if auto_test_csv and not _contains_flag(rl_extra_args, "--test-csv"):
        rl_args.extend(["--test-csv", auto_test_csv])
    rl_args.extend(rl_extra_args)
    return rl_args


def parse_rl_direct_args(cli_args=None):
    """Parse RL arguments directly (standalone RL invocation)."""
    return _build_rl_parser().parse_args(cli_args)


def parse_rl_args(main_args, rl_extra_args, auto_test_csv=None):
    """Parse RL args synthesized from top-level args and pass-through extras."""
    rl_cli_args = build_rl_cli_args(main_args, rl_extra_args, auto_test_csv=auto_test_csv)
    rl_parser = _build_rl_parser()
    return rl_parser.parse_args(rl_cli_args), rl_cli_args


def parse_hybrid_args(main_args, rl_extra_args, auto_test_csv=None):
    """Parse hybrid args synthesized from top-level args and RL-style extras."""
    hybrid_cli_args = build_rl_cli_args(
        main_args, rl_extra_args, auto_test_csv=auto_test_csv
    )
    hybrid_parser = _build_hybrid_parser()
    return hybrid_parser.parse_args(hybrid_cli_args), hybrid_cli_args


def build_opt_heu_parser():
    """Create parser for the Optimal+Heuristic runner."""
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
        "--test-for-opt-heu",
        nargs="?",
        const="data/test_instance.csv",
        default=None,
        help=(
            "Run opt+heu directly on a prepared test-instance CSV "
            "(depot/customers/CP rows) instead of sampling from combined-details CSV. "
            "If flag is provided without a value, defaults to data/test_instance.csv."
        ),
    )
    parser.add_argument(
        "--customer-num",
        type=int,
        required=True,
        help=(
            "Number of delivery customers to use. "
            "Ignored when --test-for-opt-heu is set."
        ),
    )
    parser.add_argument(
        "--charging-stations-num",
        type=int,
        required=True,
        help=(
            "Number of charging points to use (excluding depot). "
            "Ignored when --test-for-opt-heu is set."
        ),
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
        "--reserve-battery",
        type=float,
        default=0.0,
        help=(
            "Reserve battery in kWh. "
            "Effective usable battery is full battery minus reserve battery."
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
    parser.add_argument(
        "--no-EDF-NDF",
        "--no-edf-ndf",
        dest="no_edf_ndf",
        action="store_true",
        help=(
            "Skip EDF and NDF baseline heuristics in opt+heu runner. "
            "By default both are executed and written to results."
        ),
    )
    return parser


def parse_opt_heu_direct_args(cli_args=None):
    """Parse Opt+Heu arguments directly (standalone invocation)."""
    return build_opt_heu_parser().parse_args(cli_args)


def build_opt_heu_cli_args(main_args, opt_extra_args):
    """Build Opt+Heu CLI args from shared main args plus opt/heu extras."""
    cli_args = [
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
        "--reserve-battery",
        str(main_args.reserve_battery),
    ]
    if getattr(main_args, "test_for_opt_heu", None):
        cli_args.extend(["--test-for-opt-heu", str(main_args.test_for_opt_heu)])
    if getattr(main_args, "no_edf_ndf", False):
        cli_args.append("--no-EDF-NDF")
    cli_args.extend(opt_extra_args)
    return cli_args


def parse_opt_heu_args(main_args, opt_extra_args):
    """Parse Opt+Heu args synthesized from top-level args."""
    opt_heu_cli_args = build_opt_heu_cli_args(main_args, opt_extra_args)
    parser = build_opt_heu_parser()
    return parser.parse_args(opt_heu_cli_args), opt_heu_cli_args


def build_main_parser():
    """Create top-level parser used by `convoy_main.py`."""
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
        "--test-for-opt-heu",
        nargs="?",
        const="data/test_instance.csv",
        default=None,
        help=(
            "Use an existing test-instance CSV for opt+heu instead of sampling from "
            "--combined-details-csv. If set without value, uses data/test_instance.csv."
        ),
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
        "--reserve-battery",
        type=float,
        default=0.0,
        help=(
            "Reserve battery in kWh shared by RL and opt/heu. "
            "Effective usable battery is full battery minus reserve battery."
        ),
    )
    parser.add_argument(
        "--no-EDF-NDF",
        "--no-edf-ndf",
        dest="no_edf_ndf",
        action="store_true",
        help=(
            "Skip EDF and NDF baseline heuristics in opt+heu runner. "
            "By default both are executed."
        ),
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=1.0,
        help=(
            "RL-only charging-cost weight in reward. "
            "RL reward becomes customer_reward - cost_weight * charging_cost."
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
        "--clear-rl-checkpoints",
        action="store_true",
        help=(
            "If set, delete RL checkpoint directory before running. "
            "Uses --checkpoint-dir from --opt-rl-extra when provided; "
            "otherwise deletes CONVOY2/checkpoints_vrptw."
        ),
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help=(
            "Run EVRP baseline pipeline and append its metrics to the same results CSV. "
            "Uses the same EV count and matrix files as convoy_main."
        ),
    )
    parser.add_argument(
        "--baseline-bin",
        type=str,
        default="baseline/bin/evrp-tw-spd",
        help="Path to EVRP baseline solver binary.",
    )
    parser.add_argument(
        "--baseline-time",
        type=int,
        default=10,
        help="Baseline solver time limit in seconds.",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=5,
        help="Number of baseline solver runs.",
    )
    parser.add_argument(
        "--baseline-instance-output-path",
        type=str,
        default="baseline/data/test_instance_evrp.txt",
        help="Output path for converted EVRP instance file used by baseline.",
    )
    parser.add_argument(
        "--baseline-output-file",
        type=str,
        default="baseline/data/latest_baseline_output.txt",
        help="Stable copied baseline-output text file path.",
    )
    baseline_quiet_group = parser.add_mutually_exclusive_group()
    baseline_quiet_group.add_argument(
        "--baseline-quiet",
        dest="baseline_quiet",
        action="store_true",
        help=(
            "Redirect baseline solver stdout/stderr to --baseline-solver-log "
            "(default: on)."
        ),
    )
    baseline_quiet_group.add_argument(
        "--baseline-no-quiet",
        dest="baseline_quiet",
        action="store_false",
        help="Print baseline solver stdout/stderr to console.",
    )
    parser.add_argument(
        "--baseline-solver-log",
        type=str,
        default="baseline/data/baseline_solver.log",
        help="Solver log path used when --baseline-quiet is set.",
    )
    parser.set_defaults(baseline_quiet=True)
    parser.add_argument(
        "--baseline-print-charging-events",
        action="store_true",
        help="Print per-charging-event breakdown from baseline metric computation.",
    )
    parser.add_argument(
        "--baseline-print-route-trace",
        action="store_true",
        help=(
            "Print per-node baseline route trace "
            "(arrival/departure/wait/service/charge)."
        ),
    )
    parser.add_argument(
        "--baseline-extra",
        action="append",
        default=[],
        help=(
            "Optional quoted extra flags forwarded to baseline solver. "
            "Example: --baseline-extra \"--g_1 20 --pop_size 9\""
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
        "--skip-convoy-rl",
        action="store_true",
        help=(
            "Skip `convoy_rl_partial_ch` stage while still allowing "
            "`convoy_hybrid` (unless --only-opt-heu is set)."
        ),
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
    """Parse top-level combined-runner arguments."""
    return build_main_parser().parse_args()

#!/usr/bin/env python3
"""Run EVRP baseline pipeline end-to-end for one CONVOY2 test instance.

Steps:
1) Convert CONVOY2 test CSV to EVRP-TW-SPD-HMA instance format.
2) Run baseline solver binary.
3) Capture latest solver output into a stable file path.
4) Compute CONVOY-style reward/cost/objective from solver output.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
try:
    from .compute_baseline_metrics import compute_metrics
    from .convert_to_evrp_instance import convert_test_csv_to_evrp_instance
except ImportError:
    from compute_baseline_metrics import compute_metrics
    from convert_to_evrp_instance import convert_test_csv_to_evrp_instance


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    """Resolve a path relative to repo root when not absolute."""
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve(strict=False)


def _split_extra_args(chunks: List[str]) -> List[str]:
    """Split repeated quoted CLI chunks into flat token list."""
    out: List[str] = []
    for chunk in chunks:
        out.extend(shlex.split(chunk))
    return out


def _print_cmd(cmd: List[str]) -> None:
    """Print command in shell-like form for reproducibility."""
    print("$", " ".join(shlex.quote(part) for part in cmd))


def _read_log_tail(log_path: Path, max_lines: int = 40) -> str:
    """Read the last `max_lines` from a text log file."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _find_latest_solver_output(baseline_dir: Path, before_mtimes: Dict[Path, int]) -> Path:
    """Find newest baseline output file generated (or updated) by solver run."""
    pattern = "*timelimit=*subproblem=*.txt"
    candidates = list(baseline_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No baseline output files matching '{pattern}' found under {baseline_dir}"
        )

    changed = [
        p
        for p in candidates
        if p not in before_mtimes or p.stat().st_mtime_ns > before_mtimes[p]
    ]
    if changed:
        return max(changed, key=lambda p: p.stat().st_mtime_ns)
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _extract_latest_solution_block(raw_text: str) -> str:
    """Keep only the latest solver output block starting from 'Details of the solution:'."""
    lines = raw_text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Details of the solution:"):
            start_idx = idx
    if start_idx is None:
        return raw_text if raw_text.endswith("\n") else raw_text + "\n"
    trimmed = "\n".join(lines[start_idx:]).strip()
    return trimmed + "\n"


def run_baseline_pipeline(
    *,
    test_csv: str | Path,
    dist_matrix_csv: str | Path,
    time_matrix_csv: str | Path,
    vehicles: int,
    repo_root: str | Path | None = None,
    instance_output_path: str | Path = "baseline/data/test_instance_evrp.txt",
    latest_baseline_output: str | Path = "baseline/data/latest_baseline_output.txt",
    baseline_bin: str | Path = "baseline/bin/evrp-tw-spd",
    baseline_time: int = 10,
    baseline_runs: int = 5,
    baseline_extra: Optional[Sequence[str]] = None,
    ev_energy_rate_kwh_per_distance: float = 0.00025,
    dispatching_cost: float = 1000.0,
    unit_cost: float = 1.0,
    capacity: float = 200.0,
    electric_power: float = 30.0,
    consumption_rate: float = 0.00025,
    recharging_rate: Optional[float] = None,
    instance_name: Optional[str] = None,
    default_customer_delivery: float = 1.0,
    default_customer_pickup: float = 1.0,
    print_solver_cmd: bool = True,
    quiet_solver: bool = True,
    solver_log_path: str | Path | None = None,
) -> Dict[str, object]:
    """Run full baseline pipeline and return parsed CONVOY-style metrics.

    Pipeline:
    1) Convert `test_csv` + matrices to EVRP instance format.
    2) Execute baseline solver binary.
    3) Persist latest solver output into a stable output text file.
    4) Compute reward/cost/objective with depot top-up accounting.
    """
    repo_root_path = (
        Path(repo_root).expanduser().resolve(strict=False)
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )

    test_csv_path = _resolve_path(repo_root_path, test_csv)
    dist_csv_path = _resolve_path(repo_root_path, dist_matrix_csv)
    time_csv_path = _resolve_path(repo_root_path, time_matrix_csv)
    instance_output_path = _resolve_path(repo_root_path, instance_output_path)
    latest_baseline_output = _resolve_path(repo_root_path, latest_baseline_output)
    baseline_bin = _resolve_path(repo_root_path, baseline_bin)

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"test CSV not found: {test_csv_path}")
    if not dist_csv_path.is_file():
        raise FileNotFoundError(f"distance matrix CSV not found: {dist_csv_path}")
    if not time_csv_path.is_file():
        raise FileNotFoundError(f"time matrix CSV not found: {time_csv_path}")
    if not baseline_bin.is_file():
        raise FileNotFoundError(f"baseline solver binary not found: {baseline_bin}")
    if int(vehicles) <= 0:
        raise ValueError("vehicles must be > 0.")

    convert_result = convert_test_csv_to_evrp_instance(
        test_csv=test_csv_path,
        dist_matrix_csv=dist_csv_path,
        time_matrix_csv=time_csv_path,
        output_path=instance_output_path,
        vehicles=vehicles,
        dispatching_cost=dispatching_cost,
        unit_cost=unit_cost,
        capacity=capacity,
        electric_power=electric_power,
        consumption_rate=consumption_rate,
        recharging_rate=recharging_rate,
        instance_name=instance_name,
        default_customer_delivery=default_customer_delivery,
        default_customer_pickup=default_customer_pickup,
    )
    id_map_path = Path(convert_result["id_map_path"]).resolve(strict=False)
    instance_path = Path(convert_result["instance_path"]).resolve(strict=False)

    baseline_dir = baseline_bin.parent.parent
    output_pattern = "*timelimit=*subproblem=*.txt"
    before_mtimes = {p: p.stat().st_mtime_ns for p in baseline_dir.glob(output_pattern)}
    solver_cmd = [
        str(baseline_bin),
        "--problem",
        str(instance_path),
        "--pruning",
        "--time",
        str(baseline_time),
        "--runs",
        str(baseline_runs),
        "--related_removal",
        "--regret_insertion",
        *(list(baseline_extra) if baseline_extra else []),
    ]
    if print_solver_cmd:
        _print_cmd(solver_cmd)

    solver_log_file: Optional[Path] = None
    log_fp = None
    run_kwargs: Dict[str, object] = {"cwd": str(baseline_dir), "check": True}
    if quiet_solver:
        solver_log_file = _resolve_path(
            repo_root_path,
            solver_log_path if solver_log_path is not None else "baseline/data/baseline_solver.log",
        )
        solver_log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = solver_log_file.open("w", encoding="utf-8")
        run_kwargs["stdout"] = log_fp
        run_kwargs["stderr"] = subprocess.STDOUT

    start = time.perf_counter()
    try:
        try:
            subprocess.run(solver_cmd, **run_kwargs)
        except subprocess.CalledProcessError as exc:
            error_lines = [
                "Baseline solver execution failed.",
                "Command: {}".format(" ".join(shlex.quote(part) for part in solver_cmd)),
                f"Return code: {exc.returncode}",
            ]
            if exc.returncode < 0:
                signum = -int(exc.returncode)
                try:
                    sig_name = signal.Signals(signum).name
                except ValueError:
                    sig_name = "UNKNOWN"
                error_lines.append(
                    f"Terminated by signal: {sig_name} ({signum})"
                )
            if solver_log_file is not None and solver_log_file.is_file():
                tail = _read_log_tail(solver_log_file, max_lines=40)
                if tail:
                    error_lines.append(
                        "Last 40 lines from solver log ({}):\n{}".format(
                            solver_log_file, tail
                        )
                    )
                else:
                    error_lines.append(
                        f"Solver log file exists but is empty: {solver_log_file}"
                    )
            raise RuntimeError("\n".join(error_lines)) from exc
    finally:
        if log_fp is not None:
            log_fp.close()
    elapsed_time_ms = (time.perf_counter() - start) * 1000.0

    raw_output_path = _find_latest_solver_output(baseline_dir, before_mtimes)
    latest_baseline_output.parent.mkdir(parents=True, exist_ok=True)
    raw_text = raw_output_path.read_text(encoding="utf-8")
    latest_only_text = _extract_latest_solution_block(raw_text)
    latest_baseline_output.write_text(latest_only_text, encoding="utf-8")

    metrics = compute_metrics(
        baseline_output_file=latest_baseline_output,
        test_instance_csv=test_csv_path,
        id_map_csv=id_map_path,
        ev_energy_rate_kwh_per_distance=ev_energy_rate_kwh_per_distance,
        max_vehicles=int(vehicles),
        time_matrix_csv=time_csv_path,
        dist_matrix_csv=dist_csv_path,
        instance_txt=instance_path,
    )

    solver_time_ms = None
    run_summary = metrics.get("baseline_run_summary")
    if isinstance(run_summary, tuple) and len(run_summary) >= 2:
        try:
            solver_time_ms = float(run_summary[1]) * 1000.0
        except (TypeError, ValueError):
            solver_time_ms = None

    out: Dict[str, object] = dict(metrics)
    out.update(
        {
            "instance_path": str(instance_path),
            "id_map_path": str(id_map_path),
            "raw_output_path": str(raw_output_path),
            "stable_output_path": str(latest_baseline_output),
            "elapsed_time_ms": elapsed_time_ms,
            "solver_time_ms": solver_time_ms,
        }
    )
    if solver_log_file is not None:
        out["solver_log_path"] = str(solver_log_file)
    return out


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone baseline pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run convert -> baseline solve -> metric computation in one command."
    )
    parser.add_argument("--test-csv", default="data/test_instance.csv")
    parser.add_argument(
        "--dist-matrix-csv",
        default="data/distance_matrix_jd200_1.csv",
    )
    parser.add_argument(
        "--time-matrix-csv",
        default="data/time_matrix_jd200_1.csv",
    )
    parser.add_argument("--vehicles", type=int, required=True)

    parser.add_argument(
        "--instance-output-path",
        default="baseline/data/test_instance_evrp.txt",
        help="EVRP instance output path.",
    )
    parser.add_argument(
        "--latest-baseline-output",
        default="baseline/data/latest_baseline_output.txt",
        help="Stable copied baseline-output file path used for metric computation.",
    )
    parser.add_argument(
        "--baseline-bin",
        default="baseline/bin/evrp-tw-spd",
        help="Path to EVRP-TW-SPD-HMA solver binary.",
    )
    parser.add_argument("--baseline-time", type=int, default=10)
    parser.add_argument("--baseline-runs", type=int, default=5)
    parser.add_argument(
        "--baseline-extra",
        action="append",
        default=[],
        help=(
            "Optional quoted extra flags forwarded to baseline solver. "
            "Example: --baseline-extra \"--g_1 20 --pop_size 9\""
        ),
    )
    quiet_group = parser.add_mutually_exclusive_group()
    quiet_group.add_argument(
        "--baseline-quiet",
        dest="baseline_quiet",
        action="store_true",
        help="Redirect baseline solver stdout/stderr to --solver-log-path (default: on).",
    )
    quiet_group.add_argument(
        "--baseline-no-quiet",
        dest="baseline_quiet",
        action="store_false",
        help="Print baseline solver stdout/stderr to console.",
    )
    parser.add_argument(
        "--solver-log-path",
        default="baseline/data/baseline_solver.log",
        help="Log file path used when --baseline-quiet is enabled.",
    )
    parser.set_defaults(baseline_quiet=True)

    parser.add_argument("--ev-energy-rate-kwh-per-distance", type=float, default=0.00025)
    parser.add_argument("--print-charging-events", action="store_true")
    parser.add_argument(
        "--print-route-trace",
        action="store_true",
        help="Print per-node baseline route trace (arrival/departure/wait/service/charge).",
    )

    # Converter options
    parser.add_argument("--dispatching-cost", type=float, default=1000.0)
    parser.add_argument("--unit-cost", type=float, default=1.0)
    parser.add_argument("--capacity", type=float, default=200.0)
    parser.add_argument("--electric-power", type=float, default=30.0)
    parser.add_argument("--consumption-rate", type=float, default=0.00025)
    parser.add_argument("--recharging-rate", type=float, default=None)
    parser.add_argument("--instance-name", default=None)
    parser.add_argument("--default-customer-delivery", type=float, default=1.0)
    parser.add_argument("--default-customer-pickup", type=float, default=1.0)
    return parser


def main() -> None:
    """CLI entrypoint for one-shot baseline pipeline."""
    args = _build_parser().parse_args()
    metrics = run_baseline_pipeline(
        test_csv=args.test_csv,
        dist_matrix_csv=args.dist_matrix_csv,
        time_matrix_csv=args.time_matrix_csv,
        vehicles=args.vehicles,
        instance_output_path=args.instance_output_path,
        latest_baseline_output=args.latest_baseline_output,
        baseline_bin=args.baseline_bin,
        baseline_time=args.baseline_time,
        baseline_runs=args.baseline_runs,
        baseline_extra=_split_extra_args(args.baseline_extra),
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        dispatching_cost=args.dispatching_cost,
        unit_cost=args.unit_cost,
        capacity=args.capacity,
        electric_power=args.electric_power,
        consumption_rate=args.consumption_rate,
        recharging_rate=args.recharging_rate,
        instance_name=args.instance_name,
        default_customer_delivery=args.default_customer_delivery,
        default_customer_pickup=args.default_customer_pickup,
        quiet_solver=args.baseline_quiet,
        solver_log_path=args.solver_log_path,
    )

    print(f"Wrote EVRP instance: {metrics['instance_path']}")
    print(f"Wrote ID mapping   : {metrics['id_map_path']}")
    print(f"Copied baseline output: {metrics['raw_output_path']}")
    print(f"Saved stable output   : {metrics['stable_output_path']}")

    print(f"Total reward: {metrics['total_reward']:.6f}")
    print(f"Total cost: {metrics['total_charging_cost']:.6f}")
    print(f"Objective val: {metrics['objective_score']:.6f}")
    print(f"Visited customers: {metrics['visited_customer_count']}")
    print(
        "Reward components: "
        f"total_reward={float(metrics['total_reward']):.6f}, "
        f"total_cost={float(metrics['total_charging_cost']):.6f}, "
        f"successful_delivery={int(metrics['visited_customer_count'])}, "
        f"objective={float(metrics['objective_score']):.6f}"
    )

    baseline_total_cost_reported = metrics["baseline_total_cost_reported"]
    if baseline_total_cost_reported is not None:
        print(f"Baseline reported total cost: {baseline_total_cost_reported:.6f}")

    run_summary = metrics["baseline_run_summary"]
    if run_summary is not None:
        print(f"Baseline run summary (cost,time): {run_summary[0]:.6f}, {run_summary[1]:.6f}")

    if "solver_log_path" in metrics:
        print(f"Baseline solver log: {metrics['solver_log_path']}")

    if args.print_charging_events:
        print("Charging events:")
        events = metrics["charging_events"]
        if not events:
            print("  (none)")
        for event in events:
            print(
                "  route={route} mapped_id={mapped_id} original_id={original_id} "
                "type={node_type} arr_rd={arr_rd:.2f} dep_rd={dep_rd:.2f} "
                "charged_dist={charged_distance_units:.2f} charged_kwh={charged_energy_kwh:.6f} "
                "unit_cost={unit_charging_cost:.6f} cost={charging_cost:.6f}".format(**event)
            )

    if args.print_route_trace:
        print("Route trace:")
        route_traces = metrics.get("route_traces", [])
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
                    "    step={step:>2d} mapped={mapped_id:>3d} original={original_id:>3d} "
                    "type={node_type} arr={arrival_time:.2f} dep={departure_time:.2f} "
                    "travel_t={travel_time_from_prev:.2f} travel_d={travel_d} "
                    "energy={energy} soc_arr={soc_arr} soc_dep={soc_dep} "
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


if __name__ == "__main__":
    main()

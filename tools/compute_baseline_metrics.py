#!/usr/bin/env python3
"""Compute CONVOY-style reward/cost/objective from EVRP-TW-SPD-HMA output."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class NodeInfo:
    """Minimal node metadata from test instance used for scoring."""
    node_type: str
    reward: float
    unit_charging_cost: float
    tw_start: float
    tw_end: float
    service_time: float


@dataclass
class VisitToken:
    """One parsed token from baseline route output."""
    mapped_id: int
    arr_rd: Optional[float]
    dep_rd: Optional[float]


ROUTE_LINE_RE = re.compile(
    r"^route\s+\d+,\s*node_num\s+\d+,\s*cost\s+([0-9eE+\-\.]+),\s*nodes:\s*(.*)$"
)
NODE_TOKEN_RE = re.compile(
    r"(?P<id>-?\d+)(?:\((?P<arr>-?\d+(?:\.\d+)?),\s*(?P<dep>-?\d+(?:\.\d+)?)\))?"
)
RUN_SUMMARY_RE = re.compile(
    r"^\s*([0-9eE+\-\.]+)\s*,\s*([0-9eE+\-\.]+)\s*$"
)


def _to_int(value: str) -> int:
    """Parse integer-like text, allowing float-formatted integer strings."""
    text = value.strip()
    if text == "":
        raise ValueError("Empty integer value")
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def _to_float(value: str, default: float = 0.0) -> float:
    """Parse float-like text with fallback default for empty values."""
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def _is_float(value: str) -> bool:
    """Return True if text is parseable as float."""
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def _load_test_instance_info(test_csv: Path) -> Dict[int, NodeInfo]:
    """Load node type/reward/unit charging cost from CONVOY test-instance CSV."""
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ID", "type"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {test_csv}: {missing}")

        info: Dict[int, NodeInfo] = {}
        for row in reader:
            oid = _to_int(row["ID"])
            node_type = (row.get("type") or "").strip().lower()
            reward = _to_float(row.get("reward", "0"), default=0.0)
            unit_cost = _to_float(row.get("unit_charging_cost", "0"), default=0.0)
            if "first_receive_tm" in row and "last_receive_tm" in row:
                tw_start = _to_float(row.get("first_receive_tm", "0"), default=0.0)
                tw_end = _to_float(row.get("last_receive_tm", "0"), default=float("inf"))
            else:
                tw_start = _to_float(row.get("tw_start", "0"), default=0.0)
                tw_end = _to_float(row.get("tw_end", ""), default=float("inf"))
            service_time = _to_float(row.get("service_time", "0"), default=0.0)
            info[oid] = NodeInfo(
                node_type=node_type,
                reward=reward,
                unit_charging_cost=unit_cost,
                tw_start=tw_start,
                tw_end=tw_end,
                service_time=service_time,
            )
    return info


def _load_id_map(id_map_csv: Path) -> Dict[int, int]:
    """Load baseline mapped-id to original-id conversion table."""
    with id_map_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"mapped_id", "original_id"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {id_map_csv}: {missing}")

        id_map: Dict[int, int] = {}
        for row in reader:
            mid = _to_int(row["mapped_id"])
            oid = _to_int(row["original_id"])
            id_map[mid] = oid
    return id_map


def _parse_baseline_output(output_path: Path) -> Tuple[List[List[VisitToken]], Optional[float], Optional[Tuple[float, float]]]:
    """Parse route blocks and summary values from baseline text output."""
    routes: List[List[VisitToken]] = []
    total_cost_reported: Optional[float] = None
    run_summary: Optional[Tuple[float, float]] = None

    with output_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Total cost:"):
                total_cost_reported = float(line.split(":", 1)[1].strip())
                continue

            route_match = ROUTE_LINE_RE.match(line)
            if route_match:
                nodes_str = route_match.group(2)
                visit_tokens: List[VisitToken] = []
                for node_match in NODE_TOKEN_RE.finditer(nodes_str):
                    mapped_id = int(node_match.group("id"))
                    arr_rd = node_match.group("arr")
                    dep_rd = node_match.group("dep")
                    visit_tokens.append(
                        VisitToken(
                            mapped_id=mapped_id,
                            arr_rd=float(arr_rd) if arr_rd is not None else None,
                            dep_rd=float(dep_rd) if dep_rd is not None else None,
                        )
                    )
                routes.append(visit_tokens)
                continue

            summary_match = RUN_SUMMARY_RE.match(line)
            if summary_match:
                run_summary = (float(summary_match.group(1)), float(summary_match.group(2)))

    if not routes:
        raise ValueError(f"No route lines found in baseline output: {output_path}")

    return routes, total_cost_reported, run_summary


def _parse_instance_rates(instance_txt: Path) -> Tuple[Optional[float], Optional[float]]:
    """Parse consumption/recharging rates from EVRP instance header."""
    consumption_rate: Optional[float] = None
    recharging_rate: Optional[float] = None
    with instance_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NODE_SECTION"):
                break
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip().upper()
            val = parts[1].strip()
            if key == "CONSUMPTION_RATE":
                consumption_rate = _to_float(val)
            elif key == "RECHARGING_RATE":
                recharging_rate = _to_float(val)
    return consumption_rate, recharging_rate


def _load_matrix_by_id(matrix_csv: Path) -> Dict[int, Dict[int, float]]:
    """Load matrix CSV keyed by integer node IDs.

    Supports:
    - ID-labeled matrix (header row + first ID column),
    - plain square matrix (IDs implied as 0..n-1).
    """
    with matrix_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_rows = [[c.strip() for c in row] for row in reader if row and any(c.strip() for c in row)]

    if not raw_rows:
        raise ValueError(f"Matrix CSV is empty: {matrix_csv}")

    width = len(raw_rows[0])
    if any(len(r) != width for r in raw_rows):
        raise ValueError(f"Matrix CSV must be rectangular: {matrix_csv}")

    has_id_header = not _is_float(raw_rows[0][0])
    matrix: Dict[int, Dict[int, float]] = {}

    if not has_id_header:
        n = len(raw_rows)
        if width != n:
            raise ValueError(f"Plain matrix must be square: {matrix_csv}")
        for i, row in enumerate(raw_rows):
            matrix[i] = {j: float(v) for j, v in enumerate(row)}
        return matrix

    col_ids = [_to_int(x) for x in raw_rows[0][1:]]
    if len(raw_rows) - 1 != len(col_ids):
        raise ValueError(f"ID-labeled matrix must be square: {matrix_csv}")

    for row in raw_rows[1:]:
        row_id = _to_int(row[0])
        vals = [float(v) for v in row[1:]]
        if len(vals) != len(col_ids):
            raise ValueError(f"Inconsistent row length in matrix: {matrix_csv}")
        matrix[row_id] = {cid: vals[j] for j, cid in enumerate(col_ids)}
    return matrix


def _build_route_traces(
    routes: List[List[VisitToken]],
    id_map: Dict[int, int],
    node_info: Dict[int, NodeInfo],
    time_matrix_by_id: Dict[int, Dict[int, float]],
    baseline_consumption_rate: float,
    baseline_recharging_rate: float,
    ev_energy_rate_kwh_per_distance: float,
    dist_matrix_by_id: Optional[Dict[int, Dict[int, float]]] = None,
) -> List[List[Dict[str, object]]]:
    """Build per-route trace with timing and energy/SOC details."""
    route_traces: List[List[Dict[str, object]]] = []

    for route_idx, route_tokens in enumerate(routes):
        if not route_tokens:
            continue

        trace_rows: List[Dict[str, object]] = []
        first = route_tokens[0]
        first_original_id = id_map.get(first.mapped_id, first.mapped_id)
        first_info = node_info.get(first_original_id)
        if first_info is None:
            route_traces.append(trace_rows)
            continue

        # Baseline routes are depot-anchored; initialize at depot TW start.
        route_full_rd = max(
            (
                v
                for tok in route_tokens
                for v in (tok.arr_rd, tok.dep_rd)
                if v is not None
            ),
            default=None,
        )
        full_battery_kwh: Optional[float] = (
            float(route_full_rd) * ev_energy_rate_kwh_per_distance
            if route_full_rd is not None
            else None
        )
        current_time = max(0.0, first_info.tw_start)
        first_charge_rd = 0.0
        first_charge_time = 0.0
        if first.arr_rd is not None and first.dep_rd is not None:
            first_charge_rd = max(first.dep_rd - first.arr_rd, 0.0)
            first_charge_time = (
                first_charge_rd
                * baseline_consumption_rate
                * baseline_recharging_rate
            )
        first_charge_energy_kwh = first_charge_rd * ev_energy_rate_kwh_per_distance
        first_soc_arrival_kwh: Optional[float] = None
        if first.arr_rd is not None:
            first_soc_arrival_kwh = first.arr_rd * ev_energy_rate_kwh_per_distance
        elif full_battery_kwh is not None:
            first_soc_arrival_kwh = full_battery_kwh
        first_soc_departure_kwh: Optional[float] = None
        if first.dep_rd is not None:
            first_soc_departure_kwh = first.dep_rd * ev_energy_rate_kwh_per_distance
        elif first_soc_arrival_kwh is not None:
            first_soc_departure_kwh = first_soc_arrival_kwh + first_charge_energy_kwh
        if full_battery_kwh is not None and first_soc_departure_kwh is not None:
            first_soc_departure_kwh = min(max(first_soc_departure_kwh, 0.0), full_battery_kwh)
        first_departure = current_time + first_info.service_time + first_charge_time
        trace_rows.append(
            {
                "route": route_idx,
                "step": 0,
                "mapped_id": first.mapped_id,
                "original_id": first_original_id,
                "node_type": first_info.node_type,
                "arrival_time": current_time,
                "departure_time": first_departure,
                "travel_time_from_prev": 0.0,
                "travel_distance_from_prev": 0.0,
                "energy_used_kwh_from_prev": 0.0,
                "wait_time": 0.0,
                "service_time": first_info.service_time,
                "charge_rd": first_charge_rd,
                "charge_energy_kwh": first_charge_energy_kwh,
                "charge_time": first_charge_time,
                "soc_arrival_kwh": first_soc_arrival_kwh,
                "soc_departure_kwh": first_soc_departure_kwh,
                "arr_rd": first.arr_rd,
                "dep_rd": first.dep_rd,
                "tw_start": first_info.tw_start,
                "tw_end": first_info.tw_end,
                "on_time": True,
            }
        )
        current_time = first_departure
        current_soc_kwh: Optional[float] = first_soc_departure_kwh
        prev_original_id = first_original_id

        for step_idx, token in enumerate(route_tokens[1:], start=1):
            original_id = id_map.get(token.mapped_id, token.mapped_id)
            info = node_info.get(original_id)
            if info is None:
                prev_original_id = original_id
                continue

            travel_time = time_matrix_by_id.get(prev_original_id, {}).get(original_id)
            if travel_time is None:
                raise KeyError(
                    f"Missing time-matrix edge: {prev_original_id} -> {original_id}"
                )
            travel_distance: Optional[float] = None
            if dist_matrix_by_id is not None:
                travel_distance = dist_matrix_by_id.get(prev_original_id, {}).get(original_id)
                if travel_distance is None:
                    raise KeyError(
                        f"Missing distance-matrix edge: {prev_original_id} -> {original_id}"
                    )
            energy_used_kwh: Optional[float] = None
            if travel_distance is not None:
                energy_used_kwh = travel_distance * ev_energy_rate_kwh_per_distance
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, info.tw_start)
            wait_time = max(info.tw_start - arrival_time, 0.0)
            soc_arrival_kwh: Optional[float] = None
            if current_soc_kwh is not None and energy_used_kwh is not None:
                soc_arrival_kwh = max(current_soc_kwh - energy_used_kwh, 0.0)
            elif token.arr_rd is not None:
                soc_arrival_kwh = token.arr_rd * ev_energy_rate_kwh_per_distance

            charge_rd = 0.0
            charge_time = 0.0
            if (
                token.arr_rd is not None
                and token.dep_rd is not None
                and info.node_type in {"d", "f"}
            ):
                charge_rd = max(token.dep_rd - token.arr_rd, 0.0)
                charge_time = (
                    charge_rd
                    * baseline_consumption_rate
                    * baseline_recharging_rate
                )
            charge_energy_kwh = charge_rd * ev_energy_rate_kwh_per_distance
            soc_departure_kwh: Optional[float] = soc_arrival_kwh
            if token.dep_rd is not None:
                soc_departure_kwh = token.dep_rd * ev_energy_rate_kwh_per_distance
            elif soc_arrival_kwh is not None:
                soc_departure_kwh = soc_arrival_kwh + charge_energy_kwh
            if full_battery_kwh is not None and soc_departure_kwh is not None:
                soc_departure_kwh = min(max(soc_departure_kwh, 0.0), full_battery_kwh)

            departure_time = service_start + info.service_time + charge_time
            on_time = service_start <= (info.tw_end + 1e-9)
            trace_rows.append(
                {
                    "route": route_idx,
                    "step": step_idx,
                    "mapped_id": token.mapped_id,
                    "original_id": original_id,
                    "node_type": info.node_type,
                    "arrival_time": arrival_time,
                    "departure_time": departure_time,
                    "travel_time_from_prev": travel_time,
                    "travel_distance_from_prev": travel_distance,
                    "energy_used_kwh_from_prev": energy_used_kwh,
                    "wait_time": wait_time,
                    "service_time": info.service_time,
                    "charge_rd": charge_rd,
                    "charge_energy_kwh": charge_energy_kwh,
                    "charge_time": charge_time,
                    "soc_arrival_kwh": soc_arrival_kwh,
                    "soc_departure_kwh": soc_departure_kwh,
                    "arr_rd": token.arr_rd,
                    "dep_rd": token.dep_rd,
                    "tw_start": info.tw_start,
                    "tw_end": info.tw_end,
                    "on_time": on_time,
                }
            )
            current_time = departure_time
            current_soc_kwh = soc_departure_kwh
            prev_original_id = original_id

        route_traces.append(trace_rows)

    return route_traces


def compute_metrics(
    baseline_output_file: Path,
    test_instance_csv: Path,
    id_map_csv: Path,
    ev_energy_rate_kwh_per_distance: float,
    max_vehicles: Optional[int] = None,
    time_matrix_csv: Optional[Path] = None,
    dist_matrix_csv: Optional[Path] = None,
    instance_txt: Optional[Path] = None,
    baseline_consumption_rate: Optional[float] = None,
    baseline_recharging_rate: Optional[float] = None,
) -> Dict[str, object]:
    """Compute CONVOY-style reward/cost/objective from baseline output.

    Charging cost includes:
    - explicit charge events at depot/CP from baseline `(arr_RD, dep_RD)`,
    - final depot recharge-to-full when missing in explicit event values.
    - only first `max_vehicles` routes when the cap is provided.
    """
    node_info = _load_test_instance_info(test_instance_csv)
    id_map = _load_id_map(id_map_csv)
    routes, total_cost_reported, run_summary = _parse_baseline_output(baseline_output_file)
    if max_vehicles is not None and max_vehicles <= 0:
        raise ValueError("max_vehicles must be > 0 when provided.")

    if max_vehicles is None:
        scored_routes = routes
    else:
        scored_routes = routes[:max_vehicles]

    visited_customers: set[int] = set()
    total_reward = 0.0

    total_charging_cost = 0.0
    charging_events: List[Dict[str, object]] = []
    route_traces: List[List[Dict[str, object]]] = []
    eps = 1e-9

    for route_idx, route_tokens in enumerate(scored_routes):
        route_full_rd: Optional[float] = None
        final_depot_visit: Optional[Tuple[int, int, NodeInfo, float]] = None

        for token in route_tokens:
            original_id = id_map.get(token.mapped_id, token.mapped_id)
            info = node_info.get(original_id)
            if info is None:
                continue

            if info.node_type == "c":
                if original_id not in visited_customers:
                    visited_customers.add(original_id)
                    total_reward += info.reward

            if token.arr_rd is None or token.dep_rd is None:
                continue

            if info.node_type == "d":
                if route_full_rd is None:
                    route_full_rd = token.dep_rd
                final_depot_visit = (token.mapped_id, original_id, info, token.dep_rd)

            if info.node_type not in {"d", "f"}:
                continue

            charged_distance_units = max(token.dep_rd - token.arr_rd, 0.0)
            if charged_distance_units <= eps:
                continue

            charged_energy_kwh = charged_distance_units * ev_energy_rate_kwh_per_distance
            event_cost = charged_energy_kwh * info.unit_charging_cost
            total_charging_cost += event_cost

            charging_events.append(
                {
                    "route": route_idx,
                    "mapped_id": token.mapped_id,
                    "original_id": original_id,
                    "node_type": info.node_type,
                    "arr_rd": token.arr_rd,
                    "dep_rd": token.dep_rd,
                    "charged_distance_units": charged_distance_units,
                    "charged_energy_kwh": charged_energy_kwh,
                    "unit_charging_cost": info.unit_charging_cost,
                    "charging_cost": event_cost,
                }
            )

        # Baseline output may end at depot without explicit final top-up to full.
        # Add the missing depot recharge (if any) so cost aligns with CONVOY MILP/heuristic accounting.
        if route_full_rd is not None and final_depot_visit is not None:
            mapped_id, original_id, depot_info, final_dep_rd = final_depot_visit
            topup_distance_units = max(route_full_rd - final_dep_rd, 0.0)
            if topup_distance_units > eps:
                topup_energy_kwh = topup_distance_units * ev_energy_rate_kwh_per_distance
                topup_cost = topup_energy_kwh * depot_info.unit_charging_cost
                total_charging_cost += topup_cost
                charging_events.append(
                    {
                        "route": route_idx,
                        "mapped_id": mapped_id,
                        "original_id": original_id,
                        "node_type": depot_info.node_type,
                        "arr_rd": final_dep_rd,
                        "dep_rd": route_full_rd,
                        "charged_distance_units": topup_distance_units,
                        "charged_energy_kwh": topup_energy_kwh,
                        "unit_charging_cost": depot_info.unit_charging_cost,
                        "charging_cost": topup_cost,
                    }
                )

    objective_score = total_reward - total_charging_cost

    # Optional route-time trace reconstruction for detailed debugging.
    if baseline_consumption_rate is None or baseline_recharging_rate is None:
        if instance_txt is not None and instance_txt.is_file():
            parsed_cons, parsed_rech = _parse_instance_rates(instance_txt)
            if baseline_consumption_rate is None:
                baseline_consumption_rate = parsed_cons
            if baseline_recharging_rate is None:
                baseline_recharging_rate = parsed_rech

    if (
        time_matrix_csv is not None
        and baseline_consumption_rate is not None
        and baseline_recharging_rate is not None
    ):
        time_matrix_by_id = _load_matrix_by_id(time_matrix_csv)
        dist_matrix_by_id = (
            _load_matrix_by_id(dist_matrix_csv) if dist_matrix_csv is not None else None
        )
        route_traces = _build_route_traces(
            routes=scored_routes,
            id_map=id_map,
            node_info=node_info,
            time_matrix_by_id=time_matrix_by_id,
            baseline_consumption_rate=float(baseline_consumption_rate),
            baseline_recharging_rate=float(baseline_recharging_rate),
            ev_energy_rate_kwh_per_distance=float(ev_energy_rate_kwh_per_distance),
            dist_matrix_by_id=dist_matrix_by_id,
        )

    return {
        "total_reward": total_reward,
        "total_charging_cost": total_charging_cost,
        "objective_score": objective_score,
        "visited_customers": sorted(visited_customers),
        "visited_customer_count": len(visited_customers),
        "charging_events": charging_events,
        "baseline_total_cost_reported": total_cost_reported,
        "baseline_run_summary": run_summary,
        "baseline_route_count_reported": len(routes),
        "baseline_route_count_scored": len(scored_routes),
        "baseline_max_vehicles_scored": max_vehicles,
        "route_traces": route_traces,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone baseline metric computation."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute reward, charging cost, and objective score from EVRP-TW-SPD-HMA output "
            "using CONVOY test_instance reward/cost columns."
        )
    )
    parser.add_argument("--baseline-output-file", required=True, help="Path to baseline output text file")
    parser.add_argument("--test-instance-csv", required=True, help="Path to test_instance.csv used for rewards/costs")
    parser.add_argument(
        "--id-map-csv",
        required=True,
        help="Path to mapping file produced by converter (e.g., baseline/data/test_instance_evrp.id_map.csv)",
    )
    parser.add_argument(
        "--ev-energy-rate-kwh-per-distance",
        type=float,
        default=0.00025,
        help="Energy consumption rate used to convert charged distance units into kWh",
    )
    parser.add_argument(
        "--time-matrix-csv",
        default=None,
        help="Optional time-matrix CSV used to print per-node arrival/departure trace.",
    )
    parser.add_argument(
        "--dist-matrix-csv",
        default=None,
        help="Optional distance-matrix CSV used for per-step distance/SOC fields in route trace.",
    )
    parser.add_argument(
        "--instance-file",
        default=None,
        help=(
            "Optional EVRP instance text file (for CONSUMPTION_RATE/RECHARGING_RATE) "
            "used in route-trace reconstruction."
        ),
    )
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=None,
        help=(
            "If set, compute metrics using only the first N baseline routes "
            "(vehicle routes) in solver output."
        ),
    )
    parser.add_argument(
        "--print-charging-events",
        action="store_true",
        help="Print per-charging-event breakdown",
    )
    parser.add_argument(
        "--print-route-trace",
        action="store_true",
        help="Print per-node route trace (arrival/departure/wait/service/charge).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for baseline metric computation."""
    args = _build_parser().parse_args()

    metrics = compute_metrics(
        baseline_output_file=Path(args.baseline_output_file),
        test_instance_csv=Path(args.test_instance_csv),
        id_map_csv=Path(args.id_map_csv),
        ev_energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
        max_vehicles=args.max_vehicles,
        time_matrix_csv=Path(args.time_matrix_csv) if args.time_matrix_csv else None,
        dist_matrix_csv=Path(args.dist_matrix_csv) if args.dist_matrix_csv else None,
        instance_txt=Path(args.instance_file) if args.instance_file else None,
    )

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
    print(
        "Routes (reported/scored): "
        f"{metrics['baseline_route_count_reported']}/"
        f"{metrics['baseline_route_count_scored']}"
    )

    if args.print_charging_events:
        print("Charging events:")
        events: List[Dict[str, object]] = metrics["charging_events"]  # type: ignore[assignment]
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
        route_traces: List[List[Dict[str, object]]] = metrics.get("route_traces", [])  # type: ignore[assignment]
        if not route_traces:
            print("  (trace unavailable; provide --time-matrix-csv and --instance-file)")
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

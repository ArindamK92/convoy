"""CP insertion postprocessing for decoded hybrid routes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.convoy_rl_partial_ch.myparser import parse_customer, parse_distance_matrix_csv

from convoy_hybrid.convoy_hybrid_kdtree import create_kd_tree, find_nearest_cp


@dataclass
class NodeInfo:
    """Basic node metadata used in trace reconstruction."""

    node_id: int
    node_type: str
    x: float
    y: float
    tw_start: float
    tw_end: float
    service_time: float
    reward: float
    charging_cost_per_kwh: float
    charge_rate_kwh_per_hour: float


def _as_int(v, default=0):
    if v is None:
        return int(default)
    return int(float(v))


def _build_node_info_from_test_csv(csv_path: str, default_charge_rate: float) -> tuple[dict[int, NodeInfo], list[dict], int]:
    depot, customers, charging_points = parse_customer(csv_path)

    depot_id = _as_int(depot.get("node_id", 0), 0)
    node_info: dict[int, NodeInfo] = {
        depot_id: NodeInfo(
            node_id=depot_id,
            node_type="depot",
            x=float(depot["x"]),
            y=float(depot["y"]),
            tw_start=float(depot["tw_start"]),
            tw_end=float(depot["tw_end"]),
            service_time=float(depot.get("service_time", 0.0)),
            reward=float(depot.get("reward", 0.0)),
            charging_cost_per_kwh=float(depot.get("charging_cost_per_kwh", 0.0)),
            charge_rate_kwh_per_hour=float(default_charge_rate),
        )
    }

    for c in customers:
        cid = _as_int(c.get("customer_id"), -1)
        if cid < 0:
            continue
        node_info[cid] = NodeInfo(
            node_id=cid,
            node_type="customer",
            x=float(c["x"]),
            y=float(c["y"]),
            tw_start=float(c["tw_start"]),
            tw_end=float(c["tw_end"]),
            service_time=float(c.get("service_time", 0.0)),
            reward=float(c.get("reward", 0.0)),
            charging_cost_per_kwh=0.0,
            charge_rate_kwh_per_hour=0.0,
        )

    cp_rows = []
    for cp in charging_points:
        cp_id = _as_int(cp.get("cp_id"), -1)
        if cp_id < 0:
            continue
        cp_rate = float(cp.get("charge_rate_kwh_per_hour", default_charge_rate))
        cp_row = {
            "cp_id": cp_id,
            "x": float(cp["x"]),
            "y": float(cp["y"]),
            "charging_cost_per_kwh": float(cp.get("charging_cost_per_kwh", 0.0)),
            "charge_rate_kwh_per_hour": cp_rate,
            "tw_start": float(cp.get("tw_start", 0.0)),
            "tw_end": float(cp.get("tw_end", 1e9)),
            "service_time": float(cp.get("service_time", 0.0)),
        }
        cp_rows.append(cp_row)
        node_info[cp_id] = NodeInfo(
            node_id=cp_id,
            node_type="cp",
            x=cp_row["x"],
            y=cp_row["y"],
            tw_start=cp_row["tw_start"],
            tw_end=cp_row["tw_end"],
            service_time=cp_row["service_time"],
            reward=0.0,
            charging_cost_per_kwh=float(cp_row["charging_cost_per_kwh"]),
            charge_rate_kwh_per_hour=cp_rate,
        )

    return node_info, cp_rows, depot_id


def _to_global_routes(routes_local: list[list[int]], global_ids: torch.Tensor) -> list[list[int]]:
    out: list[list[int]] = []
    for route in routes_local:
        converted = []
        for node_local in route:
            node_local = int(node_local)
            if 0 <= node_local < int(global_ids.shape[0]):
                converted.append(int(global_ids[node_local].item()))
            else:
                converted.append(node_local)
        out.append(converted)
    return out


def _get_matrix_value(matrix: torch.Tensor, from_id: int, to_id: int) -> float:
    if from_id < 0 or to_id < 0:
        return float("inf")
    if from_id >= int(matrix.shape[0]) or to_id >= int(matrix.shape[1]):
        return float("inf")
    return float(matrix[from_id, to_id].item())


def _append_leg(
    trace: list[dict],
    route_aug: list[int],
    node_info: dict[int, NodeInfo],
    from_id: int,
    to_id: int,
    current_soc: float,
    current_time: float,
    effective_battery_kwh: float,
    energy_rate: float,
    default_charge_rate_kwh_per_hour: float,
    dist_full: torch.Tensor,
    time_full: torch.Tensor,
    step: int,
    inserted: bool,
) -> tuple[float, float, int]:
    dist = _get_matrix_value(dist_full, from_id, to_id)
    travel_t = _get_matrix_value(time_full, from_id, to_id)
    energy = max(0.0, dist * energy_rate)

    arr = current_time + travel_t
    ninfo = node_info[to_id]
    service_start = max(arr, ninfo.tw_start)
    wait_t = max(service_start - arr, 0.0)

    soc_deficit = max(energy - current_soc, 0.0)
    soc_arr = max(current_soc - energy, 0.0)

    if ninfo.node_type in {"depot", "cp"}:
        rate = max(
            float(ninfo.charge_rate_kwh_per_hour or default_charge_rate_kwh_per_hour),
            1e-6,
        )
        charge_needed = max(effective_battery_kwh - soc_arr, 0.0)
        charge_t = (charge_needed / rate) * 60.0
        depart = service_start + charge_t
        soc_dep = effective_battery_kwh
    else:
        charge_needed = 0.0
        charge_t = 0.0
        depart = service_start + float(ninfo.service_time)
        soc_dep = soc_arr

    trace.append(
        {
            "step": step,
            "from_node_id": from_id,
            "node_id": to_id,
            "node_type": ninfo.node_type,
            "inserted_cp": bool(inserted and ninfo.node_type == "cp"),
            "travel_distance": dist,
            "energy_used_kwh": energy,
            "arrival_time": arr,
            "service_start_time": service_start,
            "depart_time": depart,
            "wait_time": wait_t,
            "soc_kwh": soc_arr,
            "soc_after_depart_kwh": soc_dep,
            "charge_needed_kwh": charge_needed,
            "soc_deficit_kwh": soc_deficit,
            "tw_end": float(ninfo.tw_end),
            "customer_reward": float(ninfo.reward),
            "charging_cost_per_kwh": float(ninfo.charging_cost_per_kwh),
        }
    )

    route_aug.append(to_id)
    return soc_dep, depart, step + 1


def _compute_reward_components(trace: list[dict]) -> dict:
    """Compute reward/cost/objective on augmented path using full charging at CP/depot."""
    served_customers: set[int] = set()
    total_reward = 0.0
    total_cost = 0.0
    successful_delivery = 0

    for rec in trace:
        node_type = rec["node_type"]
        if node_type == "customer":
            customer_id = int(rec["node_id"])
            first_visit = customer_id not in served_customers
            if first_visit:
                served_customers.add(customer_id)
            on_time = float(rec["service_start_time"]) <= (float(rec["tw_end"]) + 1e-6)
            success = bool(first_visit and on_time)
            step_reward = float(rec["customer_reward"]) if success else 0.0
            total_reward += step_reward
            if success:
                successful_delivery += 1
            rec["first_visit"] = bool(first_visit)
            rec["on_time"] = bool(on_time)
            rec["successful_delivery"] = bool(success)
            rec["step_reward"] = float(step_reward)
            rec["step_cost"] = 0.0
        else:
            step_cost = float(rec["charge_needed_kwh"]) * float(rec["charging_cost_per_kwh"])
            total_cost += step_cost
            rec["first_visit"] = False
            rec["on_time"] = False
            rec["successful_delivery"] = False
            rec["step_reward"] = -float(step_cost)
            rec["step_cost"] = float(step_cost)

    objective = total_reward - total_cost
    return {
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "objective": float(objective),
        "successful_delivery": int(successful_delivery),
    }


def _compute_partial_charging_cost_from_augmented_routes(
    augmented_routes: list[list[int]],
    node_info: dict[int, NodeInfo],
    depot_id: int,
    dist_full: torch.Tensor,
    effective_battery_kwh: float,
    energy_rate_kwh_per_distance: float,
) -> tuple[float, list[dict]]:
    """Recompute charging cost with partial-charge lookahead across CP/depot visits."""
    if not augmented_routes:
        return 0.0, []

    max_soc = max(float(effective_battery_kwh), 0.0)
    energy_rate = float(energy_rate_kwh_per_distance)
    total_cost = 0.0
    charge_events: list[dict] = []

    for route_idx, route in enumerate(augmented_routes, start=1):
        if not route or len(route) < 2:
            continue

        cp_sequence = [int(route[0])]
        subtrip_energy: list[float] = []
        energy_since_last_cp = 0.0

        prev = int(route[0])
        for node_raw in route[1:]:
            node = int(node_raw)
            travel_dist = _get_matrix_value(dist_full, prev, node)
            if travel_dist == float("inf"):
                prev = node
                continue
            energy_since_last_cp += max(0.0, travel_dist * energy_rate)
            ninfo = node_info.get(node)
            is_cp_or_depot = bool(node == depot_id or (ninfo is not None and ninfo.node_type == "cp"))
            if is_cp_or_depot:
                cp_sequence.append(node)
                subtrip_energy.append(energy_since_last_cp)
                energy_since_last_cp = 0.0
            prev = node

        if len(cp_sequence) < 2 or len(subtrip_energy) != (len(cp_sequence) - 1):
            continue

        soc = max_soc
        for l in range(1, len(cp_sequence)):
            cp_node = int(cp_sequence[l])
            energy_used = float(subtrip_energy[l - 1])
            soc = max(soc - energy_used, 0.0)
            cp_info = node_info.get(cp_node)
            unit_cost = float(cp_info.charging_cost_per_kwh) if cp_info is not None else 0.0
            charge_amount = 0.0

            if l == len(cp_sequence) - 1:
                # Match heuristic step-3 behavior: always top up at final CP/depot.
                charge_amount = max(max_soc - soc, 0.0)
            else:
                energy_to_next_best_cp = 0.0
                for l2 in range(l + 1, len(cp_sequence)):
                    energy_to_next_best_cp += float(subtrip_energy[l2 - 1])
                    if energy_to_next_best_cp > max_soc:
                        charge_amount = max(max_soc - soc, 0.0)
                        break
                    next_cp_node = int(cp_sequence[l2])
                    next_cp_info = node_info.get(next_cp_node)
                    next_cost = (
                        float(next_cp_info.charging_cost_per_kwh)
                        if next_cp_info is not None
                        else 0.0
                    )
                    if next_cost < unit_cost:
                        charge_amount = max(energy_to_next_best_cp - soc, 0.0)
                        break

            charge_amount = min(max(charge_amount, 0.0), max(max_soc - soc, 0.0))
            charge_cost = charge_amount * unit_cost
            soc += charge_amount
            total_cost += charge_cost

            if charge_amount > 1e-9:
                charge_events.append(
                    {
                        "route_index": route_idx,
                        "vehicle_id": route_idx,
                        "cp_node": cp_node,
                        "cp_id": 0 if cp_node == depot_id else cp_node,
                        "charge_amount_kwh": charge_amount,
                        "charge_cost": charge_cost,
                        "unit_cost_per_kwh": unit_cost,
                    }
                )

    return total_cost, charge_events


def augment_path_with_nearest_cp(
    routes_local: list[list[int]],
    global_ids: torch.Tensor,
    test_csv: str,
    distance_matrix_csv: str,
    time_matrix_csv: str,
    battery_capacity_kwh: float,
    reserve_soc_kwh: float,
    energy_rate_kwh_per_distance: float,
    charge_rate_kwh_per_hour: float,
) -> dict | None:
    """Insert nearest CP when direct next-leg is not feasible under current SoC."""
    node_info, cp_rows, depot_id = _build_node_info_from_test_csv(
        test_csv, default_charge_rate=float(charge_rate_kwh_per_hour)
    )
    if not cp_rows:
        return None

    kd_tree = create_kd_tree(cp_rows)

    routes_global = _to_global_routes(routes_local, global_ids)
    dist_full = parse_distance_matrix_csv(distance_matrix_csv)
    time_full = parse_distance_matrix_csv(time_matrix_csv)

    effective_battery_kwh = max(float(battery_capacity_kwh) - float(reserve_soc_kwh), 0.0)
    energy_rate = float(energy_rate_kwh_per_distance)

    customer_to_nearest_cp: dict[int, int] = {}
    unique_customers = {
        nid
        for route in routes_global
        for nid in route
        if nid in node_info and node_info[nid].node_type == "customer"
    }
    for cid in sorted(unique_customers):
        nearest_cp = find_nearest_cp(kd_tree, node_info[cid].x, node_info[cid].y)
        customer_to_nearest_cp[cid] = int(nearest_cp["cp_id"])

    augmented_routes: list[list[int]] = []
    trace: list[dict] = []
    skipped_late_customers: list[dict] = []
    step = 1

    for route_idx, route in enumerate(routes_global, start=1):
        if not route:
            continue
        route_aug = [route[0]]
        current_id = int(route[0])
        current_soc = effective_battery_kwh
        current_time = 0.0

        for target_id in route[1:]:
            target_id = int(target_id)
            target_info = node_info.get(target_id)
            if target_info is None:
                continue

            direct_energy = max(
                0.0, _get_matrix_value(dist_full, current_id, target_id) * energy_rate
            )

            if direct_energy <= (current_soc + 1e-9):
                if target_info.node_type == "customer":
                    direct_arrival = current_time + _get_matrix_value(
                        time_full, current_id, target_id
                    )
                    if direct_arrival > (target_info.tw_end + 1e-6):
                        skipped_late_customers.append(
                            {
                                "route_id": int(route_idx),
                                "customer_id": int(target_id),
                                "from_id": int(current_id),
                                "arrival_time": float(direct_arrival),
                                "tw_end": float(target_info.tw_end),
                                "reason": "late_if_direct",
                            }
                        )
                        continue
                current_soc, current_time, step = _append_leg(
                    trace,
                    route_aug,
                    node_info,
                    current_id,
                    target_id,
                    current_soc,
                    current_time,
                    effective_battery_kwh,
                    energy_rate,
                    charge_rate_kwh_per_hour,
                    dist_full,
                    time_full,
                    step,
                    inserted=False,
                )
                current_id = target_id
                continue

            # Need a recharge stop before target. Prefer nearest CP of current node.
            if current_id in node_info:
                nearest_cp = find_nearest_cp(kd_tree, node_info[current_id].x, node_info[current_id].y)
                cp_id = int(nearest_cp["cp_id"])
            else:
                cp_id = int(cp_rows[0]["cp_id"])

            energy_to_cp = max(0.0, _get_matrix_value(dist_full, current_id, cp_id) * energy_rate)

            # If nearest CP is unreachable, try any reachable CP.
            if energy_to_cp > (current_soc + 1e-9):
                reachable_cps = []
                for cp in cp_rows:
                    cpid = int(cp["cp_id"])
                    e_req = max(0.0, _get_matrix_value(dist_full, current_id, cpid) * energy_rate)
                    if e_req <= (current_soc + 1e-9):
                        reachable_cps.append((e_req, cpid))
                if reachable_cps:
                    reachable_cps.sort(key=lambda x: x[0])
                    cp_id = int(reachable_cps[0][1])
                    energy_to_cp = float(reachable_cps[0][0])

            # Insert CP leg only if reachable; otherwise force direct leg and mark deficit.
            if energy_to_cp <= (current_soc + 1e-9):
                current_soc, current_time, step = _append_leg(
                    trace,
                    route_aug,
                    node_info,
                    current_id,
                    cp_id,
                    current_soc,
                    current_time,
                    effective_battery_kwh,
                    energy_rate,
                    charge_rate_kwh_per_hour,
                    dist_full,
                    time_full,
                    step,
                    inserted=True,
                )
                current_id = cp_id

            if target_info.node_type == "customer":
                arrival_if_next = current_time + _get_matrix_value(
                    time_full, current_id, target_id
                )
                if arrival_if_next > (target_info.tw_end + 1e-6):
                    skipped_late_customers.append(
                        {
                            "route_id": int(route_idx),
                            "customer_id": int(target_id),
                            "from_id": int(current_id),
                            "arrival_time": float(arrival_if_next),
                            "tw_end": float(target_info.tw_end),
                            "reason": (
                                "late_after_recharge"
                                if energy_to_cp <= (current_soc + 1e-9)
                                else "late_with_soc_deficit_path"
                            ),
                        }
                    )
                    continue

            current_soc, current_time, step = _append_leg(
                trace,
                route_aug,
                node_info,
                current_id,
                target_id,
                current_soc,
                current_time,
                effective_battery_kwh,
                energy_rate,
                charge_rate_kwh_per_hour,
                dist_full,
                time_full,
                step,
                inserted=False,
            )
            current_id = target_id

        # Ensure each augmented route ends at depot for reporting consistency.
        if route_aug[-1] != depot_id:
            current_soc, current_time, step = _append_leg(
                trace,
                route_aug,
                node_info,
                current_id,
                depot_id,
                current_soc,
                current_time,
                effective_battery_kwh,
                energy_rate,
                charge_rate_kwh_per_hour,
                dist_full,
                time_full,
                step,
                inserted=False,
            )
        augmented_routes.append(route_aug)

    reward_components_full = _compute_reward_components(trace)
    partial_charge_cost, partial_charge_events = _compute_partial_charging_cost_from_augmented_routes(
        augmented_routes=augmented_routes,
        node_info=node_info,
        depot_id=depot_id,
        dist_full=dist_full,
        effective_battery_kwh=effective_battery_kwh,
        energy_rate_kwh_per_distance=energy_rate,
    )
    reward_components_partial = {
        "total_reward": float(reward_components_full.get("total_reward", 0.0)),
        "total_cost": float(partial_charge_cost),
        "objective": float(reward_components_full.get("total_reward", 0.0) - partial_charge_cost),
        "successful_delivery": int(reward_components_full.get("successful_delivery", 0)),
    }

    return {
        "customer_to_nearest_cp": customer_to_nearest_cp,
        "augmented_routes": augmented_routes,
        "trace": trace,
        "cp_ids": sorted(int(cp["cp_id"]) for cp in cp_rows),
        "skipped_late_customers": skipped_late_customers,
        "reward_components": reward_components_full,
        "reward_components_full": reward_components_full,
        "reward_components_partial": reward_components_partial,
        "partial_charge_events": partial_charge_events,
    }

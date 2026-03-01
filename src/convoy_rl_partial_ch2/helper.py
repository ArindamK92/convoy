"""Printing and decode-trace helpers for convoy evaluation output."""

import torch

from tensordict import TensorDict
from rl4co.envs import CVRPTWEnv
from rl4co.models import AttentionModel

from src.convoy_rl_partial_ch2.convoy import convoy


def split_routes_with_vehicle_ids(
    actions_1d: torch.Tensor, step_to_vehicle: dict[int, int]
) -> list[tuple[int | None, list[int]]]:
    """Split flat actions into routes and attach the executing EV id per route."""
    routes: list[tuple[int | None, list[int]]] = []
    current_route = [0]
    first_non_depot_step: int | None = None

    for step_idx, node in enumerate(actions_1d.tolist(), start=1):
        node = int(node)
        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append((step_to_vehicle.get(first_non_depot_step), current_route))
                current_route = [0]
                first_non_depot_step = None
        else:
            if first_non_depot_step is None:
                first_non_depot_step = step_idx
            current_route.append(node)

    if len(current_route) > 1:
        current_route.append(0)
        routes.append((step_to_vehicle.get(first_non_depot_step), current_route))

    return routes


def format_node_label(
    node: int,
    station_mask_1d: torch.Tensor | None = None,
    cp_id_per_node_1d: torch.Tensor | None = None,
) -> str:
    """Format node id to distinguish depot/customer/charging-station nodes."""
    node = int(node)
    if node == 0:
        return "DEPOT"
    if (
        station_mask_1d is not None
        and 0 <= node < int(station_mask_1d.shape[0])
        and bool(station_mask_1d[node].item())
    ):
        cp_id = -1
        if (
            cp_id_per_node_1d is not None
            and 0 <= node < int(cp_id_per_node_1d.shape[0])
        ):
            cp_id = int(cp_id_per_node_1d[node].item())
        return f"CP{cp_id}" if cp_id >= 0 else f"CP_NODE_{node}"
    return f"CUST{node}"


def build_customer_visit_trace(
    env: convoy,
    td_state: TensorDict,
    actions_1d: torch.Tensor,
) -> list[dict]:
    """Simulate one decoded solution and collect per-visit trace for customers/CPs."""
    if "dist_matrix" not in td_state.keys() or "travel_time_matrix" not in td_state.keys():
        raise ValueError(
            "Combined mode requires dist_matrix and travel_time_matrix for trace printing."
        )
    dist_matrix = td_state["dist_matrix"][0]
    travel_matrix = td_state["travel_time_matrix"][0]

    durations = td_state["durations"][0]
    tw_starts = td_state["time_windows"][0, :, 0]
    tw_ends = td_state["time_windows"][0, :, 1]
    if "charge_nodes_mask" in td_state.keys():
        charge_nodes_mask = td_state["charge_nodes_mask"][0]
    else:
        charge_nodes_mask = torch.zeros(dist_matrix.shape[0], dtype=torch.bool)
        charge_nodes_mask[0] = True
    if "station_mask" in td_state.keys():
        station_mask = td_state["station_mask"][0]
    else:
        station_mask = torch.zeros_like(charge_nodes_mask)
    if "charge_rate_per_node" in td_state.keys():
        charge_rate_per_node = td_state["charge_rate_per_node"][0]
    else:
        charge_rate_per_node = torch.zeros(dist_matrix.shape[0], dtype=torch.float32)
        charge_rate_per_node[0] = env.charge_rate_kwh_per_hour
    if "charge_cost_per_kwh_per_node" in td_state.keys():
        charge_cost_per_kwh_per_node = td_state["charge_cost_per_kwh_per_node"][0]
    else:
        charge_cost_per_kwh_per_node = torch.zeros(dist_matrix.shape[0], dtype=torch.float32)
        charge_cost_per_kwh_per_node[0] = float(
            getattr(env, "depot_charge_cost_per_kwh", 0.0)
        )
    if "cp_id_per_node" in td_state.keys():
        cp_id_per_node = td_state["cp_id_per_node"][0]
    else:
        cp_id_per_node = torch.full((dist_matrix.shape[0],), -1, dtype=torch.long)
    if "customer_reward_per_node" in td_state.keys():
        customer_reward_per_node = td_state["customer_reward_per_node"][0]
    else:
        customer_reward_per_node = torch.zeros(dist_matrix.shape[0], dtype=torch.float32)
    if "global_node_ids" in td_state.keys():
        global_node_ids = td_state["global_node_ids"][0].to(torch.long)
    else:
        global_node_ids = torch.arange(
            dist_matrix.shape[0], dtype=torch.long, device=dist_matrix.device
        )

    battery_cap = float(
        getattr(env, "effective_battery_kwh", env.battery_capacity_kwh)
    )
    energy_rate = float(env.energy_rate_kwh_per_distance)
    num_evs = int(env.num_evs)
    time_units_per_hour = float(env.time_units_per_hour)

    current_node = 0
    current_time = 0.0
    current_soc = battery_cap
    current_vehicle_idx = 0
    vehicle_ready_times = [0.0 for _ in range(num_evs)]
    served_customers = torch.zeros(dist_matrix.shape[0], dtype=torch.bool)
    trace: list[dict] = []

    for step_idx, node in enumerate(actions_1d.tolist(), start=1):
        node = int(node)
        if node < 0 or node >= int(dist_matrix.shape[0]):
            continue

        prev_node = int(current_node)
        prev_node_id = (
            int(global_node_ids[prev_node].item())
            if 0 <= prev_node < int(global_node_ids.shape[0])
            else prev_node
        )
        node_id = (
            int(global_node_ids[node].item())
            if 0 <= node < int(global_node_ids.shape[0])
            else node
        )
        travel_dist = float(dist_matrix[current_node, node].item())
        travel_time = float(travel_matrix[current_node, node].item())
        energy_used = max(0.0, travel_dist * energy_rate)
        arrival_time = current_time + travel_time
        service_start = max(arrival_time, float(tw_starts[node].item()))
        on_time = service_start <= (float(tw_ends[node].item()) + 1e-6)
        soc_after_arrival = max(current_soc - energy_used, 0.0)

        is_charge_node = bool(charge_nodes_mask[node].item())
        is_station = bool(station_mask[node].item())
        charge_needed = 0.0
        if is_charge_node:
            selected_rate = max(float(charge_rate_per_node[node].item()), 1e-6)
            charge_needed = max(battery_cap - soc_after_arrival, 0.0)
            charge_time = (charge_needed / selected_rate) * time_units_per_hour
            depart_time = service_start + charge_time
            soc_after_depart = battery_cap
        else:
            depart_time = service_start + float(durations[node].item())
            soc_after_depart = soc_after_arrival

        if node == 0:
            # Include depot charging in reward/cost decomposition so CSV totals
            # match the environment reward definition.
            depot_charge_cost = float(charge_cost_per_kwh_per_node[node].item())
            depot_step_reward = -(charge_needed * depot_charge_cost)
            if charge_needed > 1e-9:
                trace.append(
                    {
                        "step": step_idx,
                        "node_type": "cp",
                        "node_id": node_id,
                        "node_local_idx": node,
                        "from_node_id": prev_node_id,
                        "from_local_idx": prev_node,
                        "vehicle_id": current_vehicle_idx + 1,
                        "arrival_time": arrival_time,
                        "depart_time": depart_time,
                        "travel_distance": travel_dist,
                        "energy_used_kwh": energy_used,
                        "soc_kwh": soc_after_arrival,
                        "step_reward": depot_step_reward,
                        "on_time": False,
                        "first_visit": False,
                        "successful_delivery": False,
                    }
                )
            vehicle_ready_times[current_vehicle_idx] = depart_time
            current_vehicle_idx = min(
                range(num_evs), key=lambda idx: vehicle_ready_times[idx]
            )
            current_time = vehicle_ready_times[current_vehicle_idx]
            current_soc = battery_cap
            current_node = 0
            continue

        if is_station:
            station_cost = float(charge_cost_per_kwh_per_node[node].item())
            step_reward = -(charge_needed * station_cost)
            cp_id = (
                int(cp_id_per_node[node].item())
                if node < int(cp_id_per_node.shape[0])
                else -1
            )
            cp_id = cp_id if cp_id >= 0 else node_id
            trace.append(
                {
                    "step": step_idx,
                    "node_type": "cp",
                    "node_id": cp_id,
                    "node_local_idx": node,
                    "from_node_id": prev_node_id,
                    "from_local_idx": prev_node,
                    "vehicle_id": current_vehicle_idx + 1,
                    "arrival_time": arrival_time,
                    "depart_time": depart_time,
                    "travel_distance": travel_dist,
                    "energy_used_kwh": energy_used,
                    "soc_kwh": soc_after_arrival,
                    "step_reward": step_reward,
                    "on_time": False,
                    "first_visit": False,
                    "successful_delivery": False,
                }
            )
        else:
            first_visit = not bool(served_customers[node].item())
            successful_delivery = bool(on_time and first_visit)
            step_reward = (
                float(customer_reward_per_node[node].item()) if successful_delivery else 0.0
            )
            served_customers[node] = True
            trace.append(
                {
                    "step": step_idx,
                    "node_type": "customer",
                    "node_id": node_id,
                    "node_local_idx": node,
                    "from_node_id": prev_node_id,
                    "from_local_idx": prev_node,
                    "vehicle_id": current_vehicle_idx + 1,
                    "arrival_time": arrival_time,
                    "depart_time": depart_time,
                    "travel_distance": travel_dist,
                    "energy_used_kwh": energy_used,
                    "soc_kwh": soc_after_arrival,
                    "step_reward": step_reward,
                    "on_time": bool(on_time),
                    "first_visit": bool(first_visit),
                    "successful_delivery": successful_delivery,
                }
            )
        current_time = depart_time
        current_soc = soc_after_depart
        current_node = node

    return trace


def _compute_partial_charging_cost_from_routes(
    routes: list[tuple[int | None, list[int]]],
    td_state: TensorDict,
    env: convoy,
) -> tuple[float, list[dict]]:
    """Recompute charging cost using partial-charge lookahead over CP/depot visits."""
    if not routes:
        return 0.0, []

    if "dist_matrix" not in td_state.keys():
        return 0.0, []

    dist_matrix = td_state["dist_matrix"][0].detach().cpu()
    total_nodes = int(dist_matrix.shape[0])

    if "station_mask" in td_state.keys():
        station_mask = td_state["station_mask"][0].detach().cpu().to(torch.bool)
    else:
        station_mask = torch.zeros(total_nodes, dtype=torch.bool)
    if "charge_cost_per_kwh_per_node" in td_state.keys():
        charge_cost_per_node = (
            td_state["charge_cost_per_kwh_per_node"][0].detach().cpu().to(torch.float32)
        )
    else:
        charge_cost_per_node = torch.zeros(total_nodes, dtype=torch.float32)
        charge_cost_per_node[0] = float(getattr(env, "depot_charge_cost_per_kwh", 0.0))
    if "cp_id_per_node" in td_state.keys():
        cp_id_per_node = td_state["cp_id_per_node"][0].detach().cpu().to(torch.long)
    else:
        cp_id_per_node = torch.full((total_nodes,), -1, dtype=torch.long)

    max_soc = float(getattr(env, "effective_battery_kwh", env.battery_capacity_kwh))
    energy_rate = float(env.energy_rate_kwh_per_distance)

    total_cost = 0.0
    charge_events: list[dict] = []

    for route_idx, (vehicle_id, route) in enumerate(routes, start=1):
        if not route or len(route) < 2:
            continue

        cp_sequence = [int(route[0])]
        subtrip_energy: list[float] = []
        energy_since_last_cp = 0.0

        prev = int(route[0])
        for node_raw in route[1:]:
            node = int(node_raw)
            if not (0 <= prev < total_nodes and 0 <= node < total_nodes):
                prev = node
                continue
            travel_dist = float(dist_matrix[prev, node].item())
            energy_since_last_cp += max(0.0, travel_dist * energy_rate)
            is_cp_or_depot = bool(node == 0 or station_mask[node].item())
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
            unit_cost = float(charge_cost_per_node[cp_node].item())
            charge_amount = 0.0

            if l == len(cp_sequence) - 1:
                # Match heuristic Step-3 behavior: always top up at final CP/depot.
                charge_amount = max(max_soc - soc, 0.0)
            else:
                energy_to_next_best_cp = 0.0
                for l2 in range(l + 1, len(cp_sequence)):
                    energy_to_next_best_cp += float(subtrip_energy[l2 - 1])
                    if energy_to_next_best_cp > max_soc:
                        charge_amount = max(max_soc - soc, 0.0)
                        break
                    next_cp_node = int(cp_sequence[l2])
                    next_cost = float(charge_cost_per_node[next_cp_node].item())
                    if next_cost < unit_cost:
                        charge_amount = max(energy_to_next_best_cp - soc, 0.0)
                        break

            charge_amount = min(max(charge_amount, 0.0), max(max_soc - soc, 0.0))
            charge_cost = charge_amount * unit_cost
            soc += charge_amount
            total_cost += charge_cost

            cp_id = (
                int(cp_id_per_node[cp_node].item())
                if 0 <= cp_node < int(cp_id_per_node.shape[0])
                else -1
            )
            if charge_amount > 1e-9:
                charge_events.append(
                    {
                        "route_index": route_idx,
                        "vehicle_id": vehicle_id,
                        "cp_node": cp_node,
                        "cp_id": cp_id if cp_node != 0 and cp_id >= 0 else 0,
                        "charge_amount_kwh": charge_amount,
                        "charge_cost": charge_cost,
                        "unit_cost_per_kwh": unit_cost,
                    }
                )

    return total_cost, charge_events


def print_quality_table(
    initial_reward: float,
    fixed_history: list[tuple[int, float]],
    best_ckpt_reward: float | None,
    final_model_reward: float,
    best_test_reward: float | None,
) -> None:
    """Print a compact reward trend table to verify quality improvement."""
    print("\nQuality trend on fixed evaluation set (higher is better):")
    print("stage                epoch   reward")
    print(f"initial              0       {initial_reward:.6f}")
    for epoch, reward in fixed_history:
        print(f"periodic             {epoch:<7d} {reward:.6f}")
    if best_ckpt_reward is not None:
        print(f"best_checkpoint      -       {best_ckpt_reward:.6f}")
    print(f"final_model          -       {final_model_reward:.6f}")
    if best_test_reward is not None:
        print(f"best_checkpoint_test -       {best_test_reward:.6f}")


def print_one_solution(
    model: AttentionModel,
    env: CVRPTWEnv,
    instance=None,
    title: str = "One test-instance solution",
    return_details: bool = False,
    decode_kwargs: dict | None = None,
) -> float | dict:
    """Decode and print one full test solution from a given or sampled instance."""
    if instance is None:
        dataset = env.dataset(1, phase="test")
        instance = dataset.collate_fn([dataset[0]])
    return print_one_solution_from_instance(
        model,
        env,
        instance,
        title=title,
        return_details=return_details,
        decode_kwargs=decode_kwargs,
    )


def print_one_solution_from_instance(
    model: AttentionModel,
    env: CVRPTWEnv,
    instance,
    title: str,
    return_details: bool = False,
    decode_kwargs: dict | None = None,
) -> float | dict:
    """Decode and print route details for one given instance."""
    instance = instance.to(model.device)
    if decode_kwargs is None:
        decode_kwargs = {"decode_type": "greedy"}
    decode_label = decode_kwargs.get("decode_type", "greedy")

    with torch.inference_mode():
        td = env.reset(instance)
        out = model.policy(td, env, phase="test", **decode_kwargs)

    actions = out["actions"][0].detach().cpu()
    reward = float(out["reward"][0].detach().cpu())
    visit_trace = build_customer_visit_trace(env, td, actions)
    step_to_vehicle = {int(rec["step"]): int(rec["vehicle_id"]) for rec in visit_trace}
    routes = split_routes_with_vehicle_ids(actions, step_to_vehicle)
    station_mask_1d = td["station_mask"][0].detach().cpu() if "station_mask" in td.keys() else None
    cp_id_per_node_1d = (
        td["cp_id_per_node"][0].detach().cpu() if "cp_id_per_node" in td.keys() else None
    )
    action_labels = [
        format_node_label(int(node), station_mask_1d, cp_id_per_node_1d)
        for node in actions.tolist()
    ]

    print(f"\n{title} ({decode_label} decode):")
    print(f"Reward: {reward:.6f}")
    print(f"Flat action sequence: {actions.tolist()}")
    print(f"Flat action labels:   {action_labels}")
    if not routes:
        print("Vehicle routes: []")
    else:
        print("Vehicle routes:")
        ev_trip_count: dict[int, int] = {}
        for i, (vehicle_id, route) in enumerate(routes, start=1):
            labels = [
                format_node_label(node, station_mask_1d, cp_id_per_node_1d)
                for node in route
            ]
            if vehicle_id is None:
                route_name = f"Route {i}"
            else:
                ev_trip_count[vehicle_id] = ev_trip_count.get(vehicle_id, 0) + 1
                trip_no = ev_trip_count[vehicle_id]
                suffix = f" Trip {trip_no}" if trip_no > 1 else ""
                route_name = f"Vehicle {vehicle_id}{suffix}"
            print(f"  {route_name}: {route}")
            print(f"             {labels}")
    if visit_trace:
        print("Visit trace (customer + CP; includes distance + energy per step):")
        for rec in visit_trace:
            if rec["node_type"] == "cp":
                if int(rec.get("node_local_idx", -1)) == 0:
                    node_txt = f"depot_id={int(rec['node_id']):>3d}"
                else:
                    node_txt = f"cp_id={int(rec['node_id']):>3d}"
            else:
                node_txt = f"customer_id={int(rec['node_id']):>3d}"
            print(
                f"  step={rec['step']:>3d} {node_txt} "
                f"vehicle={rec['vehicle_id']:>2d} "
                f"from_id={int(rec.get('from_node_id', -1)):>3d} "
                f"dist={float(rec.get('travel_distance', 0.0)):.2f} "
                f"energy={float(rec.get('energy_used_kwh', 0.0)):.2f}kWh "
                f"arrival={rec['arrival_time']:.2f} "
                f"depart={rec['depart_time']:.2f} "
                f"soc={rec['soc_kwh']:.2f}kWh "
                f"reward={rec['step_reward']:.2f}"
            )
    else:
        print("Visit trace: []")
    customer_reward_sum = 0.0
    full_charge_cost_sum = 0.0
    total_successful_delivery = 0
    for rec in visit_trace:
        step_reward = float(rec["step_reward"])
        if rec["node_type"] == "customer":
            customer_reward_sum += max(step_reward, 0.0)
            if bool(rec.get("successful_delivery", False)):
                total_successful_delivery += 1
        elif rec["node_type"] == "cp":
            full_charge_cost_sum += max(-step_reward, 0.0)
    full_charge_objective_val = customer_reward_sum - full_charge_cost_sum
    charging_cost_sum, partial_charge_events = _compute_partial_charging_cost_from_routes(
        routes=routes,
        td_state=td,
        env=env,
    )
    objective_val = customer_reward_sum - charging_cost_sum
    print(
        "Reward components (before partial charging): "
        f"total_reward={customer_reward_sum:.6f}, "
        f"total_cost={full_charge_cost_sum:.6f}, "
        f"successful_delivery={total_successful_delivery}, "
        f"objective={full_charge_objective_val:.6f}"
    )
    if partial_charge_events:
        print("Partial charging summary:")
        for evt in partial_charge_events:
            vehicle_id = evt["vehicle_id"] if evt["vehicle_id"] is not None else "?"
            cp_txt = "DEPOT" if int(evt["cp_node"]) == 0 else f"CP{int(evt['cp_id'])}"
            print(
                f"  route={evt['route_index']:>2d} vehicle={vehicle_id} "
                f"node={cp_txt} charge={evt['charge_amount_kwh']:.4f}kWh "
                f"cost={evt['charge_cost']:.4f}"
            )
    print(
        "Reward components (after partial charging): "
        f"total_reward={customer_reward_sum:.6f}, "
        f"total_cost={charging_cost_sum:.6f}, "
        f"successful_delivery={total_successful_delivery}, "
        f"objective={objective_val:.6f}"
    )
    if return_details:
        return {
            "objective_val": objective_val,
            "total_reward": customer_reward_sum,
            "total_cost": charging_cost_sum,
            "total_successful_delivery": total_successful_delivery,
            "full_charge_objective_val": full_charge_objective_val,
            "full_charge_total_reward": customer_reward_sum,
            "full_charge_total_cost": full_charge_cost_sum,
            "full_charge_total_successful_delivery": total_successful_delivery,
        }
    return objective_val

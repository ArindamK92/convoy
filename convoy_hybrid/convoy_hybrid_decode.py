"""Decode and route-print helpers for hybrid runner."""

from __future__ import annotations

import torch

from convoy_hybrid.convoy_hybrid_cp_path import augment_path_with_nearest_cp


def split_routes(actions_1d: torch.Tensor) -> list[list[int]]:
    """Split flat action list into depot-delimited routes."""
    routes: list[list[int]] = []
    route = [0]
    for node in actions_1d.tolist():
        node = int(node)
        if node == 0:
            if len(route) > 1:
                route.append(0)
                routes.append(route)
                route = [0]
        else:
            route.append(node)
    if len(route) > 1:
        route.append(0)
        routes.append(route)
    return routes


def format_node_label(node_local_idx: int, global_ids: torch.Tensor | None) -> str:
    """Format local node index to human-readable depot/customer label."""
    if node_local_idx == 0:
        return "DEPOT"
    if global_ids is not None and 0 <= node_local_idx < int(global_ids.shape[0]):
        return f"CUST{int(global_ids[node_local_idx].item())}"
    return f"CUST{node_local_idx}"


def _build_distance_and_time_matrices(td_state):
    """Resolve distance/time matrices from state, or derive them from coordinates."""
    if "dist_matrix" in td_state.keys() and "travel_time_matrix" in td_state.keys():
        return td_state["dist_matrix"][0], td_state["travel_time_matrix"][0]
    if "dist_matrix" in td_state.keys():
        dist = td_state["dist_matrix"][0]
        return dist, dist

    depot_xy = td_state["depot"][0].to(torch.float32).reshape(1, -1)
    locs = td_state["locs"][0].to(torch.float32)
    all_xy = torch.cat([depot_xy, locs], dim=0)
    delta = all_xy[:, None, :] - all_xy[None, :, :]
    dist = torch.linalg.norm(delta, dim=-1)
    return dist, dist


def build_visit_trace(
    td_state,
    routes: list[list[int]],
    battery_capacity_kwh: float,
    reserve_soc_kwh: float,
    energy_rate_kwh_per_distance: float,
    charge_rate_kwh_per_hour: float,
    time_units_per_hour: float = 60.0,
) -> list[dict]:
    """Build SoC/time trace using convoy-style timing and depot recharge logic."""
    dist_matrix, time_matrix = _build_distance_and_time_matrices(td_state)

    total_nodes = int(dist_matrix.shape[0])
    if "durations" in td_state.keys():
        durations = td_state["durations"][0]
    else:
        durations = torch.zeros(total_nodes, dtype=torch.float32, device=dist_matrix.device)

    if "time_windows" in td_state.keys():
        time_windows = td_state["time_windows"][0]
    else:
        zeros = torch.zeros(total_nodes, dtype=torch.float32, device=dist_matrix.device)
        inf = torch.full_like(zeros, float("inf"))
        time_windows = torch.stack([zeros, inf], dim=-1)

    if "global_node_ids" in td_state.keys():
        global_ids = td_state["global_node_ids"][0].to(torch.long)
    else:
        global_ids = torch.arange(total_nodes, dtype=torch.long, device=dist_matrix.device)

    effective_battery_kwh = max(float(battery_capacity_kwh) - float(reserve_soc_kwh), 0.0)
    energy_rate = float(energy_rate_kwh_per_distance)
    charge_rate = max(float(charge_rate_kwh_per_hour), 1e-6)
    time_units = float(time_units_per_hour)

    trace: list[dict] = []
    step = 0

    for vehicle_id, route in enumerate(routes, start=1):
        current_node = 0
        current_time = 0.0
        current_soc = effective_battery_kwh

        for node in route[1:]:
            node = int(node)
            step += 1

            from_node = int(current_node)
            from_id = int(global_ids[from_node].item())
            node_id = int(global_ids[node].item())

            travel_distance = float(dist_matrix[from_node, node].item())
            travel_time = float(time_matrix[from_node, node].item())
            energy_used = max(0.0, travel_distance * energy_rate)

            arrival_time = current_time + travel_time
            tw_start = float(time_windows[node, 0].item())
            service_time = float(durations[node].item())
            service_start = max(arrival_time, tw_start)
            wait_time = max(service_start - arrival_time, 0.0)

            soc_deficit_kwh = max(energy_used - current_soc, 0.0)
            soc_at_arrival = max(current_soc - energy_used, 0.0)

            if node == 0:
                charge_needed = max(effective_battery_kwh - soc_at_arrival, 0.0)
                charge_time = (charge_needed / charge_rate) * time_units
                depart_time = service_start + charge_time
                soc_after_depart = effective_battery_kwh
                node_type = "depot"
            else:
                charge_needed = 0.0
                charge_time = 0.0
                depart_time = service_start + service_time
                soc_after_depart = soc_at_arrival
                node_type = "customer"

            trace.append(
                {
                    "step": step,
                    "vehicle_id": vehicle_id,
                    "node_type": node_type,
                    "node_local_idx": node,
                    "node_id": node_id,
                    "from_local_idx": from_node,
                    "from_node_id": from_id,
                    "travel_distance": travel_distance,
                    "energy_used_kwh": energy_used,
                    "arrival_time": arrival_time,
                    "wait_time": wait_time,
                    "service_time": service_time,
                    "charge_time": charge_time,
                    "depart_time": depart_time,
                    "soc_kwh": soc_at_arrival,
                    "soc_after_depart_kwh": soc_after_depart,
                    "charge_needed_kwh": charge_needed,
                    "soc_deficit_kwh": soc_deficit_kwh,
                }
            )

            current_node = node
            current_time = depart_time
            current_soc = soc_after_depart

    return trace


def decode_and_print_solution(
    model,
    env,
    instance,
    title: str,
    decode_kwargs: dict,
    trace_settings: dict | None = None,
    cp_postprocess_settings: dict | None = None,
) -> dict:
    """Decode one instance, print path details, and return summary dict."""
    with torch.inference_mode():
        td = env.reset(instance.to(model.device))

    decode_kwargs_local = dict(decode_kwargs)
    decode_type_local = decode_kwargs_local.get("decode_type", "greedy")
    # Cap beam/multistart counts by per-instance available starts to avoid OOB indices.
    try:
        max_starts_raw = env.get_num_starts(td)
        if isinstance(max_starts_raw, torch.Tensor):
            max_starts = int(max_starts_raw.min().item())
        else:
            max_starts = int(max_starts_raw)
    except Exception:  # pragma: no cover - defensive fallback
        max_starts = int(td["action_mask"].shape[-1] - 1)
    max_starts = max(max_starts, 1)

    if "num_starts" in decode_kwargs_local:
        requested_num_starts = int(decode_kwargs_local["num_starts"])
        if requested_num_starts > max_starts:
            decode_kwargs_local["num_starts"] = max_starts
            print(
                "Decode note: capped num_starts from "
                f"{requested_num_starts} to {max_starts} for this instance."
            )

    if decode_type_local == "beam_search":
        requested_beam = int(decode_kwargs_local.get("beam_width", 1))
        if requested_beam > max_starts:
            capped_beam = max_starts
            if capped_beam <= 1:
                decode_kwargs_local.pop("beam_width", None)
                decode_kwargs_local["decode_type"] = "greedy"
                print(
                    "Decode note: beam_search requested but only one feasible start "
                    "is available for this instance; falling back to greedy."
                )
            else:
                decode_kwargs_local["beam_width"] = capped_beam
                print(
                    "Decode note: capped beam_width from "
                    f"{requested_beam} to {capped_beam} for this instance."
                )

    decode_label = decode_kwargs_local.get("decode_type", "greedy")
    with torch.inference_mode():
        out = model.policy(td, env, phase="test", **decode_kwargs_local)

    actions = out["actions"][0].detach().cpu()
    reward = float(out["reward"][0].detach().cpu())
    global_ids = (
        td["global_node_ids"][0].detach().cpu() if "global_node_ids" in td.keys() else None
    )

    action_labels = [format_node_label(int(node), global_ids) for node in actions.tolist()]
    routes = split_routes(actions)

    print(f"\n{title} ({decode_label} decode):")
    print(f"Reward: {reward:.6f}")
    print(f"Flat action sequence: {actions.tolist()}")
    print(f"Flat action labels:   {action_labels}")

    if not routes:
        print("Vehicle routes: []")
    else:
        print("Vehicle routes:")
        for i, route in enumerate(routes, start=1):
            labels = [format_node_label(int(n), global_ids) for n in route]
            print(f"  Vehicle {i}: {route}")
            print(f"             {labels}")

    if trace_settings is None:
        trace_settings = {}
    visit_trace = build_visit_trace(
        td_state=td,
        routes=routes,
        battery_capacity_kwh=float(trace_settings.get("battery_capacity_kwh", 30.0)),
        reserve_soc_kwh=float(trace_settings.get("reserve_soc_kwh", 0.0)),
        energy_rate_kwh_per_distance=float(
            trace_settings.get("energy_rate_kwh_per_distance", 0.00025)
        ),
        charge_rate_kwh_per_hour=float(
            trace_settings.get("charge_rate_kwh_per_hour", 120.0)
        ),
        time_units_per_hour=float(trace_settings.get("time_units_per_hour", 60.0)),
    )

    if visit_trace:
        print("Visit trace (customer + depot; includes distance + energy per step):")
        for rec in visit_trace:
            if rec["node_type"] == "depot":
                node_txt = f"depot_id={int(rec['node_id']):>3d}"
            else:
                node_txt = f"customer_id={int(rec['node_id']):>3d}"
            deficit_note = " [soc_deficit]" if rec["soc_deficit_kwh"] > 1e-9 else ""
            print(
                f"  step={rec['step']:>3d} {node_txt} "
                f"vehicle={rec['vehicle_id']:>2d} "
                f"from_id={int(rec['from_node_id']):>3d} "
                f"dist={rec['travel_distance']:.2f} "
                f"energy={rec['energy_used_kwh']:.2f}kWh "
                f"arrival={rec['arrival_time']:.2f} "
                f"depart={rec['depart_time']:.2f} "
                f"soc={rec['soc_kwh']:.2f}kWh"
                f"{deficit_note}"
            )

    visited_customer_ids: list[int] = []
    for node in actions.tolist():
        node = int(node)
        if node == 0:
            continue
        if global_ids is not None and 0 <= node < int(global_ids.shape[0]):
            visited_customer_ids.append(int(global_ids[node].item()))
        else:
            visited_customer_ids.append(node)

    first_visit_customer_ids = list(dict.fromkeys(visited_customer_ids))
    print(f"Visited customer IDs (actual, action order): {visited_customer_ids}")
    print(f"Visited customer IDs (actual, first visit): {first_visit_customer_ids}")

    cp_augmented = None
    if cp_postprocess_settings:
        test_csv = cp_postprocess_settings.get("test_csv")
        dist_csv = cp_postprocess_settings.get("distance_matrix_csv")
        time_csv = cp_postprocess_settings.get("time_matrix_csv")
        if test_csv and dist_csv and time_csv and global_ids is not None:
            cp_augmented = augment_path_with_nearest_cp(
                routes_local=routes,
                global_ids=global_ids,
                test_csv=str(test_csv),
                distance_matrix_csv=str(dist_csv),
                time_matrix_csv=str(time_csv),
                battery_capacity_kwh=float(
                    trace_settings.get("battery_capacity_kwh", 30.0) if trace_settings else 30.0
                ),
                reserve_soc_kwh=float(
                    trace_settings.get("reserve_soc_kwh", 0.0) if trace_settings else 0.0
                ),
                energy_rate_kwh_per_distance=float(
                    trace_settings.get("energy_rate_kwh_per_distance", 0.00025)
                    if trace_settings
                    else 0.00025
                ),
                charge_rate_kwh_per_hour=float(
                    trace_settings.get("charge_rate_kwh_per_hour", 120.0)
                    if trace_settings
                    else 120.0
                ),
            )
            if cp_augmented is not None:
                print("Nearest CP per visited customer (KD-tree):")
                for cid in sorted(cp_augmented["customer_to_nearest_cp"].keys()):
                    print(
                        f"  customer_id={cid:>3d} -> cp_id={cp_augmented['customer_to_nearest_cp'][cid]:>3d}"
                    )

                print("Augmented routes with inserted CPs (when needed):")
                cp_ids = set(int(x) for x in cp_augmented.get("cp_ids", []))
                depot_id = int(global_ids[0].item())
                for i, route in enumerate(cp_augmented["augmented_routes"], start=1):
                    labels = []
                    for nid in route:
                        nid = int(nid)
                        if nid == depot_id:
                            labels.append("DEPOT")
                        elif nid in cp_ids:
                            labels.append(f"CP{nid}")
                        else:
                            labels.append(f"CUST{nid}")
                    print(f"  Vehicle {i}: {route}")
                    print(f"             {labels}")

                print("Augmented trace (with CP insertion and updated SoC):")
                for rec in cp_augmented["trace"]:
                    node_prefix = (
                        "depot_id"
                        if rec["node_type"] == "depot"
                        else ("cp_id" if rec["node_type"] == "cp" else "customer_id")
                    )
                    inserted_txt = " inserted_cp" if rec.get("inserted_cp", False) else ""
                    deficit_note = " [soc_deficit]" if rec["soc_deficit_kwh"] > 1e-9 else ""
                    print(
                        f"  step={rec['step']:>3d} {node_prefix}={int(rec['node_id']):>3d} "
                        f"from_id={int(rec['from_node_id']):>3d} "
                        f"dist={rec['travel_distance']:.2f} "
                        f"energy={rec['energy_used_kwh']:.2f}kWh "
                        f"arrival={rec['arrival_time']:.2f} "
                        f"depart={rec['depart_time']:.2f} "
                        f"soc={rec['soc_kwh']:.2f}kWh "
                        f"reward={float(rec.get('step_reward', 0.0)):.2f}"
                        f"{inserted_txt}{deficit_note}"
                    )
                skipped = cp_augmented.get("skipped_late_customers", [])
                if skipped:
                    print("Skipped late customers (no travel applied):")
                    for rec in skipped:
                        print(
                            f"  route={int(rec.get('route_id', 0)):>2d} "
                            f"customer_id={int(rec.get('customer_id', -1)):>3d} "
                            f"from_id={int(rec.get('from_id', -1)):>3d} "
                            f"arrival_if_visited={float(rec.get('arrival_time', 0.0)):.2f} "
                            f"tw_end={float(rec.get('tw_end', 0.0)):.2f} "
                            f"reason={str(rec.get('reason', 'late'))}"
                        )
                comp_full = cp_augmented.get("reward_components_full") or cp_augmented.get(
                    "reward_components", {}
                )
                print(
                    "Reward components (CP-augmented full charging): "
                    f"total_reward={float(comp_full.get('total_reward', 0.0)):.6f}, "
                    f"total_cost={float(comp_full.get('total_cost', 0.0)):.6f}, "
                    f"successful_delivery={int(comp_full.get('successful_delivery', 0))}, "
                    f"objective={float(comp_full.get('objective', 0.0)):.6f}"
                )
                partial_events = cp_augmented.get("partial_charge_events", [])
                if partial_events:
                    print("Partial charging summary:")
                    for evt in partial_events:
                        cp_txt = (
                            "DEPOT"
                            if int(evt.get("cp_node", -1)) == int(global_ids[0].item())
                            else f"CP{int(evt.get('cp_id', -1))}"
                        )
                        print(
                            f"  route={int(evt.get('route_index', 0)):>2d} "
                            f"vehicle={int(evt.get('vehicle_id', 0)):>2d} "
                            f"node={cp_txt} "
                            f"charge={float(evt.get('charge_amount_kwh', 0.0)):.4f}kWh "
                            f"cost={float(evt.get('charge_cost', 0.0)):.4f}"
                        )
                comp_partial = cp_augmented.get("reward_components_partial", {})
                if comp_partial:
                    print(
                        "Reward components (CP-augmented partial charging): "
                        f"total_reward={float(comp_partial.get('total_reward', 0.0)):.6f}, "
                        f"total_cost={float(comp_partial.get('total_cost', 0.0)):.6f}, "
                        f"successful_delivery={int(comp_partial.get('successful_delivery', 0))}, "
                        f"objective={float(comp_partial.get('objective', 0.0)):.6f}"
                    )

    return {
        "reward": reward,
        "visited_customer_ids": visited_customer_ids,
        "visited_customer_ids_first_visit": first_visit_customer_ids,
        "visit_trace": visit_trace,
        "cp_augmented": cp_augmented,
    }

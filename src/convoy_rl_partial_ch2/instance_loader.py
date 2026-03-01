"""Helpers to build one test instance TensorDict from test CSV + combined matrices."""

import torch

from tensordict import TensorDict

from src.convoy_rl_partial_ch2.myparser import parse_customer, parse_distance_matrix_csv


def load_vrptw_instance_from_csv(
    csv_path: str,
    vehicle_capacity: float,
    distance_matrix_csv: str | None = None,
    time_matrix_csv: str | None = None,
    depot_charge_rate_kwh_per_hour: float = 120.0,
    depot_charge_cost_per_kwh: float = 0.0,
    device: torch.device | str = "cpu",
):
    """Load one VRPTW instance from CSV into a TensorDict batch of size 1.

    Uses combined matrix indexing by node IDs:
      depot -> depot node_id from test CSV (default 0 if not provided)
      customers -> customer_id from test CSV
      charging points -> cp_id from test CSV
    """
    if vehicle_capacity <= 0:
        raise ValueError("--csv-vehicle-capacity must be > 0.")
    if distance_matrix_csv is None:
        raise ValueError("Combined mode requires test distance matrix CSV.")
    if depot_charge_rate_kwh_per_hour <= 0:
        raise ValueError("--ev-charge-rate-kwh-per-hour must be > 0.")
    if depot_charge_cost_per_kwh < 0:
        raise ValueError("depot_charge_cost_per_kwh must be >= 0.")

    depot, customers, charging_points = parse_customer(csv_path)
    loc_nodes = customers + charging_points
    all_nodes = [depot] + loc_nodes

    customer_ids = [c.get("customer_id") for c in customers]
    if any(cid is None for cid in customer_ids):
        raise ValueError("test-csv must include customer_id for combined matrix lookup.")
    customer_ids_int = [int(cid) for cid in customer_ids]
    cp_ids_int = [int(cp["cp_id"]) for cp in charging_points]
    depot_id = int(depot.get("node_id", 0))
    depot_cost = float(depot.get("charging_cost_per_kwh", depot_charge_cost_per_kwh))

    depot_xy = torch.tensor([[depot["x"], depot["y"]]], dtype=torch.float32, device=device)
    locs = torch.tensor(
        [[n["x"], n["y"]] for n in loc_nodes], dtype=torch.float32, device=device
    ).unsqueeze(0)
    demand_abs = torch.tensor(
        [n.get("demand", 0.0) for n in loc_nodes], dtype=torch.float32, device=device
    ).unsqueeze(0)
    demand = demand_abs / float(vehicle_capacity)
    durations = torch.tensor(
        [n["service_time"] for n in all_nodes], dtype=torch.float32, device=device
    ).unsqueeze(0)
    time_windows = torch.tensor(
        [[n["tw_start"], n["tw_end"]] for n in all_nodes],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    customer_reward_per_node = torch.tensor(
        [[n.get("reward", 0.0) for n in all_nodes]],
        dtype=torch.float32,
        device=device,
    )
    capacity = torch.tensor([[float(vehicle_capacity)]], dtype=torch.float32, device=device)

    global_nodes = torch.tensor(
        [[depot_id] + customer_ids_int + cp_ids_int], dtype=torch.long, device=device
    )

    dist_full = parse_distance_matrix_csv(distance_matrix_csv).to(device)
    idx_i = global_nodes[:, :, None].expand(-1, -1, global_nodes.shape[1])
    idx_j = global_nodes[:, None, :].expand(-1, global_nodes.shape[1], -1)
    inst_dist = dist_full[idx_i, idx_j]

    if time_matrix_csv is not None:
        time_full = parse_distance_matrix_csv(time_matrix_csv).to(device)
        inst_time = time_full[idx_i, idx_j]
    else:
        inst_time = inst_dist

    out = {
        "depot": depot_xy,
        "locs": locs,
        "demand": demand,
        "durations": durations,
        "time_windows": time_windows,
        "customer_reward_per_node": customer_reward_per_node,
        "capacity": capacity,
        "global_node_ids": global_nodes,
        "dist_matrix": inst_dist,
        "travel_time_matrix": inst_time,
    }

    # If CPs are explicitly present in test CSV, use them directly (no random sampling).
    if charging_points:
        total_nodes = 1 + len(loc_nodes)
        station_start = 1 + len(customers)
        station_mask = torch.zeros((1, total_nodes), dtype=torch.bool, device=device)
        station_mask[:, station_start:] = True

        charge_nodes_mask = station_mask.clone()
        charge_nodes_mask[:, 0] = True

        charge_rate_per_node = torch.zeros((1, total_nodes), dtype=torch.float32, device=device)
        charge_rate_per_node[:, 0] = float(depot_charge_rate_kwh_per_hour)
        charge_rate_per_node[:, station_start:] = torch.tensor(
            [float(cp["charge_rate_kwh_per_hour"]) for cp in charging_points],
            dtype=torch.float32,
            device=device,
        )

        charge_cost_per_kwh_per_node = torch.zeros(
            (1, total_nodes), dtype=torch.float32, device=device
        )
        charge_cost_per_kwh_per_node[:, 0] = depot_cost
        charge_cost_per_kwh_per_node[:, station_start:] = torch.tensor(
            [float(cp["charging_cost_per_kwh"]) for cp in charging_points],
            dtype=torch.float32,
            device=device,
        )

        cp_id_per_node = torch.full((1, total_nodes), -1, dtype=torch.long, device=device)
        cp_id_per_node[:, station_start:] = torch.tensor(
            cp_ids_int, dtype=torch.long, device=device
        )

        out["station_mask"] = station_mask
        out["charge_nodes_mask"] = charge_nodes_mask
        out["charge_rate_per_node"] = charge_rate_per_node
        out["charge_cost_per_kwh_per_node"] = charge_cost_per_kwh_per_node
        out["cp_id_per_node"] = cp_id_per_node

    return TensorDict(out, batch_size=[1])

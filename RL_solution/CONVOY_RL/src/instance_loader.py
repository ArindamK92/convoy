"""Helpers to build one test instance TensorDict from test CSV + combined matrices."""

import torch

from tensordict import TensorDict

from src.myparser import parse_customer, parse_distance_matrix_csv


def load_vrptw_instance_from_csv(
    csv_path: str,
    vehicle_capacity: float,
    distance_matrix_csv: str | None = None,
    time_matrix_csv: str | None = None,
    device: torch.device | str = "cpu",
):
    """Load one VRPTW instance from CSV into a TensorDict batch of size 1.

    Uses combined matrix indexing by node IDs:
      depot -> 0
      customers -> customer_id from test CSV
    """
    if vehicle_capacity <= 0:
        raise ValueError("--csv-vehicle-capacity must be > 0.")
    if distance_matrix_csv is None:
        raise ValueError("Combined mode requires test distance matrix CSV.")

    depot, customers = parse_customer(csv_path)
    all_nodes = [depot] + customers

    customer_ids = [c.get("customer_id") for c in customers]
    if any(cid is None for cid in customer_ids):
        raise ValueError("test-csv must include customer_id for combined matrix lookup.")

    depot_xy = torch.tensor([[depot["x"], depot["y"]]], dtype=torch.float32, device=device)
    locs = torch.tensor(
        [[c["x"], c["y"]] for c in customers], dtype=torch.float32, device=device
    ).unsqueeze(0)
    demand_abs = torch.tensor(
        [c["demand"] for c in customers], dtype=torch.float32, device=device
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
        [[0] + [int(cid) for cid in customer_ids]],
        dtype=torch.long,
        device=device,
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

    return TensorDict(out, batch_size=[1])

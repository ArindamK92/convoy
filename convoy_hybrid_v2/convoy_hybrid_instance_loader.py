"""CSV instance loading helpers for hybrid runner (customers+depot only)."""

from __future__ import annotations

import torch

from tensordict import TensorDict
from rl4co.envs import CVRPTWEnv

from src.convoy_rl_partial_ch.myparser import parse_customer, parse_distance_matrix_csv


def load_customers_only_instance_from_csv(
    csv_path: str,
    vehicle_capacity: float,
    distance_matrix_csv: str | None = None,
    time_matrix_csv: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[TensorDict, int]:
    """Load one CSV instance as depot+customers only, ignoring CP rows."""
    if vehicle_capacity <= 0:
        raise ValueError("--csv-vehicle-capacity must be > 0.")

    depot, customers, charging_points = parse_customer(csv_path)
    if not customers:
        raise ValueError("test-csv must contain at least one customer row.")

    customer_ids = [c.get("customer_id") for c in customers]
    if any(cid is None for cid in customer_ids):
        raise ValueError("test-csv must include customer_id for all customer rows.")

    customer_ids_int = [int(cid) for cid in customer_ids]
    depot_id = int(depot.get("node_id", 0))
    all_nodes = [depot] + customers
    cp_ids_int = [int(cp.get("cp_id")) for cp in charging_points if cp.get("cp_id") is not None]
    cp_locs = [
        [float(cp.get("x", 0.0)), float(cp.get("y", 0.0))]
        for cp in charging_points
        if cp.get("cp_id") is not None
    ]
    cp_rates = [
        float(cp.get("charge_rate_kwh_per_hour", 120.0))
        for cp in charging_points
        if cp.get("cp_id") is not None
    ]

    depot_xy = torch.tensor([[depot["x"], depot["y"]]], dtype=torch.float32, device=device)
    locs = torch.tensor(
        [[n["x"], n["y"]] for n in customers], dtype=torch.float32, device=device
    ).unsqueeze(0)
    demand_abs = torch.tensor(
        [n.get("demand", 0.0) for n in customers], dtype=torch.float32, device=device
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
        [[depot_id] + customer_ids_int], dtype=torch.long, device=device
    )

    out = {
        "depot": depot_xy,
        "locs": locs,
        "demand": demand,
        "durations": durations,
        "time_windows": time_windows,
        "customer_reward_per_node": customer_reward_per_node,
        "capacity": capacity,
        "global_node_ids": global_nodes,
    }
    if cp_ids_int:
        out["cp_global_ids"] = torch.tensor([cp_ids_int], dtype=torch.long, device=device)
        out["cp_locs"] = torch.tensor([cp_locs], dtype=torch.float32, device=device)
        out["cp_charge_rate_kwh_per_hour"] = torch.tensor(
            [cp_rates], dtype=torch.float32, device=device
        )
    if distance_matrix_csv is not None:
        dist_full = parse_distance_matrix_csv(distance_matrix_csv).to(device)
        idx_i = global_nodes[:, :, None].expand(-1, -1, global_nodes.shape[1])
        idx_j = global_nodes[:, None, :].expand(-1, global_nodes.shape[1], -1)
        out["dist_matrix"] = dist_full[idx_i, idx_j]
        out["full_dist_matrix"] = dist_full.unsqueeze(0)
        if time_matrix_csv is not None:
            time_full = parse_distance_matrix_csv(time_matrix_csv).to(device)
            out["travel_time_matrix"] = time_full[idx_i, idx_j]
            out["full_time_matrix"] = time_full.unsqueeze(0)
        else:
            out["travel_time_matrix"] = out["dist_matrix"]
            out["full_time_matrix"] = dist_full.unsqueeze(0)
    return TensorDict(out, batch_size=[1]), len(charging_points)


def build_fixed_instance(env: CVRPTWEnv, args) -> tuple[TensorDict, int]:
    """Create one fixed TensorDict instance used for train/val/test datasets."""
    prev_state = torch.random.get_rng_state()
    try:
        if args.fixed_instance_seed is not None:
            torch.manual_seed(int(args.fixed_instance_seed))

        if args.fixed_instance_csv:
            fixed_dist_csv = args.test_distance_matrix_csv or args.combined_dist_matrix_csv
            fixed_time_csv = (
                args.test_time_matrix_csv
                or args.combined_time_matrix_csv
                or args.combined_dist_matrix_csv
            )
            td_fixed, ignored_cp = load_customers_only_instance_from_csv(
                args.fixed_instance_csv,
                vehicle_capacity=args.csv_vehicle_capacity,
                distance_matrix_csv=fixed_dist_csv,
                time_matrix_csv=fixed_time_csv,
                device="cpu",
            )
            return td_fixed, ignored_cp

        td_fixed = env.generator._generate([1])
        return td_fixed, 0
    finally:
        torch.random.set_rng_state(prev_state)

"""Helpers to build one test instance TensorDict from CSV inputs."""

import torch

from tensordict import TensorDict

from src.convoy import convoy
from src.myparser import parse_customer, parse_distance_matrix_csv


def load_vrptw_instance_from_csv(
    csv_path: str,
    vehicle_capacity: float,
    distance_mode: str = "manhattan",
    distance_matrix_csv: str | None = None,
    time_matrix_csv: str | None = None,
    device: torch.device | str = "cpu",
):
    """Load one VRPTW instance from CSV into a TensorDict batch of size 1.

    CSV schema:
        is_depot,x,y,demand,tw_start,tw_end,service_time
        optional: reward
    """
    if vehicle_capacity <= 0:
        raise ValueError("--csv-vehicle-capacity must be > 0.")

    depot, customers = parse_customer(csv_path)
    all_nodes = [depot] + customers

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
    out = {
        "depot": depot_xy,
        "locs": locs,
        "demand": demand,
        "durations": durations,
        "time_windows": time_windows,
        "customer_reward_per_node": customer_reward_per_node,
        "capacity": capacity,
    }

    if distance_matrix_csv is not None:
        mat = parse_distance_matrix_csv(distance_matrix_csv)
        num_customers = locs.shape[1]
        if mat.shape[0] == num_customers + 1:
            inst = mat
        elif mat.shape[0] == num_customers:
            inst = torch.zeros((num_customers + 1, num_customers + 1), dtype=torch.float32)
            inst[1:, 1:] = mat
            depot_to_cust = convoy._distance(
                depot_xy[0:1], locs[0], distance_mode
            ).squeeze(0)
            inst[0, 1:] = depot_to_cust
            inst[1:, 0] = depot_to_cust
        elif mat.shape[0] > num_customers + 1:
            customer_ids = [c.get("customer_id") for c in customers]
            if any(cid is None for cid in customer_ids):
                raise ValueError(
                    "For larger test distance matrices (e.g., 201x201), add "
                    "`customer_id` column to test CSV so rows/cols can be selected."
                )
            if mat.shape[0] == mat.shape[1]:
                # Convention: if matrix includes depot, depot index is 0 and
                # customers are indexed by their customer_id (typically 1..N).
                if mat.shape[0] >= (max(customer_ids) + 1):
                    global_nodes = [0] + customer_ids
                    inst = mat[global_nodes][:, global_nodes]
                # Alternative convention: customer-only matrix with 0-based customer_id.
                elif mat.shape[0] > max(customer_ids):
                    cust_nodes = customer_ids
                    inst = torch.zeros((num_customers + 1, num_customers + 1), dtype=torch.float32)
                    cc = mat[cust_nodes][:, cust_nodes]
                    inst[1:, 1:] = cc
                    depot_to_cust = convoy._distance(
                        depot_xy[0:1], locs[0], distance_mode
                    ).squeeze(0)
                    inst[0, 1:] = depot_to_cust
                    inst[1:, 0] = depot_to_cust
                else:
                    raise ValueError("customer_id values exceed matrix size.")
            else:
                raise ValueError("distance matrix CSV must be square.")
        else:
            raise ValueError(
                "test distance matrix size must be either "
                f"{num_customers}x{num_customers} or "
                f"{num_customers + 1}x{num_customers + 1}"
            )
        out["dist_matrix"] = inst.unsqueeze(0).to(device)
    if time_matrix_csv is not None:
        tmat = parse_distance_matrix_csv(time_matrix_csv)
        num_customers = locs.shape[1]
        if tmat.shape[0] == num_customers + 1:
            inst_t = tmat
        elif tmat.shape[0] == num_customers:
            inst_t = torch.zeros((num_customers + 1, num_customers + 1), dtype=torch.float32)
            inst_t[1:, 1:] = tmat
            depot_to_cust = convoy._distance(
                depot_xy[0:1], locs[0], distance_mode
            ).squeeze(0)
            inst_t[0, 1:] = depot_to_cust
            inst_t[1:, 0] = depot_to_cust
        elif tmat.shape[0] > num_customers + 1:
            customer_ids = [c.get("customer_id") for c in customers]
            if any(cid is None for cid in customer_ids):
                raise ValueError(
                    "For larger test time matrices, add `customer_id` column "
                    "to test CSV so rows/cols can be selected."
                )
            if tmat.shape[0] >= (max(customer_ids) + 1):
                nodes = [0] + customer_ids
                inst_t = tmat[nodes][:, nodes]
            elif tmat.shape[0] > max(customer_ids):
                cust_nodes = customer_ids
                inst_t = torch.zeros((num_customers + 1, num_customers + 1), dtype=torch.float32)
                cc = tmat[cust_nodes][:, cust_nodes]
                inst_t[1:, 1:] = cc
                depot_to_cust = convoy._distance(
                    depot_xy[0:1], locs[0], distance_mode
                ).squeeze(0)
                inst_t[0, 1:] = depot_to_cust
                inst_t[1:, 0] = depot_to_cust
            else:
                raise ValueError("customer_id values exceed time matrix size.")
        else:
            raise ValueError(
                "test time matrix size must be either "
                f"{num_customers}x{num_customers} or "
                f"{num_customers + 1}x{num_customers + 1}"
            )
        out["travel_time_matrix"] = inst_t.unsqueeze(0).to(device)

    return TensorDict(out, batch_size=[1])

"""Combined-details-backed customer pool generator for sampled CVRPTW instances."""

import csv

import torch

from tensordict import TensorDict
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator

from src.convoy_rl_partial_ch2.myparser import parse_distance_matrix_csv


def parse_combined_details_rows(csv_path: str) -> list[dict]:
    """Parse rows from combined details schema."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Combined details CSV must contain headers.")
        required = {
            "id",
            "type",
            "lng",
            "lat",
            "first_receive_tm",
            "last_receive_tm",
            "service_time",
            "reward",
            "unit_charging_cost",
            "charge_rate_kwh_per_hour",
        }
        fields = {name.strip().lower() for name in reader.fieldnames}
        if not required.issubset(fields):
            raise ValueError(
                "Combined details CSV must contain columns: "
                "ID,type,lng,lat,first_receive_tm,last_receive_tm,service_time,reward,"
                "unit_charging_cost,charge_rate_kwh_per_hour"
            )

        rows: list[dict] = []
        for row in reader:
            row_norm = {k.strip().lower(): v for k, v in row.items()}
            rec = {
                "id": int(float(row_norm["id"])),
                "type": row_norm["type"].strip().lower(),
                "x": float(row_norm["lng"]),
                "y": float(row_norm["lat"]),
                "tw_start": float(row_norm["first_receive_tm"]),
                "tw_end": float(row_norm["last_receive_tm"]),
                "service_time": float(row_norm["service_time"]),
                "reward": float(row_norm["reward"]),
                "unit_charging_cost": float(row_norm["unit_charging_cost"]),
                "charge_rate_kwh_per_hour": float(row_norm["charge_rate_kwh_per_hour"]),
            }
            # Combined schema has no explicit demand. Use unit demand for customers.
            rec["demand"] = 1.0 if rec["type"] == "c" else 0.0
            rows.append(rec)
    if not rows:
        raise ValueError("Combined details CSV is empty.")
    return rows


class CSVCustomerPoolGenerator(CVRPTWGenerator):
    """Generate CVRPTW instances by sampling customers from combined details CSV."""

    def __init__(
        self,
        csv_path: str,
        sample_size: int,
        vehicle_capacity: float,
        max_time: float | None = None,
        distance_matrix_csv: str | None = None,
        time_matrix_csv: str | None = None,
    ):
        if sample_size <= 0:
            raise ValueError("--customer-num must be > 0.")
        if vehicle_capacity <= 0:
            raise ValueError("--pool-vehicle-capacity must be > 0.")
        if distance_matrix_csv is None:
            raise ValueError("Combined mode requires a distance matrix CSV.")

        rows = parse_combined_details_rows(csv_path)
        depot_rows = [r for r in rows if r["type"] == "d"]
        customer_rows = [r for r in rows if r["type"] == "c"]
        cp_rows = [r for r in rows if r["type"] == "f"]
        if len(depot_rows) != 1:
            raise ValueError("Combined details CSV must contain exactly one depot row (type=d).")
        if not customer_rows:
            raise ValueError("Combined details CSV must contain customer rows (type=c).")
        if sample_size > len(customer_rows):
            raise ValueError(
                f"--customer-num={sample_size} is larger than available "
                f"customers in CSV ({len(customer_rows)})."
            )

        depot = depot_rows[0]
        self.depot_id = int(depot["id"])
        self.depot_charge_cost_per_kwh = float(depot["unit_charging_cost"])
        self.cp_rows = [
            {
                "cp_id": int(r["id"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "charge_rate_kwh_per_hour": float(r["charge_rate_kwh_per_hour"]),
                "charging_cost_per_kwh": float(r["unit_charging_cost"]),
            }
            for r in cp_rows
        ]

        csv_max_time = max([depot["tw_end"], *[c["tw_end"] for c in customer_rows]])
        chosen_max_time = float(max_time if max_time is not None else csv_max_time)

        super().__init__(
            num_loc=sample_size,
            vehicle_capacity=vehicle_capacity,
            max_time=chosen_max_time,
            scale=False,
        )

        self.pool_size = len(customer_rows)
        self.sample_size = sample_size
        self.depot_xy = torch.tensor([depot["x"], depot["y"]], dtype=torch.float32)
        self.depot_tw = torch.tensor(
            [depot["tw_start"], depot["tw_end"]], dtype=torch.float32
        )
        self.depot_service = float(depot["service_time"])
        self.depot_reward = float(depot.get("reward", 0.0))

        self.pool_locs = torch.tensor(
            [[c["x"], c["y"]] for c in customer_rows], dtype=torch.float32
        )
        self.pool_customer_ids = [int(c["id"]) for c in customer_rows]
        self.pool_demand_abs = torch.tensor(
            [c["demand"] for c in customer_rows], dtype=torch.float32
        )
        self.pool_durations = torch.tensor(
            [c["service_time"] for c in customer_rows], dtype=torch.float32
        )
        self.pool_time_windows = torch.tensor(
            [[c["tw_start"], c["tw_end"]] for c in customer_rows], dtype=torch.float32
        )
        self.pool_rewards = torch.tensor(
            [c.get("reward", 0.0) for c in customer_rows], dtype=torch.float32
        )

        self.min_time = float(
            min(self.depot_tw[0].item(), self.pool_time_windows[:, 0].min().item())
        )
        self.max_time = chosen_max_time

        self.distance_matrix_pool = parse_distance_matrix_csv(distance_matrix_csv)
        self.time_matrix_pool = (
            parse_distance_matrix_csv(time_matrix_csv)
            if time_matrix_csv
            else self.distance_matrix_pool
        )
        max_needed = max([self.depot_id] + self.pool_customer_ids)
        if self.distance_matrix_pool.shape[0] <= max_needed:
            raise ValueError(
                f"Combined distance matrix must include index {max_needed}; "
                f"got shape {tuple(self.distance_matrix_pool.shape)}."
            )
        if self.time_matrix_pool.shape[0] <= max_needed:
            raise ValueError(
                f"Combined time matrix must include index {max_needed}; "
                f"got shape {tuple(self.time_matrix_pool.shape)}."
            )

    def _generate(self, batch_size):
        bs = int(batch_size[0]) if isinstance(batch_size, list) else int(batch_size)

        idx = torch.stack(
            [torch.randperm(self.pool_size)[: self.sample_size] for _ in range(bs)],
            dim=0,
        )

        locs = self.pool_locs[idx]
        demand = self.pool_demand_abs[idx] / float(self.capacity)
        durations_customers = self.pool_durations[idx]
        tw_customers = self.pool_time_windows[idx]
        rewards_customers = self.pool_rewards[idx]

        depot = self.depot_xy.unsqueeze(0).expand(bs, -1)
        depot_duration = torch.full((bs, 1), self.depot_service, dtype=torch.float32)
        durations = torch.cat([depot_duration, durations_customers], dim=1)

        depot_tw = self.depot_tw.unsqueeze(0).unsqueeze(1).expand(bs, 1, 2)
        time_windows = torch.cat([depot_tw, tw_customers], dim=1)
        depot_reward = torch.full((bs, 1), self.depot_reward, dtype=torch.float32)
        customer_reward_per_node = torch.cat([depot_reward, rewards_customers], dim=1)

        global_nodes = torch.zeros((bs, self.sample_size + 1), dtype=torch.long)
        global_nodes[:, 0] = int(self.depot_id)
        for b in range(bs):
            selected = idx[b].tolist()
            selected_ids = [self.pool_customer_ids[i] for i in selected]
            global_nodes[b, 1:] = torch.tensor(selected_ids, dtype=torch.long)

        idx_i = global_nodes[:, :, None].expand(-1, -1, global_nodes.shape[1])
        idx_j = global_nodes[:, None, :].expand(-1, global_nodes.shape[1], -1)
        dist_matrix = self.distance_matrix_pool[idx_i, idx_j]
        time_matrix = self.time_matrix_pool[idx_i, idx_j]

        capacity = torch.full((bs, 1), float(self.capacity), dtype=torch.float32)
        out = {
            "locs": locs,
            "depot": depot,
            "demand": demand,
            "capacity": capacity,
            "durations": durations,
            "time_windows": time_windows,
            "customer_reward_per_node": customer_reward_per_node,
            "dist_matrix": dist_matrix,
            "travel_time_matrix": time_matrix,
            "global_node_ids": global_nodes,
        }
        return TensorDict(out, batch_size=[bs])

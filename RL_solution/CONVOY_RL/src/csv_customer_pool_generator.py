"""CSV-backed customer pool generator for sampled CVRPTW training instances."""

import torch

from tensordict import TensorDict
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator

from src.convoy import convoy
from src.myparser import parse_customer, parse_distance_matrix_csv


class CSVCustomerPoolGenerator(CVRPTWGenerator):
    """Generate CVRPTW instances by sampling customers from a CSV pool.

    For each generated instance, sample `sample_size` customers uniformly without
    replacement from the full CSV customer pool.
    """

    def __init__(
        self,
        csv_path: str,
        sample_size: int,
        vehicle_capacity: float,
        max_time: float | None = None,
        distance_matrix_csv: str | None = None,
        time_matrix_csv: str | None = None,
        distance_mode_for_depot: str = "manhattan",
    ):
        """Load customer pool data/matrices and configure sampling-based generation."""
        if sample_size <= 0:
            raise ValueError("--pool-sample-size must be > 0.")
        if vehicle_capacity <= 0:
            raise ValueError("--pool-vehicle-capacity must be > 0.")

        depot, customers = parse_customer(csv_path)
        if sample_size > len(customers):
            raise ValueError(
                f"--pool-sample-size={sample_size} is larger than available "
                f"customers in CSV ({len(customers)})."
            )

        csv_max_time = max([depot["tw_end"], *[c["tw_end"] for c in customers]])
        chosen_max_time = float(max_time if max_time is not None else csv_max_time)

        super().__init__(
            num_loc=sample_size,
            vehicle_capacity=vehicle_capacity,
            max_time=chosen_max_time,
            scale=False,
        )

        self.pool_size = len(customers)
        self.sample_size = sample_size
        self.distance_mode_for_depot = distance_mode_for_depot
        self.depot_xy = torch.tensor([depot["x"], depot["y"]], dtype=torch.float32)
        self.depot_tw = torch.tensor(
            [depot["tw_start"], depot["tw_end"]], dtype=torch.float32
        )
        self.depot_service = float(depot["service_time"])
        self.depot_reward = float(depot.get("reward", 0.0))

        self.pool_locs = torch.tensor(
            [[c["x"], c["y"]] for c in customers], dtype=torch.float32
        )
        self.pool_customer_ids = [c.get("customer_id") for c in customers]
        self.pool_demand_abs = torch.tensor(
            [c["demand"] for c in customers], dtype=torch.float32
        )
        self.pool_durations = torch.tensor(
            [c["service_time"] for c in customers], dtype=torch.float32
        )
        self.pool_time_windows = torch.tensor(
            [[c["tw_start"], c["tw_end"]] for c in customers], dtype=torch.float32
        )
        self.pool_rewards = torch.tensor(
            [c.get("reward", 0.0) for c in customers], dtype=torch.float32
        )

        self.min_time = float(
            min(self.depot_tw[0].item(), self.pool_time_windows[:, 0].min().item())
        )
        self.max_time = chosen_max_time
        self.distance_matrix_pool = None
        self.matrix_has_depot = False
        if distance_matrix_csv:
            mat = parse_distance_matrix_csv(distance_matrix_csv)
            if mat.shape[0] == self.pool_size:
                self.distance_matrix_pool = mat
                self.matrix_has_depot = False
            elif mat.shape[0] == self.pool_size + 1:
                self.distance_matrix_pool = mat
                self.matrix_has_depot = True
            else:
                raise ValueError(
                    "Distance matrix size must be either "
                    f"{self.pool_size}x{self.pool_size} (customers only) or "
                    f"{self.pool_size + 1}x{self.pool_size + 1} (with depot)."
                )
        self.time_matrix_pool = None
        self.time_matrix_has_depot = False
        if time_matrix_csv:
            tmat = parse_distance_matrix_csv(time_matrix_csv)
            if tmat.shape[0] == self.pool_size:
                self.time_matrix_pool = tmat
                self.time_matrix_has_depot = False
            elif tmat.shape[0] == self.pool_size + 1:
                self.time_matrix_pool = tmat
                self.time_matrix_has_depot = True
            else:
                raise ValueError(
                    "Time matrix size must be either "
                    f"{self.pool_size}x{self.pool_size} (customers only) or "
                    f"{self.pool_size + 1}x{self.pool_size + 1} (with depot)."
                )

    def _generate(self, batch_size):
        """Sample customer subsets and build a batch of CVRPTW training instances."""
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

        capacity = torch.full((bs, 1), float(self.capacity), dtype=torch.float32)
        out = {
            "locs": locs,
            "depot": depot,
            "demand": demand,
            "capacity": capacity,
            "durations": durations,
            "time_windows": time_windows,
            "customer_reward_per_node": customer_reward_per_node,
        }

        def build_instance_matrix(
            full_matrix: torch.Tensor,
            has_depot: bool,
            selected_idx: torch.Tensor,
            selected_ids: list[int | None],
        ) -> torch.Tensor:
            """Create one depot+customers matrix aligned to the sampled instance."""
            inst_mat = torch.zeros((self.sample_size + 1, self.sample_size + 1))
            if has_depot:
                if all(cid is not None for cid in selected_ids):
                    nodes = torch.tensor([0] + [int(x) for x in selected_ids], dtype=torch.long)
                else:
                    nodes = torch.cat([torch.tensor([0], dtype=torch.long), selected_idx + 1])
                inst_mat = full_matrix[nodes][:, nodes]
            else:
                if all(cid is not None for cid in selected_ids) and max(selected_ids) < self.pool_size:
                    cust_nodes = torch.tensor([int(x) for x in selected_ids], dtype=torch.long)
                else:
                    cust_nodes = selected_idx
                cc = full_matrix[cust_nodes][:, cust_nodes]
                inst_mat[1:, 1:] = cc
                depot_to_cust = convoy._distance(
                    self.depot_xy[None, :], self.pool_locs[selected_idx], self.distance_mode_for_depot
                ).squeeze(0)
                inst_mat[0, 1:] = depot_to_cust
                inst_mat[1:, 0] = depot_to_cust
            return inst_mat

        if self.distance_matrix_pool is not None:
            dmat = torch.zeros((bs, self.sample_size + 1, self.sample_size + 1))
            for b in range(bs):
                selected = idx[b]
                selected_ids = [self.pool_customer_ids[i] for i in selected.tolist()]
                dmat[b] = build_instance_matrix(
                    self.distance_matrix_pool, self.matrix_has_depot, selected, selected_ids
                )
            out["dist_matrix"] = dmat
        if self.time_matrix_pool is not None:
            tmat = torch.zeros((bs, self.sample_size + 1, self.sample_size + 1))
            for b in range(bs):
                selected = idx[b]
                selected_ids = [self.pool_customer_ids[i] for i in selected.tolist()]
                tmat[b] = build_instance_matrix(
                    self.time_matrix_pool, self.time_matrix_has_depot, selected, selected_ids
                )
            out["travel_time_matrix"] = tmat

        return TensorDict(out, batch_size=[bs])

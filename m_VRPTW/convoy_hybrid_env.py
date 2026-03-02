"""Matrix-aware CVRPTW environment for hybrid runner."""

from __future__ import annotations

import torch

from tensordict import TensorDict
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from rl4co.utils.ops import gather_by_index


class MatrixCVRPTWEnv(CVRPTWEnv):
    """CVRPTW variant that uses provided distance/time matrices when available."""

    # Keep canonical env name so RL4CO can pick the correct init embeddings.
    name = "cvrptw"

    def __init__(self, *args, num_evs: int | None = None, **kwargs):
        """Initialize matrix-aware env with optional fleet-size cap."""
        super().__init__(*args, **kwargs)
        self.num_evs = int(num_evs) if num_evs is not None else None

    def _reset(self, td: TensorDict | None = None, batch_size=None) -> TensorDict:
        """Reset state and carry matrix/global-id tensors for matrix-based rollout."""
        device = td.device
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.generator.vehicle_capacity, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=device,
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
                "used_vehicles": torch.zeros(
                    (*batch_size, 1), dtype=torch.long, device=device
                ),
            },
            batch_size=batch_size,
        )
        for key in [
            "dist_matrix",
            "travel_time_matrix",
            "global_node_ids",
            "customer_reward_per_node",
        ]:
            if key in td.keys():
                td_reset.set(key, td[key])
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Mask by capacity/visited + matrix travel-time feasibility when available."""
        not_masked = CVRPEnv.get_action_mask(td)
        if "travel_time_matrix" not in td.keys():
            mask = super().get_action_mask(td)
        else:
            current_loc = gather_by_index(td["locs"], td["current_node"])
            travel_row = gather_by_index(td["travel_time_matrix"], td["current_node"], dim=1)
            travel_time = travel_row.squeeze(1)
            if "dist_matrix" in td.keys():
                dist_row = gather_by_index(td["dist_matrix"], td["current_node"], dim=1)
                dist = dist_row.squeeze(1)
            else:
                dist = travel_time

            td.update({"current_loc": current_loc, "distances": dist, "travel_times": travel_time})
            can_reach_in_time = td["current_time"] + travel_time <= td["time_windows"][..., 1]
            mask = not_masked & can_reach_in_time

        if self.num_evs is not None and self.num_evs > 0:
            at_depot = td["current_node"].squeeze(-1) == 0
            unserved_exists = (td["visited"][..., 1:] == 0).any(dim=-1)
            vehicles_exhausted = td["used_vehicles"].squeeze(-1) >= self.num_evs
            block_new_route = at_depot & unserved_exists & vehicles_exhausted
            if block_new_route.any():
                mask = mask.clone()
                block_expanded = block_new_route.unsqueeze(-1)
                mask[..., 1:] = torch.where(
                    block_expanded,
                    torch.zeros_like(mask[..., 1:]),
                    mask[..., 1:],
                )
                # Keep depot feasible so decoder never sees an all-false row.
                mask[..., 0] = torch.where(
                    block_new_route,
                    torch.ones_like(mask[..., 0], dtype=torch.bool),
                    mask[..., 0],
                )

        if "done" in td.keys():
            done = td["done"].to(torch.bool)
            if done.any():
                mask = mask.clone()
                done_expanded = done.unsqueeze(-1)
                mask = torch.where(done_expanded, torch.zeros_like(mask), mask)
                mask[..., 0] = torch.where(
                    done,
                    torch.ones_like(mask[..., 0], dtype=torch.bool),
                    mask[..., 0],
                )

        # Safety fallback for decode strategies: never return all-false action rows.
        no_action = ~mask.any(dim=-1)
        if no_action.any():
            mask = mask.clone()
            no_action_expanded = no_action.unsqueeze(-1)
            mask = torch.where(no_action_expanded, torch.zeros_like(mask), mask)
            mask[..., 0] = torch.where(
                no_action,
                torch.ones_like(mask[..., 0], dtype=torch.bool),
                mask[..., 0],
            )

        return mask

    def _step(self, td: TensorDict) -> TensorDict:
        """Update current time using matrix travel time, then apply CVRP transition."""
        prev_node = td["current_node"].clone()
        action = td["action"].clone()

        if "travel_time_matrix" not in td.keys():
            td_out = super()._step(td)
        else:
            batch_size = td["locs"].shape[0]
            if "travel_times" in td.keys():
                travel_time = gather_by_index(td["travel_times"], td["action"]).reshape(
                    [batch_size, 1]
                )
            else:
                travel_row = gather_by_index(td["travel_time_matrix"], td["current_node"], dim=1)
                travel_time = gather_by_index(travel_row.squeeze(1), td["action"]).reshape(
                    [batch_size, 1]
                )

            duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
            start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
                [batch_size, 1]
            )
            td["current_time"] = (td["action"][:, None] != 0) * (
                torch.max(td["current_time"] + travel_time, start_times) + duration
            )
            td_out = CVRPEnv._step(self, td)

        route_start = (prev_node.squeeze(-1) == 0) & (action != 0)
        used_vehicles = td_out["used_vehicles"] + route_start.to(torch.long).unsqueeze(-1)
        td_out.set("used_vehicles", used_vehicles)

        if self.num_evs is not None and self.num_evs > 0:
            unserved_exists = (td_out["visited"][..., 1:] == 0).any(dim=-1)
            at_depot_now = td_out["current_node"].squeeze(-1) == 0
            vehicles_exhausted = used_vehicles.squeeze(-1) >= self.num_evs
            forced_done = at_depot_now & unserved_exists & vehicles_exhausted
            if forced_done.any():
                td_out.set("done", td_out["done"] | forced_done)

        td_out.set("action_mask", self.get_action_mask(td_out))
        return td_out

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Use matrix tour length reward and penalize unserved customers."""
        actions = actions.to(torch.long)
        if "dist_matrix" not in td.keys():
            reward = super()._get_reward(td, actions)
        else:
            batch_size = actions.shape[0]
            device = actions.device
            depot = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            seq = torch.cat([depot, actions], dim=1)

            from_nodes = seq[:, :-1]
            to_nodes = seq[:, 1:]
            batch_idx = torch.arange(batch_size, device=device)[:, None]
            step_dist = td["dist_matrix"][batch_idx, from_nodes, to_nodes]

            last_nodes = seq[:, -1]
            close_dist = td["dist_matrix"][
                torch.arange(batch_size, device=device),
                last_nodes,
                torch.zeros_like(last_nodes),
            ]
            reward = -(step_dist.sum(dim=1) + close_dist)

        n_customers = int(td["demand"].shape[-1])
        valid_customer = (actions > 0) & (actions <= n_customers)
        visited_by_actions = torch.zeros(
            (actions.shape[0], n_customers + 1), dtype=torch.bool, device=actions.device
        )
        customer_actions = torch.where(
            valid_customer, actions, torch.zeros_like(actions)
        )
        visited_by_actions.scatter_(1, customer_actions, True)
        served_count = visited_by_actions[:, 1:].sum(dim=-1)
        unserved_count = n_customers - served_count
        if unserved_count.any():
            if "dist_matrix" in td.keys():
                penalty_unit = td["dist_matrix"][:, 0, 1:].mean(dim=-1) * 2.0
            else:
                penalty_unit = torch.full_like(reward, 1_000.0)
            unserved_penalty = unserved_count.to(reward.dtype) * penalty_unit
            reward = reward - unserved_penalty
        return reward

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

    def __init__(
        self,
        *args,
        num_evs: int | None = None,
        battery_capacity_kwh: float = 30.0,
        reserve_soc_kwh: float = 0.0,
        energy_rate_kwh_per_distance: float = 0.00025,
        default_cp_charge_rate_kwh_per_hour: float = 120.0,
        **kwargs,
    ):
        """Initialize matrix-aware env with EV/SOC dynamics and optional fleet-size cap."""
        super().__init__(*args, **kwargs)
        self.num_evs = int(num_evs) if num_evs is not None else None
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.reserve_soc_kwh = float(reserve_soc_kwh)
        self.effective_battery_kwh = max(
            self.battery_capacity_kwh - self.reserve_soc_kwh, 0.0
        )
        self.energy_rate_kwh_per_distance = float(energy_rate_kwh_per_distance)
        self.default_cp_charge_rate_kwh_per_hour = float(
            default_cp_charge_rate_kwh_per_hour
        )
        self._cached_full_dist: dict[str, torch.Tensor] = {}
        self._cached_full_time: dict[str, torch.Tensor] = {}

    @staticmethod
    def _matrix_lookup(
        matrix: torch.Tensor, from_ids: torch.Tensor, to_ids: torch.Tensor
    ) -> torch.Tensor:
        """Lookup values from [N,N] or [B,N,N] matrix for vectorized indices."""
        if matrix.dim() == 2:
            if to_ids.dim() == 1:
                return matrix[from_ids, to_ids]
            if to_ids.dim() == 2 and from_ids.dim() == 1:
                return matrix[from_ids.unsqueeze(-1), to_ids]
            if to_ids.dim() == 3 and from_ids.dim() == 2:
                return matrix[from_ids.unsqueeze(-1), to_ids]
            return matrix[from_ids, to_ids]

        bsz = from_ids.shape[0]
        bidx = torch.arange(bsz, device=from_ids.device)
        if to_ids.dim() == 1:
            return matrix[bidx, from_ids, to_ids]
        if to_ids.dim() == 2 and from_ids.dim() == 1:
            return matrix[bidx.unsqueeze(-1), from_ids.unsqueeze(-1), to_ids]
        if to_ids.dim() == 3 and from_ids.dim() == 2:
            return matrix[
                bidx.view(bsz, 1, 1), from_ids.unsqueeze(-1), to_ids
            ]
        return matrix[bidx, from_ids, to_ids]

    def _get_full_dist_matrix(self, td: TensorDict) -> torch.Tensor | None:
        """Return full global distance matrix from td or generator."""
        ref_device = (
            td["global_node_ids"].device
            if "global_node_ids" in td.keys()
            else td.device
        )
        if "full_dist_matrix" in td.keys():
            return td["full_dist_matrix"].to(ref_device)
        if not hasattr(self.generator, "distance_matrix_pool"):
            return None
        device_key = str(ref_device)
        if device_key not in self._cached_full_dist:
            self._cached_full_dist[device_key] = self.generator.distance_matrix_pool.to(
                ref_device
            )
        return self._cached_full_dist[device_key]

    def _get_full_time_matrix(self, td: TensorDict) -> torch.Tensor | None:
        """Return full global time matrix from td or generator."""
        ref_device = (
            td["global_node_ids"].device
            if "global_node_ids" in td.keys()
            else td.device
        )
        if "full_time_matrix" in td.keys():
            return td["full_time_matrix"].to(ref_device)
        if not hasattr(self.generator, "time_matrix_pool"):
            return None
        device_key = str(ref_device)
        if device_key not in self._cached_full_time:
            self._cached_full_time[device_key] = self.generator.time_matrix_pool.to(
                ref_device
            )
        return self._cached_full_time[device_key]

    def _get_cp_metadata(
        self, td: TensorDict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return CP ids, charge-rates, and CP coordinates, batched by instance."""
        bs = int(td["global_node_ids"].shape[0]) if "global_node_ids" in td.keys() else 1
        device = (
            td["locs"].device
            if "locs" in td.keys()
            else (td["global_node_ids"].device if "global_node_ids" in td.keys() else td.device)
        )

        if "cp_global_ids" in td.keys():
            cp_ids = td["cp_global_ids"]
            if cp_ids.dim() == 1:
                cp_ids = cp_ids.unsqueeze(0)
            if cp_ids.shape[0] == 1 and bs > 1:
                cp_ids = cp_ids.expand(bs, -1)

            if "cp_charge_rate_kwh_per_hour" in td.keys():
                cp_rates = td["cp_charge_rate_kwh_per_hour"]
                if cp_rates.dim() == 1:
                    cp_rates = cp_rates.unsqueeze(0)
                if cp_rates.shape[0] == 1 and bs > 1:
                    cp_rates = cp_rates.expand(bs, -1)
            else:
                cp_rates = torch.full(
                    cp_ids.shape,
                    float(self.default_cp_charge_rate_kwh_per_hour),
                    dtype=torch.float32,
                    device=device,
                )

            cp_locs = td["cp_locs"] if "cp_locs" in td.keys() else None
            if cp_locs is not None:
                if cp_locs.dim() == 2:
                    cp_locs = cp_locs.unsqueeze(0)
                if cp_locs.shape[0] == 1 and bs > 1:
                    cp_locs = cp_locs.expand(bs, -1, -1)
            return cp_ids.to(device), cp_rates.to(device), (
                cp_locs.to(device) if cp_locs is not None else None
            )

        cp_rows = getattr(self.generator, "cp_rows", None)
        if not cp_rows:
            return None, None, None

        cp_ids_1d = torch.tensor(
            [int(cp["cp_id"]) for cp in cp_rows], dtype=torch.long, device=device
        )
        cp_rates_1d = torch.tensor(
            [
                float(cp.get("charge_rate_kwh_per_hour", self.default_cp_charge_rate_kwh_per_hour))
                for cp in cp_rows
            ],
            dtype=torch.float32,
            device=device,
        )
        cp_locs_2d = torch.tensor(
            [[float(cp["x"]), float(cp["y"])] for cp in cp_rows],
            dtype=torch.float32,
            device=device,
        )
        return (
            cp_ids_1d.unsqueeze(0).expand(bs, -1),
            cp_rates_1d.unsqueeze(0).expand(bs, -1),
            cp_locs_2d.unsqueeze(0).expand(bs, -1, -1),
        )

    def _initialize_cp_support(self, td: TensorDict) -> None:
        """Precompute nearest-CP map for all local nodes and min energy to any CP."""
        if "global_node_ids" not in td.keys() or "locs" not in td.keys():
            return

        cp_ids, cp_rates, cp_locs = self._get_cp_metadata(td)
        if cp_ids is None or cp_ids.numel() == 0 or cp_locs is None:
            return

        node_xy = td["locs"]  # [B, N_local, 2], local node 0 is depot.
        delta = node_xy.unsqueeze(2) - cp_locs.unsqueeze(1)  # [B, N_local, C, 2]
        nearest_cp_idx = torch.argmin((delta * delta).sum(dim=-1), dim=-1)  # [B, N_local]
        nearest_cp_global = torch.gather(cp_ids, 1, nearest_cp_idx)
        nearest_cp_rate = torch.gather(cp_rates, 1, nearest_cp_idx)

        td.set("cp_global_ids", cp_ids)
        td.set("cp_charge_rate_kwh_per_hour", cp_rates)
        td.set("nearest_cp_global_id_per_local_node", nearest_cp_global)
        td.set("nearest_cp_charge_rate_per_local_node", nearest_cp_rate)

        full_dist = self._get_full_dist_matrix(td)
        if full_dist is None:
            return

        global_nodes = td["global_node_ids"]  # [B, N_local]
        if full_dist.dim() == 2:
            dist_to_cp = full_dist[global_nodes.unsqueeze(-1), cp_ids.unsqueeze(1)]
        else:
            bs = global_nodes.shape[0]
            bidx = torch.arange(bs, device=td.device).view(bs, 1, 1)
            dist_to_cp = full_dist[
                bidx, global_nodes.unsqueeze(-1), cp_ids.unsqueeze(1)
            ]

        min_dist_to_cp = dist_to_cp.min(dim=-1).values
        td.set(
            "min_energy_to_any_cp_per_local_node",
            torch.clamp(min_dist_to_cp * self.energy_rate_kwh_per_distance, min=0.0),
        )

    def _reset(self, td: TensorDict | None = None, batch_size=None) -> TensorDict:
        """Reset state and carry matrix/global-id tensors for matrix-based rollout."""
        if td is None:
            raise ValueError("MatrixCVRPTWEnv._reset expected a TensorDict, got None.")
        device = (
            td["depot"].device
            if "depot" in td.keys()
            else td.device
        )
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
                "current_soc_kwh": torch.full(
                    (*batch_size, 1),
                    float(self.effective_battery_kwh),
                    dtype=torch.float32,
                    device=device,
                ),
                "travel_distance_accum": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
            },
            batch_size=batch_size,
        )
        for key in [
            "dist_matrix",
            "travel_time_matrix",
            "global_node_ids",
            "customer_reward_per_node",
            "cp_global_ids",
            "cp_locs",
            "cp_charge_rate_kwh_per_hour",
            "full_dist_matrix",
            "full_time_matrix",
        ]:
            if key in td.keys():
                value = td[key]
                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                td_reset.set(key, value)
        self._initialize_cp_support(td_reset)
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

        # SoC-aware feasibility with CP-assist: allow target if direct is safe, or if a
        # detour through at least one reachable CP can make target reachable.
        if "current_soc_kwh" in td.keys() and "global_node_ids" in td.keys():
            if "nearest_cp_global_id_per_local_node" not in td.keys():
                self._initialize_cp_support(td)

            if (
                "distances" in td.keys()
                and "min_energy_to_any_cp_per_local_node" in td.keys()
                and "cp_global_ids" in td.keys()
                and td["cp_global_ids"].shape[-1] > 0
            ):
                eps = 1e-9
                rate = float(self.energy_rate_kwh_per_distance)
                bs = td["global_node_ids"].shape[0]
                batch_idx = torch.arange(bs, device=td.device)

                soc = td["current_soc_kwh"].squeeze(-1)  # [B]
                direct_energy = torch.clamp(td["distances"] * rate, min=0.0)  # [B, N]
                current_local = td["current_node"].squeeze(-1)
                action_global = td["global_node_ids"]  # all local candidates as global IDs
                current_global = action_global[batch_idx, current_local]
                min_energy_to_cp_target = td["min_energy_to_any_cp_per_local_node"]  # [B, N]

                is_depot_target = torch.zeros_like(direct_energy, dtype=torch.bool)
                is_depot_target[:, 0] = True

                direct_reachable = direct_energy <= (soc.unsqueeze(-1) + eps)
                safe_after_customer = (
                    (soc.unsqueeze(-1) - direct_energy)
                    >= (min_energy_to_cp_target - eps)
                )
                direct_safe = torch.where(
                    is_depot_target, direct_reachable, direct_reachable & safe_after_customer
                )

                cp_ids = td["cp_global_ids"]  # [B, C]
                full_dist = self._get_full_dist_matrix(td)
                if full_dist is not None:
                    # Current -> each CP
                    if full_dist.dim() == 2:
                        dist_cur_to_cp = full_dist[
                            current_global.unsqueeze(-1), cp_ids
                        ]  # [B, C]
                        dist_cp_to_target = full_dist[
                            cp_ids.unsqueeze(-1), action_global.unsqueeze(1)
                        ]  # [B, C, N]
                    else:
                        bidx = batch_idx.view(bs, 1)
                        dist_cur_to_cp = full_dist[bidx, current_global.unsqueeze(-1), cp_ids]
                        bidx3 = batch_idx.view(bs, 1, 1)
                        dist_cp_to_target = full_dist[
                            bidx3, cp_ids.unsqueeze(-1), action_global.unsqueeze(1)
                        ]

                    energy_cur_to_cp = torch.clamp(dist_cur_to_cp * rate, min=0.0)
                    energy_cp_to_target = torch.clamp(dist_cp_to_target * rate, min=0.0)

                    cp_reachable_now = energy_cur_to_cp <= (soc.unsqueeze(-1) + eps)  # [B, C]
                    cp_to_target_feasible = energy_cp_to_target <= (
                        float(self.effective_battery_kwh) + eps
                    )  # [B, C, N]
                    feasible_via_cp = (
                        cp_reachable_now.unsqueeze(-1) & cp_to_target_feasible
                    ).any(dim=1)  # [B, N]

                    mask = mask & (direct_safe | feasible_via_cp)

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
        """Apply action with time-window dynamics and optional CP detour for SoC safety."""
        prev_node = td["current_node"].clone()
        action = td["action"].clone()

        if "travel_time_matrix" not in td.keys():
            td_out = super()._step(td)
        else:
            batch_size = td["locs"].shape[0]
            batch_idx = torch.arange(batch_size, device=td.device)
            eps = 1e-9
            rate = float(self.energy_rate_kwh_per_distance)

            current_local = td["current_node"].squeeze(-1)
            action_local = td["action"].to(torch.long)
            global_nodes = td["global_node_ids"]
            current_global = global_nodes[batch_idx, current_local]
            target_global = global_nodes[batch_idx, action_local]

            full_dist = self._get_full_dist_matrix(td)
            full_time = self._get_full_time_matrix(td)
            if full_dist is None or full_time is None:
                # Fallback to local matrices if full global matrices are unavailable.
                direct_dist = gather_by_index(
                    gather_by_index(td["dist_matrix"], td["current_node"], dim=1).squeeze(1),
                    td["action"],
                ).reshape([batch_size])
                direct_time = gather_by_index(
                    gather_by_index(td["travel_time_matrix"], td["current_node"], dim=1).squeeze(1),
                    td["action"],
                ).reshape([batch_size])
            else:
                direct_dist = self._matrix_lookup(full_dist, current_global, target_global).reshape(
                    [batch_size]
                )
                direct_time = self._matrix_lookup(full_time, current_global, target_global).reshape(
                    [batch_size]
                )

            direct_energy = torch.clamp(direct_dist * rate, min=0.0)
            soc_prev = td["current_soc_kwh"].squeeze(-1)
            min_energy_target_cp = td.get(
                "min_energy_to_any_cp_per_local_node",
                torch.zeros_like(global_nodes, dtype=torch.float32),
            )[batch_idx, action_local]

            is_customer_action = action_local != 0
            needs_cp = (direct_energy > (soc_prev + eps)) | (
                is_customer_action
                & ((soc_prev - direct_energy) < (min_energy_target_cp - eps))
            )

            # Defaults: direct leg.
            step_dist = direct_dist
            step_time = direct_time
            soc_at_target = torch.clamp(soc_prev - direct_energy, min=0.0)
            cp_detour_used = torch.zeros_like(needs_cp, dtype=torch.bool)

            if needs_cp.any():
                if "nearest_cp_global_id_per_local_node" not in td.keys():
                    self._initialize_cp_support(td)

                cp_ids, cp_rates, _ = self._get_cp_metadata(td)
                if cp_ids is not None and cp_ids.shape[-1] > 0 and full_dist is not None and full_time is not None:
                    chosen_cp = td["nearest_cp_global_id_per_local_node"][batch_idx, current_local]
                    dist_cur_cp = self._matrix_lookup(full_dist, current_global, chosen_cp).reshape(
                        [batch_size]
                    )
                    energy_cur_cp = torch.clamp(dist_cur_cp * rate, min=0.0)
                    cp_reachable = energy_cur_cp <= (soc_prev + eps)

                    # If nearest CP is unreachable, pick closest reachable CP by distance.
                    if full_dist.dim() == 2:
                        dist_cur_all_cp = full_dist[current_global.unsqueeze(-1), cp_ids]
                    else:
                        dist_cur_all_cp = full_dist[
                            batch_idx.view(batch_size, 1),
                            current_global.unsqueeze(-1),
                            cp_ids,
                        ]
                    energy_cur_all_cp = torch.clamp(dist_cur_all_cp * rate, min=0.0)
                    reachable_any = energy_cur_all_cp <= (soc_prev.unsqueeze(-1) + eps)
                    any_reachable = reachable_any.any(dim=-1)
                    masked_dist = torch.where(
                        reachable_any,
                        dist_cur_all_cp,
                        torch.full_like(dist_cur_all_cp, float("inf")),
                    )
                    best_reachable_idx = torch.argmin(masked_dist, dim=-1)
                    fallback_cp = cp_ids[batch_idx, best_reachable_idx]

                    use_fallback = needs_cp & (~cp_reachable) & any_reachable
                    chosen_cp = torch.where(use_fallback, fallback_cp, chosen_cp)

                    dist_cur_cp = self._matrix_lookup(full_dist, current_global, chosen_cp).reshape(
                        [batch_size]
                    )
                    time_cur_cp = self._matrix_lookup(full_time, current_global, chosen_cp).reshape(
                        [batch_size]
                    )
                    energy_cur_cp = torch.clamp(dist_cur_cp * rate, min=0.0)
                    cp_reachable = energy_cur_cp <= (soc_prev + eps)

                    cp_rate_candidates = torch.where(
                        cp_ids == chosen_cp.unsqueeze(-1),
                        cp_rates,
                        torch.zeros_like(cp_rates),
                    )
                    cp_rate = cp_rate_candidates.max(dim=-1).values
                    cp_rate = torch.where(
                        cp_rate > 0.0,
                        cp_rate,
                        torch.full_like(cp_rate, float(self.default_cp_charge_rate_kwh_per_hour)),
                    )

                    soc_arrival_cp = torch.clamp(soc_prev - energy_cur_cp, min=0.0)
                    charge_needed = torch.clamp(
                        float(self.effective_battery_kwh) - soc_arrival_cp, min=0.0
                    )
                    charge_time = (charge_needed / torch.clamp(cp_rate, min=1e-6)) * 60.0

                    dist_cp_target = self._matrix_lookup(full_dist, chosen_cp, target_global).reshape(
                        [batch_size]
                    )
                    time_cp_target = self._matrix_lookup(full_time, chosen_cp, target_global).reshape(
                        [batch_size]
                    )
                    energy_cp_target = torch.clamp(dist_cp_target * rate, min=0.0)

                    can_detour = needs_cp & cp_reachable
                    step_dist = torch.where(can_detour, dist_cur_cp + dist_cp_target, step_dist)
                    step_time = torch.where(
                        can_detour, time_cur_cp + charge_time + time_cp_target, step_time
                    )
                    soc_at_target = torch.where(
                        can_detour,
                        torch.clamp(float(self.effective_battery_kwh) - energy_cp_target, min=0.0),
                        soc_at_target,
                    )
                    cp_detour_used = can_detour

            duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size])
            start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
                [batch_size]
            )
            arrival = td["current_time"].squeeze(-1) + step_time
            service_start = torch.max(arrival, start_times)
            next_time_non_depot = service_start + duration
            td["current_time"] = torch.where(
                action_local != 0,
                next_time_non_depot,
                torch.zeros_like(next_time_non_depot),
            ).unsqueeze(-1)

            soc_after_depart = torch.where(
                action_local == 0,
                torch.full_like(soc_at_target, float(self.effective_battery_kwh)),
                soc_at_target,
            )
            td["current_soc_kwh"] = soc_after_depart.unsqueeze(-1)
            td["travel_distance_accum"] = td["travel_distance_accum"] + step_dist.unsqueeze(-1)
            td["cp_detour_used"] = cp_detour_used.to(torch.uint8).unsqueeze(-1)

            td_out = CVRPEnv._step(self, td)
            td_out.set("current_soc_kwh", td["current_soc_kwh"])
            td_out.set("travel_distance_accum", td["travel_distance_accum"])
            td_out.set("cp_detour_used", td["cp_detour_used"])

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
        if "travel_distance_accum" in td.keys():
            reward = -td["travel_distance_accum"].squeeze(-1).to(torch.float32)
        elif "dist_matrix" not in td.keys():
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

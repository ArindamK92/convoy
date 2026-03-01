"""Convoy environment implementation."""

import torch

from tensordict import TensorDict
from rl4co.envs import CVRPTWEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.utils.ops import gather_by_index
from src.convoy_rl_partial_ch2.myparser import parse_distance_matrix_csv


def parse_square_matrix_csv(matrix_csv_path: str) -> torch.Tensor:
    """Parse matrix CSV into a torch tensor (plain or ID-labeled)."""
    return parse_distance_matrix_csv(matrix_csv_path)


class convoy(CVRPTWEnv):
    """CVRPTW environment backed by combined distance/time matrices."""

    def __init__(
        self,
        battery_capacity_kwh: float = 30.0,
        energy_rate_kwh_per_distance: float = 0.5,
        charge_rate_kwh_per_hour: float = 120.0,
        cost_weight: float = 1.0,
        depot_charge_cost_per_kwh: float = 0.0,
        reserve_soc_kwh: float = 0.0,
        num_evs: int = 1,
        charging_pool_rows: list[dict] | None = None,
        charging_pool_sample_size: int = 5,
        combined_dist_matrix_csv: str | None = None,
        combined_time_matrix_csv: str | None = None,
        **kwargs,
    ):
        """Initialize EV parameters, charging pool, and combined matrices."""
        super().__init__(**kwargs)
        if battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be > 0.")
        if energy_rate_kwh_per_distance <= 0:
            raise ValueError("energy_rate_kwh_per_distance must be > 0.")
        if charge_rate_kwh_per_hour <= 0:
            raise ValueError("charge_rate_kwh_per_hour must be > 0.")
        if cost_weight < 0:
            raise ValueError("cost_weight must be >= 0.")
        if depot_charge_cost_per_kwh < 0:
            raise ValueError("depot_charge_cost_per_kwh must be >= 0.")
        if reserve_soc_kwh < 0:
            raise ValueError("reserve_soc_kwh must be >= 0.")
        if num_evs <= 0:
            raise ValueError("num_evs must be > 0.")
        if charging_pool_sample_size < 0:
            raise ValueError("charging_pool_sample_size must be >= 0.")
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.energy_rate_kwh_per_distance = float(energy_rate_kwh_per_distance)
        self.charge_rate_kwh_per_hour = float(charge_rate_kwh_per_hour)
        self.cost_weight = float(cost_weight)
        self.depot_charge_cost_per_kwh = float(depot_charge_cost_per_kwh)
        self.reserve_soc_kwh = float(reserve_soc_kwh)
        # Reserve SOC is modeled by shrinking usable battery to (full - reserve).
        self.effective_battery_kwh = self.battery_capacity_kwh - self.reserve_soc_kwh
        if self.effective_battery_kwh <= 0:
            raise ValueError(
                "reserve_soc_kwh must be smaller than battery_capacity_kwh."
            )
        self.num_evs = int(num_evs)
        self.charging_pool_sample_size = int(charging_pool_sample_size)
        # Script time values (TW, service, travel) are treated as minutes.
        self.time_units_per_hour = 60.0
        self.cp_pool_locs: torch.Tensor | None = None
        self.cp_pool_rates: torch.Tensor | None = None
        self.cp_pool_costs: torch.Tensor | None = None
        self.cp_pool_ids: torch.Tensor | None = None
        self.combined_dist_matrix: torch.Tensor | None = None
        self.combined_time_matrix: torch.Tensor | None = None

        if combined_dist_matrix_csv is not None:
            self.combined_dist_matrix = parse_square_matrix_csv(combined_dist_matrix_csv)
        if combined_time_matrix_csv is not None:
            self.combined_time_matrix = parse_square_matrix_csv(combined_time_matrix_csv)
        elif self.combined_dist_matrix is not None:
            self.combined_time_matrix = self.combined_dist_matrix
        if self.combined_dist_matrix is None:
            raise ValueError("Combined mode requires --combined-dist-matrix-csv.")

        # Load charging-point pool once; each reset samples a subset per instance.
        if self.charging_pool_sample_size > 0:
            if charging_pool_rows is None:
                raise ValueError(
                    "Combined mode requires charging_pool_rows from combined details."
                )
            cp_rows = charging_pool_rows
            if self.charging_pool_sample_size > len(cp_rows):
                raise ValueError(
                    f"--charging-stations-num={self.charging_pool_sample_size} "
                    f"is larger than available charging points ({len(cp_rows)})."
                )
            self.cp_pool_locs = torch.tensor(
                [[r["x"], r["y"]] for r in cp_rows], dtype=torch.float32
            )
            self.cp_pool_rates = torch.tensor(
                [r["charge_rate_kwh_per_hour"] for r in cp_rows], dtype=torch.float32
            )
            self.cp_pool_costs = torch.tensor(
                [r["charging_cost_per_kwh"] for r in cp_rows], dtype=torch.float32
            )
            self.cp_pool_ids = torch.tensor([r["cp_id"] for r in cp_rows], dtype=torch.long)

    def _attach_best_cp_metadata(self, td: TensorDict) -> TensorDict:
        """Attach per-customer best CP ID and its unit charging cost."""
        if "station_mask" not in td.keys():
            return td

        station_mask = td["station_mask"].to(torch.bool)
        batch_shape = station_mask.shape[:-1]
        total_nodes = station_mask.shape[-1]
        device = station_mask.device
        flat_batch = station_mask.numel() // total_nodes

        # Align node coordinates to [depot + non-depot nodes].
        all_locs = torch.cat((td["depot"][..., None, :], td["locs"]), dim=-2)
        all_locs_flat = all_locs.reshape(flat_batch, total_nodes, 2)
        station_mask_flat = station_mask.reshape(flat_batch, total_nodes)

        if "charge_cost_per_kwh_per_node" in td.keys():
            cp_cost_flat = td["charge_cost_per_kwh_per_node"].reshape(flat_batch, total_nodes)
        else:
            cp_cost_flat = torch.zeros((flat_batch, total_nodes), dtype=torch.float32, device=device)

        if "cp_id_per_node" in td.keys():
            cp_id_flat = td["cp_id_per_node"].to(torch.long).reshape(flat_batch, total_nodes)
        else:
            fallback_idx = torch.arange(total_nodes, device=device, dtype=torch.long).view(1, -1)
            fallback_idx = fallback_idx.expand(flat_batch, -1)
            cp_id_flat = torch.where(
                station_mask_flat,
                fallback_idx,
                torch.full_like(fallback_idx, -1),
            )

        best_cp_flat = torch.full((flat_batch, total_nodes), -1, dtype=torch.long, device=device)
        best_cp_unit_cost_flat = torch.zeros(
            (flat_batch, total_nodes), dtype=torch.float32, device=device
        )
        best_cp_node_idx_flat = torch.full(
            (flat_batch, total_nodes), -1, dtype=torch.long, device=device
        )

        customer_mask_flat = ~station_mask_flat
        customer_mask_flat[:, 0] = False

        has_station = station_mask_flat.any(dim=1)
        has_customer = customer_mask_flat.any(dim=1)
        if has_station.any() and has_customer.any():
            node_idx = torch.arange(total_nodes, device=device, dtype=torch.long).view(1, -1)
            node_idx = node_idx.expand(flat_batch, -1)
            sentinel = torch.full_like(node_idx, total_nodes)

            cp_pos_all = torch.where(station_mask_flat, node_idx, sentinel)
            cp_pos_all, _ = torch.sort(cp_pos_all, dim=1)
            max_cp = int(station_mask_flat.sum(dim=1).max().item())

            cust_pos_all = torch.where(customer_mask_flat, node_idx, sentinel)
            cust_pos_all, _ = torch.sort(cust_pos_all, dim=1)
            max_cust = int(customer_mask_flat.sum(dim=1).max().item())

            if max_cp > 0 and max_cust > 0:
                cp_pos = cp_pos_all[:, :max_cp]
                valid_cp = cp_pos < total_nodes
                cp_pos_safe = cp_pos.clamp(max=total_nodes - 1)

                cust_pos = cust_pos_all[:, :max_cust]
                valid_cust = cust_pos < total_nodes
                cust_pos_safe = cust_pos.clamp(max=total_nodes - 1)

                x_nodes = all_locs_flat[..., 0]
                y_nodes = all_locs_flat[..., 1]

                cp_x = torch.gather(x_nodes, 1, cp_pos_safe)
                cp_y = torch.gather(y_nodes, 1, cp_pos_safe)
                cp_cost = torch.gather(cp_cost_flat, 1, cp_pos_safe)
                cp_id = torch.gather(cp_id_flat, 1, cp_pos_safe)

                inf = torch.full_like(cp_x, float("inf"))
                ninf = torch.full_like(cp_x, float("-inf"))
                min_x = torch.where(valid_cp, cp_x, inf).min(dim=1).values
                max_x = torch.where(valid_cp, cp_x, ninf).max(dim=1).values
                min_y = torch.where(valid_cp, cp_y, inf).min(dim=1).values
                max_y = torch.where(valid_cp, cp_y, ninf).max(dim=1).values
                min_cost = torch.where(valid_cp, cp_cost, inf).min(dim=1).values
                max_cost = torch.where(valid_cp, cp_cost, ninf).max(dim=1).values

                zero = torch.zeros_like(min_x)
                one = torch.ones_like(min_x)
                min_x = torch.where(has_station, min_x, zero)
                max_x = torch.where(has_station, max_x, zero)
                min_y = torch.where(has_station, min_y, zero)
                max_y = torch.where(has_station, max_y, zero)
                min_cost = torch.where(has_station, min_cost, zero)
                max_cost = torch.where(has_station, max_cost, zero)

                denom_x = max_x - min_x
                denom_y = max_y - min_y
                denom_cost = max_cost - min_cost
                denom_x = torch.where(denom_x.abs() > 1e-12, denom_x, one)
                denom_y = torch.where(denom_y.abs() > 1e-12, denom_y, one)
                denom_cost = torch.where(denom_cost.abs() > 1e-12, denom_cost, one)

                cp_feat = torch.stack(
                    [
                        (cp_x - min_x[:, None]) / denom_x[:, None],
                        (cp_y - min_y[:, None]) / denom_y[:, None],
                        (cp_cost - min_cost[:, None]) / denom_cost[:, None],
                    ],
                    dim=-1,
                )

                cust_x = torch.gather(x_nodes, 1, cust_pos_safe)
                cust_y = torch.gather(y_nodes, 1, cust_pos_safe)
                cust_feat = torch.stack(
                    [
                        (cust_x - min_x[:, None]) / denom_x[:, None],
                        (cust_y - min_y[:, None]) / denom_y[:, None],
                        torch.zeros_like(cust_x),
                    ],
                    dim=-1,
                )

                distances = torch.cdist(cust_feat, cp_feat, p=2)
                distances = torch.where(
                    valid_cp[:, None, :],
                    distances,
                    torch.full_like(distances, float("inf")),
                )
                best_local_cp = distances.argmin(dim=-1)

                best_cp_ids = torch.gather(cp_id, 1, best_local_cp)
                best_cp_costs = torch.gather(cp_cost, 1, best_local_cp)
                best_cp_nodes = torch.gather(cp_pos_safe, 1, best_local_cp).to(torch.long)

                row_idx = torch.arange(flat_batch, device=device).unsqueeze(1).expand_as(cust_pos_safe)
                assign_mask = valid_cust & has_station[:, None]
                target_nodes = cust_pos_safe.to(torch.long)

                best_cp_flat[row_idx[assign_mask], target_nodes[assign_mask]] = best_cp_ids[
                    assign_mask
                ].to(torch.long)
                best_cp_unit_cost_flat[row_idx[assign_mask], target_nodes[assign_mask]] = (
                    best_cp_costs[assign_mask]
                )
                best_cp_node_idx_flat[row_idx[assign_mask], target_nodes[assign_mask]] = (
                    best_cp_nodes[assign_mask]
                )

        td.set("best_CP", best_cp_flat.reshape(*batch_shape, total_nodes))
        td.set(
            "best_CP_unit_cost",
            best_cp_unit_cost_flat.reshape(*batch_shape, total_nodes),
        )
        td.set(
            "best_CP_node_idx",
            best_cp_node_idx_flat.reshape(*batch_shape, total_nodes),
        )
        return td

    def _augment_instance_with_charging_stations(self, td: TensorDict) -> TensorDict:
        """Append sampled charging stations and related metadata to one batch."""
        # Test instances may already include explicit charging-node metadata.
        # In that case, use it as-is instead of randomly sampling from the CP pool.
        explicit_station_keys = {
            "station_mask",
            "charge_nodes_mask",
            "charge_rate_per_node",
            "charge_cost_per_kwh_per_node",
        }
        if explicit_station_keys.issubset(set(td.keys())):
            return self._attach_best_cp_metadata(td)

        batch_size = td["locs"].shape[0]
        device = td["locs"].device
        num_customers = td["locs"].shape[1]

        # Start with depot-only charging metadata (works even when stations are disabled).
        charge_nodes_mask = torch.zeros(
            (batch_size, 1 + num_customers), dtype=torch.bool, device=device
        )
        charge_nodes_mask[:, 0] = True
        station_mask = torch.zeros_like(charge_nodes_mask)
        charge_rate_per_node = torch.zeros(
            (batch_size, 1 + num_customers), dtype=torch.float32, device=device
        )
        charge_rate_per_node[:, 0] = self.charge_rate_kwh_per_hour
        charge_cost_per_kwh_per_node = torch.zeros_like(charge_rate_per_node)
        charge_cost_per_kwh_per_node[:, 0] = self.depot_charge_cost_per_kwh

        if (
            self.charging_pool_sample_size <= 0
            or self.cp_pool_locs is None
            or self.cp_pool_rates is None
            or self.cp_pool_costs is None
            or self.cp_pool_ids is None
        ):
            td.set("charge_nodes_mask", charge_nodes_mask)
            td.set("station_mask", station_mask)
            td.set("charge_rate_per_node", charge_rate_per_node)
            td.set("charge_cost_per_kwh_per_node", charge_cost_per_kwh_per_node)
            return self._attach_best_cp_metadata(td)

        # Sample charging stations per instance from combined details pool.
        sample_idx = torch.stack(
            [
                torch.randperm(self.cp_pool_locs.shape[0])[: self.charging_pool_sample_size]
                for _ in range(batch_size)
            ],
            dim=0,
        )
        sampled_station_locs = self.cp_pool_locs[sample_idx].to(device)
        sampled_station_rates = self.cp_pool_rates[sample_idx].to(device)
        sampled_station_costs = self.cp_pool_costs[sample_idx].to(device)
        sampled_station_ids = self.cp_pool_ids[sample_idx].to(device)

        locs_aug = torch.cat([td["locs"], sampled_station_locs], dim=1)
        zero_demand = torch.zeros(
            (batch_size, self.charging_pool_sample_size),
            dtype=td["demand"].dtype,
            device=device,
        )
        demand_aug = torch.cat([td["demand"], zero_demand], dim=1)

        # Station stop duration is computed dynamically from SOC and charging power.
        zero_duration = torch.zeros(
            (batch_size, self.charging_pool_sample_size),
            dtype=td["durations"].dtype,
            device=device,
        )
        durations_aug = torch.cat([td["durations"], zero_duration], dim=1)

        # Stations are always open over the instance planning horizon.
        station_tw = torch.zeros(
            (batch_size, self.charging_pool_sample_size, 2),
            dtype=td["time_windows"].dtype,
            device=device,
        )
        max_tw_end = td["time_windows"][..., 1].amax(dim=1, keepdim=True)
        station_tw[..., 1] = max_tw_end.expand(-1, self.charging_pool_sample_size)
        time_windows_aug = torch.cat([td["time_windows"], station_tw], dim=1)

        td.set("locs", locs_aug)
        td.set("demand", demand_aug)
        td.set("durations", durations_aug)
        td.set("time_windows", time_windows_aug)
        if "customer_reward_per_node" in td.keys():
            zero_reward = torch.zeros(
                (batch_size, self.charging_pool_sample_size),
                dtype=td["customer_reward_per_node"].dtype,
                device=device,
            )
            reward_aug = torch.cat([td["customer_reward_per_node"], zero_reward], dim=1)
            td.set("customer_reward_per_node", reward_aug)

        # Build metadata masks aligned to [depot + customers + stations].
        total_nodes = 1 + num_customers + self.charging_pool_sample_size
        station_mask = torch.zeros((batch_size, total_nodes), dtype=torch.bool, device=device)
        station_start = 1 + num_customers
        station_mask[:, station_start:] = True

        charge_nodes_mask = station_mask.clone()
        charge_nodes_mask[:, 0] = True

        charge_rate_per_node = torch.zeros(
            (batch_size, total_nodes), dtype=torch.float32, device=device
        )
        charge_rate_per_node[:, 0] = self.charge_rate_kwh_per_hour
        charge_rate_per_node[:, station_start:] = sampled_station_rates
        charge_cost_per_kwh_per_node = torch.zeros(
            (batch_size, total_nodes), dtype=torch.float32, device=device
        )
        charge_cost_per_kwh_per_node[:, 0] = self.depot_charge_cost_per_kwh
        charge_cost_per_kwh_per_node[:, station_start:] = sampled_station_costs
        cp_id_per_node = torch.full(
            (batch_size, total_nodes), -1, dtype=torch.long, device=device
        )
        cp_id_per_node[:, station_start:] = sampled_station_ids

        td.set("charge_nodes_mask", charge_nodes_mask)
        td.set("station_mask", station_mask)
        td.set("charge_rate_per_node", charge_rate_per_node)
        td.set("charge_cost_per_kwh_per_node", charge_cost_per_kwh_per_node)
        td.set("cp_id_per_node", cp_id_per_node)

        if "global_node_ids" not in td.keys():
            raise ValueError("Combined mode requires global_node_ids in instance TensorDict.")
        base_global_nodes = td["global_node_ids"].to(torch.long)
        all_global_nodes = torch.cat([base_global_nodes, sampled_station_ids.to(torch.long)], dim=1)
        idx_i = all_global_nodes[:, :, None].expand(-1, -1, all_global_nodes.shape[1])
        idx_j = all_global_nodes[:, None, :].expand(-1, all_global_nodes.shape[1], -1)
        dist_full = self.combined_dist_matrix.to(device)
        time_full = self.combined_time_matrix.to(device)
        td.set("dist_matrix", dist_full[idx_i, idx_j])
        td.set("travel_time_matrix", time_full[idx_i, idx_j])
        return self._attach_best_cp_metadata(td)

    def get_action_mask(self, td):
        """Build feasibility mask with time-window + EV + charging-station constraints."""
        not_masked = CVRPEnv.get_action_mask(td)
        if "station_mask" in td.keys():
            # Charging stations are revisitable, so ignore CVRP visited masking for them.
            # Only keep the current node's best CP open; other CPs stay masked.
            if "best_CP_node_idx" in td.keys():
                station_candidates = torch.zeros_like(td["station_mask"], dtype=torch.bool)
                best_cp_for_current = gather_by_index(
                    td["best_CP_node_idx"].to(torch.long), td["current_node"]
                ).reshape(-1)
                valid_best = best_cp_for_current >= 0
                if valid_best.any():
                    rows = torch.nonzero(valid_best, as_tuple=False).squeeze(-1)
                    station_candidates[rows, best_cp_for_current[rows]] = True
                station_candidates = station_candidates & td["station_mask"].to(torch.bool)
            else:
                station_candidates = td["station_mask"].clone()

            # Keep the current node masked to avoid zero-distance self-loop actions.
            same_node = torch.zeros_like(station_candidates, dtype=torch.bool)
            same_node.scatter_(1, td["current_node"], True)
            station_candidates = station_candidates & ~same_node
            # Never allow direct depot -> charging-station moves.
            at_depot = td["current_node"].squeeze(-1) == 0
            station_candidates = station_candidates & (~at_depot.unsqueeze(-1))
            if "done" in td.keys():
                done = td["done"]
                if done.dim() == 1:
                    done = done.unsqueeze(-1)
                station_candidates = station_candidates & (~done.expand_as(station_candidates))
            not_masked = torch.where(
                station_candidates,
                torch.ones_like(not_masked, dtype=torch.bool),
                not_masked,
            )

        if "dist_matrix" not in td.keys() or "travel_time_matrix" not in td.keys():
            raise ValueError(
                "Combined mode requires dist_matrix and travel_time_matrix in rollout state."
            )
        current_loc = gather_by_index(td["locs"], td["current_node"])
        time_row = gather_by_index(td["travel_time_matrix"], td["current_node"], dim=1)
        travel_time = time_row.squeeze(1)
        pairwise_dist = td["dist_matrix"]
        row = gather_by_index(pairwise_dist, td["current_node"], dim=1)
        dist = row.squeeze(1)
        td.update(
            {"current_loc": current_loc, "distances": dist, "travel_times": travel_time}
        )
        can_reach_in_time = td["current_time"] + travel_time <= td["time_windows"][..., 1]

        if "charge_nodes_mask" in td.keys():
            charge_nodes_mask = td["charge_nodes_mask"]
        else:
            charge_nodes_mask = torch.zeros_like(dist, dtype=torch.bool)
            charge_nodes_mask[..., 0] = True
        if "station_mask" in td.keys():
            station_mask = td["station_mask"]
        else:
            station_mask = torch.zeros_like(dist, dtype=torch.bool)
        nearest_charge_dist = torch.where(
            charge_nodes_mask[:, None, :],
            pairwise_dist,
            torch.full_like(pairwise_dist, float("inf")),
        ).min(dim=-1).values
        # Customer move is feasible only if after serving that customer,
        # the EV can still reach that customer's best CP.
        if "best_CP_node_idx" in td.keys():
            best_cp_node_idx = td["best_CP_node_idx"].to(torch.long)
            valid_best_cp = best_cp_node_idx >= 0
            safe_best_cp_idx = best_cp_node_idx.clamp(min=0)
            dist_to_best_cp = torch.gather(
                pairwise_dist, 2, safe_best_cp_idx.unsqueeze(-1)
            ).squeeze(-1)
            charge_recovery_dist = torch.where(
                valid_best_cp, dist_to_best_cp, nearest_charge_dist
            )
        else:
            charge_recovery_dist = nearest_charge_dist
        energy_to_node = dist * self.energy_rate_kwh_per_distance
        energy_for_customer = (
            dist + charge_recovery_dist
        ) * self.energy_rate_kwh_per_distance
        customer_mask = ~charge_nodes_mask
        required_energy = torch.where(customer_mask, energy_for_customer, energy_to_node)
        battery_ok = td["current_battery"] >= required_energy

        mask = not_masked & can_reach_in_time & battery_ok
        # If nothing is feasible, force return to depot; do not relax TW filtering.
        fallback = ~mask.any(dim=-1, keepdim=True)
        depot_only = torch.zeros_like(mask, dtype=torch.bool)
        depot_only[:, 0] = True
        mask = torch.where(fallback, depot_only, mask)

        # If at least one customer is feasible now, allow customer + station moves.
        # Keep anti-bounce by blocking station->station transitions only.
        customer_nodes_mask = ~station_mask
        customer_nodes_mask[:, 0] = False
        customer_feasible = mask & customer_nodes_mask
        has_customer_option = customer_feasible.any(dim=-1, keepdim=True)
        station_feasible = mask & station_mask

        # Block only station->station. Station->depot remains allowed.
        at_station_now = (
            gather_by_index(station_mask.to(torch.float32), td["current_node"]).reshape(-1)
            > 0.5
        )
        station_allowed_when_customer_feasible = (
            station_feasible & (~at_station_now.unsqueeze(-1))
        )

        customer_or_station = customer_feasible | station_allowed_when_customer_feasible
        mask = torch.where(
            has_customer_option,
            customer_or_station,
            mask,
        )

        # If all customers are served, prefer depot return when feasible.
        # If depot is not feasible yet, allow feasible charge-node moves (CP/depot)
        # so the EV can top up first and then return physically.
        unserved_customers = customer_nodes_mask & (~td["visited"].to(torch.bool))
        has_unserved = unserved_customers.any(dim=-1, keepdim=True)
        charge_feasible = mask & charge_nodes_mask
        depot_feasible = charge_feasible[:, 0:1]
        all_served_mask = torch.where(depot_feasible, depot_only, charge_feasible)
        mask = torch.where(has_unserved, mask, all_served_mask)

        # Final safety: if at depot and nothing feasible, keep depot selectable.
        # Do not force impossible moves away from depot.
        needs_fallback = ~mask.any(dim=-1, keepdim=True)
        at_depot_now = (td["current_node"].squeeze(-1) == 0).unsqueeze(-1)
        fallback_to_depot = needs_fallback & at_depot_now
        return torch.where(fallback_to_depot, depot_only, mask)

    def _step(self, td):
        """Advance transition, update EV SOC/charging state, then refresh mask."""
        batch_size = td["locs"].shape[0]
        device = td["locs"].device
        # Enforce action feasibility against current mask.
        # This keeps rollout physically valid even if a decoder emits an invalid
        # action (e.g., rare beam backtracking artifacts).
        current_mask = self.get_action_mask(td)
        action_now = td["action"].reshape(batch_size).long()
        action_feasible = current_mask.gather(1, action_now.unsqueeze(-1)).squeeze(-1).to(torch.bool)
        if (~action_feasible).any():
            fixed_action = action_now.clone()
            for b in torch.nonzero(~action_feasible, as_tuple=False).squeeze(-1).tolist():
                feasible_idx = torch.nonzero(current_mask[b], as_tuple=False).squeeze(-1)
                if feasible_idx.numel() == 0:
                    fixed_action[b] = 0
                    continue
                if "distances" in td.keys():
                    nearest_pos = torch.argmin(td["distances"][b, feasible_idx]).item()
                    fixed_action[b] = int(feasible_idx[nearest_pos].item())
                else:
                    fixed_action[b] = int(feasible_idx[0].item())
            td["action"] = fixed_action
        travel_time = gather_by_index(td["travel_times"], td["action"]).reshape(
            [batch_size, 1]
        )
        travel_distance = gather_by_index(td["distances"], td["action"]).reshape(
            [batch_size, 1]
        )
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        arrival_or_wait = torch.max(td["current_time"] + travel_time, start_times)
        energy_used = travel_distance * self.energy_rate_kwh_per_distance
        remaining_battery = torch.clamp(td["current_battery"] - energy_used, min=0.0)

        if "charge_nodes_mask" in td.keys():
            at_charge_node = gather_by_index(
                td["charge_nodes_mask"].to(torch.float32), td["action"]
            ).reshape([batch_size, 1]) > 0.5
            at_station = gather_by_index(
                td["station_mask"].to(torch.float32), td["action"]
            ).reshape([batch_size, 1]) > 0.5
            selected_charge_rate = gather_by_index(
                td["charge_rate_per_node"], td["action"]
            ).reshape([batch_size, 1])
        else:
            at_charge_node = td["action"][:, None] == 0
            at_station = torch.zeros_like(at_charge_node)
            selected_charge_rate = torch.full_like(
                remaining_battery, self.charge_rate_kwh_per_hour
            )
        charge_needed = torch.clamp(self.effective_battery_kwh - remaining_battery, min=0.0)
        charge_time = torch.zeros_like(remaining_battery)
        if at_charge_node.any():
            safe_charge_rate = torch.clamp(selected_charge_rate, min=1e-6)
            charge_time = torch.where(
                at_charge_node,
                (charge_needed / safe_charge_rate) * self.time_units_per_hour,
                charge_time,
            )
        # For depot/station visits, service time is replaced by charge-to-full time.
        finish_time = arrival_or_wait + torch.where(at_charge_node, charge_time, duration)

        at_depot = td["action"][:, None] == 0

        vehicle_ready_time = td["ev_vehicle_ready_time"].clone()
        current_vehicle_idx = td["current_vehicle_idx"]
        batch_idx = torch.arange(batch_size, device=device)
        active_idx = current_vehicle_idx.squeeze(-1)
        active_ready = vehicle_ready_time[batch_idx, active_idx].unsqueeze(-1)
        # Depot has a shared fleet: after returning + charging this EV, pick earliest-ready EV.
        active_ready = torch.where(at_depot, finish_time, active_ready)
        vehicle_ready_time[batch_idx, active_idx] = active_ready.squeeze(-1)

        next_ready_time, next_vehicle_idx = vehicle_ready_time.min(dim=1)
        full_battery = torch.full(
            (batch_size, 1),
            self.effective_battery_kwh,
            dtype=remaining_battery.dtype,
            device=device,
        )
        battery_non_depot = torch.where(at_station, full_battery, remaining_battery)

        td["current_time"] = torch.where(
            at_depot, next_ready_time.unsqueeze(-1), finish_time
        )
        td["current_battery"] = torch.where(at_depot, full_battery, battery_non_depot)
        td["current_vehicle_idx"] = torch.where(
            at_depot, next_vehicle_idx.unsqueeze(-1), current_vehicle_idx
        )
        td["ev_vehicle_ready_time"] = vehicle_ready_time
        td = super(CVRPTWEnv, self)._step(td)
        action_mask = self.get_action_mask(td)

        # Enforce explicit route closure: even after all customers are served,
        # rollout should terminate only after the active EV reaches depot.
        if "station_mask" in td.keys():
            customer_nodes_mask_done = ~td["station_mask"]
            customer_nodes_mask_done[:, 0] = False
        else:
            customer_nodes_mask_done = torch.ones_like(action_mask, dtype=torch.bool)
            customer_nodes_mask_done[:, 0] = False
        unserved_customers_done = customer_nodes_mask_done & (~td["visited"].to(torch.bool))
        all_customers_served = ~unserved_customers_done.any(dim=-1)
        at_depot_done = td["current_node"].squeeze(-1) == 0
        must_return_to_depot = all_customers_served & (~at_depot_done)
        if must_return_to_depot.any():
            td["done"] = td["done"] & (~must_return_to_depot)
            depot_only = torch.zeros_like(action_mask, dtype=torch.bool)
            depot_only[:, 0] = True
            depot_feasible_now = action_mask[:, 0]
            force_depot_now = must_return_to_depot & depot_feasible_now
            if force_depot_now.any():
                action_mask = torch.where(force_depot_now[:, None], depot_only, action_mask)

        if "station_mask" in td.keys() and "charge_nodes_mask" in td.keys():
            station_mask = td["station_mask"]
            customer_nodes_mask = ~station_mask
            customer_nodes_mask[:, 0] = False
            unserved_customers = customer_nodes_mask & (~td["visited"].to(torch.bool))
            has_unserved = unserved_customers.any(dim=-1)
            feasible_customer_now = (action_mask & customer_nodes_mask).any(dim=-1)
            at_depot_now = td["current_node"].squeeze(-1) == 0
            at_station_now = gather_by_index(
                station_mask.to(torch.float32), td["current_node"]
            ).squeeze(-1) > 0.5
            full_battery_now = td["current_battery"].squeeze(-1) >= (
                self.effective_battery_kwh - 1e-6
            )
            depot_only = torch.zeros_like(action_mask, dtype=torch.bool)
            depot_only[:, 0] = True
            # Dead-end guard: at a charging station with full SOC and no customer feasible.
            # Force depot return so the decoder can switch to the next ready EV
            # instead of terminating the rollout immediately.
            dead_end = has_unserved & (~feasible_customer_now) & at_station_now & full_battery_now
            if dead_end.any():
                action_mask = torch.where(dead_end[:, None], depot_only, action_mask)
            # If we are already at depot and no customer is feasible anymore,
            # terminate even if some customers remain unserved.
            stranded_at_depot = has_unserved & (~feasible_customer_now) & at_depot_now
            if stranded_at_depot.any():
                td["done"] = td["done"] | stranded_at_depot
                action_mask = torch.where(stranded_at_depot[:, None], depot_only, action_mask)
            # If no physically feasible move remains away from depot, terminate.
            no_action_now = ~action_mask.any(dim=-1)
            stranded_away = no_action_now & (~at_depot_now)
            if stranded_away.any():
                td["done"] = td["done"] | stranded_away
                action_mask = torch.where(stranded_away[:, None], depot_only, action_mask)

        td.set("action_mask", action_mask)
        return td

    def _get_reward(self, td, actions):
        """Compute rollout reward using step-local travel/charge contributions.

        Per-step rules:
        - Customer move reward: customer_reward(on-time, first-visit) - 0.6 * energy_used
        - Charge-node move reward: +0.6 * energy_used + cost_weight * charging_penalty
          where charging_penalty = charge_needed(kWh) * (0.6 - unit_charge_cost).

          **COnsidering $0.6 per kWh is the max energy cost at any CP 

        If rollout ends away from depot, an implicit final return-to-depot step uses
        the same charge-node rule.
        """
        if "customer_reward_per_node" not in td.keys():
            raise ValueError("Combined mode requires customer_reward_per_node in TensorDict.")

        batch_size, num_steps = actions.shape
        device = actions.device
        batch_idx = torch.arange(batch_size, device=device)
        num_nodes = td["locs"].shape[1]

        if "dist_matrix" not in td.keys() or "travel_time_matrix" not in td.keys():
            raise ValueError("Combined mode requires dist_matrix and travel_time_matrix.")
        dist_matrix = td["dist_matrix"]
        travel_matrix = td["travel_time_matrix"]

        durations = td["durations"]
        time_windows = td["time_windows"]
        customer_reward_per_node = td["customer_reward_per_node"]
        if "station_mask" in td.keys():
            station_mask = td["station_mask"].to(torch.bool)
        else:
            station_mask = torch.zeros(
                (batch_size, num_nodes), dtype=torch.bool, device=device
            )
        if "charge_nodes_mask" in td.keys():
            charge_nodes_mask = td["charge_nodes_mask"].to(torch.bool)
        else:
            charge_nodes_mask = torch.zeros(
                (batch_size, num_nodes), dtype=torch.bool, device=device
            )
            charge_nodes_mask[:, 0] = True
        if "charge_rate_per_node" in td.keys():
            charge_rate_per_node = td["charge_rate_per_node"]
        else:
            charge_rate_per_node = torch.zeros(
                (batch_size, num_nodes), dtype=torch.float32, device=device
            )
            charge_rate_per_node[:, 0] = self.charge_rate_kwh_per_hour
        if "charge_cost_per_kwh_per_node" in td.keys():
            charge_cost_per_kwh_per_node = td["charge_cost_per_kwh_per_node"]
        else:
            charge_cost_per_kwh_per_node = torch.zeros(
                (batch_size, num_nodes), dtype=torch.float32, device=device
            )
            charge_cost_per_kwh_per_node[:, 0] = self.depot_charge_cost_per_kwh

        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_time = torch.zeros(batch_size, dtype=torch.float32, device=device)
        current_battery = torch.full(
            (batch_size,),
            self.effective_battery_kwh,
            dtype=torch.float32,
            device=device,
        )
        current_vehicle_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        vehicle_ready_time = torch.zeros(
            (batch_size, self.num_evs), dtype=torch.float32, device=device
        )
        served_customers = torch.zeros(
            (batch_size, num_nodes), dtype=torch.bool, device=device
        )
        total_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
        zero_float = torch.zeros(batch_size, dtype=torch.float32, device=device)

        for step_idx in range(num_steps):
            next_node = actions[:, step_idx].long().clamp(min=0, max=num_nodes - 1)

            travel_distance = dist_matrix[batch_idx, current_node, next_node]
            travel_time = travel_matrix[batch_idx, current_node, next_node]
            arrival_time = current_time + travel_time

            tw_start = time_windows[batch_idx, next_node, 0]
            tw_end = time_windows[batch_idx, next_node, 1]
            service_start = torch.maximum(arrival_time, tw_start)
            on_time = service_start <= (tw_end + 1e-6)

            energy_used = travel_distance * self.energy_rate_kwh_per_distance
            remaining_battery = torch.clamp(current_battery - energy_used, min=0.0)

            is_charge_node = charge_nodes_mask[batch_idx, next_node]
            is_station = station_mask[batch_idx, next_node]
            selected_rate = charge_rate_per_node[batch_idx, next_node]

            charge_needed = torch.clamp(
                self.effective_battery_kwh - remaining_battery, min=0.0
            )
            safe_charge_rate = torch.clamp(selected_rate, min=1e-6)
            charge_time = torch.where(
                is_charge_node,
                (charge_needed / safe_charge_rate) * self.time_units_per_hour,
                zero_float,
            )

            duration = durations[batch_idx, next_node]
            finish_time = service_start + torch.where(is_charge_node, charge_time, duration)

            is_customer = (~station_mask[batch_idx, next_node]) & (next_node != 0)
            first_visit = ~served_customers[batch_idx, next_node]
            collect_reward = is_customer & first_visit & on_time
            step_reward = torch.where(
                collect_reward,
                customer_reward_per_node[batch_idx, next_node],
                zero_float,
            )
            # step_reward = 0 ##AK: modified to remove customer reward and only consider energy cost and charging penalty
            customer_step_reward = step_reward - (0.6 * energy_used)

            station_charge_cost = charge_cost_per_kwh_per_node[batch_idx, next_node]
            charging_penalty = torch.where(
                is_charge_node,
                charge_needed * (0.6 - station_charge_cost),
                zero_float,
            )
            charge_step_reward = (
                -(0.6 * energy_used)
                + (self.cost_weight * charging_penalty)
            )

            # customer_step_reward = 0 ##AK: modified to remove customer reward and only consider energy cost and charging penalty

            total_reward = total_reward + torch.where(
                is_charge_node, charge_step_reward, customer_step_reward
            )

            visit_mask = torch.zeros_like(served_customers)
            visit_mask.scatter_(1, next_node.unsqueeze(-1), is_customer.unsqueeze(-1))
            served_customers = served_customers | visit_mask

            at_depot = next_node == 0
            active_ready = vehicle_ready_time[batch_idx, current_vehicle_idx]
            active_ready = torch.where(at_depot, finish_time, active_ready)
            vehicle_ready_time[batch_idx, current_vehicle_idx] = active_ready

            next_ready_time, next_vehicle_idx = vehicle_ready_time.min(dim=1)
            full_battery = torch.full_like(remaining_battery, self.effective_battery_kwh)
            battery_non_depot = torch.where(is_station, full_battery, remaining_battery)

            current_time = torch.where(at_depot, next_ready_time, finish_time)
            current_battery = torch.where(at_depot, full_battery, battery_non_depot)
            current_vehicle_idx = torch.where(at_depot, next_vehicle_idx, current_vehicle_idx)
            current_node = next_node

        # Rollout can terminate with active EV away from depot after final customer.
        # Account for mandatory final return-to-depot recharge cost explicitly.
        need_final_return = current_node != 0
        if need_final_return.any():
            depot_node = torch.zeros_like(current_node)
            travel_back_distance = dist_matrix[batch_idx, current_node, depot_node]
            energy_back = travel_back_distance * self.energy_rate_kwh_per_distance
            battery_at_depot = torch.clamp(current_battery - energy_back, min=0.0)
            final_charge_needed = torch.clamp(
                self.effective_battery_kwh - battery_at_depot, min=0.0
            )
            depot_charge_cost = charge_cost_per_kwh_per_node[:, 0]
            final_depot_penalty = torch.where(
                need_final_return,
                final_charge_needed * (0.6 - depot_charge_cost),
                zero_float,
            )
            final_step_reward = (
                (0.6 * energy_back)
                + (self.cost_weight * final_depot_penalty)
            )
            total_reward = total_reward + torch.where(
                need_final_return, final_step_reward, zero_float
            )

        return total_reward

    def _reset(self, td=None, batch_size=None):
        """Initialize the rollout state TensorDict for a new episode."""
        # Add sampled charging stations (if configured) before RL state construction.
        td = self._augment_instance_with_charging_stations(td)
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
                "current_battery": torch.full(
                    (*batch_size, 1),
                    self.effective_battery_kwh,
                    dtype=torch.float32,
                    device=device,
                ),
                "current_vehicle_idx": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "ev_vehicle_ready_time": torch.zeros(
                    (*batch_size, self.num_evs),
                    dtype=torch.float32,
                    device=device,
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
            },
            batch_size=batch_size,
        )
        if "dist_matrix" in td.keys():
            td_reset.set("dist_matrix", td["dist_matrix"])
        if "travel_time_matrix" in td.keys():
            td_reset.set("travel_time_matrix", td["travel_time_matrix"])
        if "station_mask" in td.keys():
            td_reset.set("station_mask", td["station_mask"])
            # Mark stations as already visited so CVRP completion still depends on customers.
            td_reset["visited"] = torch.maximum(
                td_reset["visited"], td["station_mask"].to(td_reset["visited"].dtype)
            )
        if "charge_nodes_mask" in td.keys():
            td_reset.set("charge_nodes_mask", td["charge_nodes_mask"])
        if "charge_rate_per_node" in td.keys():
            td_reset.set("charge_rate_per_node", td["charge_rate_per_node"])
        if "charge_cost_per_kwh_per_node" in td.keys():
            td_reset.set("charge_cost_per_kwh_per_node", td["charge_cost_per_kwh_per_node"])
        if "cp_id_per_node" in td.keys():
            td_reset.set("cp_id_per_node", td["cp_id_per_node"])
        if "best_CP" in td.keys():
            td_reset.set("best_CP", td["best_CP"])
        if "best_CP_unit_cost" in td.keys():
            td_reset.set("best_CP_unit_cost", td["best_CP_unit_cost"])
        if "best_CP_node_idx" in td.keys():
            td_reset.set("best_CP_node_idx", td["best_CP_node_idx"])
        if "customer_reward_per_node" in td.keys():
            td_reset.set("customer_reward_per_node", td["customer_reward_per_node"])
        if "global_node_ids" in td.keys():
            td_reset.set("global_node_ids", td["global_node_ids"])
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

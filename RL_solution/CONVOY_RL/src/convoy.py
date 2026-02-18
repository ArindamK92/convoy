"""Convoy environment implementation."""

import csv
import os

import torch

from tensordict import TensorDict
from rl4co.envs import CVRPTWEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.utils.ops import gather_by_index


def parse_charging_station_csv_rows(csv_path: str) -> list[dict]:
    """Parse charging-point rows with x/y, charging rate, and charging cost."""
    stations: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(
                "Charging station CSV must contain headers including x, y and charge rate."
            )
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        if "x" not in field_map or "y" not in field_map:
            raise ValueError("Charging station CSV must contain columns: x,y,<charge_rate>.")
        rate_col = None
        for candidate in (
            "charge_rate_kwh_per_hour",
            "charge_rate",
            "charging_rate_kwh_per_hour",
            "charging_rate",
        ):
            if candidate in field_map:
                rate_col = field_map[candidate]
                break
        if rate_col is None:
            raise ValueError(
                "Charging station CSV must include one rate column: "
                "charge_rate_kwh_per_hour/charge_rate/charging_rate_kwh_per_hour/charging_rate."
            )
        cost_col = None
        for candidate in (
            "charging_cost_per_kwh",
            "charging_cost_per_kwhr",
            "charge_cost_per_kwh",
            "charge_cost",
        ):
            if candidate in field_map:
                cost_col = field_map[candidate]
                break
        if cost_col is None:
            raise ValueError(
                "Charging station CSV must include charging cost column: "
                "charging_cost_per_kWh (or equivalent)."
            )
        for i, row in enumerate(reader, start=1):
            cp_id = i
            if "cp_id" in row and row["cp_id"] != "":
                try:
                    cp_id = int(row["cp_id"])
                except ValueError:
                    cp_id = i
            rec = {
                "cp_id": cp_id,
                "x": float(row[field_map["x"]]),
                "y": float(row[field_map["y"]]),
                "charge_rate_kwh_per_hour": float(row[rate_col]),
                "charging_cost_per_kwh": float(row[cost_col]),
            }
            if rec["charge_rate_kwh_per_hour"] <= 0:
                raise ValueError("Charging station rate must be > 0 for every row.")
            if rec["charging_cost_per_kwh"] < 0:
                raise ValueError("Charging cost per kWh must be >= 0 for every row.")
            stations.append(rec)
    if not stations:
        raise ValueError("Charging station CSV is empty.")
    return stations


class convoy(CVRPTWEnv):
    """CVRPTW environment with selectable distance computation.

    Distance modes:
        - `euclidean`: sqrt((x1-x2)^2 + (y1-y2)^2)
        - `manhattan`: abs(x1-x2) + abs(y1-y2)
        - `linear_sum`: (x1-x2) + (y1-y2)
    """

    def __init__(
        self,
        distance_mode: str = "euclidean",
        battery_capacity_kwh: float = 60.0,
        energy_rate_kwh_per_distance: float = 0.5,
        charge_rate_kwh_per_hour: float = 120.0,
        reserve_soc_kwh: float = 0.0,
        num_evs: int = 1,
        charging_pool_csv: str | None = "CP_details.csv",
        charging_pool_sample_size: int = 5,
        **kwargs,
    ):
        """Initialize distance mode, EV parameters, and charging-station pool."""
        super().__init__(**kwargs)
        if battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be > 0.")
        if energy_rate_kwh_per_distance <= 0:
            raise ValueError("energy_rate_kwh_per_distance must be > 0.")
        if charge_rate_kwh_per_hour <= 0:
            raise ValueError("charge_rate_kwh_per_hour must be > 0.")
        if reserve_soc_kwh < 0:
            raise ValueError("reserve_soc_kwh must be >= 0.")
        if num_evs <= 0:
            raise ValueError("num_evs must be > 0.")
        if charging_pool_sample_size < 0:
            raise ValueError("charging_pool_sample_size must be >= 0.")
        self.distance_mode = distance_mode
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.energy_rate_kwh_per_distance = float(energy_rate_kwh_per_distance)
        self.charge_rate_kwh_per_hour = float(charge_rate_kwh_per_hour)
        self.reserve_soc_kwh = float(reserve_soc_kwh)
        self.num_evs = int(num_evs)
        self.charging_pool_sample_size = int(charging_pool_sample_size)
        self.charging_pool_csv = charging_pool_csv
        # Script time values (TW, service, travel) are treated as minutes.
        self.time_units_per_hour = 60.0
        self.cp_pool_locs: torch.Tensor | None = None
        self.cp_pool_rates: torch.Tensor | None = None
        self.cp_pool_costs: torch.Tensor | None = None
        self.cp_pool_ids: torch.Tensor | None = None

        # Load charging-point pool once; each reset samples a subset per instance.
        if self.charging_pool_sample_size > 0:
            if not self.charging_pool_csv or not os.path.exists(self.charging_pool_csv):
                raise ValueError(
                    "Charging pool CSV not found. Set --charging-pool-csv or "
                    "set --charging-pool-sample-size 0."
                )
            cp_rows = parse_charging_station_csv_rows(self.charging_pool_csv)
            if self.charging_pool_sample_size > len(cp_rows):
                raise ValueError(
                    f"--charging-pool-sample-size={self.charging_pool_sample_size} "
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

    @staticmethod
    def _distance(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute pairwise distances according to the selected mode."""
        dx = a[..., 0] - b[..., 0]
        dy = a[..., 1] - b[..., 1]
        if mode == "euclidean":
            return torch.sqrt(dx * dx + dy * dy + 1e-12)
        if mode == "manhattan":
            return torch.abs(dx) + torch.abs(dy)
        if mode == "linear_sum":
            return dx + dy
        raise ValueError(f"Unknown distance mode: {mode}")

    def _build_pairwise_distance_matrix(self, td: TensorDict) -> torch.Tensor:
        """Return an all-node pairwise distance matrix for energy calculations."""
        if "dist_matrix" in td.keys():
            return td["dist_matrix"]
        return self._distance(
            td["locs"][:, :, None, :], td["locs"][:, None, :, :], self.distance_mode
        )

    def _augment_instance_with_charging_stations(self, td: TensorDict) -> TensorDict:
        """Append sampled charging stations and related metadata to one batch."""
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
            return td

        # Sample charging stations per instance from CP_details.csv pool.
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

        # Extend optional matrices so stations can be selected as valid nodes.
        if "dist_matrix" in td.keys() or "travel_time_matrix" in td.keys():
            all_nodes = torch.cat([td["depot"][:, None, :], td["locs"]], dim=1)
            geo = self._distance(
                all_nodes[:, :, None, :], all_nodes[:, None, :, :], self.distance_mode
            )
            if "dist_matrix" in td.keys():
                old_d = td["dist_matrix"]
                old_n = old_d.shape[-1]
                new_d = geo.clone()
                new_d[:, :old_n, :old_n] = old_d
                td.set("dist_matrix", new_d)
            if "travel_time_matrix" in td.keys():
                old_t = td["travel_time_matrix"]
                old_n = old_t.shape[-1]
                new_t = geo.clone()
                new_t[:, :old_n, :old_n] = old_t
                td.set("travel_time_matrix", new_t)
        return td

    def get_action_mask(self, td):
        """Build feasibility mask with time-window + EV + charging-station constraints."""
        not_masked = CVRPEnv.get_action_mask(td)
        if "station_mask" in td.keys():
            # Charging stations are revisitable, so ignore CVRP visited masking for them.
            # Keep the current node masked to avoid zero-distance self-loop actions.
            station_candidates = td["station_mask"].clone()
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

        current_loc = gather_by_index(td["locs"], td["current_node"])
        if "travel_time_matrix" in td.keys():
            time_row = gather_by_index(td["travel_time_matrix"], td["current_node"], dim=1)
            travel_time = time_row.squeeze(1)
        else:
            travel_time = self._distance(
                current_loc[..., None, :], td["locs"], self.distance_mode
            )
        pairwise_dist = self._build_pairwise_distance_matrix(td)
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
        # the EV can still reach any charging node (depot or station).
        energy_to_node = dist * self.energy_rate_kwh_per_distance
        energy_for_customer = (
            dist + nearest_charge_dist
        ) * self.energy_rate_kwh_per_distance
        customer_mask = ~charge_nodes_mask
        required_energy = torch.where(customer_mask, energy_for_customer, energy_to_node)
        battery_ok = td["current_battery"] >= (required_energy + self.reserve_soc_kwh)

        mask = not_masked & can_reach_in_time & battery_ok
        # If nothing is feasible, force return to depot; do not relax TW filtering.
        fallback = ~mask.any(dim=-1, keepdim=True)
        depot_only = torch.zeros_like(mask, dtype=torch.bool)
        depot_only[:, 0] = True
        mask = torch.where(fallback, depot_only, mask)

        # If at least one customer is feasible now, force serving a customer.
        # This prevents infinite charge-node bouncing when stations are revisitable.
        customer_nodes_mask = ~station_mask
        customer_nodes_mask[:, 0] = False
        customer_feasible = mask & customer_nodes_mask
        has_customer_option = customer_feasible.any(dim=-1, keepdim=True)
        mask = torch.where(has_customer_option, customer_feasible, mask)

        # If all customers are already served, force return to depot to terminate route.
        unserved_customers = customer_nodes_mask & (~td["visited"].to(torch.bool))
        has_unserved = unserved_customers.any(dim=-1, keepdim=True)
        mask = torch.where(has_unserved, mask, depot_only)

        # Final safety: ensure at least one feasible action exists.
        needs_fallback = ~mask.any(dim=-1, keepdim=True)
        return torch.where(needs_fallback, depot_only, mask)

    def _step(self, td):
        """Advance transition, update EV SOC/charging state, then refresh mask."""
        batch_size = td["locs"].shape[0]
        device = td["locs"].device
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
        charge_needed = torch.clamp(self.battery_capacity_kwh - remaining_battery, min=0.0)
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
            self.battery_capacity_kwh,
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
                self.battery_capacity_kwh - 1e-6
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

        td.set("action_mask", action_mask)
        return td

    def _get_reward(self, td, actions):
        """Reward = on-time customer rewards - charging cost paid at charging stations."""
        if "customer_reward_per_node" not in td.keys():
            # Backward-compatible fallback: distance-minimization objective.
            if "dist_matrix" in td.keys():
                tour = torch.cat(
                    [
                        torch.zeros(
                            actions.size(0), 1, dtype=actions.dtype, device=actions.device
                        ),
                        actions,
                    ],
                    dim=1,
                )
                frm = tour[:, :-1]
                to = tour[:, 1:]
                b = torch.arange(actions.size(0), device=actions.device)[:, None]
                seg_len = td["dist_matrix"][b, frm, to]
                return -seg_len.sum(dim=-1)
            locs_ordered = torch.cat(
                [td["locs"][..., 0:1, :], gather_by_index(td["locs"], actions)],
                dim=1,
            )
            seg_len = self._distance(
                locs_ordered[..., :-1, :], locs_ordered[..., 1:, :], self.distance_mode
            )
            return -seg_len.sum(dim=-1)

        batch_size, num_steps = actions.shape
        device = actions.device
        batch_idx = torch.arange(batch_size, device=device)
        num_nodes = td["locs"].shape[1]

        if "dist_matrix" in td.keys():
            dist_matrix = td["dist_matrix"]
        else:
            dist_matrix = self._distance(
                td["locs"][:, :, None, :], td["locs"][:, None, :, :], self.distance_mode
            )
        if "travel_time_matrix" in td.keys():
            travel_matrix = td["travel_time_matrix"]
        else:
            travel_matrix = dist_matrix

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

        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_time = torch.zeros(batch_size, dtype=torch.float32, device=device)
        current_battery = torch.full(
            (batch_size,),
            self.battery_capacity_kwh,
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

            charge_needed = torch.clamp(self.battery_capacity_kwh - remaining_battery, min=0.0)
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
            total_reward = total_reward + step_reward

            station_charge_cost = charge_cost_per_kwh_per_node[batch_idx, next_node]
            charging_penalty = torch.where(
                is_station,
                charge_needed * station_charge_cost,
                zero_float,
            )
            total_reward = total_reward - charging_penalty

            visit_mask = torch.zeros_like(served_customers)
            visit_mask.scatter_(1, next_node.unsqueeze(-1), is_customer.unsqueeze(-1))
            served_customers = served_customers | visit_mask

            at_depot = next_node == 0
            active_ready = vehicle_ready_time[batch_idx, current_vehicle_idx]
            active_ready = torch.where(at_depot, finish_time, active_ready)
            vehicle_ready_time[batch_idx, current_vehicle_idx] = active_ready

            next_ready_time, next_vehicle_idx = vehicle_ready_time.min(dim=1)
            full_battery = torch.full_like(remaining_battery, self.battery_capacity_kwh)
            battery_non_depot = torch.where(is_station, full_battery, remaining_battery)

            current_time = torch.where(at_depot, next_ready_time, finish_time)
            current_battery = torch.where(at_depot, full_battery, battery_non_depot)
            current_vehicle_idx = torch.where(at_depot, next_vehicle_idx, current_vehicle_idx)
            current_node = next_node

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
                    self.battery_capacity_kwh,
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
        if "customer_reward_per_node" in td.keys():
            td_reset.set("customer_reward_per_node", td["customer_reward_per_node"])
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

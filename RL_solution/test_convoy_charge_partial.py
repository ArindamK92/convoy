"""
Run:
python test_convoy_charge_partial.py
python test_convoy_charge_partial.py

Print solution paths:
python test_convoy_charge_partial.py --print-solution

Check solution quality using Fixed-set evaluation callback:
python test_convoy_charge_partial.py --epochs 100 --fixed-eval-every 5 --fixed-eval-size 1000


Test using real data (.csv):
Training: synthetic/generated instances
Validation/fixed-eval during training: synthetic/generated instances
python test_convoy_charge_partial.py --test-csv vrptw_data.csv --csv-vehicle-capacity 30

Distance computation options:
python test_convoy_charge_partial.py --distance-mode euclidean
python test_convoy_charge_partial.py --distance-mode linear_sum
python test_convoy_charge_partial.py --distance-mode manhattan --test-csv vrptw_data.csv  --print-solution

Train by sampling from a 200-customer CSV pool (30 per instance):
python test_convoy_charge_partial.py --train-pool-csv vrptw_pool_200.csv --pool-sample-size 30

Use external distance matrix CSV with the sampled customer pool:
python test_convoy_charge_partial.py --train-pool-csv vrptw_pool_200.csv --pool-sample-size 30 --distance-matrix-csv dist_200x200.csv

Train using input csv:
python test_convoy_charge_partial.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --epochs 100  --test-csv vrptw_data.csv  --print-solution


Use distance from input csv:
python test_convoy_charge_partial.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv dist_201x201.csv  --test-csv vrptw_data.csv  --print-solution

Use distance for test data also:
python test_convoy_charge_partial.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv dist_201x201.csv \
  --test-csv vrptw_data.csv \
  --test-distance-matrix-csv dist_201x201.csv \
  --print-solution

Use time to travel from input csv:
python test_convoy_charge_partial.py \
  --train-pool-csv vrptw_pool_200.csv \
  --pool-sample-size 30 \
  --distance-matrix-csv dist_201x201.csv \
  --time-matrix-csv time_201x201.csv \
  --test-csv vrptw_data.csv \
  --test-distance-matrix-csv dist_201x201.csv \
  --test-time-matrix-csv time_201x201.csv \
  --print-solution


Use custom CP   
python3 test_convoy_charge_partial.py \
  --train-pool-csv vrptw_pool_200.csv \
  --distance-matrix-csv dist_201x201.csv \
  --time-matrix-csv time_201x201.csv \
  --test-csv vrptw_data.csv \
  --test-distance-matrix-csv dist_201x201.csv \
  --test-time-matrix-csv time_201x201.csv \
  --charging-pool-csv CP_details.csv \
  --charging-pool-sample-size 10 \
  --ev-num-vehicles 5 \
  --print-solution

Use discrete partial charging amounts (kWh):
python3 test_convoy_charge_partial.py \
  --test-csv vrptw_data.csv \
  --csv-vehicle-capacity 30 \
  --charging-pool-csv CP_details.csv \
  --charging-pool-sample-size 10 \
  --charge-levels-kwh 10,20,30,40,60 \
  --ev-num-vehicles 5 \
  --print-solution

all together:
python3 test_convoy_charge_partial.py \
  --train-pool-csv vrptw_pool_200.csv \
  --distance-matrix-csv dist_201x201.csv \
  --time-matrix-csv time_201x201.csv \
  --test-csv vrptw_data.csv \
  --test-distance-matrix-csv dist_201x201.csv \
  --test-time-matrix-csv time_201x201.csv \
  --charging-pool-csv CP_details.csv \
  --charging-pool-sample-size 10 \
  --ev-num-vehicles 5 \
  --charge-levels-kwh 10,20,30,40,60 \
  --print-solution


  
"""

import argparse
import csv
import os

import lightning as L
import torch

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from tensordict import TensorDict
from rl4co.envs import CVRPTWEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer
from rl4co.utils.ops import gather_by_index
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader


class CVRPTWCustomDistanceEnv(CVRPTWEnv):
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
        charge_levels_kwh: list[float] | None = None,
        **kwargs,
    ):
        """Initialize distance mode, EV parameters, stations, and charge levels."""
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
        if charge_levels_kwh is None:
            charge_levels_kwh = [0.25, 0.5, 0.75, 1.0]
        self.distance_mode = distance_mode
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.energy_rate_kwh_per_distance = float(energy_rate_kwh_per_distance)
        self.charge_rate_kwh_per_hour = float(charge_rate_kwh_per_hour)
        self.reserve_soc_kwh = float(reserve_soc_kwh)
        self.num_evs = int(num_evs)
        self.charging_pool_sample_size = int(charging_pool_sample_size)
        self.charging_pool_csv = charging_pool_csv
        parsed_levels = []
        for level in charge_levels_kwh:
            val = float(level)
            if val <= 0:
                continue
            # If values look like fractions, treat them as SOC fractions.
            parsed_levels.append(
                val * self.battery_capacity_kwh if 0 < val <= 1.0 else val
            )
        if not parsed_levels:
            raise ValueError("At least one positive charge level is required.")
        self.charge_levels_kwh = torch.tensor(
            sorted(set(parsed_levels)), dtype=torch.float32
        )
        self.num_charge_levels = int(self.charge_levels_kwh.numel())
        # Script time values (TW, service, travel) are treated as minutes.
        self.time_units_per_hour = 60.0
        self.cp_pool_locs: torch.Tensor | None = None
        self.cp_pool_rates: torch.Tensor | None = None
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
        """Append sampled charging stations plus discrete charge-decision actions."""
        batch_size = td["locs"].shape[0]
        device = td["locs"].device
        num_customers = td["locs"].shape[1]
        max_tw_end = td["time_windows"][..., 1].amax(dim=1, keepdim=True)

        n_stations = 0
        sampled_station_locs = torch.zeros((batch_size, 0, 2), dtype=td["locs"].dtype, device=device)
        sampled_station_rates = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
        sampled_station_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        if (
            self.charging_pool_sample_size > 0
            and self.cp_pool_locs is not None
            and self.cp_pool_rates is not None
            and self.cp_pool_ids is not None
        ):
            n_stations = self.charging_pool_sample_size
            sample_idx = torch.stack(
                [
                    torch.randperm(self.cp_pool_locs.shape[0])[:n_stations]
                    for _ in range(batch_size)
                ],
                dim=0,
            )
            sampled_station_locs = self.cp_pool_locs[sample_idx].to(device)
            sampled_station_rates = self.cp_pool_rates[sample_idx].to(device)
            sampled_station_ids = self.cp_pool_ids[sample_idx].to(device)

        # Append sampled stations.
        locs_aug = td["locs"]
        demand_aug = td["demand"]
        durations_aug = td["durations"]
        tw_aug = td["time_windows"]
        if n_stations > 0:
            locs_aug = torch.cat([locs_aug, sampled_station_locs], dim=1)
            demand_aug = torch.cat(
                [
                    demand_aug,
                    torch.zeros((batch_size, n_stations), dtype=td["demand"].dtype, device=device),
                ],
                dim=1,
            )
            durations_aug = torch.cat(
                [
                    durations_aug,
                    torch.zeros(
                        (batch_size, n_stations), dtype=td["durations"].dtype, device=device
                    ),
                ],
                dim=1,
            )
            station_tw = torch.zeros(
                (batch_size, n_stations, 2), dtype=td["time_windows"].dtype, device=device
            )
            station_tw[..., 1] = max_tw_end.expand(-1, n_stations)
            tw_aug = torch.cat([tw_aug, station_tw], dim=1)

        # Append pseudo nodes used only for discrete charge amount selection.
        n_levels = self.num_charge_levels
        if n_levels > 0:
            decision_locs = td["depot"][:, None, :].expand(batch_size, n_levels, 2)
            locs_aug = torch.cat([locs_aug, decision_locs], dim=1)
            demand_aug = torch.cat(
                [
                    demand_aug,
                    torch.zeros((batch_size, n_levels), dtype=td["demand"].dtype, device=device),
                ],
                dim=1,
            )
            durations_aug = torch.cat(
                [
                    durations_aug,
                    torch.zeros((batch_size, n_levels), dtype=td["durations"].dtype, device=device),
                ],
                dim=1,
            )
            decision_tw = torch.zeros(
                (batch_size, n_levels, 2), dtype=td["time_windows"].dtype, device=device
            )
            decision_tw[..., 1] = max_tw_end.expand(-1, n_levels)
            tw_aug = torch.cat([tw_aug, decision_tw], dim=1)

        td.set("locs", locs_aug)
        td.set("demand", demand_aug)
        td.set("durations", durations_aug)
        td.set("time_windows", tw_aug)

        total_nodes = 1 + locs_aug.shape[1]
        station_start = 1 + num_customers
        station_end = station_start + n_stations
        decision_start = station_end
        decision_end = decision_start + n_levels

        station_mask = torch.zeros((batch_size, total_nodes), dtype=torch.bool, device=device)
        if n_stations > 0:
            station_mask[:, station_start:station_end] = True
        decision_mask = torch.zeros_like(station_mask)
        if n_levels > 0:
            decision_mask[:, decision_start:decision_end] = True

        charge_nodes_mask = station_mask.clone()
        charge_nodes_mask[:, 0] = True  # Depot is always a charging node.

        charge_rate_per_node = torch.zeros((batch_size, total_nodes), dtype=torch.float32, device=device)
        charge_rate_per_node[:, 0] = self.charge_rate_kwh_per_hour
        if n_stations > 0:
            charge_rate_per_node[:, station_start:station_end] = sampled_station_rates

        charge_amount_per_node = torch.zeros(
            (batch_size, total_nodes), dtype=torch.float32, device=device
        )
        if n_levels > 0:
            charge_amount_per_node[:, decision_start:decision_end] = self.charge_levels_kwh.to(
                device
            )[None, :]

        cp_id_per_node = torch.full((batch_size, total_nodes), -1, dtype=torch.long, device=device)
        if n_stations > 0:
            cp_id_per_node[:, station_start:station_end] = sampled_station_ids

        td.set("charge_nodes_mask", charge_nodes_mask)
        td.set("station_mask", station_mask)
        td.set("decision_mask", decision_mask)
        td.set("charge_rate_per_node", charge_rate_per_node)
        td.set("charge_amount_per_node", charge_amount_per_node)
        td.set("cp_id_per_node", cp_id_per_node)

        # Extend optional matrices so sampled stations/decision nodes share one action space.
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
        """Build feasibility mask for movement plus discrete charging decisions."""
        not_masked = CVRPEnv.get_action_mask(td)
        decision_mask = td["decision_mask"] if "decision_mask" in td.keys() else torch.zeros_like(not_masked)
        station_mask = td["station_mask"] if "station_mask" in td.keys() else torch.zeros_like(not_masked)
        charge_nodes_mask = (
            td["charge_nodes_mask"] if "charge_nodes_mask" in td.keys() else torch.zeros_like(not_masked)
        )
        if charge_nodes_mask.shape[-1] > 0:
            charge_nodes_mask[:, 0] = True
        needs_charge_decision = (
            td["needs_charge_decision"]
            if "needs_charge_decision" in td.keys()
            else torch.zeros((not_masked.shape[0], 1), dtype=torch.bool, device=not_masked.device)
        )

        # Decision pseudo-actions are never valid during movement mode.
        not_masked = not_masked & (~decision_mask)

        # Charging stations are revisitable in movement mode.
        if station_mask.any():
            station_candidates = station_mask.clone()
            same_node = torch.zeros_like(station_candidates, dtype=torch.bool)
            same_node.scatter_(1, td["current_node"], True)
            station_candidates = station_candidates & ~same_node
            if "done" in td.keys():
                done = td["done"]
                if done.dim() == 1:
                    done = done.unsqueeze(-1)
                station_candidates = station_candidates & (~done.expand_as(station_candidates))
            not_masked = torch.where(station_candidates, torch.ones_like(not_masked), not_masked)

        current_loc = gather_by_index(td["locs"], td["current_node"])
        if "travel_time_matrix" in td.keys():
            time_row = gather_by_index(td["travel_time_matrix"], td["current_node"], dim=1)
            travel_time = time_row.squeeze(1)
        else:
            travel_time = self._distance(current_loc[..., None, :], td["locs"], self.distance_mode)
        pairwise_dist = self._build_pairwise_distance_matrix(td)
        dist = gather_by_index(pairwise_dist, td["current_node"], dim=1).squeeze(1)
        td.update({"current_loc": current_loc, "distances": dist, "travel_times": travel_time})
        can_reach_in_time = td["current_time"] + travel_time <= td["time_windows"][..., 1]

        nearest_charge_dist = torch.where(
            charge_nodes_mask[:, None, :],
            pairwise_dist,
            torch.full_like(pairwise_dist, float("inf")),
        ).min(dim=-1).values
        energy_to_node = dist * self.energy_rate_kwh_per_distance
        energy_for_customer = (dist + nearest_charge_dist) * self.energy_rate_kwh_per_distance
        customer_move_mask = (~charge_nodes_mask) & (~decision_mask)
        required_energy = torch.where(customer_move_mask, energy_for_customer, energy_to_node)
        battery_ok = td["current_battery"] >= (required_energy + self.reserve_soc_kwh)

        move_mask = not_masked & can_reach_in_time & battery_ok
        fallback = ~move_mask.any(dim=-1, keepdim=True)
        relaxed_mask = not_masked & battery_ok
        move_mask = torch.where(fallback, relaxed_mask, move_mask)

        # If customer actions exist, prioritize customer service over charging hops.
        customer_nodes_mask = customer_move_mask.clone()
        if customer_nodes_mask.shape[-1] > 0:
            customer_nodes_mask[:, 0] = False
        customer_feasible = move_mask & customer_nodes_mask
        has_customer_option = customer_feasible.any(dim=-1, keepdim=True)
        move_mask = torch.where(has_customer_option, customer_feasible, move_mask)

        # Once all customers are served, force returning depot in movement mode.
        unserved_customers = customer_nodes_mask & (~td["visited"].to(torch.bool))
        has_unserved = unserved_customers.any(dim=-1, keepdim=True)
        depot_only = torch.zeros_like(move_mask, dtype=torch.bool)
        depot_only[:, 0] = True
        move_mask = torch.where(has_unserved, move_mask, depot_only)
        move_mask = torch.where(~move_mask.any(dim=-1, keepdim=True), depot_only, move_mask)

        # Charge-decision mode: only discrete charge amount actions are feasible.
        charge_decision_mask = decision_mask.clone()
        if "done" in td.keys():
            done = td["done"]
            if done.dim() == 1:
                done = done.unsqueeze(-1)
            charge_decision_mask = charge_decision_mask & (~done.expand_as(charge_decision_mask))

        needs_decision = needs_charge_decision.expand_as(move_mask)
        final_mask = torch.where(needs_decision, charge_decision_mask, move_mask)
        return final_mask

    def _step(self, td):
        """Advance transition with movement actions and charge-decision actions."""
        batch_size = td["locs"].shape[0]
        device = td["locs"].device
        raw_action = td["action"]

        needs_charge_decision = (
            td["needs_charge_decision"]
            if "needs_charge_decision" in td.keys()
            else torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        )
        current_charge_station = (
            td["current_charge_station"]
            if "current_charge_station" in td.keys()
            else torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        )
        effective_action = torch.where(
            needs_charge_decision.squeeze(-1), current_charge_station.squeeze(-1), raw_action
        )

        travel_time = gather_by_index(td["travel_times"], effective_action).reshape([batch_size, 1])
        travel_distance = gather_by_index(td["distances"], effective_action).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], effective_action).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], effective_action)[..., 0].reshape([batch_size, 1])
        arrival_or_wait = torch.max(td["current_time"] + travel_time, start_times)

        process_move = ~needs_charge_decision
        energy_used = torch.where(
            process_move, travel_distance * self.energy_rate_kwh_per_distance, torch.zeros_like(travel_distance)
        )
        remaining_battery = torch.clamp(td["current_battery"] - energy_used, min=0.0)

        charge_nodes_mask = td["charge_nodes_mask"] if "charge_nodes_mask" in td.keys() else None
        station_mask = td["station_mask"] if "station_mask" in td.keys() else None
        if charge_nodes_mask is not None:
            at_charge_node = gather_by_index(
                charge_nodes_mask.to(torch.float32), effective_action
            ).reshape([batch_size, 1]) > 0.5
            selected_charge_rate = gather_by_index(
                td["charge_rate_per_node"], effective_action
            ).reshape([batch_size, 1])
        else:
            at_charge_node = effective_action[:, None] == 0
            selected_charge_rate = torch.full_like(remaining_battery, self.charge_rate_kwh_per_hour)
        if station_mask is not None:
            at_station = gather_by_index(
                station_mask.to(torch.float32), effective_action
            ).reshape([batch_size, 1]) > 0.5
        else:
            at_station = torch.zeros_like(at_charge_node)
        at_depot = effective_action[:, None] == 0
        at_station_non_depot = process_move & at_station & (~at_depot)

        # Movement step time updates.
        finish_move = torch.where(
            at_station_non_depot,  # arrival at CP triggers a later charge-decision step.
            arrival_or_wait,
            arrival_or_wait + duration,
        )
        depot_charge_needed = torch.clamp(self.battery_capacity_kwh - remaining_battery, min=0.0)
        safe_charge_rate = torch.clamp(selected_charge_rate, min=1e-6)
        depot_charge_time = (depot_charge_needed / safe_charge_rate) * self.time_units_per_hour
        finish_move = torch.where(process_move & at_depot, arrival_or_wait + depot_charge_time, finish_move)

        # Discrete charge-decision step updates.
        selected_charge_amount = torch.zeros_like(remaining_battery)
        if "charge_amount_per_node" in td.keys():
            selected_charge_amount = gather_by_index(
                td["charge_amount_per_node"], raw_action
            ).reshape([batch_size, 1])
        station_rate_for_decision = gather_by_index(
            td["charge_rate_per_node"], current_charge_station.squeeze(-1)
        ).reshape([batch_size, 1])
        station_rate_for_decision = torch.clamp(station_rate_for_decision, min=1e-6)
        decision_charge_add = torch.clamp(
            torch.minimum(selected_charge_amount, self.battery_capacity_kwh - remaining_battery),
            min=0.0,
        )
        decision_charge_time = (decision_charge_add / station_rate_for_decision) * self.time_units_per_hour
        finish_decision = td["current_time"] + decision_charge_time

        finish_time = torch.where(needs_charge_decision, finish_decision, finish_move)
        full_battery = torch.full(
            (batch_size, 1),
            self.battery_capacity_kwh,
            dtype=remaining_battery.dtype,
            device=device,
        )
        battery_after = remaining_battery
        battery_after = torch.where(process_move & at_depot, full_battery, battery_after)
        battery_after = torch.where(needs_charge_decision, remaining_battery + decision_charge_add, battery_after)

        # Fleet readiness changes only when moving to depot.
        vehicle_ready_time = td["ev_vehicle_ready_time"].clone()
        current_vehicle_idx = td["current_vehicle_idx"]
        batch_idx = torch.arange(batch_size, device=device)
        active_idx = current_vehicle_idx.squeeze(-1)
        active_ready = vehicle_ready_time[batch_idx, active_idx].unsqueeze(-1)
        active_ready = torch.where(process_move & at_depot, finish_time, active_ready)
        vehicle_ready_time[batch_idx, active_idx] = active_ready.squeeze(-1)
        next_ready_time, next_vehicle_idx = vehicle_ready_time.min(dim=1)

        td["current_time"] = torch.where(process_move & at_depot, next_ready_time.unsqueeze(-1), finish_time)
        td["current_battery"] = battery_after
        td["current_vehicle_idx"] = torch.where(
            process_move & at_depot, next_vehicle_idx.unsqueeze(-1), current_vehicle_idx
        )
        td["ev_vehicle_ready_time"] = vehicle_ready_time

        # Base CVRP updates should use movement target while in decision mode.
        td["action"] = effective_action
        td = super(CVRPTWEnv, self)._step(td)

        # Enter decision mode after reaching a non-depot charging station.
        td["needs_charge_decision"] = torch.where(
            needs_charge_decision, torch.zeros_like(needs_charge_decision), at_station_non_depot
        )
        td["current_charge_station"] = torch.where(
            at_station_non_depot, effective_action[:, None], torch.zeros_like(current_charge_station)
        )

        action_mask = self.get_action_mask(td)
        if "station_mask" in td.keys() and "charge_nodes_mask" in td.keys():
            decision_mask = td["decision_mask"] if "decision_mask" in td.keys() else torch.zeros_like(td["station_mask"])
            customer_nodes_mask = (~td["station_mask"]) & (~decision_mask)
            customer_nodes_mask[:, 0] = False
            unserved_customers = customer_nodes_mask & (~td["visited"].to(torch.bool))
            has_unserved = unserved_customers.any(dim=-1)
            feasible_customer_now = (action_mask & customer_nodes_mask).any(dim=-1)
            at_charge_node_now = gather_by_index(
                td["charge_nodes_mask"].to(torch.float32), td["current_node"]
            ).squeeze(-1) > 0.5
            dead_end = (
                has_unserved
                & (~feasible_customer_now)
                & at_charge_node_now
                & (~td["needs_charge_decision"].squeeze(-1))
            )
            if dead_end.any():
                td["done"] = td["done"] | dead_end
                depot_only = torch.zeros_like(action_mask, dtype=torch.bool)
                depot_only[:, 0] = True
                action_mask = torch.where(dead_end[:, None], depot_only, action_mask)

        td.set("action_mask", action_mask)
        td["action"] = raw_action
        return td

    def _get_reward(self, td, actions):
        """Return negative route length computed from matrix or coordinate distances."""
        actions_eff = actions.clone()
        if "decision_mask" in td.keys():
            decision_flags = torch.gather(td["decision_mask"], 1, actions)
            for t in range(actions.shape[1]):
                prev = torch.zeros_like(actions_eff[:, t]) if t == 0 else actions_eff[:, t - 1]
                actions_eff[:, t] = torch.where(decision_flags[:, t], prev, actions_eff[:, t])
        if "dist_matrix" in td.keys():
            tour = torch.cat(
                [
                    torch.zeros(
                        actions.size(0), 1, dtype=actions.dtype, device=actions.device
                    ),
                    actions_eff,
                ],
                dim=1,
            )
            frm = tour[:, :-1]
            to = tour[:, 1:]
            b = torch.arange(actions.size(0), device=actions.device)[:, None]
            seg_len = td["dist_matrix"][b, frm, to]
            return -seg_len.sum(dim=-1)
        locs_ordered = torch.cat(
            [td["locs"][..., 0:1, :], gather_by_index(td["locs"], actions_eff)],
            dim=1,
        )
        seg_len = self._distance(
            locs_ordered[..., :-1, :], locs_ordered[..., 1:, :], self.distance_mode
        )
        return -seg_len.sum(dim=-1)

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
                "needs_charge_decision": torch.zeros(
                    *batch_size, 1, dtype=torch.bool, device=device
                ),
                "current_charge_station": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
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
        if "decision_mask" in td.keys():
            td_reset.set("decision_mask", td["decision_mask"])
            # Decision pseudo-nodes must never be required for episode completion.
            td_reset["visited"] = torch.maximum(
                td_reset["visited"], td["decision_mask"].to(td_reset["visited"].dtype)
            )
        if "charge_nodes_mask" in td.keys():
            td_reset.set("charge_nodes_mask", td["charge_nodes_mask"])
        if "charge_rate_per_node" in td.keys():
            td_reset.set("charge_rate_per_node", td["charge_rate_per_node"])
        if "charge_amount_per_node" in td.keys():
            td_reset.set("charge_amount_per_node", td["charge_amount_per_node"])
        if "cp_id_per_node" in td.keys():
            td_reset.set("cp_id_per_node", td["cp_id_per_node"])
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset


def parse_vrptw_csv_rows(csv_path: str):
    """Parse VRPTW CSV rows and return depot row + customer rows."""
    depot = None
    customers = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "is_depot",
            "x",
            "y",
            "demand",
            "tw_start",
            "tw_end",
            "service_time",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV must contain columns: "
                "is_depot,x,y,demand,tw_start,tw_end,service_time"
            )
        for row in reader:
            is_depot = int(row["is_depot"])
            rec = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "demand": float(row["demand"]),
                "tw_start": float(row["tw_start"]),
                "tw_end": float(row["tw_end"]),
                "service_time": float(row["service_time"]),
                "customer_id": (
                    int(row["customer_id"])
                    if "customer_id" in row and row["customer_id"] != ""
                    else None
                ),
            }
            if rec["tw_end"] <= rec["tw_start"]:
                raise ValueError("Each row must satisfy tw_end > tw_start.")
            if is_depot == 1:
                if depot is not None:
                    raise ValueError("CSV must contain exactly one depot row.")
                depot = rec
            else:
                customers.append(rec)
    if depot is None:
        raise ValueError("CSV must contain one row with is_depot=1.")
    if not customers:
        raise ValueError("CSV must contain at least one customer row (is_depot=0).")
    return depot, customers


def parse_distance_matrix_csv(matrix_csv_path: str) -> torch.Tensor:
    """Parse a square numeric distance matrix CSV into a torch tensor.

    Accepted shapes:
        - [N, N] for customer-to-customer only.
        - [N+1, N+1] where row/col 0 correspond to depot.
    """
    rows = []
    with open(matrix_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        raise ValueError("Distance matrix CSV is empty.")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("Distance matrix CSV must be rectangular.")
    if len(rows) != n:
        raise ValueError("Distance matrix CSV must be square.")
    return torch.tensor(rows, dtype=torch.float32)


def parse_charging_station_csv_rows(csv_path: str) -> list[dict]:
    """Parse charging-point rows with x/y coordinates and charging rate."""
    stations: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(
                "Charging station CSV must contain headers including x, y and charge rate."
            )
        fieldnames = set(reader.fieldnames)
        if "x" not in fieldnames or "y" not in fieldnames:
            raise ValueError("Charging station CSV must contain columns: x,y,<charge_rate>.")
        rate_col = None
        for candidate in (
            "charge_rate_kwh_per_hour",
            "charge_rate",
            "charging_rate_kwh_per_hour",
            "charging_rate",
        ):
            if candidate in fieldnames:
                rate_col = candidate
                break
        if rate_col is None:
            raise ValueError(
                "Charging station CSV must include one rate column: "
                "charge_rate_kwh_per_hour/charge_rate/charging_rate_kwh_per_hour/charging_rate."
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
                "x": float(row["x"]),
                "y": float(row["y"]),
                "charge_rate_kwh_per_hour": float(row[rate_col]),
            }
            if rec["charge_rate_kwh_per_hour"] <= 0:
                raise ValueError("Charging station rate must be > 0 for every row.")
            stations.append(rec)
    if not stations:
        raise ValueError("Charging station CSV is empty.")
    return stations


def parse_charge_levels_kwh(raw: str) -> list[float]:
    """Parse comma-separated discrete charging amounts in kWh."""
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("--charge-levels-kwh must contain at least one value.")
    levels = sorted({float(x) for x in parts})
    if any(x <= 0 for x in levels):
        raise ValueError("--charge-levels-kwh values must be > 0.")
    return levels


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

        depot, customers = parse_vrptw_csv_rows(csv_path)
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

        depot = self.depot_xy.unsqueeze(0).expand(bs, -1)
        depot_duration = torch.full((bs, 1), self.depot_service, dtype=torch.float32)
        durations = torch.cat([depot_duration, durations_customers], dim=1)

        depot_tw = self.depot_tw.unsqueeze(0).unsqueeze(1).expand(bs, 1, 2)
        time_windows = torch.cat([depot_tw, tw_customers], dim=1)

        capacity = torch.full((bs, 1), float(self.capacity), dtype=torch.float32)
        out = {
            "locs": locs,
            "depot": depot,
            "demand": demand,
            "capacity": capacity,
            "durations": durations,
            "time_windows": time_windows,
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
                depot_to_cust = CVRPTWCustomDistanceEnv._distance(
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

        from tensordict import TensorDict

        return TensorDict(out, batch_size=[bs])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CVRPTW training/testing.

    Supported arguments:
        --num-loc (int, default=20):
            Number of customer nodes per generated instance (depot is added separately).
        --epochs (int, default=100):
            Number of training epochs.
        --batch-size (int, default=256):
            Training batch size.
        --eval-batch-size (int, default=512):
            Batch size for validation and testing.
        --train-data-size (int, default=4096):
            Number of generated training instances per epoch.
        --val-data-size (int, default=1024):
            Number of generated validation instances.
        --test-data-size (int, default=1024):
            Number of generated test instances used to compute final test reward.
        --lr (float, default=1e-4):
            Optimizer learning rate.
        --max-time (float, default=480.0):
            Maximum time horizon for CVRPTW time windows.
        --seed (int, default=42):
            Random seed used for reproducibility.
        --baseline (str, default="exponential"):
            REINFORCE baseline type. Choices:
            "exponential", "rollout", "shared", "mean", "no", "critic".
        --accelerator (str, default="auto"):
            Lightning accelerator backend. Choices:
            "auto" (use GPU if available else CPU), "cpu", "gpu".
        --print-solution (flag, default=False):
            If set, run one greedy decode on a single test instance and print
            the full action sequence and split routes for each vehicle.
        --fixed-eval-size (int, default=512):
            Size of the fixed validation-like set used to track quality over time.
            This set is generated once before training and reused at each check.
        --fixed-eval-every (int, default=5):
            Run fixed-set quality evaluation every N epochs (also at final epoch).
        --checkpoint-dir (str, default="checkpoints_vrptw"):
            Directory where best/last checkpoints are saved.
        --test-csv (str, default=None):
            Optional CSV file for a custom single test instance. If provided, the
            script decodes and reports reward/routes on this instance after testing.
            Expected columns per row:
            is_depot,x,y,demand,tw_start,tw_end,service_time
            Exactly one row must have is_depot=1. Demand is interpreted as absolute
            demand and normalized by --csv-vehicle-capacity.
        --csv-vehicle-capacity (float, default=30.0):
            Vehicle capacity used to normalize CSV demands into [0, 1] for RL4CO.
        --distance-mode (str, default="manhattan"):
            Distance function used inside CVRPTW transitions and reward.
            Choices:
            "euclidean" -> sqrt((x1-x2)^2 + (y1-y2)^2)
            "manhattan" -> abs(x1-x2) + abs(y1-y2)
            "linear_sum" -> (x1-x2) + (y1-y2)
        --ev-battery-capacity-kwh (float, default=60.0):
            Full battery capacity (kWh) for each EV.
        --ev-energy-rate-kwh-per-distance (float, default=0.5):
            Energy usage per distance unit traveled.
        --ev-charge-rate-kwh-per-hour (float, default=120.0):
            Depot charging rate. Charging time is converted to minutes.
        --ev-reserve-soc-kwh (float, default=0.0):
            Minimum reserve state-of-charge required after feasibility checks.
        --ev-num-vehicles (int, default=1):
            Number of EVs in fleet (all start full at time 0).
        --charging-pool-csv (str, default="CP_details.csv"):
            Charging-station pool CSV path. Expected columns:
            x,y,charge_rate_kwh_per_hour (or charge_rate).
        --charging-pool-sample-size (int, default=5):
            Number of charging stations sampled per instance from the pool CSV.
        --charge-levels-kwh (str, default="15,30,45,60"):
            Comma-separated discrete charging amounts used as RL decisions at CPs.
        --train-pool-csv (str, default=None):
            Optional CSV pool (e.g. 200 customers + 1 depot) used to generate
            train/val/test instances by random customer subsampling.
        --pool-sample-size (int, default=30):
            Number of customers sampled per generated instance from
            --train-pool-csv.
        --pool-vehicle-capacity (float, default=30.0):
            Vehicle capacity used to normalize customer demands from
            --train-pool-csv.
        --distance-matrix-csv (str, default=None):
            Optional external distance matrix CSV for --train-pool-csv mode.
            Supported shapes:
            [N,N] for customer-to-customer distances only, or
            [N+1,N+1] including depot at row/col 0.
            If [N,N] is provided, depot-customer distances are derived from
            coordinates using --distance-mode.
        --time-matrix-csv (str, default=None):
            Optional external travel-time matrix CSV for --train-pool-csv mode.
            Same shape rules as --distance-matrix-csv.
        --test-distance-matrix-csv (str, default=None):
            Optional distance matrix CSV for --test-csv instance.
            Supported shapes:
            [M,M] for customer-to-customer only, or
            [M+1,M+1] including depot at row/col 0.
            If [M,M] is provided, depot-customer distances are derived from
            coordinates using --distance-mode.
            Larger square matrices are also supported when --test-csv includes
            `customer_id` column for selecting the proper submatrix.
        --test-time-matrix-csv (str, default=None):
            Optional travel-time matrix CSV for --test-csv instance.
            Same shape/submatrix rules as --test-distance-matrix-csv.

    Returns:
        argparse.Namespace: Parsed command-line values.
    """
    parser = argparse.ArgumentParser(
        description="Train and test RL4CO on VRPTW (CVRPTW in rl4co)."
    )
    parser.add_argument("--num-loc", type=int, default=20, help="Number of customers.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Train batch size.")
    parser.add_argument(
        "--eval-batch-size", type=int, default=512, help="Val/Test batch size."
    )
    parser.add_argument(
        "--train-data-size", type=int, default=4096, help="Train samples per epoch."
    )
    parser.add_argument(
        "--val-data-size", type=int, default=1024, help="Validation sample count."
    )
    parser.add_argument(
        "--test-data-size", type=int, default=1024, help="Test sample count."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max-time", type=float, default=480.0, help="TW horizon.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="exponential",
        choices=["exponential", "rollout", "shared", "mean", "no", "critic"],
        help="REINFORCE baseline type.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Lightning accelerator.",
    )
    parser.add_argument(
        "--print-solution",
        action="store_true",
        help="Print one decoded VRPTW solution path after testing.",
    )
    parser.add_argument(
        "--fixed-eval-size",
        type=int,
        default=512,
        help="Fixed-set size used to track solution quality over epochs.",
    )
    parser.add_argument(
        "--fixed-eval-every",
        type=int,
        default=5,
        help="Evaluate on fixed set every N epochs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_vrptw",
        help="Directory for best/last model checkpoints.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to a custom single-instance VRPTW CSV for final testing.",
    )
    parser.add_argument(
        "--csv-vehicle-capacity",
        type=float,
        default=30.0,
        help="Vehicle capacity used to normalize CSV demands.",
    )
    parser.add_argument(
        "--distance-mode",
        type=str,
        default="manhattan",
        choices=["euclidean", "manhattan", "linear_sum"],
        help="Distance function for travel time and reward.",
    )
    parser.add_argument(
        "--ev-battery-capacity-kwh",
        type=float,
        default=60.0,
        help="EV battery capacity in kWh.",
    )
    parser.add_argument(
        "--ev-energy-rate-kwh-per-distance",
        type=float,
        default=0.5,
        help="Energy use per distance unit (kWh / distance-unit).",
    )
    parser.add_argument(
        "--ev-charge-rate-kwh-per-hour",
        type=float,
        default=120.0,
        help="Charging rate at depot (kWh/hour).",
    )
    parser.add_argument(
        "--ev-reserve-soc-kwh",
        type=float,
        default=0.0,
        help="Reserve SOC that must remain (kWh).",
    )
    parser.add_argument(
        "--ev-num-vehicles",
        type=int,
        default=1,
        help="Number of EVs available in fleet.",
    )
    parser.add_argument(
        "--charging-pool-csv",
        type=str,
        default="CP_details.csv",
        help="Charging station pool CSV (x,y,charge_rate).",
    )
    parser.add_argument(
        "--charging-pool-sample-size",
        type=int,
        default=5,
        help="Number of charging stations sampled per instance.",
    )
    parser.add_argument(
        "--charge-levels-kwh",
        type=str,
        default="15,30,45,60",
        help="Comma-separated discrete charging amounts in kWh.",
    )
    parser.add_argument(
        "--train-pool-csv",
        type=str,
        default=None,
        help="CSV customer pool used to sample instances during training.",
    )
    parser.add_argument(
        "--pool-sample-size",
        type=int,
        default=30,
        help="Number of customers sampled per generated instance from pool CSV.",
    )
    parser.add_argument(
        "--pool-vehicle-capacity",
        type=float,
        default=30.0,
        help="Vehicle capacity used for demand normalization in pool CSV mode.",
    )
    parser.add_argument(
        "--distance-matrix-csv",
        type=str,
        default=None,
        help="External distance matrix CSV for train-pool mode.",
    )
    parser.add_argument(
        "--time-matrix-csv",
        type=str,
        default=None,
        help="External travel-time matrix CSV for train-pool mode.",
    )
    parser.add_argument(
        "--test-distance-matrix-csv",
        type=str,
        default=None,
        help="External distance matrix CSV for test-csv instance.",
    )
    parser.add_argument(
        "--test-time-matrix-csv",
        type=str,
        default=None,
        help="External travel-time matrix CSV for test-csv instance.",
    )
    return parser.parse_args()


def extract_reward(metrics: dict) -> float:
    """Extract a scalar reward from metric dictionaries with varying key names."""
    for key in ("test/reward", "test/reward/0"):
        if key in metrics:
            return float(metrics[key])
    reward_keys = [k for k in metrics if "reward" in k]
    if not reward_keys:
        raise RuntimeError(f"Could not find reward key in test metrics: {metrics}")
    return float(metrics[reward_keys[0]])


def split_routes_from_actions(
    actions_1d: torch.Tensor,
    decision_mask_1d: torch.Tensor | None = None,
    include_decision_actions: bool = False,
) -> list[list[int]]:
    """Split a flat CVRPTW action sequence into per-vehicle routes.

    In RL4CO CVRPTW, node 0 is the depot and depot visits separate vehicle tours.
    When `include_decision_actions=True`, partial-charge actions are kept in routes.
    """
    routes: list[list[int]] = []
    current_route = [0]

    for node in actions_1d.tolist():
        node = int(node)
        is_decision = (
            decision_mask_1d is not None
            and 0 <= node < int(decision_mask_1d.shape[0])
            and bool(decision_mask_1d[node].item())
        )
        if is_decision and not include_decision_actions:
            # Hide pseudo-actions in movement-only routes if requested.
            continue
        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
        else:
            current_route.append(node)

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes


def format_node_label(
    node: int,
    station_mask_1d: torch.Tensor | None = None,
    cp_id_per_node_1d: torch.Tensor | None = None,
    decision_mask_1d: torch.Tensor | None = None,
    charge_amount_per_node_1d: torch.Tensor | None = None,
) -> str:
    """Format node id to distinguish depot/customer/charging-station nodes."""
    node = int(node)
    if node == 0:
        return "DEPOT"
    if (
        decision_mask_1d is not None
        and 0 <= node < int(decision_mask_1d.shape[0])
        and bool(decision_mask_1d[node].item())
    ):
        charge_kwh = 0.0
        if (
            charge_amount_per_node_1d is not None
            and 0 <= node < int(charge_amount_per_node_1d.shape[0])
        ):
            charge_kwh = float(charge_amount_per_node_1d[node].item())
        return f"CHG+{charge_kwh:.1f}kWh"
    if (
        station_mask_1d is not None
        and 0 <= node < int(station_mask_1d.shape[0])
        and bool(station_mask_1d[node].item())
    ):
        cp_id = -1
        if (
            cp_id_per_node_1d is not None
            and 0 <= node < int(cp_id_per_node_1d.shape[0])
        ):
            cp_id = int(cp_id_per_node_1d[node].item())
        return f"CP{cp_id}" if cp_id >= 0 else f"CP_NODE_{node}"
    return f"CUST{node}"


def build_customer_visit_trace(
    env: CVRPTWCustomDistanceEnv,
    td_state: TensorDict,
    actions_1d: torch.Tensor,
) -> list[dict]:
    """Simulate one decoded solution and collect per-customer time/SOC trace."""
    all_locs = td_state["locs"][0]
    if "dist_matrix" in td_state.keys():
        dist_matrix = td_state["dist_matrix"][0]
    else:
        dist_matrix = env._distance(
            all_locs[:, None, :], all_locs[None, :, :], env.distance_mode
        )
    if "travel_time_matrix" in td_state.keys():
        travel_matrix = td_state["travel_time_matrix"][0]
    else:
        travel_matrix = dist_matrix

    durations = td_state["durations"][0]
    tw_starts = td_state["time_windows"][0, :, 0]
    if "charge_nodes_mask" in td_state.keys():
        charge_nodes_mask = td_state["charge_nodes_mask"][0]
    else:
        charge_nodes_mask = torch.zeros(dist_matrix.shape[0], dtype=torch.bool)
        charge_nodes_mask[0] = True
    if "station_mask" in td_state.keys():
        station_mask = td_state["station_mask"][0]
    else:
        station_mask = torch.zeros_like(charge_nodes_mask)
    if "decision_mask" in td_state.keys():
        decision_mask = td_state["decision_mask"][0]
    else:
        decision_mask = torch.zeros_like(charge_nodes_mask)
    if "charge_rate_per_node" in td_state.keys():
        charge_rate_per_node = td_state["charge_rate_per_node"][0]
    else:
        charge_rate_per_node = torch.zeros(dist_matrix.shape[0], dtype=torch.float32)
        charge_rate_per_node[0] = env.charge_rate_kwh_per_hour
    if "charge_amount_per_node" in td_state.keys():
        charge_amount_per_node = td_state["charge_amount_per_node"][0]
    else:
        charge_amount_per_node = torch.zeros_like(charge_rate_per_node)

    battery_cap = float(env.battery_capacity_kwh)
    energy_rate = float(env.energy_rate_kwh_per_distance)
    num_evs = int(env.num_evs)
    time_units_per_hour = float(env.time_units_per_hour)

    current_node = 0
    current_time = 0.0
    current_soc = battery_cap
    current_vehicle_idx = 0
    vehicle_ready_times = [0.0 for _ in range(num_evs)]
    trace: list[dict] = []

    for step_idx, node in enumerate(actions_1d.tolist(), start=1):
        node = int(node)
        if node < 0 or node >= int(dist_matrix.shape[0]):
            continue
        is_decision = bool(decision_mask[node].item())
        if is_decision:
            if not bool(charge_nodes_mask[current_node].item()):
                continue
            selected_rate = max(float(charge_rate_per_node[current_node].item()), 1e-6)
            selected_amount = max(float(charge_amount_per_node[node].item()), 0.0)
            addable = max(battery_cap - current_soc, 0.0)
            charged = min(selected_amount, addable)
            charge_time = (charged / selected_rate) * time_units_per_hour
            current_time += charge_time
            current_soc += charged
            continue

        travel_dist = float(dist_matrix[current_node, node].item())
        travel_time = float(travel_matrix[current_node, node].item())
        arrival_time = current_time + travel_time
        service_start = max(arrival_time, float(tw_starts[node].item()))
        soc_after_arrival = max(current_soc - (travel_dist * energy_rate), 0.0)

        is_charge_node = bool(charge_nodes_mask[node].item())
        is_station = bool(station_mask[node].item())
        if node == 0:
            selected_rate = max(float(charge_rate_per_node[node].item()), 1e-6)
            charge_needed = max(battery_cap - soc_after_arrival, 0.0)
            charge_time = (charge_needed / selected_rate) * time_units_per_hour
            depart_time = service_start + charge_time
            soc_after_depart = battery_cap
        else:
            depart_time = service_start + float(durations[node].item())
            soc_after_depart = soc_after_arrival

        if node == 0:
            vehicle_ready_times[current_vehicle_idx] = depart_time
            current_vehicle_idx = min(
                range(num_evs), key=lambda idx: vehicle_ready_times[idx]
            )
            current_time = vehicle_ready_times[current_vehicle_idx]
            current_soc = battery_cap
            current_node = 0
            continue

        if not is_station:
            trace.append(
                {
                    "step": step_idx,
                    "customer_id": node,
                    "vehicle_id": current_vehicle_idx + 1,
                    "arrival_time": arrival_time,
                    "depart_time": depart_time,
                    "soc_kwh": soc_after_arrival,
                }
            )
        current_time = depart_time
        current_soc = soc_after_depart
        current_node = node

    return trace


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
    """
    if vehicle_capacity <= 0:
        raise ValueError("--csv-vehicle-capacity must be > 0.")

    depot, customers = parse_vrptw_csv_rows(csv_path)
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
    capacity = torch.tensor([[float(vehicle_capacity)]], dtype=torch.float32, device=device)
    out = {
        "depot": depot_xy,
        "locs": locs,
        "demand": demand,
        "durations": durations,
        "time_windows": time_windows,
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
            depot_to_cust = CVRPTWCustomDistanceEnv._distance(
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
                    depot_to_cust = CVRPTWCustomDistanceEnv._distance(
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
            depot_to_cust = CVRPTWCustomDistanceEnv._distance(
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
                depot_to_cust = CVRPTWCustomDistanceEnv._distance(
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


def evaluate_policy_on_dataset(
    model: AttentionModel,
    env: CVRPTWEnv,
    dataset,
    batch_size: int,
) -> float:
    """Compute mean reward of a model on a given dataset with greedy decoding."""
    was_training = model.training
    model.eval()
    rewards = []
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            td = env.reset(batch)
            out = model.policy(td, env, phase="test", decode_type="greedy")
            rewards.append(out["reward"].detach().cpu())
    if was_training:
        model.train()
    return float(torch.cat(rewards, dim=0).mean())


class FixedSetEvalCallback(Callback):
    """Track quality trend by periodically evaluating on one fixed dataset.

    This reduces noise from changing random instances and gives a clearer signal
    whether route quality is improving across training epochs.
    """

    def __init__(
        self,
        env: CVRPTWEnv,
        dataset,
        batch_size: int,
        every_n_epochs: int,
    ):
        """Configure periodic fixed-set evaluation during model training."""
        super().__init__()
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.every_n_epochs = max(1, every_n_epochs)
        self.history: list[tuple[int, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Run fixed-set evaluation at configured epochs and record reward history."""
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        should_eval = (
            epoch % self.every_n_epochs == 0 or epoch == int(trainer.max_epochs)
        )
        if not should_eval:
            return
        reward = evaluate_policy_on_dataset(
            pl_module, self.env, self.dataset, self.batch_size
        )
        self.history.append((epoch, reward))
        pl_module.log("fixed_eval/reward", reward, on_epoch=True, prog_bar=True)
        print(f"[fixed-eval] epoch={epoch} reward={reward:.6f}")


def print_quality_table(
    initial_reward: float,
    fixed_history: list[tuple[int, float]],
    best_ckpt_reward: float | None,
    final_model_reward: float,
    best_test_reward: float | None,
) -> None:
    """Print a compact reward trend table to verify quality improvement."""
    print("\nQuality trend on fixed evaluation set (higher is better):")
    print("stage                epoch   reward")
    print(f"initial              0       {initial_reward:.6f}")
    for epoch, reward in fixed_history:
        print(f"periodic             {epoch:<7d} {reward:.6f}")
    if best_ckpt_reward is not None:
        print(f"best_checkpoint      -       {best_ckpt_reward:.6f}")
    print(f"final_model          -       {final_model_reward:.6f}")
    if best_test_reward is not None:
        print(f"best_checkpoint_test -       {best_test_reward:.6f}")


def print_one_solution(model: AttentionModel, env: CVRPTWEnv) -> None:
    """Decode and print one full test solution (flat actions + per-vehicle routes)."""
    dataset = env.dataset(1, phase="test")
    instance = dataset.collate_fn([dataset[0]])
    print_one_solution_from_instance(model, env, instance, title="One test-instance solution")


def print_one_solution_from_instance(
    model: AttentionModel,
    env: CVRPTWEnv,
    instance,
    title: str,
) -> float:
    """Decode and print route details for one given instance."""
    instance = instance.to(model.device)

    with torch.inference_mode():
        td = env.reset(instance)
        out = model.policy(td, env, phase="test", decode_type="greedy")

    actions = out["actions"][0].detach().cpu()
    reward = float(out["reward"][0].detach().cpu())
    station_mask_1d = td["station_mask"][0].detach().cpu() if "station_mask" in td.keys() else None
    decision_mask_1d = td["decision_mask"][0].detach().cpu() if "decision_mask" in td.keys() else None
    charge_amount_per_node_1d = (
        td["charge_amount_per_node"][0].detach().cpu()
        if "charge_amount_per_node" in td.keys()
        else None
    )
    routes = split_routes_from_actions(
        actions, decision_mask_1d, include_decision_actions=True
    )
    cp_id_per_node_1d = (
        td["cp_id_per_node"][0].detach().cpu() if "cp_id_per_node" in td.keys() else None
    )
    action_labels = [
        format_node_label(
            int(node),
            station_mask_1d,
            cp_id_per_node_1d,
            decision_mask_1d,
            charge_amount_per_node_1d,
        )
        for node in actions.tolist()
    ]

    print(f"\n{title} (greedy decode):")
    print(f"Reward: {reward:.6f}")
    print(f"Flat action sequence: {actions.tolist()}")
    print(f"Flat action labels:   {action_labels}")
    if not routes:
        print("Vehicle routes: []")
    else:
        print("Vehicle routes:")
        for i, route in enumerate(routes, start=1):
            labels = [
                format_node_label(
                    node,
                    station_mask_1d,
                    cp_id_per_node_1d,
                    decision_mask_1d,
                    charge_amount_per_node_1d,
                )
                for node in route
            ]
            print(f"  Vehicle {i}: {route}")
            print(f"             {labels}")
    visit_trace = build_customer_visit_trace(env, td, actions)
    if visit_trace:
        print("Customer visit trace (time units are the same as TW/travel-time inputs):")
        for rec in visit_trace:
            print(
                f"  step={rec['step']:>3d} customer={rec['customer_id']:>3d} "
                f"vehicle={rec['vehicle_id']:>2d} "
                f"arrival={rec['arrival_time']:.2f} "
                f"depart={rec['depart_time']:.2f} "
                f"soc={rec['soc_kwh']:.2f}kWh"
            )
    else:
        print("Customer visit trace: []")
    return reward


def main() -> None:
    """Train, evaluate, and optionally decode solutions for the configured CVRPTW run."""
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    charge_levels_kwh = parse_charge_levels_kwh(args.charge_levels_kwh)

    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator

    if args.train_pool_csv:
        pool_generator = CSVCustomerPoolGenerator(
            csv_path=args.train_pool_csv,
            sample_size=args.pool_sample_size,
            vehicle_capacity=args.pool_vehicle_capacity,
            max_time=args.max_time,
            distance_matrix_csv=args.distance_matrix_csv,
            time_matrix_csv=args.time_matrix_csv,
            distance_mode_for_depot=args.distance_mode,
        )
        env = CVRPTWCustomDistanceEnv(
            distance_mode=args.distance_mode,
            battery_capacity_kwh=args.ev_battery_capacity_kwh,
            energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
            charge_rate_kwh_per_hour=args.ev_charge_rate_kwh_per_hour,
            reserve_soc_kwh=args.ev_reserve_soc_kwh,
            num_evs=args.ev_num_vehicles,
            charging_pool_csv=args.charging_pool_csv,
            charging_pool_sample_size=args.charging_pool_sample_size,
            charge_levels_kwh=charge_levels_kwh,
            check_solution=False,
            generator=pool_generator,
        )
    else:
        env = CVRPTWCustomDistanceEnv(
            distance_mode=args.distance_mode,
            battery_capacity_kwh=args.ev_battery_capacity_kwh,
            energy_rate_kwh_per_distance=args.ev_energy_rate_kwh_per_distance,
            charge_rate_kwh_per_hour=args.ev_charge_rate_kwh_per_hour,
            reserve_soc_kwh=args.ev_reserve_soc_kwh,
            num_evs=args.ev_num_vehicles,
            charging_pool_csv=args.charging_pool_csv,
            charging_pool_sample_size=args.charging_pool_sample_size,
            charge_levels_kwh=charge_levels_kwh,
            check_solution=False,
            generator_params={
                "num_loc": args.num_loc,
                "max_time": args.max_time,
                "scale": False,
            },
        )

    model = AttentionModel(
        env=env,
        baseline=args.baseline,
        batch_size=args.batch_size,
        val_batch_size=args.eval_batch_size,
        test_batch_size=args.eval_batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.test_data_size,
        optimizer_kwargs={"lr": args.lr},
    )

    fixed_eval_dataset = env.dataset(args.fixed_eval_size, phase="test")
    initial_fixed_reward = evaluate_policy_on_dataset(
        model, env, fixed_eval_dataset, args.eval_batch_size
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="vrptw-{epoch:03d}",
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    fixed_eval_callback = FixedSetEvalCallback(
        env=env,
        dataset=fixed_eval_dataset,
        batch_size=args.eval_batch_size,
        every_n_epochs=args.fixed_eval_every,
    )

    trainer = RL4COTrainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision=32,
        logger=False,
        callbacks=[checkpoint_callback, fixed_eval_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
    )

    trainer.fit(model)
    best_ckpt_path = checkpoint_callback.best_model_path
    test_results = trainer.test(model, verbose=False)
    metrics = test_results[0]
    test_reward = extract_reward(metrics)
    final_fixed_reward = evaluate_policy_on_dataset(
        model, env, fixed_eval_dataset, args.eval_batch_size
    )

    best_ckpt_fixed_reward = None
    best_test_reward = None
    model_for_solution = model
    if best_ckpt_path:
        try:
            add_safe_globals([CVRPTWEnv, CVRPTWCustomDistanceEnv])
            best_model = AttentionModel.load_from_checkpoint(best_ckpt_path, env=env)
            best_ckpt_fixed_reward = evaluate_policy_on_dataset(
                best_model, env, fixed_eval_dataset, args.eval_batch_size
            )
            best_test_results = trainer.test(best_model, verbose=False)
            best_test_reward = extract_reward(best_test_results[0])
            model_for_solution = best_model
        except Exception as exc:
            print(f"Warning: could not evaluate best checkpoint ({exc}).")

    print("Finished training and testing.")
    print(f"Accelerator: {accelerator}")
    print(f"Distance mode: {args.distance_mode}")
    print(
        "EV params: "
        f"battery={args.ev_battery_capacity_kwh}kWh, "
        f"energy_rate={args.ev_energy_rate_kwh_per_distance}kWh/dist, "
        f"charge_rate={args.ev_charge_rate_kwh_per_hour}kWh/h, "
        f"reserve={args.ev_reserve_soc_kwh}kWh, "
        f"fleet_size={args.ev_num_vehicles}"
    )
    print(
        "Charging pool: "
        f"csv={args.charging_pool_csv}, "
        f"sample_size={args.charging_pool_sample_size}"
    )
    print(f"Discrete charge levels (kWh): {charge_levels_kwh}")
    if args.train_pool_csv:
        print(
            "Training pool mode: "
            f"csv={args.train_pool_csv}, sample_size={args.pool_sample_size}"
        )
    print(f"Best checkpoint: {best_ckpt_path if best_ckpt_path else 'not found'}")
    print(f"Test reward: {test_reward:.6f}")
    print(f"All test metrics: {metrics}")
    print_quality_table(
        initial_reward=initial_fixed_reward,
        fixed_history=fixed_eval_callback.history,
        best_ckpt_reward=best_ckpt_fixed_reward,
        final_model_reward=final_fixed_reward,
        best_test_reward=best_test_reward,
    )
    if args.test_csv:
        custom_instance = load_vrptw_instance_from_csv(
            args.test_csv,
            vehicle_capacity=args.csv_vehicle_capacity,
            distance_mode=args.distance_mode,
            distance_matrix_csv=args.test_distance_matrix_csv,
            time_matrix_csv=args.test_time_matrix_csv,
            device=model.device,
        )
        custom_reward = print_one_solution_from_instance(
            model_for_solution,
            env,
            custom_instance,
            title=f"CSV test-instance solution ({args.test_csv})",
        )
        if args.test_distance_matrix_csv:
            print(
                "CSV test distance source: "
                f"matrix ({args.test_distance_matrix_csv})"
            )
        else:
            print(f"CSV test distance source: mode ({args.distance_mode})")
        if args.test_time_matrix_csv:
            print(
                "CSV test travel-time source: "
                f"matrix ({args.test_time_matrix_csv})"
            )
        else:
            print("CSV test travel-time source: distance source")
        print(f"CSV instance reward: {custom_reward:.6f}")
    if args.print_solution:
        print_one_solution(model_for_solution, env)


if __name__ == "__main__":
    main()

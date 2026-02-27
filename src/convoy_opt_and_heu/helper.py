"""Preprocessing helpers for building Opt+Heu instances from CSV inputs."""

import math
import time 
import random
import pandas as pd
import os
try:
    from .delivery import Delivery
    from .cp import CP
    from .ev import EV
except ImportError:
    from delivery import Delivery
    from cp import CP
    from ev import EV

class Point:
    """Simple 2D point utility used for geometric helpers."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        

        
        
def _write_test_instance_csv(data_dir, node_info_by_global_id, deliveries, cp):
    columns = [
        "ID",
        "type",
        "lng",
        "lat",
        "first_receive_tm",
        "last_receive_tm",
        "service_time",
        "reward",
        "unit_charging_cost",
        "charge_rate_kwh_per_hour",
    ]
    rows = []

    # Depot first
    depot_global_id = int(cp[0].global_id)
    rows.append(node_info_by_global_id[depot_global_id])

    # Then selected deliveries
    for d in deliveries:
        rows.append(node_info_by_global_id[int(d.global_id)])

    # Then selected charging points (excluding depot)
    for c in cp[1:]:
        rows.append(node_info_by_global_id[int(c.global_id)])

    output_path = os.path.join(data_dir, "test_instance.csv")
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def _resolve_data_dir():
    # Priority: explicit override -> ~/CONVOY/data -> legacy layout paths -> local sibling data dir
    data_dir_override = os.environ.get("CONVOY_DATA_DIR")
    candidates = []
    if data_dir_override:
        candidates.append(data_dir_override)
    candidates.extend(
        [
            os.path.join(os.path.expanduser("~"), "CONVOY", "data"),
            os.path.join(os.path.expanduser("~"), "CONVOY", "optimal_and_heuristic", "data"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
        ]
    )
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find data directory. Checked: {}".format(", ".join(candidates))
    )


def preProcess(
    nD,
    nC,
    nS,
    nE,
    combined_details_csv,
    combined_dist_matrix_csv,
    combined_time_matrix_csv,
    ev_energy_rate_kwh_per_distance=0.00025,
    alpha1=1,
    alpha2=1,
    random_seed=None,
    use_full_instance=False,
):
    # Use an isolated RNG to avoid cross-run interference from other modules.
    rng = random.Random(random_seed) if random_seed is not None else random

    # Read the combined details CSV
    csv_path = os.path.abspath(combined_details_csv)
    output_data_dir = os.path.dirname(csv_path)
    data = pd.read_csv(csv_path)
    if "charge_rate_kwh_per_hour" not in data.columns:
        raise ValueError(
            "Missing required column 'charge_rate_kwh_per_hour' in {}".format(csv_path)
        )


    # Constants
    nW = 1  # Number of Warehouse depots
    if ev_energy_rate_kwh_per_distance <= 0:
        raise ValueError("ev_energy_rate_kwh_per_distance must be > 0.")
    # Convert from kWh/m to km/kWh.
    mj = 1.0 / (float(ev_energy_rate_kwh_per_distance) * 1000.0)
    beta_f = 30 # EV full capacity 30 kWh
    M = 99999 # very large M for Big M constraint



    # objects of Delivery and CP classes
    deliveries = []
    cp = []
    EVs = []
    theta = [] # unit cost at CPs including depot
    cp_charge_rate = [] # charging rate at CPs including depot (kW)
    reward = []
    cp_node_types = []
    node_info_by_global_id = {}

    # Maps to track local ID to global ID and global ID to local ID
    delivery_map_local_to_global = {}
    delivery_map_global_to_local = {}
    cp_map_local_to_global = {}
    cp_map_global_to_local = {}

    # Iterate over the DataFrame and create objects based on the type column
    for index, row in data.iterrows():
        lat = row['lat']
        long = row['lng']
        global_id = int(row['ID'])  # global ID from input
        first_receive_tm = row['first_receive_tm']
        last_receive_tm = row['last_receive_tm']
        service_time = float(row["service_time"]) if not pd.isna(row["service_time"]) else 0.0
        reward_ = row['reward']
        unit_charging_cost = row['unit_charging_cost']
        charge_rate_kwh_per_hour = row['charge_rate_kwh_per_hour']
        node_info_by_global_id[global_id] = {
            "ID": global_id,
            "type": row["type"],
            "lng": row["lng"],
            "lat": row["lat"],
            "first_receive_tm": first_receive_tm,
            "last_receive_tm": last_receive_tm,
            "service_time": row["service_time"],
            "reward": reward_,
            "unit_charging_cost": unit_charging_cost,
            "charge_rate_kwh_per_hour": charge_rate_kwh_per_hour,
        }

        # Check the type column and create the appropriate object
        if row['type'] == 'c':
            local_id = len(deliveries)  # Local ID is the index in the deliveries array
            deliveries.append(
                Delivery(
                    lat,
                    long,
                    local_id,
                    global_id,
                    first_receive_tm,
                    last_receive_tm,
                    reward_,
                    service_time=service_time,
                )
            )
            delivery_map_local_to_global[local_id] = global_id
            delivery_map_global_to_local[global_id] = local_id
            reward.append(reward_)
        elif row['type'] == 'd' or row['type'] == 'f':
            local_id = len(cp)  # Local ID is the index in the cp array
            cp.append(CP(lat, long, local_id, global_id))
            cp_node_types.append(row["type"])
            cp_map_local_to_global[local_id] = global_id
            cp_map_global_to_local[global_id] = local_id
            theta.append(unit_charging_cost)
            if pd.isna(charge_rate_kwh_per_hour):
                raise ValueError(
                    "Missing charge_rate_kwh_per_hour for charging node ID {}".format(global_id)
                )
            cp_charge_rate.append(float(charge_rate_kwh_per_hour))
            
          
    dist = {}
    time1 = {}
    dist_file_path = combined_dist_matrix_csv
    df = pd.read_csv(dist_file_path, index_col=0)

    # Convert to dictionary with (x, y) as key and distance as value
    dist = {(int(row), int(col)): float(df.loc[row, col])/1000
                     for row in df.index
                     for col in df.columns} # Store distance in Km

    duration_file_path = combined_time_matrix_csv
    df_dur = pd.read_csv(duration_file_path, index_col=0)
    # Convert to duration with (x, y) as key and distance as value
    time1 = {(int(row), int(col)): df_dur.loc[row, col]
                     for row in df_dur.index
                     for col in df_dur.columns} # Store travel duration in min



    # Select deliveries and charging points as per user input.
    if use_full_instance:
        nD = len(deliveries)
    if nD > len(deliveries):
        raise ValueError(
            "Requested nD={} deliveries, but only {} customers are available.".format(
                nD, len(deliveries)
            )
        )
    if use_full_instance:
        selected_delivery_idx = list(range(len(deliveries)))
    else:
        selected_delivery_idx = rng.sample(range(len(deliveries)), k=nD)
    deliveries = [deliveries[i] for i in selected_delivery_idx]
    reward = [reward[i] for i in selected_delivery_idx]

    depot_indices = [i for i, t in enumerate(cp_node_types) if t == "d"]
    if len(depot_indices) != 1:
        raise ValueError(
            "Expected exactly one depot row (type=d), found {}.".format(
                len(depot_indices)
            )
        )
    depot_idx = depot_indices[0]
    station_indices = [i for i, t in enumerate(cp_node_types) if t == "f"]
    if use_full_instance:
        nC = len(station_indices)
    if nC > len(station_indices):
        raise ValueError(
            "Requested nC={} charging stations, but only {} are available.".format(
                nC, len(station_indices)
            )
        )
    if use_full_instance:
        selected_station_idx = list(station_indices)
    else:
        selected_station_idx = rng.sample(station_indices, k=nC)
    selected_cp_idx = [depot_idx] + selected_station_idx
    cp = [cp[i] for i in selected_cp_idx]
    theta = [theta[i] for i in selected_cp_idx]
    cp_charge_rate = [cp_charge_rate[i] for i in selected_cp_idx]

    # Reindex selected nodes so downstream sets/maps (D/C/C1) remain contiguous.
    delivery_map_local_to_global = {}
    delivery_map_global_to_local = {}
    for new_local_id, d_obj in enumerate(deliveries):
        d_obj.local_id = new_local_id
        delivery_map_local_to_global[new_local_id] = int(d_obj.global_id)
        delivery_map_global_to_local[int(d_obj.global_id)] = new_local_id

    cp_map_local_to_global = {}
    cp_map_global_to_local = {}
    for new_local_id, cp_obj in enumerate(cp):
        cp_obj.local_id = new_local_id
        cp_map_local_to_global[new_local_id] = int(cp_obj.global_id)
        cp_map_global_to_local[int(cp_obj.global_id)] = new_local_id

    # Export selected depot/customers/CPs for downstream testing.
    # Overwrites if file already exists.
    _write_test_instance_csv(output_data_dir, node_info_by_global_id, deliveries, cp)

    # List of entities (after selection/reindex).
    W = {0}  # Warehouse
    D = set(range(0, len(deliveries)))
    C = set(range(1, len(cp)))  # CP starts from 1 (0 is depot)
    C1 = W.union(C)
    E = set(range(0, nE))
    S = set(range(0, nS))


    # create EV object
    for j in E:
        EVs.append(EV(j, cp[0], beta_f)) # cp[0] is the depot



    # Initialize window time for every delivery point
    # random.seed(seed_val)
    tau_start = {}
    tau_end = {}
    for y in D:
        tau_start[y] = deliveries[y].tau_start
        tau_end[y] = deliveries[y].tau_end

    delivery_end_time = max(tau_end.values())



    # # Per kWh energy cost at a charging station
    # theta = [round(random.uniform(0.4, 0.6),2) for _ in C1]
    # theta[0] = 0.36



    psi_DD = {} # Energy requirement to travel between two delivery points
    psi_DC = {} # Energy requirement to travel between a delivery point and a charging point
    gamma_DD = {} # Time to reach from one delivery point to another
    gamma_DC = {} # Time to reach from one delivery point to charging point or opposite
    psi_C0 = {}  # Energy requirement to travel from CP to depot
    gamma_C0 = {}  # Time requirement to travel from CP to depot



    # Store energy and time requirement for traveling each edge
    for d1 in D:
        for d2 in D:
            d1_global = delivery_map_local_to_global[d1]
            d2_global = delivery_map_local_to_global[d2]
            dist12 = dist[(d1_global, d2_global)] # in Km
            energy12 = round(dist12 /mj, 2) # in kWh
            psi_DD[(d1,d2)] = energy12
            time12 = time1[(d1_global, d2_global)]
            gamma_DD[(d1,d2)] = time12
        for c in C1:
            d1_global = delivery_map_local_to_global[d1]
            c_global = cp_map_local_to_global[c]
            dist1c = dist[(d1_global, c_global)]
            energy1c = round(dist1c /mj, 2) # in kWh
            psi_DC[(d1,c)] = energy1c
            # time12 = round(dist1c /vj, 2) # in min
            time12 = time1[(d1_global, c_global)]
            gamma_DC[(d1,c)] = time12

    depot_global = cp_map_local_to_global[0]
    for c in C:
        c_global = cp_map_local_to_global[c]
        dist_c0 = dist[(c_global, depot_global)]
        psi_C0[c] = round(dist_c0 / mj, 2)
        gamma_C0[c] = time1[(c_global, depot_global)]
            
    rateE = {} # charge acceptance rate at EV : equivalent to power in kW
    for j in E:
        rateE[j] = rng.randint(300, 350) # in kW #Kept high to ignore it
        
    rateC = {} # charging rate at CP = charging power in kW
    for y in C:
        rateC[y] = cp_charge_rate[y] # in kW (from CSV)


    # Tunable parameters
    alpha1 = alpha1 # parameter for tuning number of successful deliveries
    alpha2 = alpha2 # parameter for tuning the energy cost
    
    return cp, deliveries, theta, reward, C, D, E, C1, S, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC, alpha1, alpha2, gamma_C0, psi_C0

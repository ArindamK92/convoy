import math
import time 
import random
import pandas as pd
from delivery import Delivery
from cp import CP
from ev import EV
import os


seed_val = 123
random.seed(seed_val)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        

        
        
def preProcess(base_file, nD, nC, nS, nE, alpha1=30, alpha2=1):
    # Read the CSV file
    # csv_file = 'combined_data.csv'  # Replace with your CSV file path
    csv_path = os.path.join("..", "data", "combined_data_{}.csv".format(base_file))  # Relative path
    data = pd.read_csv(csv_path)


    # Constants
    nW = 1  # Number of Warehouse depots
    mj = 4 # mileage in Km per kWh
    beta_f = 30 # EV full capacity 30 kWh
    M = 99999 # very large M for Big M constraint



    # objects of Delivery and CP classes
    deliveries = []
    cp = []
    EVs = []
    theta = [] # unit cost at CPs including depot
    reward = []

    # Maps to track local ID to global ID and global ID to local ID
    delivery_map_local_to_global = {}
    delivery_map_global_to_local = {}
    cp_map_local_to_global = {}
    cp_map_global_to_local = {}

    # Iterate over the DataFrame and create objects based on the type column
    for index, row in data.iterrows():
        lat = row['lat']
        long = row['lng']
        global_id = row['ID']  # Assuming 'index' column is the global ID
        first_receive_tm = row['first_receive_tm']
        last_receive_tm = row['last_receive_tm']
        reward_ = row['reward']
        unit_charging_cost = row['unit_charging_cost']

        # Check the type column and create the appropriate object
        if row['type'] == 'c':
            local_id = len(deliveries)  # Local ID is the index in the deliveries array
            deliveries.append(Delivery(lat, long, local_id, global_id, first_receive_tm, last_receive_tm, reward_))
            delivery_map_local_to_global[local_id] = global_id
            delivery_map_global_to_local[global_id] = local_id
            reward.append(reward_)
        elif row['type'] == 'd' or row['type'] == 'f':
            local_id = len(cp)  # Local ID is the index in the cp array
            cp.append(CP(lat, long, local_id, global_id))
            cp_map_local_to_global[local_id] = global_id
            cp_map_global_to_local[global_id] = local_id
            theta.append(unit_charging_cost)
            
          
    dist = {}
    time1 = {}
    dist_file_path = os.path.join("..", "data", "distance_matrix_{}.csv".format(base_file))  # Relative path
    df = pd.read_csv(dist_file_path, index_col=0)

    # Convert to dictionary with (x, y) as key and distance as value
    dist = {(int(row), int(col)): float(df.loc[row, col])/1000
                     for row in df.index
                     for col in df.columns} # Store distance in Km

    duration_file_path = os.path.join("..", "data", "time_matrix_{}.csv".format(base_file))  # Relative path
    df_dur = pd.read_csv(duration_file_path, index_col=0)
    # Convert to duration with (x, y) as key and distance as value
    time1 = {(int(row), int(col)): df_dur.loc[row, col]
                     for row in df_dur.index
                     for col in df_dur.columns} # Store travel duration in min



    # List of entities
    W = {0} # Warehouse
    D = set(range(0, nD)) 
    C = set(range(1, nC + 1)) # CP starts from 1 as W is also considered as a CP and with id 0
    C1 = W.union(C)
    E = set(range(0, nE))
    S = set(range(0, nS))

    # We trim and use the list as per user input
    deliveries = deliveries[:nD]
    cp = cp[:(nW + nC)]
    theta = theta[:(nW + nC)]
    reward = reward[:nD]


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
            
    rateE = {} # charge acceptance rate at EV : equivalent to power in kW
    for j in E:
        rateE[j] = random.randint(90, 350) # in kW
        
    rateC = {} # charging rate at CP = charging power in kW
    for y in C:
        rateC[y] = random.randint(50, 150) # in kW


    # Tunable parameters
    alpha1 = alpha1 # parameter for tuning number of successful deliveries
    alpha2 = alpha2 # parameter for tuning the energy cost
    
    return cp, deliveries, theta, reward, C, D, E, C1, S, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC, alpha1, alpha2
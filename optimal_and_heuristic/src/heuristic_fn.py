# import pandas as pd
# import math
# import os
import time 
# import random
from st_dbscan import ST_DBSCAN
from collections import defaultdict
from utility import create_kd_tree, find_nearest_static_fog
# from delivery import Delivery
# from cp import CP
# from ev import EV

DEBUG = False
def printDebug(string, val):
    if DEBUG == True:
        print(string, val)

def heuristic(cp, deliveries, theta, C, D, E, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC):
    start_time = time.time() # Algorithm start time
    
    # Create the KD-Tree
    lats = [c.latitude for c in cp[1:]] + [d.latitude for d in deliveries] # first cp is the depot
    lons = [c.longitude for c in cp[1:]] + [d.longitude for d in deliveries]
    min_lat = min(lats)
    min_lon = min(lons)
    min_theta = min(theta[1:])
    max_lat = max(lats)
    max_lon = max(lons)
    max_theta = max(theta[1:])
    denom_lat = max_lat - min_lat
    denom_lon = max_lon - min_lon
    denom_theta = max_theta - min_theta
    kd_tree = create_kd_tree(cp, C, theta, min_lat, max_lat, min_lon, max_lon, min_theta, max_theta, denom_lat, denom_lon, denom_theta)

    # Find the nearest fog for each delivery and 
    # prepare a list of spatio-temporal parameters of the deliveries for ST-DBSCAN 
    deliveries_spatiotemporal_params = []
    min_deadline = min([tau_end[y] for y in D])
    max_deadline = max([tau_end[y] for y in D])
    denom_deadline = max_deadline - min_deadline

    for d in deliveries:
        nearest_fog_index = find_nearest_static_fog(kd_tree, d.latitude, d.longitude, min_lat, max_lat, min_lon, max_lon, min_theta, max_theta, denom_lat, denom_lon, denom_theta)
        d.nearest_CP_local_id = nearest_fog_index
        local_id_d = d.local_id
        # create a list of waypoints with the deadlines (spatio-temporal parameters)
        param = [(tau_end[local_id_d]-min_deadline)/denom_deadline, (d.latitude-min_lat)/denom_lat, (d.longitude-min_lon)/denom_lon]
        deliveries_spatiotemporal_params.append(param)
        

    # Cluster the deliveries depending on their spatio-temporal parameters
    # Hence eps1, that is the spatial density threshold (maximum spatial distance) between 
    # 
    # eps2 is the temporal threshold (maximum temporal distance) between two 
    # points to be considered related.     
    ## Avg Euclidean dist among the fog lat lon values: 0.0037424493333129533

    # tune eps1 and eps2 depending on the values of  deliveries_spatiotemporal_params
    st_dbscan = ST_DBSCAN(eps1 = 0.2, eps2 = 0.2, min_samples = 1) 
    st_dbscan.fit(deliveries_spatiotemporal_params) 
    labels = st_dbscan.labels
    clusters = defaultdict(list)
    clusters_min_deadline = {}
    for i in D:
        delivery = deliveries[i]
        label = labels[i]
        clusters[label].append(delivery)

    # Check the output and tune parameters eps1, eps2, and min_samples
    for cluster_label, incluster_deliveries in clusters.items():
        clusters_min_deadline[cluster_label] = min([tau_end[d.local_id] for d in incluster_deliveries])
        # print(f"Cluster {cluster_label}:")
        # for delivery in incluster_deliveries:
        #     print(f"  {delivery.local_id}")
            
    # Sort the cluster ids depending on their min deadlines
    sorted_cluster_ids = sorted(clusters_min_deadline, key=lambda k: clusters_min_deadline[k])


    # Assign deliveries to the EVs
    max_subtrip_per_EV = nS
    for c_id in sorted_cluster_ids:
        # sort the deliveries in each cluster in the descending order of their deadline
        sorted_deliveries = sorted(clusters[c_id], key=lambda d: tau_end[d.local_id])
        
        for delivery in sorted_deliveries:
            d_id = delivery.local_id
            #print("cluster:", c_id, " delivery: ", d_id, "delivery start time: ", tau_start[d_id])
            
            energy_cost_min = 9999999 # cost is measured in terms of energy used
            assigned_EV = None
            time_req_for_delivery = -1
            for e in EVs:
                if e.totalSubtrip >= max_subtrip_per_EV:
                    printDebug("Max delivery reached. EV: ", e.local_id)
                    continue
                last_loc = e.A[-1]
                duration_lastLoc2delivery = 0
                energy_req = 0 # energy required to reach the current delivery point from the last point
                energy_req_d2W = 0 # energy required to reach warehouse from the current delivery point
                if last_loc.type == "delivery":
                    duration_lastLoc2delivery = gamma_DD[(last_loc.local_id, d_id)]
                    energy_req = psi_DD[(last_loc.local_id, d_id)]
                    
                else:
                    duration_lastLoc2delivery = gamma_DC[(d_id, last_loc.local_id)]
                    energy_req = psi_DC[(d_id, last_loc.local_id)]
                
                energy_req_d2W = psi_DC[(d_id, 0)]
                nearest_cp = cp[delivery.nearest_CP_local_id]
                
                if energy_req < energy_cost_min:
                    # Check the battery constraints 
                    if energy_req + energy_req_d2W > e.B_res:
                        if last_loc.type == "delivery": # if already at a CP then charge is already full
                            # Send it to the nearest CP for charging
                            nearest_cp_id_from_last_point = last_loc.nearest_CP_local_id
                            e_id = e.local_id
                            e.A.append(cp[nearest_cp_id_from_last_point])
                            req_charge = beta_f - e.B_res - psi_DC[(last_loc.local_id, nearest_cp_id_from_last_point)]
                            e.T.append(e.t_ed)
                            e.t_ed = gamma_DC[(last_loc.local_id, nearest_cp_id_from_last_point)] +  req_charge/min(rateE[e_id], rateC[nearest_cp_id_from_last_point]) # reaching time + charging time
                            e.B_res = beta_f # reset battery
                            e.B_trace.append(beta_f)
                        continue
                    
                    reaching_time = e.t_ed + duration_lastLoc2delivery
                    if reaching_time <= tau_end[d_id]:
                        assigned_EV = e
                        energy_cost_min = energy_req
                        time_req_for_delivery = duration_lastLoc2delivery
            if assigned_EV != None:
                delivery.assigned_EV = assigned_EV
                assigned_EV.assigneddeliveries.append(delivery.local_id)
                # last_point = assigned_EV.A[-1]
                # time_flying = getFlyingTime(last_point, delivery, p.horizontalSpeed) # time to fly from last loc to delivery waypoint
                t_ed = assigned_EV.t_ed # earliest departure time from the last location
                #t_dep = max((delivery.latest_sensing_start_timeinstance - time_flying), t_ed) # actual departure time from the last location
                t_dep = t_ed # As there is no sensing start time
                assigned_EV.A.append(delivery)
                assigned_EV.T.append(t_dep)
                #assigned_EV.t_ed = max((t_dep +  time_req_for_delivery),  tau_end[d_id]) # old version
                assigned_EV.t_ed = min(max((t_dep +  time_req_for_delivery), tau_start[d_id]), tau_end[d_id])# earliest departure time at the delivery waypoint ## AK: New change
                #assigned_EV.t_ed = (t_dep +  time_req_for_delivery) # if only deadline exists. no slot start time
                # assigned_EV.totalEnergyUsed = assigned_EV.totalEnergyUsed + energy_req_for_delivery
                B_res = assigned_EV.B_res - energy_cost_min
                assigned_EV.B_res = B_res
                assigned_EV.B_trace.append(B_res)
                
                
                printDebug("energy_cost_min: ", energy_cost_min)


    ####            
    unassigned_deliveries = [t for t in deliveries if t.assigned_EV==None]



    for e in EVs:
        last_loc = e.A[-1]
        if last_loc.type == "delivery":
            e.A.append(cp[0]) # return to the warehouse
        if last_loc.type == "cp":
            if last_loc.local_id != 0:
                A1 = e.A[:-1] # remove the last CP
                A1.append(cp[0]) # Add the depot as last charging point
                e.A = A1

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000


    # result variables    
    total_deliveries_completed = 0
    total_cost = 0
    total_energy = 0
    delivery_distribution = {}
    for e in EVs:
        A = e.A
        # total_cost_e = 0
        energy_per_trip = 0
        delivery_done = 0
        last_loc = A[0]
        # print("Trace::")
        # print(last_loc.local_id)
        for loc in A[1:]:
            energy_req = 0
            if loc.type == "delivery":
                delivery_done += 1
                if last_loc.type == "delivery":
                    energy_req = psi_DD[(last_loc.local_id, loc.local_id)]
                else:
                    energy_req = psi_DC[(loc.local_id, last_loc.local_id)]
                energy_per_trip += energy_req
                last_loc = loc
                # print("->",loc.local_id, " energy: ", energy_req)
            else:
                if last_loc.type == "delivery":
                    energy_req = psi_DC[(last_loc.local_id, loc.local_id)]
                else:
                    print("ERROR: coming to a cp from a cp")
                # print("->",loc.local_id, " energy: ", energy_req)
                energy_per_trip += energy_req
                total_energy += energy_per_trip
                total_cost += energy_per_trip * theta[loc.local_id]
                last_loc = loc
                energy_per_trip = 0
        total_deliveries_completed += delivery_done
        delivery_distribution[e.local_id] = delivery_done

    print("****Heuristic****")          
    # print("Total delivery: ", nD, " Total CP: ", nC+1, " Total EV: ", nE, " time_slot_max:", time_slot_max)          
    print(f"Elapsed time: {elapsed_time:.2f} ms")
    print(f"Total cost: {total_cost}")
    print(f"Total successful delivery: {total_deliveries_completed}")
    print("::Delivery Distribution::")
    for j in E:
        print(f"delivery_distribution[{j}]:", delivery_distribution.get(j, 0))
    
    return elapsed_time, total_cost, total_deliveries_completed
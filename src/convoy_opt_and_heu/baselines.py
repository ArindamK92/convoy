"""Baseline dispatch policies used by the Optimal+Heuristic module."""

import time 
# import random


def NDF(cp, deliveries, theta, C, D, E, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC):
    start_time = time.time() # Algorithm start time
            
            
    # Find the nearest CP. The CP which requires the least energy to reach is the nearest
    for d in deliveries:
        c_id = min((key for key in psi_DC if key[0] == d.local_id and key[1] > 0 ),  # Filter keys with the fixed x
                    key=lambda k: psi_DC[k]  # Use the dictionary value for comparison
                    )[1]  # Get the y value from the key with the minimum value
        nearest_fog_index = c_id
        d.nearest_CP_local_id = nearest_fog_index
    
    
    
    
    sorted_deliveries = sorted(deliveries, key=lambda x: psi_DC[(x.local_id, 0)])
    for delivery in sorted_deliveries:
        d_id = delivery.local_id
        service_time_d = float(getattr(delivery, "service_time", 0.0))
        # print("cluster:", c_id, " delivery: ", d_id, "delivery start time: ", tau_start[d_id])
        
        energy_cost_min = 9999999 # cost is measured in terms of energy used
        assigned_EV = None
        departure_after_service = -1
        for e in EVs:
            # if e.totalSubtrip >= max_subtrip_per_EV:
            #     print("Max delivery reached. EV: ", e.local_id)
            #     printDebug("Max delivery reached. EV: ", e.local_id)
            #     continue
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
            
            # if energy_req < energy_cost_min:
                # Check the battery constraints 
            if energy_req + energy_req_d2W > e.B_res:
                if last_loc.type == "delivery": # if already at a CP then charge is already full
                    # Send it to the nearest CP for charging
                    nearest_cp_id_from_last_point = last_loc.nearest_CP_local_id
                    e_id = e.local_id
                    e.A.append(cp[nearest_cp_id_from_last_point])
                    energy_to_cp = psi_DC[(last_loc.local_id, nearest_cp_id_from_last_point)]
                    soc_at_cp = e.B_res - energy_to_cp
                    req_charge = max(beta_f - soc_at_cp, 0.0)
                    e.T.append(e.t_ed)
                    e.t_ed = (
                        e.t_ed
                        + gamma_DC[(last_loc.local_id, nearest_cp_id_from_last_point)]
                        + req_charge / min(rateE[e_id], rateC[nearest_cp_id_from_last_point])
                    )  # accumulated time + reaching time + charging time
                    e.B_res = beta_f # reset battery
                    e.B_trace.append(beta_f)
                continue
            
            reaching_time = e.t_ed + duration_lastLoc2delivery
            service_start = max(reaching_time, tau_start[d_id])
            if (service_start + service_time_d) <= tau_end[d_id]:
                assigned_EV = e
                energy_cost_min = energy_req
                departure_after_service = service_start + service_time_d
                break
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
            assigned_EV.t_ed = departure_after_service
            
            # assigned_EV.totalEnergyUsed = assigned_EV.totalEnergyUsed + energy_req_for_delivery
            B_res = assigned_EV.B_res - energy_cost_min
            assigned_EV.B_res = B_res
            assigned_EV.B_trace.append(B_res)
            
            
            # printDebug("energy_cost_min: ", energy_cost_min)
    
    
    ####            
    unassigned_deliveries = [t for t in deliveries if t.assigned_EV==None]
    
    # Return to depot
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
    total_reward = 0
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
                total_reward += float(getattr(loc, "reward", 0.0))
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
    
    print("****NDF****")          
    # print("Total delivery: ", nD, " Total CP: ", nC+1, " Total EV: ", nE, " Time slot max: ", time_slot_max, " delivery2ev_ratio: ", delivery2ev_ratio)          
    print(f"Elapsed time: {elapsed_time:.2f} ms")
    print(f"Total Reward: {total_reward}")
    print(f"Total cost: {total_cost}")
    print("Objective val: ", (total_reward - total_cost))
    print(f"Total successful delivery: {total_deliveries_completed}")
    print("::Delivery Distribution::")
    for j in E:
        print(f"delivery_distribution[{j}]:", delivery_distribution.get(j, 0))
    # print(EVs[1].A)
    # print(rateE[0])
    
    return elapsed_time, total_cost, total_reward, total_deliveries_completed
    
    
    
def EDF(cp, deliveries, theta, C, D, E, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC):
    start_time = time.time() # Algorithm start time

            
    # Find the nearest CP. The CP which requires the least energy to reach is the nearest
    for d in deliveries:
        c_id = min((key for key in psi_DC if key[0] == d.local_id and key[1] > 0 ),  # Filter keys with the fixed x
                    key=lambda k: psi_DC[k]  # Use the dictionary value for comparison
                    )[1]  # Get the y value from the key with the minimum value
        nearest_fog_index = c_id
        d.nearest_CP_local_id = nearest_fog_index
    
    sorted_deliveries = sorted(deliveries, key=lambda x: tau_end[x.local_id])
    for delivery in sorted_deliveries:
        d_id = delivery.local_id
        service_time_d = float(getattr(delivery, "service_time", 0.0))
        # print("cluster:", c_id, " delivery: ", d_id, "delivery start time: ", tau_start[d_id])
        
        energy_cost_min = 9999999 # cost is measured in terms of energy used
        assigned_EV = None
        departure_after_service = -1
        for e in EVs:
            
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
            
            # if energy_req < energy_cost_min:
                # Check the battery constraints 
            if energy_req + energy_req_d2W > e.B_res:
                if last_loc.type == "delivery": # if already at a CP then charge is already full
                    # Send it to the nearest CP for charging
                    nearest_cp_id_from_last_point = last_loc.nearest_CP_local_id
                    e_id = e.local_id
                    e.A.append(cp[nearest_cp_id_from_last_point])
                    energy_to_cp = psi_DC[(last_loc.local_id, nearest_cp_id_from_last_point)]
                    soc_at_cp = e.B_res - energy_to_cp
                    req_charge = max(beta_f - soc_at_cp, 0.0)
                    e.T.append(e.t_ed)
                    e.t_ed = (
                        e.t_ed
                        + gamma_DC[(last_loc.local_id, nearest_cp_id_from_last_point)]
                        + req_charge / min(rateE[e_id], rateC[nearest_cp_id_from_last_point])
                    )  # accumulated time + reaching time + charging time
                    e.B_res = beta_f # reset battery
                    e.B_trace.append(beta_f)
                continue
            
            reaching_time = e.t_ed + duration_lastLoc2delivery
            service_start = max(reaching_time, tau_start[d_id])
            if (service_start + service_time_d) <= tau_end[d_id]:
                assigned_EV = e
                energy_cost_min = energy_req
                departure_after_service = service_start + service_time_d
                break
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
            assigned_EV.t_ed = departure_after_service
            
            # assigned_EV.totalEnergyUsed = assigned_EV.totalEnergyUsed + energy_req_for_delivery
            B_res = assigned_EV.B_res - energy_cost_min
            assigned_EV.B_res = B_res
            assigned_EV.B_trace.append(B_res)
            
            
            # printDebug("energy_cost_min: ", energy_cost_min)
    
    
    ####            
    unassigned_deliveries = [t for t in deliveries if t.assigned_EV==None]
    
    # Return to depot
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
    total_reward = 0
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
                total_reward += float(getattr(loc, "reward", 0.0))
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
    
    print("****EDF****")          
    # print("Total delivery: ", nD, " Total CP: ", nC+1, " Total EV: ", nE, " Time slot max: ", time_slot_max, " delivery2ev_ratio: ", delivery2ev_ratio)          
    print(f"Elapsed time: {elapsed_time:.4f} ms")
    print(f"Total Reward: {total_reward}")
    print(f"Total cost: {total_cost}")
    print("Objective val: ", (total_reward - total_cost))
    print(f"Total successful delivery: {total_deliveries_completed}")
    # print("::Delivery Distribution::")
    for j in E:
        print(f"delivery_distribution[{j}]:", delivery_distribution.get(j, 0))
    # print(EVs[1].A)
    # print(rateE[0])
    
    return elapsed_time, total_cost, total_reward, total_deliveries_completed

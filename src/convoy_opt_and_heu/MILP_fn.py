"""Gurobi MILP model construction and solve utilities for CONVOY2."""

import gurobipy as gp
from gurobipy import GRB
# import math
import json
import os
import time

DEBUG = False


def _load_gurobi_options():
    options = {
        "OutputFlag": 0,
        "LogToConsole": 0,
    }

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_config_path = os.path.join(project_root, "config", "gurobi_wls.json")
    config_path = os.environ.get("CONVOY_GUROBI_CONFIG", default_config_path)

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            raise ValueError(
                "Invalid Gurobi config format in {}: expected JSON object.".format(
                    config_path
                )
            )
        for key in ["WLSACCESSID", "WLSSECRET", "LICENSEID", "OutputFlag", "LogToConsole"]:
            value = config_data.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    continue
            if key in {"LICENSEID", "OutputFlag", "LogToConsole"}:
                value = int(value)
            options[key] = value

    env_overrides = {
        "GUROBI_WLSACCESSID": "WLSACCESSID",
        "GUROBI_WLSSECRET": "WLSSECRET",
        "GUROBI_LICENSEID": "LICENSEID",
        "GUROBI_OUTPUTFLAG": "OutputFlag",
        "GUROBI_LOGTOCONSOLE": "LogToConsole",
    }
    for env_key, option_key in env_overrides.items():
        env_value = os.environ.get(env_key)
        if env_value is None:
            continue
        env_value = env_value.strip()
        if env_value == "":
            continue
        if option_key in {"LICENSEID", "OutputFlag", "LogToConsole"}:
            env_value = int(env_value)
        options[option_key] = env_value

    return options



def MILP(
    cp,
    deliveries,
    theta,
    reward,
    C,
    D,
    E,
    C1,
    S,
    tau_start,
    tau_end,
    nS,
    EVs,
    gamma_DD,
    psi_DD,
    gamma_DC,
    psi_DC,
    beta_f,
    rateE,
    rateC,
    alpha1,
    alpha2,
    reserve_battery=0.0,
    gamma_C0=None,
    psi_C0=None,
    return_objective=False,
):
    resB = float(reserve_battery)  # reserve battery in kWh
    if resB < 0:
        raise ValueError("reserve_battery must be >= 0.")
    if resB >= beta_f:
        raise ValueError("reserve_battery must be smaller than full battery.")
    service_time_D = {
        int(d.local_id): float(getattr(d, "service_time", 0.0)) for d in deliveries
    }
    if gamma_C0 is None:
        gamma_C0 = {}
    if psi_C0 is None:
        psi_C0 = {}
    
    
    objective_value = 0


    options = _load_gurobi_options()

    with gp.Env(params=options) as env, gp.Model("TripPlanner", env=env) as model:
        model.Params.OutputFlag = 0
        start_time = time.time()
        chiCD = model.addVars(C1, D, E, S, vtype=GRB.BINARY, name="chiCD") # Decision var for edge b/w C and D
        chiDC = model.addVars(D, C1, E, S, vtype=GRB.BINARY, name="chiDC")
        # Optional direct return from CP -> depot (0) at the end of a subtrip.
        chiC0 = model.addVars(C, E, S, vtype=GRB.BINARY, name="chiC0")
        chiDD = model.addVars(D, D, E, S, vtype=GRB.BINARY, name="chiDD")

        TarrC = model.addVars(C1, E, S, vtype=GRB.CONTINUOUS, lb=0, ub= GRB.INFINITY, name="arrival_time_at_C")
        TarrD = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, ub= GRB.INFINITY, name="arrival_time_at_D")
        TdepC = model.addVars(C1, E, S, vtype=GRB.CONTINUOUS, lb=0, ub= GRB.INFINITY, name="departure_time_at_C")
        TdepD = model.addVars(D, vtype=GRB.CONTINUOUS, lb=0, ub= GRB.INFINITY, name="departure_time_at_D")
        
        reqB = model.addVars(E, S, vtype=GRB.CONTINUOUS, lb=0, ub= GRB.INFINITY, name="reqB")       # battery required  
        
        
        u = model.addVars(D, E, S, vtype=GRB.CONTINUOUS, lb=1, ub=len(D), name="u")
        
        
        # new 
        SoC = model.addVars(E, S, vtype=GRB.CONTINUOUS, lb=0, ub= beta_f, name="SoC")       # State of Charge  
        reqCh = model.addVars(E, S, vtype=GRB.CONTINUOUS, lb=0, ub= beta_f, name="req partial Charge")       # required partial Charge \rho
        _lambda = model.addVars(D, C1, E, S, vtype=GRB.BINARY, name="lambda")
        # 1 when EV j returns to depot at the end of subtrip l.
        return_to_depot = model.addVars(E, S, vtype=GRB.BINARY, name="return_to_depot")
        # Recharge amount at depot right after the final returning subtrip.
        final_depot_charge = model.addVars(
            E, S, vtype=GRB.CONTINUOUS, lb=0, ub=beta_f, name="final_depot_charge"
        )


        #obj = model.addVar(lb=0, name="obj")  # lb=0 ensures Z is non-negative
        #objective = gp.LinExpr()
        objective = gp.QuadExpr()
        
        ## Objective 2
        for j in E:
            for l in S:
                for x in C1:
                    for y in D:
                        objective += chiCD[x, y, j, l] * reward[y] * alpha1
                for x in D:
                    for y in D:
                        if x != y:
                            objective += chiDD[x, y, j, l] * reward[y] * alpha1
        # reqCh[j,l] is the recharge performed between subtrip (l-1) and l.
        # Therefore charging cost for reqCh[j,l] must be paid at the CP where
        # subtrip l starts (chiCD), not at the CP where subtrip l ends (chiDC).
        for j in E:
            for l in S:
                if l == 0:
                    continue
                for x in C:
                    starts_from_cp_x = gp.quicksum(chiCD[x, y, j, l] for y in D)
                    objective -= theta[x] * starts_from_cp_x * reqCh[j, l] * alpha2
        for j in E:
            for l in S:
                objective -= theta[0] * final_depot_charge[j, l] * alpha2
        
        # for j in E:
        #     for l in S:
        #         for x in D:
        #             for y in C1:
        #                 term0_component = _lambda[x, y, j, l] * theta[y]  # This is symbolic during model building
        #                 objective -= term0_component * alpha2
                        
        model.setObjective(objective, GRB.MAXIMIZE)
        
        
        
        
    
        
        
        
        #### Constraints ####
        
        
        
        # Additional Constraint: All self edges to be 0
        for j in E:
            for l in S:
                for x in D:
                    model.addConstr(chiDD[x, x, j, l] == 0, name=f"chiDD_{j}_{l}_{x}_{x}")
                    

        # Miller-Tucker-Zemlin (MTZ) Formulation: sub-tour elimination constraints
        n = len(D)
        for j in E:
            for l in S:
                for x in D:
                    for y in D:
                        if x != y:
                            model.addConstr(u[x, j, l] - u[y, j, l] + n * chiDD[x, y, j, l] <= n - 1, f"mtz_{j}_{l}_{x}_{y}")


        # Every subtrip starts with a CS or D0 and end at a CS or D0  (Eq 5,6,7,8)
        # Create a linear expression
        # Initialize z_notIsEmptySubRoute variable: 1 when subtrip is not empty, 0 otherwise
        z_notIsEmptySubRoute = model.addVars(
            E, S, vtype=GRB.BINARY, name="z_notisemptysubroute")
        
        for j in E:
            for l in S:
                
                # Initialize emptySubRouteCheck as a linear expression
                emptySubRouteCheck = gp.LinExpr()

                for x in D:
                    for y in D:
                        emptySubRouteCheck += chiDD[x, y, j, l]
                    for y in C1:
                        emptySubRouteCheck += chiCD[y, x, j, l]
        
                # Big M Implementation for the constraints
                M = len(D) + 10000
                emptySubRouteCheck -= M * z_notIsEmptySubRoute[j, l]
                
                # First constraint: (sum of all edges to delivery points in a subroute) - M * z <= 0
                model.addConstr(emptySubRouteCheck <= 0, name=f"empty_sub_route_check1_{j}_{l}")
        
                # Second constraint: (sum of all edges to delivery points in a subroute) + M - M * z >= 1e-6
                emptySubRouteCheck += M  # Adding the M constant
                epsilon = 1e-6
                model.addConstr(emptySubRouteCheck >= epsilon, name=f"empty_sub_route_check2_{j}_{l}")
                
                # Constraints 1 and 2 for CP nodes at the start and end of subroutes
                constraint1 = gp.LinExpr()
                constraint2 = gp.LinExpr()
        
                for x in C1:
                    for y in D:
                        constraint1 += chiCD[x, y, j, l]
        
                for x in D:
                    for y in C1:
                        constraint2 += chiDC[x, y, j, l]
        
                # Adding constraints 5 and 6 to the model
                model.addConstr(constraint1 == z_notIsEmptySubRoute[j, l], name=f"c1_{j}_{l}")
                model.addConstr(constraint2 == z_notIsEmptySubRoute[j, l], name=f"c2_{j}_{l}")
        
        # Constraint 3: A delivery point should be visited at most once (Eq. 15)
        for x in D:
            constraint3 = gp.LinExpr()
            
            for j in E:
                for l in S:
                    for y in D:
                        constraint3 += chiDD[y, x, j, l]
                    
                    for y in C1:
                        constraint3 += chiCD[y, x, j, l]
        
            # Add the constraint to the model: constraint3 <= 1
            model.addConstr(constraint3 <= 1, name=f"c3_{x}")
        
        
        # Constraint 4: An EV visiting a delivery point 'x' in trip 'l' should also come out from there (Eq. 14) # problem
        for j in E:
            for l in S:
                for x in D:
                    constraint4A = gp.LinExpr()  # Initialize a linear expression
                    constraint4B = gp.LinExpr()  # Initialize a linear expression
                    
                    for y in D:
                        constraint4A += chiDD[y, x, j, l]
                    for y in D:
                        constraint4B += chiDD[x, y, j, l]
                        
                    for y in C1:
                        constraint4A += chiCD[y, x, j, l]
                    for y in C1:
                        constraint4B += chiDC[x, y, j, l]
        
                    # Add the constraint to the model: constraint3 == 0
                    model.addConstr(constraint4A - constraint4B == 0, name=f"c4_{j}_{l}_{x}")
                    
        # Constraint 5, 6, 7: An EV starts from a depot and return at the end only once (Eq. 12)
        for j in E:
            constraint5 = gp.LinExpr()  # Initialize a linear expression
            for l in S:
                for x in D:
                    constraint5 += chiCD[0, x, j, l]   
            model.addConstr(constraint5 <= 1, name=f"c5_{j}_{l}")
            
            constraint6 = gp.LinExpr()  # Initialize a linear expression
            for l in S:
                for x in D:
                    constraint6 += chiDC[x, 0, j, l]
                for c in C:
                    constraint6 += chiC0[c, j, l]
            model.addConstr(constraint6 <= 1, name=f"c6_{j}_{l}")

                # Add the constraint to the model: constraint3 == 0
            model.addConstr(constraint5 - constraint6 == 0, name=f"c7_{j}_{l}")
        
        # CP -> depot direct return can be used only when that subtrip ended at the same CP.
        for j in E:
            for l in S:
                for c in C:
                    model.addConstr(
                        chiC0[c, j, l]
                        <= gp.quicksum(chiDC[x, c, j, l] for x in D),
                        name=f"cp_to_depot_link_{c}_{j}_{l}",
                    )
                    model.addConstr(
                        chiC0[c, j, l] <= z_notIsEmptySubRoute[j, l],
                        name=f"cp_to_depot_nonempty_{c}_{j}_{l}",
                    )

        # Track which subtrip returns to depot (either D->depot or CP->depot).
        for j in E:
            for l in S:
                model.addConstr(
                    return_to_depot[j, l]
                    == gp.quicksum(chiDC[x, 0, j, l] for x in D)
                    + gp.quicksum(chiC0[c, j, l] for c in C),
                    name=f"return_to_depot_def_{j}_{l}",
                )
            
        
        # Constraint 8: An EV visiting a CP in trip 'l' should also come out from there at trip 'l+1'(Eq. 13)
        for j in E:
            for l in range(1, len(S)):
                for x in C:
                    constraint8A = gp.LinExpr()  # Initialize a linear expression
                    constraint8B = gp.LinExpr()  # Initialize a linear expression
                    for y in D:   
                        constraint8A += chiDC[y, x, j, (l-1)]
                        constraint8B += chiCD[x, y, j, l]
                    # If CP->depot return is taken at l-1, no need to start l from same CP.
                    constraint8A -= chiC0[x, j, (l-1)]
            
                    # Add the constraint to the model: constraint3 == 0
                    model.addConstr(constraint8A == constraint8B, name=f"c8_{j}_{l}")
                
        # # Additional Constraint: If a subtrip is non-empty, its previous subtrip should also be non-empty
        # for j in E:
        #     for l in range(1, len(S)):  # Start from 1 to access the previous subtrip (l-1)
        #         constraint9 = z_notIsEmptySubRoute[j, l] - z_notIsEmptySubRoute[j, l-1]
                
        #         # Add the constraint to the model: z[k, l] - z[k, l-1] <= 0 # kept less than to avoid floating point error
        #         model.addConstr(constraint9 <= 0, name=f"c9_{j}_{l}")
        
        
        # Depot Constraint 1:
        # If EV j is used at all, its first subtrip (l=0) must start at depot.
        # Unused EVs are allowed (both sides become 0).
        for j in E:
            depotConstraint1 = gp.LinExpr()
        
            # Sum over all tasks for the first subtrip (l=0)
            for y in D:
                depotConstraint1 += chiCD[0, y, j, 0]
        
            # Link with non-empty flag to avoid forcing every EV to be active.
            model.addConstr(
                depotConstraint1 == z_notIsEmptySubRoute[j, 0],
                name=f"depot_constraint1_{j}",
            )
            
            
            
        # Energy Constraint       
        for j in E:
            for l in S:
                constraint10 = gp.LinExpr()  # Initialize a linear expression
                for x in C1:
                    for y in D:
                        constraint10 += chiCD[x, y, j, l] * psi_DC[(y, x)]
                    
                    for y in D:
                        constraint10 += chiDC[y, x, j, l] * psi_DC[(y, x)]
                for x in D:
                    for y in D:
                        if x != y:
                            constraint10 += chiDD[x, y, j, l] * psi_DD[(x, y)]
                for c in C:
                    constraint10 += chiC0[c, j, l] * psi_C0.get(c, 0.0)
                            
                # Add the constraint to the model: all energy ensumption for a subtrip <= beta_f
                model.addConstr(reqB[j, l] == constraint10, name=f"energy_constraint1_{j}_{l}")
                # model.addConstr(constraint10 <= beta_f , name=f"energy_constraint_{j}_{l}")
                
        # New
        model.addConstrs((reqCh[j, 0] == 0 for j in E), name="init_reCh") # initialize req charging at the beginning of subtrip
        model.addConstrs((SoC[j, 0] == beta_f for j in E), name="init_soc") # initialize SoC
        for j in E:
            for l in S:
                model.addConstr(reqB[j, l] + resB <= SoC[j,l] , name=f"energy_constraint2_{j}_{l}")
                if l > 0:
                    model.addConstr(reqCh[j, l] == SoC[j,l] - (SoC[j,(l-1)] - reqB[j, (l-1)]) , name=f"energy_constraint3_{j}_{l}")

        # Linearize final depot recharge:
        # final_depot_charge = return_to_depot * (beta_f - (SoC - reqB))
        for j in E:
            for l in S:
                recharge_need = beta_f - SoC[j, l] + reqB[j, l]
                model.addConstr(
                    final_depot_charge[j, l] <= recharge_need,
                    name=f"final_depot_charge_ub_need_{j}_{l}",
                )
                model.addConstr(
                    final_depot_charge[j, l] <= beta_f * return_to_depot[j, l],
                    name=f"final_depot_charge_ub_ret_{j}_{l}",
                )
                model.addConstr(
                    final_depot_charge[j, l]
                    >= recharge_need - beta_f * (1 - return_to_depot[j, l]),
                    name=f"final_depot_charge_lb_{j}_{l}",
                )
                
                
        # Time Constraints
        for j in E:
            model.addConstr(TdepC[0, j, 0] >= 0, name=f"time_constraint1_{j}")
            
        for j in E:
            for l in S:
                for y in D:
                      constraint11 = gp.QuadExpr()  # Initialize a linear expression  
                      for x in D:
                          constraint11 += chiDD[x, y, j, l] * (TdepD[x] + gamma_DD[(x,y)])
                      for x in C1:
                          constraint11 += chiCD[x, y, j, l] * (TdepC[x, j, l] + gamma_DC[(y, x)])
                      model.addConstr(TarrD[y] >= constraint11, name=f"time_constraint2_{j}_{l}_{y}")
        
        
        for j in E:
            for l in S:
                for y in C:
                    constraint12 = gp.QuadExpr()  # Initialize a linear expression  
                    for x in D:
                        constraint12 += chiDC[x, y, j, l] * (TdepD[x] + gamma_DC[(x,y)])
                    model.addConstr(TarrC[y, j, l] >= constraint12, name=f"time_constraint3_{j}_{l}_{y}")
        
        
        for y in D:
            # Service starts at max(arrival_time, tau_start) and must start before tau_end.
            # Departure is service start + service_time.
            model.addConstr(TarrD[y] <= tau_end[y], name=f"time_constraint4A_{y}")
            model.addConstr(
                TdepD[y] >= TarrD[y] + service_time_D.get(y, 0.0),
                name=f"time_constraint4B_{y}",
            )
            model.addConstr(
                TdepD[y] >= tau_start[y] + service_time_D.get(y, 0.0),
                name=f"time_constraint4C_{y}",
            )
        
        for j in E:
            for l in range(1, len(S)):
                for y in C:
                    constraint13 = gp.LinExpr()  # Initialize a linear expression
                    # chargingTime = reqB[j, l] / min(rateE[j], rateC[y]) * 60 # *60 for converting to min
                    chargingTime = reqCh[j, l] / min(rateE[j], rateC[y]) * 60 # *60 for converting to min
                    constraint13 = TarrC[y, j, (l-1)] + chargingTime
                    model.addConstr(TdepC[y, j, l] >= constraint13, name=f"time_constraint5_{j}_{l}_{y}")
                    
                    
                    
        #new Linearization constraints
        # for j in E:
        #     for l in S:
        #         for x in D:
        #             for y in C1:
        #                 model.addConstr(_lambda[x, y, j, l] >= 0, name=f"_lambda_{j}_{l}_{x}_{x}")
        #                 model.addConstr(_lambda[x, y, j, l] <= reqCh[j, l], name=f"_lambda_cons2_{j}_{l}_{x}_{x}")
        #                 model.addConstr(_lambda[x, y, j, l] <= beta_f * chiDC[x, y, j, l], name=f"_lambda_cons3_{j}_{l}_{x}_{x}")
        #                 model.addConstr(_lambda[x, y, j, l] >= reqCh[j, l] - (1 - chiDC[x, y, j, l]) * beta_f, name=f"_lambda_cons3_{j}_{l}_{x}_{x}")
        
        
                        

        # Optimize the model
        model.optimize()
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        
        # Results variables
        total_deliveries_completed = 0
        total_cost = 0
        delivery_distribution = {}
        
            

        # Print term1 after optimization if an optimal solution is found
        if model.status == GRB.OPTIMAL:
            objective_value = model.ObjVal
            if DEBUG:
                for j in E:
                    for l in S:
                        for xp in D:
                            for yp in C1:
                                if chiCD[yp, xp, j, l].x == 1:
                                    print(f"chiCD[{yp}, {xp}, {j}, {l}] = {chiCD[yp, xp, j, l].x:.2f}")
                                if chiDC[xp, yp, j, l].x == 1:
                                    print(f"chiDC[{xp}, {yp}, {j}, {l}] = {chiDC[xp, yp, j, l].x:.2f}")
                            for yp in D:
                                if chiDD[xp, yp, j, l].x == 1:
                                    print(f"chiDD[{xp}, {yp}, {j}, {l}] = {chiDD[xp, yp, j, l].x:.2f}")
                        print(f"reqCh[{j},{l}] = {reqCh[j,l].x:.2f}")
                    
            for j in E:
                delivery_by_j = 0
                for l in S:
                    for x1 in C1:
                        for y in D:
                            if chiCD[x1, y, j, l].x > 0.5:
                                total_deliveries_completed  += 1
                                delivery_by_j += 1
                    for x1 in D:
                        for y in D:
                            if x1 != y and chiDD[x1, y, j, l].x > 0.5:
                                total_deliveries_completed  += 1
                                delivery_by_j += 1

                delivery_distribution[j] = delivery_by_j
                
            for j in E:
                for l in S:
                    if l > 0:
                        for x in C:
                            starts_from_cp_x = sum(chiCD[x, y, j, l].x for y in D)
                            if starts_from_cp_x > 0.5:
                                if DEBUG:
                                    print("charged(start_cp): ", j, l, x)
                                total_cost += theta[x] * reqCh[j, l].x
                    total_cost += theta[0] * final_depot_charge[j, l].x
                                
                                
                                
            # for j in E:
            #     for l in S:
            #         for x in C1:
            #             for y in D:
            #                 objective += chiCD[x, y, j, l] * reward[y] * alpha1
            #         for x in D:
            #             for y in D:
            #                 if x != y:
            #                     objective += chiDD[x, y, j, l] * reward[y] * alpha1
                                
        
        # if model.status == GRB.OPTIMAL:
        #     for j in E:
        #         for l in S:
        #             for x in D:
        #                 for y in C1:
        #                     # Evaluate term1 using the optimized values of chiCD, chiDD, and chiDC
        #                     term1_value = (
        #                         beta_f -
        #                         sum(chiCD[xp, yp, j, l].x * psi_DC[(xp, yp)] for xp in C1 for yp in D) -
        #                         sum(chiDD[xp, yp, j, l].x * psi_DD[(xp, yp)] for xp in D for yp in D) -
        #                         sum(chiDC[xp, yp, j, l].x * psi_DC[(xp, yp)] for xp in D for yp in C1)
        #                     )
                            
        #                     print(f"Evaluated term1 for j={j}, l={l}, x={x}, y={y}: {term1_value:.2f}")
        # else:
        #     print("No optimal solution found.")


    print("****Optimal****") 
    # print("Total delivery: ", nD, " Total CP: ", nC+1, " Total EV: ", nE)
    print(f"Elapsed time: {elapsed_time:.2f} ms")
    print(f"Total cost: {total_cost}")
    print(f"Objective val: {objective_value}")
    print(f"Total successful delivery: {total_deliveries_completed}")
    print("::Delivery Distribution::")
    for j in E:
        print(f"delivery_distribution[{j}]:", delivery_distribution.get(j, 0))
        
    if return_objective:
        return elapsed_time, total_cost, total_deliveries_completed, objective_value
    return elapsed_time, total_cost, total_deliveries_completed

import math
import random
import csv
import os
os.chdir(r"C:\research\imp_test_from_last_laptop\EV\EV_delivery\LCN_extension\experiments\src")

from helper import preProcess
from ev import EV
from heuristic_partial_charging_fn import heuristic
from baselines import NDF, EDF
from MILP_fn import MILP



seed_val = 123
# random.seed(time.time())
random.seed(seed_val)
DEBUG = False

def printDebug(string, val):
    if DEBUG == True:
        print(string, val)




## user input
base_file = "jd200_1" # base file name
# output_file = 'results' + '_base_file' + '.csv'
output_file = os.path.join("..", "results", "results3_{}.csv".format(base_file))



with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Methods", "Total delivery", "Total CP", "Total EV", "delivery2ev_ratio", "Elapsed time (ms)", "Total cost", "Total successful delivery", "Avg. cost per successful delivery"])
    
    # nD_list = [5, 10, 15] + [x for x in range(25, 201, 25)] # number of deliveries
    nD_list = [10]
    
    itr = 0
    
    for nD in nD_list: #deliveries
        for delivery2ev_ratio in range(5, 16, 5): 
            for nC in [5]: # number of charging points
                
                # if nD < 25 and nC > 5: # for small number of deliveries keep nC = 5
                #     continue
                
                # if int(nD/delivery2ev_ratio) < 2:
                #     continue
            
                
            
                # Number of  (We can control it as user input)
                nW = 1  # Warehouse depots
                nE = int(nD/delivery2ev_ratio)  # Electric delivery trucks
                
                # TEST
                if itr == 1:
                    break
                nE = 2
                nC = 2
                itr = itr + 1
                
                nS = math.ceil(nD / nE)  # max Subtrips per EV
                
                print("Total delivery: ", nD, " Total CP: ", nC+1, " Total EV: ", nE)
                
                
                # Initialize: read the input files
                cp, deliveries, theta, reward, C, D, E, C1, S, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC, alpha1, alpha2 = preProcess(base_file, nD, nC, nS, nE)    
                
                
                ## Optimal
                alpha1, alpha2 = 1, 1
                if nD < 20 and nC < 20:
                    elapsed_time, total_cost, total_deliveries_completed = MILP(cp, deliveries, theta, reward, C, D, E, C1, S, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC, alpha1, alpha2)
                    writer.writerow(["Optimal", nD, nC, nE, delivery2ev_ratio, elapsed_time, total_cost, total_deliveries_completed, total_cost/total_deliveries_completed])
                
                ## Heuristic CSA
                # reset EV trips
                EVs = []
                for j in E:
                    EVs.append(EV(j, cp[0], beta_f)) # cp[0] is the depot
                elapsed_time, total_cost, total_reward, total_deliveries_completed = heuristic(cp, deliveries, theta, reward, C, D, E, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC)
                writer.writerow(["CSA", nD, nC, nE, delivery2ev_ratio, elapsed_time, total_cost, total_deliveries_completed, total_cost/total_deliveries_completed])
                
                # ## NDF
                # # reset EV trips
                # EVs = []
                # for j in E:
                #     EVs.append(EV(j, cp[0], beta_f)) # cp[0] is the depot
                # elapsed_time, total_cost, total_deliveries_completed = NDF(cp, deliveries, theta, C, D, E, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC)
                # writer.writerow(["NDF", nD, nC, nE, delivery2ev_ratio, elapsed_time, total_cost, total_deliveries_completed, total_cost/total_deliveries_completed])
                
                # ## EDF  
                # # reset EV trips
                # EVs = []
                # for j in E:
                #     EVs.append(EV(j, cp[0], beta_f)) # cp[0] is the depot
                # elapsed_time, total_cost, total_deliveries_completed = EDF(cp, deliveries, theta, C, D, E, tau_start, tau_end, nS, EVs, gamma_DD, psi_DD, gamma_DC, psi_DC, beta_f, rateE, rateC)
                # writer.writerow(["EDF", nD, nC, nE, delivery2ev_ratio, elapsed_time, total_cost, total_deliveries_completed, total_cost/total_deliveries_completed])
        
        
        

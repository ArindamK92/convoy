# -*- coding: utf-8 -*-
class EV:
    def __init__(self, local_id, depot, battery_capacity):
        self.local_id = local_id
        self.T = [] # Departure time for assigned tasks. It will have |A| - 1 values
        self.A = [depot] # Assigned points
        self.t_ed = 0 # Earliest departure time from the last assigned point
        self.B = battery_capacity
        self.B_res = battery_capacity
        self.totalcost = 0 # total cost till now
        self.cost_trace = [] # cost involved per task
        self.assigneddeliveries = []
        self.B_trace = [battery_capacity] # residual battery trace
        self.totalEnergyUsed = 0
        self.totalSubtrip = 0
        self.subtrip_energy = 0
        self.visited_CP_sequence = [depot]
        self.subtrips_energy_sequence = []
        
    def reset(self, depot, battery_capacity):
        self.T = []
        self.A = [depot]
        self.assigneddeliveries = []
        self.B_trace = [battery_capacity]
        
        
        

class Delivery:
    def __init__(self, latitude, longitude, local_id, global_id, tau_start, tau_end, reward):
        self.latitude = latitude
        self.longitude = longitude
        self.local_id = local_id
        self.global_id = global_id
        self.type = "delivery"
        self.nearest_CP_local_id = -1
        self.assigned_EV = None
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.reward = reward
        
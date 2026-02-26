"""Charging-point domain model used by Optimal+Heuristic code."""

class CP:
    """Charging point with geographic and ID metadata."""

    def __init__(self, latitude, longitude, local_id, global_id):
        self.latitude = latitude
        self.longitude = longitude
        self.local_id = local_id
        self.global_id = global_id
        self.type = "cp"

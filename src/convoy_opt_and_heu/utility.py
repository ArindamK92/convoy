"""Spatial utility helpers for nearest charging-point lookups."""

import random
import math
import time
import os
from scipy.spatial import KDTree

random.seed(time.time())


def create_kd_tree(cp, C, theta, min_lat, max_lat, min_lon, max_lon, min_theta, max_theta, denom_lat, denom_lon, denom_theta):
    """
    Create a KD-Tree from a list of CP objects
    """
    # lats = [c.latitude for c in cp[1:]] # first cp is the depot
    # lons = [c.longitude for c in cp[1:]]
    # min_lat = min(lats)
    # min_lon = min(lons)
    # min_theta = min(theta[1:])
    # max_lat = max(lats)
    # max_lon = max(lons)
    # max_theta = max(theta[1:])
    # denom_lat = max_lat - min_lat
    # denom_lon = max_lon - min_lon
    # denom_theta = max_theta - min_theta
    points = []
    for i in range(1, len(C)):
        norm_lat = (cp[i].latitude - min_lat) / denom_lat
        norm_lon = (cp[i].longitude - min_lon) / denom_lon
        norm_theta = (theta[i] - min_theta) / denom_theta
        points.append((norm_lat, norm_lon, norm_theta))
    kd_tree = KDTree(points)
    return kd_tree

def find_nearest_cp(kd_tree, loc_lat, loc_lon, min_lat, max_lat, min_lon, max_lon, min_theta, max_theta, denom_lat, denom_lon, denom_theta):
    """
    Function to find the nearest CP of a delivery point using the KD-Tree.

    Parameters
    ----------
    kd_tree : TYPE
        KD-tree built on charging points (excluding depot).
    loc_lat : TYPE
        Latitude of the waypoint.
    loc_lon : TYPE
        Longitube of the waypoint.

    Returns
    -------
    nearest_cp : TYPE
        Nearest CP of the waypoint.
    distance : TYPE
        Distance from the waypoint to the CP in normalized feature space.

    """
    # Query the KD-Tree for the nearest neighbor to the waypoint
    norm_lat = (loc_lat - min_lat) / denom_lat
    norm_lon = (loc_lon - min_lon) / denom_lon
    norm_theta = 0 # (min_theta - min_theta) / denom_theta # we try to choose the cp with lowest theta
    point = (norm_lat, norm_lon, norm_theta)
    _, index = kd_tree.query(point)

    return index+1 # As while creating the KD tree we don't take the first cp, i.e. the depot

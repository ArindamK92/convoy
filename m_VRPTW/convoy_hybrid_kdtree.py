"""KD-tree helpers for nearest charging-point lookup in hybrid postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.spatial import KDTree


@dataclass
class CPKDTreeBundle:
    """Container for CP KD-tree and normalization metadata."""

    kd_tree: KDTree
    cp_rows: list[dict]
    min_lat: float
    min_lon: float
    min_theta: float
    denom_lat: float
    denom_lon: float
    denom_theta: float


def _safe_denom(value: float) -> float:
    return value if abs(value) > 1e-12 else 1.0


def create_kd_tree(cp_rows: list[dict]) -> CPKDTreeBundle:
    """Create KD-tree over CP nodes using normalized lat/lon/unit-cost features."""
    if not cp_rows:
        raise ValueError("create_kd_tree requires at least one CP row.")

    lats = [float(cp["x"]) for cp in cp_rows]
    lons = [float(cp["y"]) for cp in cp_rows]
    thetas = [float(cp.get("charging_cost_per_kwh", 0.0)) for cp in cp_rows]

    min_lat = min(lats)
    min_lon = min(lons)
    min_theta = min(thetas)
    max_lat = max(lats)
    max_lon = max(lons)
    max_theta = max(thetas)

    denom_lat = _safe_denom(max_lat - min_lat)
    denom_lon = _safe_denom(max_lon - min_lon)
    denom_theta = _safe_denom(max_theta - min_theta)

    points = []
    for cp in cp_rows:
        norm_lat = (float(cp["x"]) - min_lat) / denom_lat
        norm_lon = (float(cp["y"]) - min_lon) / denom_lon
        norm_theta = (float(cp.get("charging_cost_per_kwh", 0.0)) - min_theta) / denom_theta
        points.append((norm_lat, norm_lon, norm_theta))

    kd_tree = KDTree(points)
    return CPKDTreeBundle(
        kd_tree=kd_tree,
        cp_rows=cp_rows,
        min_lat=min_lat,
        min_lon=min_lon,
        min_theta=min_theta,
        denom_lat=denom_lat,
        denom_lon=denom_lon,
        denom_theta=denom_theta,
    )


def find_nearest_cp(tree_bundle: CPKDTreeBundle, loc_lat: float, loc_lon: float) -> dict:
    """Find nearest CP using normalized (lat, lon, theta=0) query point."""
    norm_lat = (float(loc_lat) - tree_bundle.min_lat) / tree_bundle.denom_lat
    norm_lon = (float(loc_lon) - tree_bundle.min_lon) / tree_bundle.denom_lon
    # Bias toward low charging cost like opt/heu utility: theta query fixed to 0.
    norm_theta = 0.0

    _, index = tree_bundle.kd_tree.query((norm_lat, norm_lon, norm_theta))
    return tree_bundle.cp_rows[int(index)]

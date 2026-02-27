#!/usr/bin/env python3
"""Convert CONVOY test CSV + distance/time matrices to EVRP-TW-SPD-HMA instance format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _to_int(value: object) -> int:
    """Parse integer-like values, accepting float-formatted integers."""
    text = str(value).strip()
    if text == "":
        raise ValueError("Encountered empty ID value")
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def _load_matrix(matrix_csv: Path) -> pd.DataFrame:
    """Load matrix CSV with ID header/index and validate uniqueness/numeric values."""
    matrix = pd.read_csv(matrix_csv, index_col=0)
    matrix.index = matrix.index.map(_to_int)
    matrix.columns = [_to_int(c) for c in matrix.columns]
    matrix = matrix.apply(pd.to_numeric, errors="raise")
    if matrix.index.has_duplicates:
        raise ValueError(f"Duplicate row IDs found in matrix: {matrix_csv}")
    if pd.Index(matrix.columns).has_duplicates:
        raise ValueError(f"Duplicate column IDs found in matrix: {matrix_csv}")
    return matrix


def _infer_recharging_rate_minutes_per_kwh(nodes_df: pd.DataFrame) -> float:
    """Infer baseline `RECHARGING_RATE` (minutes/kWh) from station charge rates."""
    if "charge_rate_kwh_per_hour" not in nodes_df.columns:
        return 3.47
    cp_rates = pd.to_numeric(
        nodes_df.loc[nodes_df["type"] == "f", "charge_rate_kwh_per_hour"],
        errors="coerce",
    )
    cp_rates = cp_rates[cp_rates > 0]
    if cp_rates.empty:
        return 3.47
    # EVRP-TW-SPD-HMA expects recharge time per unit energy; here we use minutes/kWh.
    return 60.0 / float(cp_rates.mean())


def _coerce_float_or_none(value: object) -> Optional[float]:
    """Convert value to float, returning None for NaN/invalid values."""
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_customer_demands(
    row: pd.Series,
    default_customer_delivery: float,
    default_customer_pickup: float,
) -> tuple[float, float]:
    """Derive delivery/pickup demand for one customer row.

    Priority:
    - delivery from `delivery`, then `demand`, then `reward` (if positive),
    - pickup from `pickup`, then `pickup_demand` (if positive),
    - otherwise fallback defaults.
    """
    if row["type"] != "c":
        return 0.0, 0.0

    delivery: Optional[float] = None
    for col in ("delivery", "demand", "reward"):
        if col in row.index:
            value = _coerce_float_or_none(row[col])
            if value is not None and value > 0:
                delivery = value
                break
    if delivery is None:
        delivery = float(default_customer_delivery)

    pickup: Optional[float] = None
    for col in ("pickup", "pickup_demand"):
        if col in row.index:
            value = _coerce_float_or_none(row[col])
            if value is not None and value > 0:
                pickup = value
                break
    if pickup is None:
        pickup = float(default_customer_pickup)

    return float(delivery), float(pickup)


def convert_test_csv_to_evrp_instance(
    test_csv: Path,
    dist_matrix_csv: Path,
    time_matrix_csv: Path,
    output_path: Path,
    vehicles: int,
    dispatching_cost: float = 1000.0,
    unit_cost: float = 1.0,
    capacity: float = 200.0,
    electric_power: float = 30.0,
    consumption_rate: float = 0.00025,
    recharging_rate: Optional[float] = None,
    instance_name: Optional[str] = None,
    default_customer_delivery: float = 1.0,
    default_customer_pickup: float = 1.0,
) -> Dict[str, Path]:
    """Convert CONVOY test CSV + matrices to EVRP-TW-SPD-HMA text instance.

    Returns paths for:
    - generated instance file (`instance_path`)
    - generated mapped-id CSV (`id_map_path`)
    """
    nodes = pd.read_csv(test_csv)
    required_cols = {
        "ID",
        "type",
        "lng",
        "lat",
        "first_receive_tm",
        "last_receive_tm",
        "service_time",
    }
    missing = sorted(required_cols.difference(nodes.columns))
    if missing:
        raise ValueError(f"Missing required columns in {test_csv}: {missing}")

    nodes = nodes.copy()
    nodes["ID"] = nodes["ID"].map(_to_int)
    nodes["type"] = nodes["type"].astype(str).str.strip().str.lower()

    invalid_types = sorted(set(nodes["type"]) - {"d", "c", "f"})
    if invalid_types:
        raise ValueError(f"Unsupported node types in {test_csv}: {invalid_types}")

    depots = nodes.index[nodes["type"] == "d"].tolist()
    if len(depots) != 1:
        raise ValueError(
            f"Expected exactly one depot (type 'd') in {test_csv}, found {len(depots)}"
        )

    depot_idx = depots[0]
    ordered_nodes = pd.concat([nodes.loc[[depot_idx]], nodes.drop(index=depot_idx)], ignore_index=True)
    ordered_nodes.insert(0, "new_id", range(len(ordered_nodes)))

    id_map: Dict[int, int] = {
        int(row["new_id"]): int(row["ID"]) for _, row in ordered_nodes.iterrows()
    }

    dist = _load_matrix(dist_matrix_csv)
    tm = _load_matrix(time_matrix_csv)

    original_ids: List[int] = [id_map[i] for i in range(len(ordered_nodes))]
    for old_id in original_ids:
        if old_id not in dist.index or old_id not in dist.columns:
            raise KeyError(f"ID {old_id} from {test_csv} not found in distance matrix {dist_matrix_csv}")
        if old_id not in tm.index or old_id not in tm.columns:
            raise KeyError(f"ID {old_id} from {test_csv} not found in time matrix {time_matrix_csv}")

    if recharging_rate is None:
        recharging_rate = _infer_recharging_rate_minutes_per_kwh(ordered_nodes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if instance_name is None:
        instance_name = output_path.stem

    lines: List[str] = []
    lines.append(f"NAME : {instance_name}")
    lines.append("TYPE : EVRP-TW-SPD")
    lines.append(f"DIMENSION : {len(ordered_nodes)}")
    lines.append(f"VEHICLES : {int(vehicles)}")
    lines.append(f"DISPATCHINGCOST : {float(dispatching_cost)}")
    lines.append(f"UNITCOST : {float(unit_cost)}")
    lines.append(f"CAPACITY : {float(capacity)}")
    lines.append(f"ELECTRIC_POWER : {float(electric_power)}")
    lines.append(f"CONSUMPTION_RATE : {float(consumption_rate)}")
    lines.append(f"RECHARGING_RATE : {float(recharging_rate)}")
    lines.append("EDGE_WEIGHT_TYPE : EXPLICIT")

    lines.append("NODE_SECTION")
    lines.append("ID,type,lng,lat,delivery,pickup,ready_time,due_date,service_time")

    for _, row in ordered_nodes.iterrows():
        node_type = row["type"]
        lng = float(row["lng"])
        lat = float(row["lat"])
        ready = float(row["first_receive_tm"])
        due = float(row["last_receive_tm"])
        service = float(row["service_time"])

        delivery, pickup = _derive_customer_demands(
            row,
            default_customer_delivery=default_customer_delivery,
            default_customer_pickup=default_customer_pickup,
        )

        lines.append(
            f"{int(row['new_id'])},{node_type},{lng:.6f},{lat:.6f},"
            f"{delivery:.6f},{pickup:.6f},{ready:.6f},{due:.6f},{service:.6f}"
        )

    lines.append("DISTANCETIME_SECTION")
    lines.append("ID,from_node,to_node,distance,spend_tm")
    edge_id = 0
    n = len(ordered_nodes)
    for i in range(n):
        old_i = id_map[i]
        for j in range(n):
            if i == j:
                continue
            old_j = id_map[j]
            d = float(dist.at[old_i, old_j])
            t = float(tm.at[old_i, old_j])
            lines.append(f"{edge_id},{i},{j},{d:.6f},{t:.6f}")
            edge_id += 1

    lines.append("DEPOT_SECTION")
    lines.append("0")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    id_map_path = output_path.with_suffix(".id_map.csv")
    ordered_nodes[["new_id", "ID", "type"]].rename(
        columns={"new_id": "mapped_id", "ID": "original_id"}
    ).to_csv(id_map_path, index=False)

    return {
        "instance_path": output_path,
        "id_map_path": id_map_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone converter usage."""
    parser = argparse.ArgumentParser(
        description="Convert CONVOY test CSV + matrices into EVRP-TW-SPD-HMA instance file."
    )
    parser.add_argument(
        "--test-csv",
        required=True,
        help="Path to test CSV (e.g., data/test_instance.csv)",
    )
    parser.add_argument(
        "--dist-matrix-csv",
        required=True,
        help="Path to distance matrix CSV with IDs as row/column labels",
    )
    parser.add_argument(
        "--time-matrix-csv",
        required=True,
        help="Path to time matrix CSV with IDs as row/column labels",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        required=True,
        help="Maximum number of vehicles to set in EVRP instance header",
    )
    parser.add_argument(
        "--output-path",
        default="baseline/data/test_instance_evrp.txt",
        help="Output EVRP instance file path",
    )
    parser.add_argument("--dispatching-cost", type=float, default=1000.0)
    parser.add_argument("--unit-cost", type=float, default=1.0)
    parser.add_argument("--capacity", type=float, default=200.0)
    parser.add_argument("--electric-power", type=float, default=30.0)
    parser.add_argument(
        "--default-customer-delivery",
        type=float,
        default=1.0,
        help="Fallback delivery demand for customers when no positive demand column is available",
    )
    parser.add_argument(
        "--default-customer-pickup",
        type=float,
        default=1.0,
        help="Fallback pickup demand for customers when no positive pickup column is available",
    )
    parser.add_argument(
        "--consumption-rate",
        type=float,
        default=0.00025,
        help="Energy consumption per distance unit (e.g., kWh/m for meter-based matrices)",
    )
    parser.add_argument(
        "--recharging-rate",
        type=float,
        default=None,
        help=(
            "Recharge time per kWh in solver time unit. If omitted, inferred as 60/avg(charge_rate_kwh_per_hour) "
            "from charging stations."
        ),
    )
    parser.add_argument(
        "--instance-name",
        default=None,
        help="Optional name to write in NAME field",
    )
    return parser


def main() -> None:
    """CLI entrypoint for test-instance to EVRP conversion."""
    parser = _build_parser()
    args = parser.parse_args()

    result = convert_test_csv_to_evrp_instance(
        test_csv=Path(args.test_csv),
        dist_matrix_csv=Path(args.dist_matrix_csv),
        time_matrix_csv=Path(args.time_matrix_csv),
        output_path=Path(args.output_path),
        vehicles=args.vehicles,
        dispatching_cost=args.dispatching_cost,
        unit_cost=args.unit_cost,
        capacity=args.capacity,
        electric_power=args.electric_power,
        consumption_rate=args.consumption_rate,
        recharging_rate=args.recharging_rate,
        instance_name=args.instance_name,
        default_customer_delivery=args.default_customer_delivery,
        default_customer_pickup=args.default_customer_pickup,
    )

    print(f"Wrote EVRP instance: {result['instance_path']}")
    print(f"Wrote ID mapping   : {result['id_map_path']}")


if __name__ == "__main__":
    main()

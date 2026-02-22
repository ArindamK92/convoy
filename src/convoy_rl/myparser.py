"""CLI argument and CSV parsing helpers for convoy training/testing."""

import argparse
import csv

import torch


def parse_customer(csv_path: str):
    """Parse test-instance CSV and return depot, customers, and charging points.

    Supported test CSV schemas:
    1) Legacy test schema with columns:
       `customer_id,is_depot,x,y,demand,tw_start,tw_end,service_time,...`
       Optional CP markers:
       `is_charging_station` or `is_cp` or `node_type` in {f,cp,station}.
    2) Combined schema with columns:
       `ID,type,lng,lat,first_receive_tm,last_receive_tm,service_time,...`
       where type is one of {d,c,f}.
    """
    depot = None
    customers = []
    charging_points = []

    def _to_float(value, default=None):
        if value is None:
            value = ""
        txt = str(value).strip()
        if txt == "":
            if default is None:
                raise ValueError("Missing required numeric value in test CSV.")
            return float(default)
        return float(txt)

    def _to_int(value, default=None):
        num = _to_float(value, default=default)
        if not num.is_integer():
            raise ValueError(f"Expected integer-like value, got '{value}'.")
        return int(num)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("test-csv must contain a header row.")
        fields = {name.strip().lower() for name in reader.fieldnames}

        legacy_required = {
            "is_depot",
            "x",
            "y",
            "demand",
            "tw_start",
            "tw_end",
            "service_time",
        }
        combined_required = {
            "id",
            "type",
            "lng",
            "lat",
            "first_receive_tm",
            "last_receive_tm",
            "service_time",
        }
        is_legacy_schema = legacy_required.issubset(fields)
        is_combined_schema = combined_required.issubset(fields)
        if not is_legacy_schema and not is_combined_schema:
            raise ValueError(
                "Unsupported test-csv schema. Use either "
                "`customer_id,is_depot,x,y,demand,tw_start,tw_end,service_time,...` "
                "or combined schema `ID,type,lng,lat,first_receive_tm,last_receive_tm,service_time,...`."
            )

        for row in reader:
            row_norm = {
                k.strip().lower(): (v if v is not None else "") for k, v in row.items()
            }

            if is_combined_schema:
                node_type = row_norm["type"].strip().lower()
                node_id = _to_int(row_norm.get("id"))
                reward = _to_float(row_norm.get("reward"), default=0.0)
                base = {
                    "x": _to_float(row_norm.get("lng")),
                    "y": _to_float(row_norm.get("lat")),
                    "demand": (
                        _to_float(row_norm.get("demand"), default=1.0)
                        if node_type == "c"
                        else 0.0
                    ),
                    "tw_start": _to_float(row_norm.get("first_receive_tm")),
                    "tw_end": _to_float(row_norm.get("last_receive_tm")),
                    "service_time": _to_float(row_norm.get("service_time"), default=0.0),
                    "reward": reward,
                    "customer_id": node_id if node_type == "c" else None,
                    "node_id": node_id,
                }
                if base["tw_end"] <= base["tw_start"]:
                    raise ValueError("Each row must satisfy tw_end > tw_start.")

                if node_type == "d":
                    if depot is not None:
                        raise ValueError("test-csv must contain exactly one depot row.")
                    depot = base
                elif node_type == "c":
                    customers.append(base)
                elif node_type == "f":
                    cp_rec = {
                        "x": base["x"],
                        "y": base["y"],
                        "demand": 0.0,
                        "tw_start": base["tw_start"],
                        "tw_end": base["tw_end"],
                        "service_time": base["service_time"],
                        "reward": 0.0,
                        "cp_id": node_id,
                        "charge_rate_kwh_per_hour": _to_float(
                            row_norm.get("charge_rate_kwh_per_hour"), default=120.0
                        ),
                        "charging_cost_per_kwh": _to_float(
                            row_norm.get("unit_charging_cost"), default=0.0
                        ),
                    }
                    charging_points.append(cp_rec)
                else:
                    raise ValueError(
                        f"Unsupported node type '{node_type}' in combined test-csv."
                    )
                continue

            # Legacy schema
            is_depot = _to_int(row_norm.get("is_depot"), default=0) == 1
            reward = _to_float(row_norm.get("reward"), default=0.0)
            customer_id = (
                _to_int(row_norm.get("customer_id"))
                if str(row_norm.get("customer_id", "")).strip() != ""
                else None
            )
            base = {
                "x": _to_float(row_norm.get("x")),
                "y": _to_float(row_norm.get("y")),
                "demand": _to_float(row_norm.get("demand")),
                "tw_start": _to_float(row_norm.get("tw_start")),
                "tw_end": _to_float(row_norm.get("tw_end")),
                "service_time": _to_float(row_norm.get("service_time"), default=0.0),
                "reward": reward,
                "customer_id": customer_id,
                "node_id": customer_id if customer_id is not None else 0,
            }
            if base["tw_end"] <= base["tw_start"]:
                raise ValueError("Each row must satisfy tw_end > tw_start.")

            node_type = row_norm.get("node_type", "").strip().lower()
            is_cp = False
            if str(row_norm.get("is_charging_station", "")).strip() != "":
                is_cp = _to_int(row_norm.get("is_charging_station"), default=0) == 1
            elif str(row_norm.get("is_cp", "")).strip() != "":
                is_cp = _to_int(row_norm.get("is_cp"), default=0) == 1
            elif node_type in {"f", "cp", "charging_station", "station"}:
                is_cp = True

            if is_depot:
                if depot is not None:
                    raise ValueError("test-csv must contain exactly one depot row.")
                depot = base
            elif is_cp:
                cp_id_txt = str(row_norm.get("cp_id", "")).strip()
                cp_id = _to_int(cp_id_txt) if cp_id_txt != "" else customer_id
                if cp_id is None:
                    raise ValueError(
                        "Charging-point row in test-csv must provide cp_id or customer_id."
                    )
                cp_rec = {
                    "x": base["x"],
                    "y": base["y"],
                    "demand": 0.0,
                    "tw_start": base["tw_start"],
                    "tw_end": base["tw_end"],
                    "service_time": base["service_time"],
                    "reward": 0.0,
                    "cp_id": cp_id,
                    "charge_rate_kwh_per_hour": _to_float(
                        row_norm.get("charge_rate_kwh_per_hour"), default=120.0
                    ),
                    "charging_cost_per_kwh": _to_float(
                        row_norm.get("unit_charging_cost")
                        if str(row_norm.get("unit_charging_cost", "")).strip() != ""
                        else row_norm.get("charging_cost_per_kwh"),
                        default=0.0,
                    ),
                }
                charging_points.append(cp_rec)
            else:
                if base["customer_id"] is None:
                    raise ValueError(
                        "Customer rows in test-csv must include customer_id for matrix lookup."
                    )
                customers.append(base)

    if depot is None:
        raise ValueError("test-csv must contain exactly one depot row.")
    if not customers:
        raise ValueError("test-csv must contain at least one customer row.")
    return depot, customers, charging_points


def _parse_int_id(cell: str) -> int:
    value = float(cell.strip())
    if not value.is_integer():
        raise ValueError(f"Matrix ID must be an integer, got '{cell}'.")
    return int(value)


def _is_float(cell: str) -> bool:
    try:
        float(cell.strip())
        return True
    except ValueError:
        return False


def parse_distance_matrix_csv(matrix_csv_path: str) -> torch.Tensor:
    """Parse distance/time matrix CSV into a torch tensor.

    Supported formats:
    1) Plain numeric square matrix (no headers).
    2) ID-labeled matrix with a header row and first ID column.
       Example header: `,0,1,2,...`
    """
    raw_rows: list[list[str]] = []
    with open(matrix_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            raw_rows.append([cell.strip() for cell in row])

    if not raw_rows:
        raise ValueError("Matrix CSV is empty.")

    width = len(raw_rows[0])
    if any(len(r) != width for r in raw_rows):
        raise ValueError("Matrix CSV must be rectangular.")

    has_id_header = not _is_float(raw_rows[0][0])

    if not has_id_header:
        rows = [[float(x) for x in r] for r in raw_rows]
        n = len(rows[0])
        if any(len(r) != n for r in rows):
            raise ValueError("Matrix CSV must be rectangular.")
        if len(rows) != n:
            raise ValueError("Matrix CSV must be square.")
        return torch.tensor(rows, dtype=torch.float32)

    if width < 2:
        raise ValueError("ID-labeled matrix must include at least one data column.")

    col_ids = [_parse_int_id(x) for x in raw_rows[0][1:]]
    if len(raw_rows) - 1 != len(col_ids):
        raise ValueError(
            "ID-labeled matrix must be square: data-row count must match header ID count."
        )
    if len(set(col_ids)) != len(col_ids):
        raise ValueError("Column IDs in matrix header must be unique.")

    row_ids: list[int] = []
    value_rows: list[list[float]] = []
    for idx, row in enumerate(raw_rows[1:], start=2):
        row_id = _parse_int_id(row[0])
        vals = [float(x) for x in row[1:]]
        if len(vals) != len(col_ids):
            raise ValueError(
                f"ID-labeled matrix row at line {idx} has inconsistent length."
            )
        row_ids.append(row_id)
        value_rows.append(vals)

    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Row IDs in matrix must be unique.")
    if set(row_ids) != set(col_ids):
        raise ValueError("Row IDs and column IDs must contain the same node IDs.")

    max_id = max(col_ids)
    dense = torch.full((max_id + 1, max_id + 1), float("inf"), dtype=torch.float32)
    for i, row_id in enumerate(row_ids):
        for j, col_id in enumerate(col_ids):
            dense[row_id, col_id] = value_rows[i][j]
    return dense


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CVRPTW training/testing."""
    parser = argparse.ArgumentParser(
        description="Train and test RL4CO on VRPTW (CVRPTW in rl4co)."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Train batch size.")
    parser.add_argument(
        "--eval-batch-size", type=int, default=512, help="Val/Test batch size."
    )
    parser.add_argument(
        "--train-data-size", type=int, default=4096, help="Train samples per epoch."
    )
    parser.add_argument(
        "--val-data-size", type=int, default=1024, help="Validation sample count."
    )
    parser.add_argument(
        "--test-data-size", type=int, default=1024, help="Test sample count."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max-time", type=float, default=480.0, help="TW horizon.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="exponential",
        choices=["exponential", "rollout", "shared", "mean", "no", "critic"],
        help="REINFORCE baseline type.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Lightning accelerator.",
    )
    parser.add_argument(
        "--print-solution",
        action="store_true",
        help="Print one decoded VRPTW solution path after testing.",
    )
    parser.add_argument(
        "--fixed-eval-size",
        type=int,
        default=512,
        help="Fixed-set size used to track solution quality over epochs.",
    )
    parser.add_argument(
        "--fixed-eval-every",
        type=int,
        default=5,
        help="Evaluate on fixed set every N epochs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_vrptw",
        help="Directory for best/last model checkpoints.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help=(
            "If set, persist best checkpoint to checkpoint-dir/best_model.ckpt "
            "and reuse it in later runs (skip training when file exists)."
        ),
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help=(
            "Path to a custom single-instance VRPTW CSV for final testing. "
            "CP rows can be included in this file and will be used directly at test time."
        ),
    )
    parser.add_argument(
        "--csv-vehicle-capacity",
        type=float,
        default=30.0,
        help="Vehicle capacity used to normalize CSV demands.",
    )
    parser.add_argument(
        "--ev-battery-capacity-kwh",
        type=float,
        default=30.0,
        help="EV battery capacity in kWh.",
    )
    parser.add_argument(
        "--ev-energy-rate-kwh-per-distance",
        type=float,
        # 4 km/kWh mileage => 0.25 kWh/km => 0.00025 kWh/m (for meter-based matrices)
        default=0.00025,
        help=(
            "Energy use per distance unit (kWh / distance-unit). "
            "Default 0.00025 corresponds to mileage 4 km/kWh for meter-based distances."
        ),
    )
    parser.add_argument(
        "--ev-charge-rate-kwh-per-hour",
        type=float,
        default=120.0,
        help="Charging rate at depot (kWh/hour).",
    )
    parser.add_argument(
        "--ev-reserve-soc-kwh",
        type=float,
        default=0.0,
        help="Reserve SOC that must remain (kWh).",
    )
    parser.add_argument(
        "--ev-num",
        type=int,
        default=1,
        help="Number of EVs available in fleet.",
    )
    parser.add_argument(
        "--charging-stations-num",
        type=int,
        default=5,
        help="Number of charging stations sampled per instance.",
    )
    parser.add_argument(
        "--combined-details-csv",
        type=str,
        required=True,
        help=(
            "Combined details CSV (ID,type,lng,lat,...) that includes depot/customers/CPs. "
            "Training customer sampling and charging-station sampling are sourced from this file."
        ),
    )
    parser.add_argument(
        "--combined-dist-matrix-csv",
        type=str,
        required=True,
        help="Combined depot+customers+CP distance matrix CSV (e.g., 251x251).",
    )
    parser.add_argument(
        "--combined-time-matrix-csv",
        type=str,
        default=None,
        help="Combined depot+customers+CP travel-time matrix CSV (e.g., 251x251).",
    )
    parser.add_argument(
        "--customer-num",
        type=int,
        default=30,
        help="Number of customers sampled per generated instance from pool CSV.",
    )
    parser.add_argument(
        "--pool-vehicle-capacity",
        type=float,
        default=30.0,
        help="Vehicle capacity used for demand normalization in pool CSV mode.",
    )
    parser.add_argument(
        "--test-distance-matrix-csv",
        type=str,
        default=None,
        help="Optional distance matrix override for test-csv instance.",
    )
    parser.add_argument(
        "--test-time-matrix-csv",
        type=str,
        default=None,
        help="Optional travel-time matrix override for test-csv instance.",
    )
    return parser


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for CVRPTW training/testing."""
    return build_parser().parse_args(cli_args)

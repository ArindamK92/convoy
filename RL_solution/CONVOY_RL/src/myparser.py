"""CLI argument and CSV parsing helpers for convoy training/testing."""

import argparse
import csv

import torch


def parse_customer(csv_path: str):
    """Parse VRPTW CSV rows and return depot row + customer rows."""
    depot = None
    customers = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "is_depot",
            "x",
            "y",
            "demand",
            "tw_start",
            "tw_end",
            "service_time",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV must contain columns: "
                "is_depot,x,y,demand,tw_start,tw_end,service_time"
            )
        for row in reader:
            is_depot = int(row["is_depot"])
            reward = 0.0
            if "reward" in row and row["reward"] != "":
                reward = float(row["reward"])
            rec = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "demand": float(row["demand"]),
                "tw_start": float(row["tw_start"]),
                "tw_end": float(row["tw_end"]),
                "service_time": float(row["service_time"]),
                "reward": reward,
                "customer_id": (
                    int(row["customer_id"])
                    if "customer_id" in row and row["customer_id"] != ""
                    else None
                ),
            }
            if rec["tw_end"] <= rec["tw_start"]:
                raise ValueError("Each row must satisfy tw_end > tw_start.")
            if is_depot == 1:
                if depot is not None:
                    raise ValueError("CSV must contain exactly one depot row.")
                depot = rec
            else:
                customers.append(rec)
    if depot is None:
        raise ValueError("CSV must contain one row with is_depot=1.")
    if not customers:
        raise ValueError("CSV must contain at least one customer row (is_depot=0).")
    return depot, customers


def parse_distance_matrix_csv(matrix_csv_path: str) -> torch.Tensor:
    """Parse a square numeric distance matrix CSV into a torch tensor.

    Accepted shapes:
        - [N, N] for customer-to-customer only.
        - [N+1, N+1] where row/col 0 correspond to depot.
    """
    rows = []
    with open(matrix_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        raise ValueError("Distance matrix CSV is empty.")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("Distance matrix CSV must be rectangular.")
    if len(rows) != n:
        raise ValueError("Distance matrix CSV must be square.")
    return torch.tensor(rows, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CVRPTW training/testing."""
    parser = argparse.ArgumentParser(
        description="Train and test RL4CO on VRPTW (CVRPTW in rl4co)."
    )
    parser.add_argument("--num-loc", type=int, default=20, help="Number of customers.")
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
        help="Path to a custom single-instance VRPTW CSV for final testing.",
    )
    parser.add_argument(
        "--csv-vehicle-capacity",
        type=float,
        default=30.0,
        help="Vehicle capacity used to normalize CSV demands.",
    )
    parser.add_argument(
        "--distance-mode",
        type=str,
        default="manhattan",
        choices=["euclidean", "manhattan", "linear_sum"],
        help="Distance function for travel time and reward.",
    )
    parser.add_argument(
        "--ev-battery-capacity-kwh",
        type=float,
        default=60.0,
        help="EV battery capacity in kWh.",
    )
    parser.add_argument(
        "--ev-energy-rate-kwh-per-distance",
        type=float,
        default=0.5,
        help="Energy use per distance unit (kWh / distance-unit).",
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
        "--ev-num-vehicles",
        type=int,
        default=1,
        help="Number of EVs available in fleet.",
    )
    parser.add_argument(
        "--charging-pool-csv",
        type=str,
        default="CP_details.csv",
        help="Charging station pool CSV (x,y,charge_rate,charging_cost_per_kWh).",
    )
    parser.add_argument(
        "--charging-pool-sample-size",
        type=int,
        default=5,
        help="Number of charging stations sampled per instance.",
    )
    parser.add_argument(
        "--train-pool-csv",
        type=str,
        default=None,
        help="CSV customer pool used to sample instances during training.",
    )
    parser.add_argument(
        "--pool-sample-size",
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
        "--distance-matrix-csv",
        type=str,
        default=None,
        help="External distance matrix CSV for train-pool mode.",
    )
    parser.add_argument(
        "--time-matrix-csv",
        type=str,
        default=None,
        help="External travel-time matrix CSV for train-pool mode.",
    )
    parser.add_argument(
        "--test-distance-matrix-csv",
        type=str,
        default=None,
        help="External distance matrix CSV for test-csv instance.",
    )
    parser.add_argument(
        "--test-time-matrix-csv",
        type=str,
        default=None,
        help="External travel-time matrix CSV for test-csv instance.",
    )
    return parser.parse_args()

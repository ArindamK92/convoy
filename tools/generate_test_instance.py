#!/usr/bin/env python3
"""Generate a sampled CONVOY test-instance CSV from combined-details CSV.

Output schema matches `data/test_instance.csv`:
ID,type,lng,lat,first_receive_tm,last_receive_tm,service_time,reward,
unit_charging_cost,charge_rate_kwh_per_hour
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "ID",
    "type",
    "lng",
    "lat",
    "first_receive_tm",
    "last_receive_tm",
    "service_time",
    "reward",
    "unit_charging_cost",
    "charge_rate_kwh_per_hour",
]


def _normalize_type(value: object) -> str:
    """Normalize node type to lower-case short code."""
    return str(value).strip().lower()


def _to_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep output columns in fixed order."""
    return df.loc[:, REQUIRED_COLUMNS].copy()


def generate_test_instance_from_combined(
    combined_details_csv: Path,
    customer_num: int,
    charging_stations_num: int,
    output_path: Path | None = None,
    seed: int | None = None,
) -> Path:
    """Sample depot + customers + charging stations and write test-instance CSV.

    - Exactly one depot row (`type=d`) is required.
    - `customer_num` rows are sampled from customers (`type=c`).
    - `charging_stations_num` rows are sampled from charging points (`type=f`).
    - Output order: depot first, then sampled customers, then sampled CPs.
    """
    if customer_num <= 0:
        raise ValueError("--customer-num must be > 0.")
    if charging_stations_num < 0:
        raise ValueError("--charging-stations-num must be >= 0.")

    combined_details_csv = combined_details_csv.expanduser().resolve()
    if not combined_details_csv.is_file():
        raise FileNotFoundError(
            f"Combined details CSV not found: {combined_details_csv}"
        )

    df = pd.read_csv(combined_details_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in combined details CSV: {}".format(
                ", ".join(missing)
            )
        )

    types = df["type"].map(_normalize_type)
    depot_df = df.loc[types == "d"]
    customers_df = df.loc[types == "c"]
    cp_df = df.loc[types == "f"]

    if len(depot_df) != 1:
        raise ValueError(
            "Expected exactly 1 depot row (type=d), found {}.".format(len(depot_df))
        )
    if customer_num > len(customers_df):
        raise ValueError(
            "Requested {} customers, but only {} customer rows are available.".format(
                customer_num, len(customers_df)
            )
        )
    if charging_stations_num > len(cp_df):
        raise ValueError(
            "Requested {} charging stations, but only {} CP rows are available.".format(
                charging_stations_num, len(cp_df)
            )
        )

    rng = random.Random(seed)

    customer_indices = list(customers_df.index)
    selected_customer_idx = rng.sample(customer_indices, customer_num)
    sampled_customers = customers_df.loc[selected_customer_idx]

    cp_indices = list(cp_df.index)
    selected_cp_idx = rng.sample(cp_indices, charging_stations_num)
    sampled_cps = cp_df.loc[selected_cp_idx]

    out_df = pd.concat([depot_df, sampled_customers, sampled_cps], axis=0)
    out_df = _to_output_columns(out_df)

    if output_path is None:
        output_path = combined_details_csv.parent / "test_instance.csv"
    else:
        output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate test_instance.csv by sampling rows from combined details CSV."
        )
    )
    parser.add_argument(
        "--combined-details-csv",
        required=True,
        help="Path to combined details CSV (ID,type,lng,lat,...).",
    )
    parser.add_argument(
        "--customer-num",
        type=int,
        required=True,
        help="Number of customer rows (type=c) to sample.",
    )
    parser.add_argument(
        "--charging-stations-num",
        type=int,
        required=True,
        help="Number of charging-station rows (type=f) to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Output CSV path. Default: <combined-details-dir>/test_instance.csv "
            "(overwritten if exists)."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_path = generate_test_instance_from_combined(
        combined_details_csv=Path(args.combined_details_csv),
        customer_num=int(args.customer_num),
        charging_stations_num=int(args.charging_stations_num),
        output_path=Path(args.output_path) if args.output_path else None,
        seed=args.seed,
    )
    print(f"Generated test instance: {out_path}")


if __name__ == "__main__":
    main()

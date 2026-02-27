#!/usr/bin/env python3
"""Convert EVRP-TW-SPD-HMA TXT instances into CONVOY CSV datasets.

Given an input file such as `jd200_1.txt`, this tool creates:
- `combined_data_jd200_1.csv`
- `distance_matrix_jd200_1.csv`
- `time_matrix_jd200_1.csv`

Synthetic-field generation rules in the combined CSV:
- Customers (`type=c`) get random reward in [5, 20].
- Charging stations (`type=f`) get random unit charging cost in [0.4, 0.6].
- Depot (`type=d`) gets fixed unit charging cost = 0.3.
- Depot and charging stations (`type in {d, f}`) get random charge rate in [55, 150].
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _find_section_index(lines: Sequence[str], marker: str) -> int:
    """Return the line index for a section marker."""
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            return idx
    raise ValueError(f"Section marker '{marker}' not found.")


def _split_csv_line(line: str) -> List[str]:
    """Split one CSV-formatted line into values."""
    return next(csv.reader([line]))


def _to_int(value: str) -> int:
    """Parse integer-like text, accepting float-formatted integers."""
    text = value.strip()
    if text == "":
        raise ValueError("Encountered empty integer value.")
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def _to_float(value: str) -> float:
    """Parse float-like text."""
    text = value.strip()
    if text == "":
        raise ValueError("Encountered empty float value.")
    return float(text)


def _resolve_tw_columns(header: Sequence[str]) -> Tuple[str, str]:
    """Return source column names for TW start/end."""
    header_set = set(header)
    if "first_receive_tm" in header_set and "last_receive_tm" in header_set:
        return "first_receive_tm", "last_receive_tm"
    if "ready_time" in header_set and "due_date" in header_set:
        return "ready_time", "due_date"
    raise ValueError(
        "Could not find time-window columns. Expected either "
        "('first_receive_tm','last_receive_tm') or ('ready_time','due_date')."
    )


def _parse_instance_file(
    input_txt: Path,
) -> Tuple[List[Dict[str, str]], List[Tuple[int, int, float, float]], str, str]:
    """Parse node rows and distance/time arcs from EVRP TXT instance."""
    lines = input_txt.read_text(encoding="utf-8").splitlines()

    node_idx = _find_section_index(lines, "NODE_SECTION")
    dist_idx = _find_section_index(lines, "DISTANCETIME_SECTION")
    depot_idx = _find_section_index(lines, "DEPOT_SECTION")
    if not (node_idx < dist_idx < depot_idx):
        raise ValueError(
            "Unexpected section ordering. Expected NODE_SECTION -> DISTANCETIME_SECTION -> DEPOT_SECTION."
        )

    if node_idx + 1 >= len(lines):
        raise ValueError("Missing NODE_SECTION header row.")
    node_header = _split_csv_line(lines[node_idx + 1].strip())
    tw_start_col, tw_end_col = _resolve_tw_columns(node_header)

    required_node_cols = {"ID", "type", "lng", "lat", "service_time"}
    missing_node_cols = sorted(required_node_cols.difference(node_header))
    if missing_node_cols:
        raise ValueError(
            f"Missing required node columns in NODE_SECTION: {missing_node_cols}"
        )

    nodes: List[Dict[str, str]] = []
    for raw in lines[node_idx + 2 : dist_idx]:
        if not raw.strip():
            continue
        values = _split_csv_line(raw.strip())
        if len(values) != len(node_header):
            raise ValueError(f"Malformed node row: {raw}")
        nodes.append(dict(zip(node_header, values)))

    if not nodes:
        raise ValueError("No node rows found in NODE_SECTION.")

    if dist_idx + 1 >= len(lines):
        raise ValueError("Missing DISTANCETIME_SECTION header row.")
    dist_header = _split_csv_line(lines[dist_idx + 1].strip())
    required_dist_cols = {"from_node", "to_node", "distance", "spend_tm"}
    missing_dist_cols = sorted(required_dist_cols.difference(dist_header))
    if missing_dist_cols:
        raise ValueError(
            f"Missing required columns in DISTANCETIME_SECTION: {missing_dist_cols}"
        )
    dist_pos = {name: idx for idx, name in enumerate(dist_header)}

    arcs: List[Tuple[int, int, float, float]] = []
    for raw in lines[dist_idx + 2 : depot_idx]:
        if not raw.strip():
            continue
        values = _split_csv_line(raw.strip())
        if len(values) < len(dist_header):
            raise ValueError(f"Malformed distance/time row: {raw}")
        from_node = _to_int(values[dist_pos["from_node"]])
        to_node = _to_int(values[dist_pos["to_node"]])
        distance = _to_float(values[dist_pos["distance"]])
        spend_tm = _to_float(values[dist_pos["spend_tm"]])
        arcs.append((from_node, to_node, distance, spend_tm))

    if not arcs:
        raise ValueError("No arcs found in DISTANCETIME_SECTION.")

    return nodes, arcs, tw_start_col, tw_end_col


def _build_combined_rows(
    node_rows: Iterable[Dict[str, str]],
    tw_start_col: str,
    tw_end_col: str,
    rng: random.Random,
) -> List[Dict[str, object]]:
    """Build combined details rows with synthetic reward/cost/charge-rate fields."""
    out_rows: List[Dict[str, object]] = []
    for row in node_rows:
        node_id = _to_int(row["ID"])
        node_type = row["type"].strip().lower()
        if node_type not in {"d", "c", "f"}:
            raise ValueError(f"Unsupported node type '{node_type}' for ID={node_id}")

        if node_type == "c":
            reward = int(rng.randint(5, 20))
            unit_charging_cost = 0.0
            charge_rate_kwh_per_hour = 0
        elif node_type == "f":
            reward = 0
            unit_charging_cost = round(rng.uniform(0.4, 0.6), 2)
            charge_rate_kwh_per_hour = int(rng.randint(55, 150))
        else:  # depot
            reward = 0
            unit_charging_cost = 0.3
            charge_rate_kwh_per_hour = int(rng.randint(55, 150))

        out_rows.append(
            {
                "ID": node_id,
                "type": node_type,
                "lng": _to_float(row["lng"]),
                "lat": _to_float(row["lat"]),
                "first_receive_tm": _to_float(row[tw_start_col]),
                "last_receive_tm": _to_float(row[tw_end_col]),
                "service_time": _to_float(row["service_time"]),
                "reward": reward,
                "unit_charging_cost": unit_charging_cost,
                "charge_rate_kwh_per_hour": charge_rate_kwh_per_hour,
            }
        )
    return out_rows


def _build_matrices(
    node_ids: Sequence[int], arcs: Iterable[Tuple[int, int, float, float]]
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    """Build dense distance and time matrices keyed by original node IDs."""
    dist_matrix: Dict[int, Dict[int, float]] = {}
    time_matrix: Dict[int, Dict[int, float]] = {}
    node_set = set(node_ids)

    for src in node_ids:
        dist_matrix[src] = {}
        time_matrix[src] = {}
        for dst in node_ids:
            if src == dst:
                dist_matrix[src][dst] = 0.0
                time_matrix[src][dst] = 0.0
            else:
                dist_matrix[src][dst] = math.inf
                time_matrix[src][dst] = math.inf

    for src, dst, distance, spend_tm in arcs:
        if src not in node_set or dst not in node_set:
            continue
        dist_matrix[src][dst] = distance
        time_matrix[src][dst] = spend_tm

    missing_pairs = []
    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            if math.isinf(dist_matrix[src][dst]) or math.isinf(time_matrix[src][dst]):
                missing_pairs.append((src, dst))
                if len(missing_pairs) >= 10:
                    break
        if len(missing_pairs) >= 10:
            break

    if missing_pairs:
        preview = ", ".join(f"({a},{b})" for a, b in missing_pairs)
        raise ValueError(
            "Distance/time section does not fully cover all node pairs. "
            f"Examples: {preview}"
        )

    return dist_matrix, time_matrix


def _write_combined_csv(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write combined node-details CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(
    output_path: Path, node_ids: Sequence[int], matrix: Dict[int, Dict[int, float]]
) -> None:
    """Write matrix CSV with IDs in both header row and first column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *node_ids])
        for src in node_ids:
            writer.writerow([src, *[matrix[src][dst] for dst in node_ids]])


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for converter script."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert EVRP-TW-SPD-HMA TXT instance into CONVOY combined-data, "
            "distance-matrix, and time-matrix CSV files."
        )
    )
    parser.add_argument(
        "--input-txt",
        required=True,
        help=(
            "Path to EVRP-TW-SPD-HMA instance txt "
            "(e.g., EVRP-TW-SPD-HMA/data/jd_instances/jd200_1.txt)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help=(
            "Directory to write output CSVs. "
            "Defaults to CONVOY/data when run from repo root."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reward/cost/charge-rate generation.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    input_txt = Path(args.input_txt).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    stem = input_txt.stem

    rng = random.Random(args.seed)
    nodes, arcs, tw_start_col, tw_end_col = _parse_instance_file(input_txt)
    combined_rows = _build_combined_rows(nodes, tw_start_col, tw_end_col, rng)
    node_ids = [int(row["ID"]) for row in combined_rows]
    dist_matrix, time_matrix = _build_matrices(node_ids, arcs)

    combined_csv = output_dir / f"combined_data_{stem}.csv"
    dist_csv = output_dir / f"distance_matrix_{stem}.csv"
    time_csv = output_dir / f"time_matrix_{stem}.csv"

    _write_combined_csv(combined_csv, combined_rows)
    _write_matrix_csv(dist_csv, node_ids, dist_matrix)
    _write_matrix_csv(time_csv, node_ids, time_matrix)

    print(f"Input TXT: {input_txt}")
    print(f"Wrote combined details CSV: {combined_csv}")
    print(f"Wrote distance matrix CSV: {dist_csv}")
    print(f"Wrote time matrix CSV: {time_csv}")
    if args.seed is None:
        print("Random seed: system-random (non-deterministic)")
    else:
        print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


def create_run_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    destination = make_unique_path(root / run_name)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 1
    while True:
        if path.suffix:
            candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        else:
            candidate = path.with_name(f"{path.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized_rows = list(rows)
    if not materialized_rows:
        raise ValueError("Cannot write an empty CSV file.")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(materialized_rows)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def format_optional_distance(value: float | None) -> str:
    if value is None:
        return "not found on the current grid"
    return f"{value:.3f} m"

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from simulation.config import SimulationConfig
from simulation.utils import (
    create_run_output_dir,
    format_optional_distance,
    make_unique_path,
    save_json,
    write_csv_rows,
)


class SimulationConfigTests(unittest.TestCase):
    def test_from_mapping_normalizes_mode_and_exposes_target_radius(self) -> None:
        config = SimulationConfig.from_mapping({"sigma_w_mode": "CONSTANT", "d_target": 0.5})

        self.assertEqual(config.sigma_w_mode, "constant")
        self.assertAlmostEqual(config.target_radius, 0.25)

    def test_merge_applies_only_non_none_overrides(self) -> None:
        config = SimulationConfig()

        merged = config.merge({"L_max": 900.0, "dL": None})

        self.assertEqual(merged.L_max, 900.0)
        self.assertEqual(merged.dL, config.dL)

    def test_from_json_loads_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            payload = {"L_max": 123.0, "sigma_w_mode": "linear", "N": 10, "M": 2}
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = SimulationConfig.from_json(config_path)

        self.assertEqual(config.L_max, 123.0)
        self.assertEqual(config.N, 10)
        self.assertEqual(config.M, 2)

    def test_validate_rejects_invalid_parameters(self) -> None:
        invalid_payloads = [
            {"L_min": 10.0, "L_max": 5.0},
            {"dL": 0.0},
            {"N": 0},
            {"M": 0},
            {"d_target": 0.0},
            {"eta_min": 1.5},
            {"alpha": 0.0},
            {"sigma_w_mode": "quadratic"},
            {"sigma_w_value": -0.1},
            {"p_required": 1.5},
            {"d0": -0.1},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises((ValueError, TypeError)):
                    SimulationConfig.from_mapping(payload)


class UtilsTests(unittest.TestCase):
    def test_make_unique_path_adds_suffix_for_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "results.csv"
            existing.write_text("header\n", encoding="utf-8")

            candidate = make_unique_path(existing)

            self.assertEqual(candidate.name, "results_1.csv")

    def test_write_csv_rows_and_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "data.csv"
            json_path = root / "data.json"

            write_csv_rows(csv_path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            save_json(json_path, {"value": 42})

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            saved_json = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["a"], "1")
        self.assertEqual(rows[1]["b"], "4")
        self.assertEqual(saved_json["value"], 42)

    def test_write_csv_rows_rejects_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "empty.csv"
            with self.assertRaises(ValueError):
                write_csv_rows(csv_path, [])

    def test_create_run_output_dir_creates_distinct_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            first = create_run_output_dir(root)
            second = create_run_output_dir(root)
            first_exists = first.exists()
            second_exists = second.exists()
            different_paths = first != second

        self.assertTrue(first_exists)
        self.assertTrue(second_exists)
        self.assertTrue(different_paths)

    def test_format_optional_distance_formats_present_and_missing_values(self) -> None:
        self.assertEqual(format_optional_distance(None), "not found on the current grid")
        self.assertEqual(format_optional_distance(12.3456), "12.346 m")


if __name__ == "__main__":
    unittest.main()

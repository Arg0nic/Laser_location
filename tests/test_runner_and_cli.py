from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

os.environ.setdefault("MPLBACKEND", "Agg")

from simulation.config import SimulationConfig
from simulation.runner import export_results, run_simulation


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_run_simulation_respects_geometric_filter_and_operating_range(self) -> None:
        config = SimulationConfig(
            L_min=0.0,
            L_max=600.0,
            dL=50.0,
            N=10,
            M=3,
            sigma_w_value=0.0,
            sigma_w_slope=0.0,
            eta_min=0.0,
            A0=1.0,
            b=0.0,
            sigma_A=0.0,
            T=0.5,
            alpha=0.5,
            p_required=1.0,
            random_seed=123,
        )

        results = run_simulation(config)

        self.assertEqual(results.max_geometric_distance, 550.0)
        self.assertEqual(results.operating_distance, 550.0)
        self.assertFalse(results.geometric_valid[-1])
        assert_allclose(results.success_probabilities[results.geometric_valid], 1.0)
        assert_allclose(results.success_probabilities[~results.geometric_valid], 0.0)

    def test_run_simulation_is_reproducible_with_fixed_seed(self) -> None:
        config = SimulationConfig(
            L_min=0.0,
            L_max=200.0,
            dL=50.0,
            N=30,
            M=4,
            sigma_w_value=0.02,
            sigma_w_slope=0.0001,
            sigma_A=0.03,
            random_seed=321,
        )

        first = run_simulation(config)
        second = run_simulation(config)

        assert_allclose(first.success_probabilities, second.success_probabilities)
        assert_allclose(first.spot_diameters, second.spot_diameters)

    def test_export_results_writes_csv_json_and_plots(self) -> None:
        config = SimulationConfig(
            L_min=0.0,
            L_max=100.0,
            dL=50.0,
            N=5,
            M=2,
            sigma_w_value=0.0,
            sigma_w_slope=0.0,
            sigma_A=0.0,
            random_seed=123,
        )
        results = run_simulation(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifacts = export_results(results, config, output_dir, save_plots=True)

            self.assertTrue(artifacts["results_csv"].exists())
            self.assertTrue(artifacts["used_config_json"].exists())
            self.assertTrue(artifacts["spot_diameter_plot"].exists())
            self.assertTrue(artifacts["success_probability_plot"].exists())

            with artifacts["results_csv"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            saved_config = json.loads(artifacts["used_config_json"].read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 3)
        self.assertEqual(saved_config["random_seed"], 123)


class MainCliTests(unittest.TestCase):
    def test_main_cli_runs_with_temp_config_and_output_dir(self) -> None:
        payload = {
            "L_min": 0.0,
            "L_max": 100.0,
            "dL": 50.0,
            "N": 5,
            "M": 2,
            "theta_0": 0.0001,
            "d_target": 0.5,
            "eta_min": 0.0,
            "A0": 1.0,
            "b": 0.0,
            "sigma_A": 0.0,
            "T": 0.5,
            "alpha": 0.5,
            "sigma_w_value": 0.0,
            "sigma_w_slope": 0.0,
            "p_required": 1.0,
            "random_seed": 123,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            config_path = temp_root / "config.json"
            output_root = temp_root / "outputs"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_root),
                    "--no-plots",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            run_directories = [path for path in output_root.iterdir() if path.is_dir()]
            has_results_csv = (run_directories[0] / "results.csv").exists()
            has_used_config = (run_directories[0] / "used_config.json").exists()

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("Laser rangefinder simulation summary", completed.stdout)
        self.assertEqual(len(run_directories), 1)
        self.assertTrue(has_results_csv)
        self.assertTrue(has_used_config)


if __name__ == "__main__":
    unittest.main()

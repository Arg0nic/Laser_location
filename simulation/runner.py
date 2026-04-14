from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .geometry import divergence_at_distance, spot_diameter
from .monte_carlo import sigma_w_at_distance, simulate_detection_probability
from .utils import make_unique_path, save_json, write_csv_rows


@dataclass(frozen=True)
class SimulationResults:
    distances: np.ndarray
    theta_values: np.ndarray
    sigma_w_values: np.ndarray
    spot_diameters: np.ndarray
    spot_radii: np.ndarray
    geometric_valid: np.ndarray
    success_probabilities: np.ndarray
    max_geometric_distance: float | None
    operating_distance: float | None


def run_simulation(config: SimulationConfig) -> SimulationResults:
    rng = np.random.default_rng(config.random_seed)
    distances = np.arange(config.L_min, config.L_max + config.dL * 0.5, config.dL, dtype=float)
    distances = distances[distances <= config.L_max + 1.0e-12]

    theta_values = divergence_at_distance(distances, config.theta_0)
    spot_diameters = spot_diameter(distances, config.theta_0)
    spot_radii = spot_diameters / 2.0
    sigma_w_values = sigma_w_at_distance(distances, config)
    geometric_valid = spot_diameters <= config.d_target

    success_probabilities = np.zeros_like(distances, dtype=float)
    valid_indices = np.flatnonzero(geometric_valid)
    for index in valid_indices:
        success_probabilities[index] = simulate_detection_probability(
            distance=float(distances[index]),
            spot_radius=float(spot_radii[index]),
            config=config,
            rng=rng,
        )

    max_geometric_distance = float(distances[valid_indices[-1]]) if valid_indices.size else None
    operating_mask = success_probabilities >= config.p_required
    operating_indices = np.flatnonzero(operating_mask)
    operating_distance = float(distances[operating_indices[-1]]) if operating_indices.size else None

    return SimulationResults(
        distances=distances,
        theta_values=np.asarray(theta_values, dtype=float),
        sigma_w_values=np.asarray(sigma_w_values, dtype=float),
        spot_diameters=np.asarray(spot_diameters, dtype=float),
        spot_radii=np.asarray(spot_radii, dtype=float),
        geometric_valid=np.asarray(geometric_valid, dtype=bool),
        success_probabilities=success_probabilities,
        max_geometric_distance=max_geometric_distance,
        operating_distance=operating_distance,
    )


def export_results(
    results: SimulationResults,
    config: SimulationConfig,
    output_dir: Path,
    *,
    save_plots: bool = True,
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}

    csv_path = make_unique_path(output_dir / "results.csv")
    rows = []
    for index, distance in enumerate(results.distances):
        rows.append(
            {
                "distance_m": float(distance),
                "theta_rad": float(results.theta_values[index]),
                "sigma_w_m": float(results.sigma_w_values[index]),
                "spot_diameter_m": float(results.spot_diameters[index]),
                "spot_radius_m": float(results.spot_radii[index]),
                "geometric_valid": bool(results.geometric_valid[index]),
                "success_probability": float(results.success_probabilities[index]),
            }
        )
    write_csv_rows(csv_path, rows)
    artifacts["results_csv"] = csv_path

    config_path = make_unique_path(output_dir / "used_config.json")
    save_json(config_path, config.to_dict())
    artifacts["used_config_json"] = config_path

    if save_plots:
        spot_plot_path = make_unique_path(output_dir / "spot_diameter.png")
        probability_plot_path = make_unique_path(output_dir / "success_probability.png")
        save_spot_diameter_plot(results, config, spot_plot_path)
        save_success_probability_plot(results, config, probability_plot_path)
        artifacts["spot_diameter_plot"] = spot_plot_path
        artifacts["success_probability_plot"] = probability_plot_path

    return artifacts


def save_spot_diameter_plot(
    results: SimulationResults,
    config: SimulationConfig,
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results.distances, results.spot_diameters, color="#1f77b4", linewidth=2.0, label="d(L)")
    ax.axhline(
        y=config.d_target,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="d_target",
    )
    ax.set_title("Laser Spot Diameter vs Distance")
    ax.set_xlabel("Distance L, m")
    ax.set_ylabel("Spot diameter d(L), m")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_success_probability_plot(
    results: SimulationResults,
    config: SimulationConfig,
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results.distances,
        results.success_probabilities,
        color="#2ca02c",
        linewidth=2.0,
        label="p(L)",
    )
    ax.axhline(
        y=config.p_required,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="p_required",
    )
    ax.set_title("Success Probability vs Distance")
    ax.set_xlabel("Distance L, m")
    ax.set_ylabel("Success probability p(L)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)

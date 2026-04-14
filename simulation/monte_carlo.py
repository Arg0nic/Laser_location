from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig
from .geometry import circle_intersection_area
from .signal_model import final_ewma_amplitudes


def sigma_w_at_distance(distance: float | np.ndarray, config: SimulationConfig) -> float | np.ndarray:
    distances = np.asarray(distance, dtype=float)
    values = config.sigma_w_value + config.sigma_w_slope * distances
    values = np.maximum(values, 0.0)

    if np.isscalar(distance):
        return float(values)
    return values


def energy_fraction_on_target(
    spot_radius: float,
    target_radius: float,
    displacements: np.ndarray,
) -> np.ndarray:
    if spot_radius < 0 or target_radius < 0:
        raise ValueError("Radii must be non-negative.")

    if spot_radius == 0:
        return (displacements <= target_radius).astype(float)

    spot_area = math.pi * spot_radius * spot_radius
    overlaps = np.fromiter(
        (
            circle_intersection_area(spot_radius, target_radius, float(distance))
            for distance in displacements
        ),
        dtype=float,
        count=displacements.size,
    )
    fractions = overlaps / spot_area
    return np.clip(fractions, 0.0, 1.0)


def simulate_detection_probability(
    distance: float,
    spot_radius: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    sigma_w = sigma_w_at_distance(distance, config)
    x_samples = rng.normal(loc=0.0, scale=sigma_w, size=config.N)
    y_samples = rng.normal(loc=0.0, scale=sigma_w, size=config.N)
    displacements = np.hypot(x_samples, y_samples)

    eta_values = energy_fraction_on_target(spot_radius, config.target_radius, displacements)
    eta_pass_mask = eta_values >= config.eta_min

    successes = np.zeros(config.N, dtype=bool)
    if np.any(eta_pass_mask):
        ideal_amplitudes = config.A0 * eta_values[eta_pass_mask]
        filtered_amplitudes = final_ewma_amplitudes(
            ideal_amplitudes=ideal_amplitudes,
            pulse_count=config.M,
            bias=config.b,
            noise_sigma=config.sigma_A,
            alpha=config.alpha,
            rng=rng,
        )
        successes[eta_pass_mask] = filtered_amplitudes >= config.T

    return float(np.mean(successes))

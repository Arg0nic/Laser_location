from __future__ import annotations

import math
from typing import Final

import numpy as np


DISTANCE_BIN_M: Final[float] = 200.0
TURBULENCE_MULTIPLIER: Final[float] = 3.0


def divergence_at_distance(distance: float | np.ndarray, theta_0: float) -> float | np.ndarray:
    distances = np.asarray(distance, dtype=float)
    bin_index = np.floor(distances / DISTANCE_BIN_M)
    values = theta_0 * np.power(TURBULENCE_MULTIPLIER, bin_index)
    if np.isscalar(distance):
        return float(values)
    return values


def spot_diameter(distance: float | np.ndarray, theta_0: float) -> float | np.ndarray:
    distances = np.asarray(distance, dtype=float)
    values = divergence_at_distance(distances, theta_0) * distances
    if np.isscalar(distance):
        return float(values)
    return values


def circle_intersection_area(radius_a: float, radius_b: float, center_distance: float) -> float:
    if radius_a < 0 or radius_b < 0:
        raise ValueError("Circle radii must be non-negative.")

    if radius_a == 0 or radius_b == 0:
        return 0.0

    r1 = float(radius_a)
    r2 = float(radius_b)
    d = abs(float(center_distance))

    if d >= r1 + r2:
        return 0.0

    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2

    if d == 0:
        return math.pi * min(r1, r2) ** 2

    cos_arg_1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)
    cos_arg_2 = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)
    cos_arg_1 = float(np.clip(cos_arg_1, -1.0, 1.0))
    cos_arg_2 = float(np.clip(cos_arg_2, -1.0, 1.0))

    part_1 = r1 * r1 * math.acos(cos_arg_1)
    part_2 = r2 * r2 * math.acos(cos_arg_2)
    radicand = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    part_3 = 0.5 * math.sqrt(max(radicand, 0.0))
    return part_1 + part_2 - part_3

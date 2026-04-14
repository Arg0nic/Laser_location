from __future__ import annotations

import numpy as np


def final_ewma_amplitudes(
    ideal_amplitudes: np.ndarray,
    pulse_count: int,
    bias: float,
    noise_sigma: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if ideal_amplitudes.size == 0:
        return np.empty(0, dtype=float)

    noise = rng.normal(loc=0.0, scale=noise_sigma, size=(ideal_amplitudes.size, pulse_count))
    amplitudes = ideal_amplitudes[:, np.newaxis] + bias + noise

    filtered = amplitudes[:, 0].copy()
    for pulse_index in range(1, pulse_count):
        filtered = alpha * amplitudes[:, pulse_index] + (1.0 - alpha) * filtered

    return filtered

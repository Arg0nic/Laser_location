from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose

from simulation.config import SimulationConfig
from simulation.monte_carlo import (
    energy_fraction_on_target,
    sigma_w_at_distance,
    simulate_detection_probability,
)
from simulation.signal_model import final_ewma_amplitudes


class SignalModelTests(unittest.TestCase):
    def test_final_ewma_amplitudes_returns_empty_array_for_empty_input(self) -> None:
        rng = np.random.default_rng(123)

        result = final_ewma_amplitudes(
            ideal_amplitudes=np.array([], dtype=float),
            pulse_count=5,
            bias=0.0,
            noise_sigma=0.1,
            alpha=0.5,
            rng=rng,
        )

        self.assertEqual(result.size, 0)

    def test_final_ewma_amplitudes_is_constant_without_noise(self) -> None:
        rng = np.random.default_rng(123)

        result = final_ewma_amplitudes(
            ideal_amplitudes=np.array([1.0, 2.0]),
            pulse_count=4,
            bias=0.5,
            noise_sigma=0.0,
            alpha=0.35,
            rng=rng,
        )

        assert_allclose(result, np.array([1.5, 2.5]))


class MonteCarloTests(unittest.TestCase):
    def test_sigma_w_at_distance_uses_linear_model(self) -> None:
        config = SimulationConfig(sigma_w_value=0.1, sigma_w_slope=0.01)

        scalar_value = sigma_w_at_distance(50.0, config)
        linear_values = sigma_w_at_distance(np.array([0.0, 10.0, 20.0]), config)

        self.assertAlmostEqual(scalar_value, 0.6)
        assert_allclose(linear_values, np.array([0.1, 0.2, 0.3]))

    def test_sigma_w_at_distance_clamps_negative_linear_values_to_zero(self) -> None:
        config = SimulationConfig(sigma_w_value=0.1, sigma_w_slope=-0.01)

        values = sigma_w_at_distance(np.array([0.0, 20.0]), config)

        assert_allclose(values, np.array([0.1, 0.0]))

    def test_energy_fraction_on_target_handles_zero_radius_and_overlap_cases(self) -> None:
        zero_radius = energy_fraction_on_target(0.0, 0.25, np.array([0.1, 0.3]))
        fractions = energy_fraction_on_target(0.25, 0.25, np.array([0.0, 0.25, 1.0]))

        assert_allclose(zero_radius, np.array([1.0, 0.0]))
        self.assertAlmostEqual(fractions[0], 1.0)
        self.assertAlmostEqual(fractions[2], 0.0)
        self.assertGreater(fractions[1], 0.0)
        self.assertLess(fractions[1], 1.0)

    def test_energy_fraction_on_target_rejects_negative_radius(self) -> None:
        with self.assertRaises(ValueError):
            energy_fraction_on_target(-0.1, 0.25, np.array([0.0]))

    def test_simulate_detection_probability_returns_one_when_all_trials_pass(self) -> None:
        config = SimulationConfig(
            N=20,
            M=4,
            sigma_w_value=0.0,
            sigma_w_slope=0.0,
            eta_min=0.5,
            A0=1.0,
            b=0.0,
            sigma_A=0.0,
            T=0.5,
            alpha=0.4,
            random_seed=123,
        )

        probability = simulate_detection_probability(
            distance=100.0,
            spot_radius=0.1,
            config=config,
            rng=np.random.default_rng(config.random_seed),
        )

        self.assertEqual(probability, 1.0)

    def test_simulate_detection_probability_returns_zero_when_eta_threshold_fails(self) -> None:
        config = SimulationConfig(
            N=20,
            M=4,
            sigma_w_value=0.0,
            sigma_w_slope=0.0,
            eta_min=0.1,
            A0=1.0,
            b=0.0,
            sigma_A=0.0,
            T=0.1,
            alpha=0.4,
            random_seed=123,
        )

        probability = simulate_detection_probability(
            distance=100.0,
            spot_radius=1.0,
            config=config,
            rng=np.random.default_rng(config.random_seed),
        )

        self.assertEqual(probability, 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from simulation.geometry import circle_intersection_area, divergence_at_distance, spot_diameter


class GeometryTests(unittest.TestCase):
    def test_divergence_at_distance_supports_scalar_and_array_inputs(self) -> None:
        scalar_value = divergence_at_distance(250.0, 1.0e-4)
        array_values = divergence_at_distance(np.array([0.0, 199.9, 200.0, 400.0]), 1.0e-4)

        self.assertAlmostEqual(scalar_value, 3.0e-4)
        assert_allclose(array_values, np.array([1.0e-4, 1.0e-4, 3.0e-4, 9.0e-4]))

    def test_spot_diameter_uses_basic_model(self) -> None:
        value = spot_diameter(300.0, 1.0e-4)

        self.assertAlmostEqual(value, 0.09)

    def test_circle_intersection_area_handles_no_overlap(self) -> None:
        self.assertEqual(circle_intersection_area(1.0, 1.0, 2.1), 0.0)

    def test_circle_intersection_area_handles_full_inclusion(self) -> None:
        area = circle_intersection_area(2.0, 1.0, 0.5)

        self.assertAlmostEqual(area, math.pi * 1.0**2)

    def test_circle_intersection_area_handles_coincident_centers(self) -> None:
        area = circle_intersection_area(1.5, 1.5, 0.0)

        self.assertAlmostEqual(area, math.pi * 1.5**2)

    def test_circle_intersection_area_handles_partial_overlap(self) -> None:
        area = circle_intersection_area(1.0, 1.0, 1.0)
        expected = (2.0 * math.pi / 3.0) - (math.sqrt(3.0) / 2.0)

        self.assertAlmostEqual(area, expected, places=12)

    def test_circle_intersection_area_rejects_negative_radii(self) -> None:
        with self.assertRaises(ValueError):
            circle_intersection_area(-1.0, 1.0, 0.0)

    def test_circle_intersection_area_returns_zero_for_zero_radius(self) -> None:
        self.assertEqual(circle_intersection_area(0.0, 1.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()

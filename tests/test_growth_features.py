#!/usr/bin/env python3
"""
This script contains all unit tests of the growth features functions script
"""
import unittest
from tests._base import CellectsUnitTest
from cellects.video.growth_features import *


class TestFindGrowthFeatures(CellectsUnitTest):
    """Test suite for :func:`find_growth_features`."""

    def test_find_growth_features_happy_path_increasing(self):
        """Happy‑path – a simple increasing curve (no zeros)."""
        # a smooth exponential‑like growth; the exact regression values are not
        # asserted because they depend on the internal “window” search.
        y = np.exp(np.linspace(0, 1, 20)) * 10.0
        time_step = 0.5
        first_growth = 0.1

        features = find_growth_features(
            y=y,
            time_step=time_step,
            first_growth=first_growth,
            first_frame=1,
        )

        # All fields of the namedtuple must be present
        self.assertIsInstance(features, dict)

        # Numeric fields must be real numbers (not NaN) for a well‑behaved series
        numeric_attrs = [
            "exp_intercept",
            "exp_growth_rate_mm2s",
            "exp_start",
            "exp_end",
            "exp_r_squared",
            "lin_intercept",
            "lin_growth_rate_mm2s",
            "lin_start",
            "lin_end",
            "lin_r_squared",
        ]

        # Rupture information is *censored* because the synthetic data never
        # contain a clear slope‑shift.
        self.assertEqual(features['growth_rupture_time_min'], "censored")
        self.assertEqual(features['growth_rupture_surface_mm2'], "censored")

    def test_find_growth_features_sigmoid(self):
        """a smooth sigmoid‑like growth (no zeros)."""
        x = np.linspace(0, 15, 300)
        y = 1 / (1 + np.exp(-x + 6))
        time_step = 0.5
        first_growth = 0.1

        features = find_growth_features(
            y=y,
            time_step=time_step,
            first_growth=first_growth,
            first_frame=1,
        )

        # All fields of the namedtuple must be present
        self.assertIsInstance(features, dict)

        # Numeric fields must be real numbers (not NaN) for a well‑behaved series
        numeric_attrs = [
            "exp_intercept",
            "exp_growth_rate_mm2s",
            "exp_start",
            "exp_end",
            "exp_r_squared",
            "lin_intercept",
            "lin_growth_rate_mm2s",
            "lin_start",
            "lin_end",
            "lin_r_squared",
        ]
        for k, v in features.items():
            self.assertTrue(np.isfinite(v))

        # Rupture information is not *censored* because the synthetic data do contain a clear slope‑shift.
        self.assertTrue(isinstance(features['growth_rupture_time_min'], float))
        self.assertTrue(isinstance(features['growth_rupture_surface_mm2'], float))

    def test_find_growth_features_empty_input_returns_nan_features(self):
        """When the series is empty the function returns a fully‑NaN record."""
        y = np.array([])
        time_step = 1.0
        first_growth = 0.2

        features = find_growth_features(
            y=y,
            time_step=time_step,
            first_growth=first_growth,
            first_frame=1,
        )

        # Every numeric field must be NaN
        for k, v in features.items():
            self.assertTrue(pd.isna(v))

    def test_find_growth_features_leading_zero_removed(self):
        """A leading zero is discarded before any other processing."""
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # leading zero
        time_step = 1.0
        first_growth = 0.1

        # The same data without the leading zero should give identical output
        expected = find_growth_features(
            y=y[1:], time_step=time_step, first_growth=first_growth
        )
        actual = find_growth_features(
            y=y, time_step=time_step, first_growth=first_growth
        )
        self.assertEqual(actual, expected)

    def test_find_growth_features_interior_zeros_replaced(self):
        """Zeros inside the series are replaced by the smallest positive value."""
        y = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 4.0])
        time_step = 1.0
        first_growth = 0.1

        # Run the function; it should not raise and should return a valid record
        features = find_growth_features(
            y=y, time_step=time_step, first_growth=first_growth
        )
        # Verify that the returned object is a GrowthFeatures instance
        self.assertIsInstance(features, dict)

    def test_find_growth_features_all_zero_raises(self):
        """A vector consisting only of zeros triggers a ValueError."""
        y = np.array([0.0, 0.0, 0.0, 0.0])
        time_step = 1.0
        first_growth = 0.1

        with self.assertRaises(ValueError) as ctx:
            find_growth_features(y=y, time_step=time_step, first_growth=first_growth)
        self.assertIn("All values are zero", str(ctx.exception))

    def test_find_growth_features_invalid_first_frame_raises(self):
        """Providing a ``first_frame`` less than 1 raises ValueError."""
        y = np.array([1, 2, 3, 4])
        time_step = 1.0
        first_growth = 0.1

        with self.assertRaises(ValueError):
            find_growth_features(
                y=y,
                time_step=time_step,
                first_growth=first_growth,
                first_frame=0,          # illegal value
            )

    def test_find_growth_features_decreasing_curve(self):
        """When ``first_growth`` is negative the algorithm treats the series as decreasing."""
        # Create a smooth decreasing series
        y = np.linspace(10, 1, 15)
        time_step = 0.5
        first_growth = -0.2   # negative → decreasing mode

        features = find_growth_features(
            y=y,
            time_step=time_step,
            first_growth=first_growth,
            first_frame=1,
        )
        self.assertIsInstance(features, dict)

        # The exponential regression should still produce finite numbers
        self.assertTrue(np.isfinite(features['exp_intercept']))
        self.assertTrue(np.isfinite(features['exp_growth_rate_mm2s']))


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python3
"""
This script contains all unit tests of the filters script
"""
import unittest
from tests._base import CellectsUnitTest
from cellects.image.filters import (
    sharpen_filter,
    mexican_hat_filter,
    gaussian_filter,
    butterworth_filter,
    farid_filter,
    hessian_filter,
    laplace_filter,
    median_filter,
    meijering_filter,
    prewitt_filter,
    roberts_filter,
    scharr_filter,
    sobel_filter,
    sato_filter,
    frangi_filter,
    masked_vessel_filters,
)
import numpy as np


class TestSharpenFilter(CellectsUnitTest):
    """Unit tests for sharpen_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.ones((3, 3), dtype=np.uint8)
        cls.zero_image = np.zeros((3, 3), dtype=np.uint8)

    def test_constant_image_is_unchanged(self):
        """A constant image should remain unchanged after sharpening."""
        result = sharpen_filter(self.image)

        self.assertEqual(result.shape, self.image.shape)
        np.testing.assert_array_equal(result, self.image)

    def test_zero_image_is_zero(self):
        """A zero image should produce a zero image."""
        result = sharpen_filter(self.zero_image)

        self.assertEqual(result.shape, self.zero_image.shape)
        np.testing.assert_array_equal(result, 0)


class TestMexicanHatFilter(CellectsUnitTest):
    """Unit tests for mexican_hat_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.ones((3, 3), dtype=np.uint8)
        cls.zero_image = np.zeros((3, 3), dtype=np.uint8)

    def test_constant_image_is_zero(self):
        """The Mexican-hat kernel sums to zero, so a constant image maps to zero."""
        result = mexican_hat_filter(self.image)

        self.assertEqual(result.shape, self.image.shape)
        np.testing.assert_array_equal(result, 0)

    def test_zero_image_is_zero(self):
        """A zero image should produce a zero image."""
        result = mexican_hat_filter(self.zero_image)

        self.assertEqual(result.shape, self.zero_image.shape)
        np.testing.assert_array_equal(result, 0)


class TestGaussianFilter(CellectsUnitTest):
    """Unit tests for gaussian_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.uint8_255 = np.full((3, 3), 255, dtype=np.uint8)
        cls.float_2 = np.full((3, 3), 2.0, dtype=np.float64)

    def test_uint8_default_normalizes_to_one(self):
        """With preserve_range=False, uint8 255 should normalize to 1.0."""
        result = gaussian_filter(self.uint8_255, sigma=1.0)

        self.assertEqual(result.shape, self.uint8_255.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 1.0, atol=1e-15)

    def test_uint8_preserve_range_true(self):
        """With preserve_range=True, uint8 255 should remain 255.0."""
        result = gaussian_filter(
            self.uint8_255,
            sigma=1.0,
            preserve_range=True,
        )

        self.assertEqual(result.shape, self.uint8_255.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 255.0, atol=1e-15)

    def test_float_constant_is_unchanged(self):
        """A floating-point constant image should remain constant."""
        result = gaussian_filter(self.float_2, sigma=1.0)

        self.assertEqual(result.shape, self.float_2.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 2.0, atol=1e-15)

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            gaussian_filter(np.ones(3))


class TestButterworthFilter(CellectsUnitTest):
    """Unit tests for butterworth_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.ones((3, 3), dtype=np.float64)

    def test_low_pass_preserves_constant(self):
        """A constant image should be preserved by a low-pass Butterworth filter."""
        result = butterworth_filter(self.image, high_pass=False)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_high_pass_zeros_constant(self):
        """A constant image should be zeroed by a high-pass Butterworth filter."""
        result = butterworth_filter(self.image, high_pass=True)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_unsquared_low_pass_preserves_constant(self):
        """The unsquared low-pass response should also preserve a constant image."""
        result = butterworth_filter(
            self.image,
            high_pass=False,
            squared_butterworth=False,
        )

        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_npad_does_not_change_constant(self):
        """Edge padding should not change a constant image."""
        result = butterworth_filter(
            self.image,
            high_pass=False,
            npad=1,
        )

        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_invalid_cutoff_ratio_raises(self):
        """Invalid cutoff ratios should raise ValueError."""
        with self.assertRaises(ValueError):
            butterworth_filter(self.image, cutoff_frequency_ratio=0.0)

        with self.assertRaises(ValueError):
            butterworth_filter(self.image, cutoff_frequency_ratio=0.51)

    def test_invalid_order_raises(self):
        """A non-positive order should raise ValueError."""
        with self.assertRaises(ValueError):
            butterworth_filter(self.image, order=0.0)

    def test_invalid_npad_raises(self):
        """A negative pad size should raise ValueError."""
        with self.assertRaises(ValueError):
            butterworth_filter(self.image, npad=-1)

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            butterworth_filter(np.ones(3))


class TestEdgeFilters(CellectsUnitTest):
    """Unit tests for the linear edge-magnitude filters."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.constant = np.ones((3, 3), dtype=np.float64)
        cls.step = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )

    def _assert_edge_behavior(self, filter_fn):
        """Assert common invariants for edge-magnitude filters."""
        constant_result = filter_fn(self.constant)
        step_result = filter_fn(self.step)

        self.assertEqual(constant_result.shape, self.constant.shape)
        self.assertEqual(constant_result.dtype, np.float64)
        np.testing.assert_allclose(constant_result, 0.0, atol=1e-15)

        self.assertEqual(step_result.shape, self.step.shape)
        self.assertEqual(step_result.dtype, np.float64)
        np.testing.assert_allclose(step_result, np.abs(step_result))
        self.assertGreater(step_result.max(), 0.0)

    def test_sobel(self):
        """Sobel edge magnitude should behave as a non-negative edge detector."""
        self._assert_edge_behavior(sobel_filter)

    def test_prewitt(self):
        """Prewitt edge magnitude should behave as a non-negative edge detector."""
        self._assert_edge_behavior(prewitt_filter)

    def test_scharr(self):
        """Scharr edge magnitude should behave as a non-negative edge detector."""
        self._assert_edge_behavior(scharr_filter)

    def test_farid(self):
        """Farid edge magnitude should behave as a non-negative edge detector."""
        self._assert_edge_behavior(farid_filter)

    def test_roberts(self):
        """Roberts edge magnitude should behave as a non-negative edge detector."""
        self._assert_edge_behavior(roberts_filter)

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError for edge filters."""
        for filter_fn in (
            sobel_filter,
            prewitt_filter,
            scharr_filter,
            farid_filter,
            roberts_filter,
        ):
            with self.assertRaises(ValueError):
                filter_fn(np.ones(3))


class TestHessianFilter(CellectsUnitTest):
    """Unit tests for hessian_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.ones((3, 3), dtype=np.float64)

    def test_constant_image_returns_ones_for_black_ridges(self):
        """A constant image has no Hessian response, so background maps to 1.0."""
        result = hessian_filter(self.image, sigmas=[1.0], black_ridges=True)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 1.0, atol=1e-15)

    def test_constant_image_returns_ones_for_bright_ridges(self):
        """A constant image should still map to 1.0 when detecting bright ridges."""
        result = hessian_filter(self.image, sigmas=[1.0], black_ridges=False)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 1.0, atol=1e-15)

    def test_invalid_sigmas_raise(self):
        """Invalid sigma sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            hessian_filter(self.image, sigmas=[])

        with self.assertRaises(ValueError):
            hessian_filter(self.image, sigmas=[0.0])

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            hessian_filter(np.ones(3), sigmas=[1.0])


class TestLaplaceFilter(CellectsUnitTest):
    """Unit tests for laplace_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.constant = np.ones((5, 5), dtype=np.float64)

        cls.impulse = np.zeros((5, 5), dtype=np.float64)
        cls.impulse[2, 2] = 1.0

    def test_constant_image_is_zero(self):
        """The Laplacian of a constant image should be zero."""
        result = laplace_filter(self.constant, ksize=3)

        self.assertEqual(result.shape, self.constant.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_impulse_response(self):
        """A center impulse should produce the standard 3x3 Laplacian response."""
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, -4.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        result = laplace_filter(self.impulse, ksize=3)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_invalid_ksize_raises(self):
        """Invalid kernel sizes should raise ValueError."""
        with self.assertRaises(ValueError):
            laplace_filter(self.constant, ksize=0)

        with self.assertRaises(ValueError):
            laplace_filter(self.constant, ksize=2)

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            laplace_filter(np.ones(3), ksize=3)


class TestMedianFilter(CellectsUnitTest):
    """Unit tests for median_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()

        cls.single_zero_image = np.ones((5, 5), dtype=np.float64)
        cls.single_zero_image[2, 2] = 0.0

        cls.zero_image = np.zeros((5, 5), dtype=np.float64)

    def test_single_zero_is_removed(self):
        """A single isolated zero should be removed by a 3x3 median filter."""
        result = median_filter(self.single_zero_image)

        self.assertEqual(result.shape, self.single_zero_image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 1.0, atol=1e-15)

    def test_zero_image_is_zero(self):
        """A zero image should remain zero."""
        result = median_filter(self.zero_image)

        self.assertEqual(result.shape, self.zero_image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            median_filter(np.ones(3))


class TestMeijeringFilter(CellectsUnitTest):
    """Unit tests for meijering_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.ones((3, 3), dtype=np.float64)

    def test_constant_image_is_zero_for_black_ridges(self):
        """A constant image has no Hessian structure, so Meijering response is zero."""
        result = meijering_filter(self.image, sigmas=[1.0], black_ridges=True)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_constant_image_is_zero_for_bright_ridges(self):
        """A constant image should also produce zero response for bright ridges."""
        result = meijering_filter(self.image, sigmas=[1.0], black_ridges=False)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_output_range(self):
        """Meijering response should remain in [0, 1]."""
        image = np.linspace(0.0, 1.0, 9).reshape(3, 3)
        result = meijering_filter(image, sigmas=[1.0, 2.0])

        self.assertEqual(result.shape, image.shape)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0 + 1e-15)

    def test_invalid_sigmas_raise(self):
        """Invalid sigma sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            meijering_filter(self.image, sigmas=[])

        with self.assertRaises(ValueError):
            meijering_filter(self.image, sigmas=[0.0])

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            meijering_filter(np.ones(3), sigmas=[1.0])


class TestSatoFilter(CellectsUnitTest):
    """Unit tests for sato_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.array(
            [[191, 231, 173], [97, 51, 73], [242, 65, 40]],
            dtype=np.uint8,
        )
        cls.sigmas = [
            [1.0],
            [2.0],
            [1.0, 2.0],
            [0.5, 1.0, 2.0],
            [1.37, 2.81, 5.23],
        ]

        cls.references = [
            np.array(
                [
                    [20.064058841308515, 11.564236475501689, 0.10080762142557076],
                    [43.10925404424466, 43.4634635132972, 22.257883313720868],
                    [29.394271220244605, 47.02259675121556, 30.528831482583918],
                ]
            ),
            np.array(
                [
                    [3.4968313449163917, 6.963551201100796, 3.75368897492622],
                    [4.527677168208166, 8.027796084151312, 4.725218134439233],
                    [4.153895687174275, 7.88141881054044, 4.595374057670962],
                ]
            ),
            np.array(
                [
                    [20.064058841308515, 11.564236475501689, 3.75368897492622],
                    [43.10925404424466, 43.4634635132972, 22.257883313720868],
                    [29.394271220244605, 47.02259675121556, 30.528831482583918],
                ]
            ),
            np.array(
                [
                    [20.064058841308515, 11.564236475501689, 3.75368897492622],
                    [43.10925404424466, 43.4634635132972, 22.257883313720868],
                    [29.394271220244605, 47.02259675121556, 30.528831482583918],
                ]
            ),
            np.array(
                [
                    [12.245701954875065, 16.340585542501923, 7.693541788628687],
                    [24.85440764245779, 31.079710162355013, 19.490669264124474],
                    [19.05239365276853, 31.698144527094133, 21.130972644436397],
                ]
            ),
        ]

    def test_matches_reference_values(self):
        """Compare Sato outputs against the reference values from the example script."""
        for sigmas, expected in zip(self.sigmas, self.references):
            result = sato_filter(self.image, sigmas)

            self.assertEqual(result.shape, self.image.shape)
            self.assertEqual(result.dtype, np.float64)
            self.assertLess(np.abs(result - expected).max(), 1e-12)
            self.assertGreater(result.max(), 0.0)

    def test_zero_image_is_zero(self):
        """A zero image should produce a zero Sato response."""
        zero_image = np.zeros((3, 3), dtype=np.uint8)
        result = sato_filter(zero_image, [1.0])

        self.assertEqual(result.shape, zero_image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_invalid_sigmas_raise(self):
        """Invalid sigma sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            sato_filter(self.image, [])

        with self.assertRaises(ValueError):
            sato_filter(self.image, [0.0])

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            sato_filter(np.ones(3), [1.0])


class TestFrangiFilter(CellectsUnitTest):
    """Unit tests for frangi_filter()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.array(
            [[191, 231, 173], [97, 51, 73], [242, 65, 40]],
            dtype=np.uint8,
        )
        cls.sigmas = [
            [1.0],
            [2.0],
            [1.0, 2.0],
            [0.5, 1.0, 2.0],
            [1.37, 2.81, 5.23],
        ]

        cls.references = [
            np.array(
                [
                    [0.22929587584135777, 0.03609335656717714, 0.0],
                    [0.7532077275049168, 0.781042738270261, 0.30977595430229865],
                    [0.43743166112919624, 0.8403551956660499, 0.45728979925864677],
                ]
            ),
            np.array(
                [
                    [0.07455677359759498, 0.17788745182642302, 0.12395079374712724],
                    [0.2235025828989898, 0.3372922896194513, 0.29237434217922076],
                    [0.16033166548395297, 0.303358682249759, 0.2676312193205446],
                ]
            ),
            np.array(
                [
                    [0.22929587584135777, 0.03609335656717714, 0.0004080565277844023],
                    [0.7532077275049168, 0.781042738270261, 0.30977595430229865],
                    [0.43743166112919624, 0.8403551956660499, 0.45728979925864677],
                ]
            ),
            np.array(
                [
                    [0.6477118469771903, 0.1821490595931513, 0.015907980148264295],
                    [0.9153997719123184, 0.9497320296654271, 0.7973784157978395],
                    [0.7455310698356454, 0.9718856099641341, 0.7436897746641659],
                ]
            ),
            np.array(
                [
                    [0.15191699150051644, 0.167872165568678, 0.07088721956780543],
                    [0.6127128460771727, 0.7113040453657281, 0.4972607201745982],
                    [0.38551758198829056, 0.7162362376145737, 0.5468893797607679],
                ]
            ),
        ]

    def test_matches_reference_values(self):
        """Compare Frangi outputs against the reference values from the example script."""
        for sigmas, expected in zip(self.sigmas, self.references):
            result = frangi_filter(self.image, sigmas)

            self.assertEqual(result.shape, self.image.shape)
            self.assertEqual(result.dtype, np.float64)
            self.assertLess(np.abs(result - expected).max(), 1e-12)
            self.assertGreater(result.max(), 0.0)
            self.assertGreaterEqual(result.min(), 0.0)
            self.assertLessEqual(result.max(), 1.0 + 1e-12)

    def test_zero_image_is_zero(self):
        """A zero image should produce a zero Frangi response."""
        zero_image = np.zeros((3, 3), dtype=np.uint8)
        result = frangi_filter(zero_image, [1.0])

        self.assertEqual(result.shape, zero_image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_invalid_sigmas_raise(self):
        """Invalid sigma sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            frangi_filter(self.image, [])

        with self.assertRaises(ValueError):
            frangi_filter(self.image, [0.0])

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            frangi_filter(np.ones(3), [1.0])


class TestMaskedVesselFilters(CellectsUnitTest):
    """Unit tests for masked_vessel_filters()."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        super().setUpClass()
        cls.image = np.array(
            [[191, 231, 173], [97, 51, 73], [242, 65, 40]],
            dtype=np.uint8,
        )
        cls.full_mask = np.ones((3, 3), dtype=bool)

        cls.partial_mask = np.zeros((3, 3), dtype=bool)
        cls.partial_mask[1, 1] = True

    def test_full_mask_frangi_matches_unmasked_frangi(self):
        """With a full mask, masked Frangi should match unmasked Frangi."""
        expected = frangi_filter(self.image, [1.0])
        result = masked_vessel_filters(self.image, "Frangi", [1.0], self.full_mask)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_full_mask_sato_matches_unmasked_sato(self):
        """With a full mask, masked Sato should match unmasked Sato."""
        expected = sato_filter(self.image, [1.0])
        result = masked_vessel_filters(self.image, "Sato", [1.0], self.full_mask)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_partial_mask_zeros_outside_region(self):
        """Pixels outside the mask should be zero."""
        result = masked_vessel_filters(
            self.image,
            "Frangi",
            [1.0],
            self.partial_mask,
        )

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.all(result[~self.partial_mask] == 0.0))

    def test_invalid_filter_name_raises(self):
        """An unsupported filter name should raise ValueError."""
        with self.assertRaises(ValueError):
            masked_vessel_filters(self.image, "Laplace", [1.0], self.full_mask)

    def test_empty_mask_raises(self):
        """An all-False mask should raise ValueError."""
        empty_mask = np.zeros((3, 3), dtype=bool)

        with self.assertRaises(ValueError):
            masked_vessel_filters(self.image, "Frangi", [1.0], empty_mask)

    def test_shape_mismatch_raises(self):
        """Mismatched image/mask shapes should raise ValueError."""
        bad_mask = np.ones((2, 2), dtype=bool)

        with self.assertRaises(ValueError):
            masked_vessel_filters(self.image, "Frangi", [1.0], bad_mask)

    def test_invalid_sigmas_raise(self):
        """Invalid sigma sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            masked_vessel_filters(self.image, "Frangi", [], self.full_mask)

        with self.assertRaises(ValueError):
            masked_vessel_filters(self.image, "Frangi", [0.0], self.full_mask)

    def test_invalid_ndim_raises(self):
        """Non-2-D input should raise ValueError."""
        with self.assertRaises(ValueError):
            masked_vessel_filters(np.ones(3), "Frangi", [1.0], self.full_mask)


if __name__ == '__main__':
    unittest.main()

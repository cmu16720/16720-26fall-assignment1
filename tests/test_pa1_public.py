"""Public contract tests for PA1.

The release defaults to ``student/pa1.py``. Course infrastructure may set the
neutral ``PA1_IMPLEMENTATION_PATH`` environment variable to grade a copied
submission without editing this test file.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATH = Path(
    os.environ.get("PA1_IMPLEMENTATION_PATH", RELEASE_ROOT / "student" / "pa1.py")
)
SPEC = importlib.util.spec_from_file_location("pa1_public_candidate", IMPLEMENTATION_PATH)
assert SPEC is not None and SPEC.loader is not None
pa1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pa1)


def test_gradients_match_theory_patch() -> None:
    image = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 3, 3, 3, 3],
            [0, 0, 1, 3, 5, 5, 5],
            [0, 0, 1, 3, 5, 7, 7],
            [0, 0, 1, 3, 5, 7, 9],
        ],
        dtype=np.float32,
    )
    Ix, Iy = pa1.compute_image_gradients(image)
    points = np.array([[3, 3], [4, 3], [3, 4], [4, 4]])
    gradients = np.column_stack((Ix[points[:, 1], points[:, 0]], Iy[points[:, 1], points[:, 0]]))
    np.testing.assert_allclose(gradients, [[1, 1], [0, 2], [2, 0], [1, 1]])
    assert Ix.shape == image.shape and Iy.shape == image.shape
    assert Ix.dtype == np.float32 and Iy.dtype == np.float32


def test_second_moment_on_constant_gradient_fields() -> None:
    Ix = np.ones((9, 11), dtype=np.float32)
    Iy = np.full((9, 11), 2.0, dtype=np.float32)
    Sxx, Sxy, Syy = pa1.compute_second_moment(Ix, Iy, sigma=1.2)
    np.testing.assert_allclose(Sxx, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(Sxy, 2.0, atol=1.0e-6)
    np.testing.assert_allclose(Syy, 4.0, atol=1.0e-6)


def test_corner_response_uses_unmodified_harris_formula() -> None:
    Sxx = np.array([[6.0, 4.0]], dtype=np.float32)
    Sxy = np.array([[2.0, 0.0]], dtype=np.float32)
    Syy = np.array([[6.0, 0.0]], dtype=np.float32)
    response = pa1.compute_corner_response(Sxx, Sxy, Syy, k=0.04)
    np.testing.assert_allclose(response, [[26.24, -0.64]], atol=1.0e-5)


def test_nms_is_greedy_disk_ordered_and_bounded() -> None:
    response = np.zeros((20, 20), dtype=np.float32)
    response[5, 5] = 10.0
    response[5, 8] = 9.0  # Exactly radius 3 from the strongest point: suppressed.
    response[5, 14] = 8.0
    response[12, 12] = 4.0  # Below the relative cutoff.
    points, scores = pa1.nonmax_suppression(response, 0.5, radius=3, max_points=10)
    np.testing.assert_array_equal(points, [[5, 5], [14, 5]])
    np.testing.assert_allclose(scores, [10.0, 8.0])
    assert points.dtype == np.int64


def test_nms_flat_field_has_contract_empty_shapes() -> None:
    points, scores = pa1.nonmax_suppression(
        np.zeros((12, 12), dtype=np.float32), 0.01, radius=2, max_points=50
    )
    assert points.shape == (0, 2)
    assert scores.shape == (0,)


def test_descriptor_horizontal_field_has_only_bin_zero() -> None:
    Ix = np.ones((24, 24), dtype=np.float32)
    Iy = np.zeros_like(Ix)
    descriptor = pa1.describe_keypoints(
        Ix, Iy, np.array([[12, 12]]), patch_size=16, cells=4, bins=8
    )[0]
    expected = np.zeros(128, dtype=np.float32)
    expected[np.arange(0, 128, 8)] = 0.25
    np.testing.assert_allclose(descriptor, expected, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(descriptor), 1.0, atol=1.0e-6)


def test_descriptor_empty_and_flat_contracts() -> None:
    zeros = np.zeros((10, 10), dtype=np.float32)
    empty = pa1.describe_keypoints(zeros, zeros, np.empty((0, 2), dtype=np.int64), 16, 4, 8)
    assert empty.shape == (0, 128)
    flat = pa1.describe_keypoints(zeros, zeros, np.array([[0, 0]]), 16, 4, 8)
    np.testing.assert_array_equal(flat, np.zeros((1, 128), dtype=np.float32))


def test_matcher_ratio_and_confidence_order() -> None:
    desc1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    desc2 = np.array(
        [[1.0, 0.0], [0.85, 0.15], [0.0, 1.0], [0.2, 0.8]], dtype=np.float32
    )
    matches, confidence = pa1.match_descriptors(desc1, desc2, ratio=0.8)
    np.testing.assert_array_equal(matches, [[0, 0], [1, 2]])
    np.testing.assert_allclose(confidence, [1.0, 1.0], atol=1.0e-6)


@pytest.mark.parametrize(
    "shape1,shape2",
    [((0, 8), (4, 8)), ((3, 8), (0, 8)), ((3, 8), (1, 8))],
)
def test_matcher_valid_empty_cases(shape1: tuple[int, int], shape2: tuple[int, int]) -> None:
    matches, confidence = pa1.match_descriptors(
        np.zeros(shape1, dtype=np.float32), np.zeros(shape2, dtype=np.float32), 0.8
    )
    assert matches.shape == (0, 2)
    assert confidence.shape == (0,)


def test_nms_ties_use_y_then_x_and_ignore_nonfinite() -> None:
    response = np.ones((3, 4), dtype=np.float32)
    response[0, 0] = np.nan
    points, scores = pa1.nonmax_suppression(response, 1.0, radius=0, max_points=4)
    np.testing.assert_array_equal(points, [[1, 0], [2, 0], [3, 0], [0, 1]])
    np.testing.assert_array_equal(scores, np.ones(4))


def test_nms_border_and_exact_distance_contract() -> None:
    response = np.zeros((12, 12), dtype=np.float32)
    response[1, 1] = 20.0  # Removed by radius-two border.
    response[3, 3] = 10.0
    response[3, 5] = 9.0  # Exactly radius two: suppressed.
    response[3, 6] = 8.0  # Distance three: accepted.
    points, _ = pa1.nonmax_suppression(response, 0.1, radius=2, max_points=20)
    np.testing.assert_array_equal(points, [[3, 3], [6, 3]])

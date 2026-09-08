"""Supplied, non-graded utilities for 16-720 PA1.

The helpers in this file remove image, plotting, perturbation, geometry, and
video plumbing from the graded feature implementation.  Students may call and
read these functions, but should not edit them for their submission.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


SEED = 16720


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Seed Python, NumPy's legacy state, OpenCV, and return a modern RNG."""
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    cv2.setRNGSeed(seed_value)
    return np.random.default_rng(seed_value)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Return the normalized Gaussian kernel used by the assignment.

    The radius is ``ceil(3 * sigma)`` and ``sigma`` is measured in pixels.
    """
    if not np.isscalar(sigma) or not np.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError("sigma must be a positive finite scalar")
    radius = int(np.ceil(3.0 * float(sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _convolve_axis_reflect101(
    image: np.ndarray, kernel: np.ndarray, axis: int
) -> np.ndarray:
    radius = kernel.size // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    mode = "reflect" if image.shape[axis] > 1 else "edge"
    padded = np.pad(image, padding, mode=mode)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, kernel.size, axis=axis
    )
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a finite 2-D float array with the assignment Gaussian convention."""
    array = np.asarray(image)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("image must be a nonempty array with shape (H, W)")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("image must have a floating-point dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    kernel = gaussian_kernel1d(sigma)
    work = array.astype(np.float64, copy=False)
    horizontal = _convolve_axis_reflect101(work, kernel, axis=1)
    result = _convolve_axis_reflect101(horizontal, kernel, axis=0)
    return result.astype(np.float32)


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB image to finite float32 grayscale in [0, 1]."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError("a color image must have three RGB or four RGBA channels")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must be numeric")
    values = array.astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("image must contain only finite values")
    if np.issubdtype(array.dtype, np.integer):
        values /= float(np.iinfo(array.dtype).max)
    elif values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
        if float(values.min()) >= 0.0 and float(values.max()) <= 255.0:
            values /= 255.0
        else:
            raise ValueError("floating-point image values must lie in [0, 1] or [0, 255]")
    if values.ndim == 3:
        rgb = values[..., :3]
        values = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image as RGB uint8 or normalized grayscale float32."""
    source = Path(path)
    # Decode from bytes because cv2.imread cannot open non-ASCII paths on Windows.
    try:
        buffer = np.fromfile(source, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(f"could not read image: {source}") from error
    encoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED) if buffer.size else None
    if encoded is None:
        raise FileNotFoundError(f"could not read image: {source}")
    if encoded.ndim == 3:
        if encoded.shape[2] == 4:
            encoded = cv2.cvtColor(encoded, cv2.COLOR_BGRA2RGBA)
        else:
            encoded = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)
    if grayscale:
        return to_gray_float(encoded)
    return encoded


def save_rgb(path: str | Path, image: np.ndarray) -> Path:
    """Write a grayscale or RGB image, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        conversion = cv2.COLOR_RGB2BGR if array.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
        array = cv2.cvtColor(array, conversion)
    # Encode in memory because cv2.imwrite cannot open non-ASCII paths on Windows.
    try:
        ok, encoded = cv2.imencode(destination.suffix or ".png", array)
    except cv2.error as error:
        raise OSError(f"could not write image: {destination}") from error
    if not ok:
        raise OSError(f"could not write image: {destination}")
    encoded.tofile(destination)
    return destination


def plot_gradients_and_response(
    image: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    response: np.ndarray,
    keypoints: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create the standard B2 input/gradient/response/keypoint figure."""
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = (
        (image, "input", "gray"),
        (Ix, "$I_x$", "coolwarm"),
        (Iy, "$I_y$", "coolwarm"),
        (response, "Harris response", "coolwarm"),
    )
    for axis, (values, title, cmap) in zip(axes, panels):
        axis.imshow(values, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    if keypoints is not None and np.asarray(keypoints).size:
        points = np.asarray(keypoints)
        axes[0].scatter(
            points[:, 0], points[:, 1], s=36, facecolors="none", edgecolors="lime"
        )
    return figure, axes


def plot_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray | None = None,
    title: str = "keypoints",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay ``(x, y)`` keypoints on an image."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        figure = ax.figure
    ax.imshow(image, cmap="gray" if np.asarray(image).ndim == 2 else None)
    points = np.asarray(keypoints)
    if points.size:
        sizes = 38.0
        if scores is not None and np.asarray(scores).size:
            score_array = np.asarray(scores, dtype=np.float64)
            span = float(np.ptp(score_array))
            sizes = 24.0 + 50.0 * (
                (score_array - float(score_array.min())) / (span + 1.0e-12)
            )
        ax.scatter(
            points[:, 0], points[:, 1], s=sizes, facecolors="none", edgecolors="lime"
        )
    ax.set_title(title)
    ax.axis("off")
    return figure, ax


def _display_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        values = to_gray_float(array)
        return np.repeat(np.rint(values[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    if np.issubdtype(array.dtype, np.floating):
        return np.rint(np.clip(array[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    return array[..., :3].astype(np.uint8, copy=False)


def plot_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    confidence: np.ndarray | None = None,
    max_lines: int = 80,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot descriptor correspondences on a side-by-side canvas."""
    left = _display_uint8(image1)
    right = _display_uint8(image2)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    axis.imshow(canvas)
    pairs = np.asarray(matches, dtype=np.int64)
    first = np.asarray(keypoints1)
    second = np.asarray(keypoints2)
    count = min(int(max_lines), pairs.shape[0])
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, max(count, 1)))
    for line_index, (index1, index2) in enumerate(pairs[:count]):
        x1, y1 = first[index1]
        x2, y2 = second[index2]
        axis.plot(
            [x1, x2 + left.shape[1]],
            [y1, y2],
            color=colors[line_index],
            linewidth=1.0,
            alpha=0.8,
        )
    suffix = f" (showing {count}/{pairs.shape[0]})"
    if confidence is not None and np.asarray(confidence).size:
        suffix += f", best confidence {float(np.asarray(confidence)[0]):.3f}"
    axis.set_title("descriptor matches" + suffix)
    axis.axis("off")
    return figure, axis


def plot_descriptor_diagnostics(
    image: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    keypoint: Sequence[int],
    descriptor: np.ndarray,
    patch_size: int = 16,
    cells: int = 4,
    bins: int = 8,
) -> tuple[plt.Figure, np.ndarray]:
    """Visualize a descriptor patch, gradients, and every cell histogram."""
    x, y = (int(keypoint[0]), int(keypoint[1]))
    half = patch_size // 2
    gray = to_gray_float(image)
    pad_before = half
    pad_after = patch_size - half
    gray_pad = np.pad(gray, ((pad_before, pad_after), (pad_before, pad_after)))
    gx_pad = np.pad(Ix, ((pad_before, pad_after), (pad_before, pad_after)))
    gy_pad = np.pad(Iy, ((pad_before, pad_after), (pad_before, pad_after)))
    patch = gray_pad[y : y + patch_size, x : x + patch_size]
    gx = gx_pad[y : y + patch_size, x : x + patch_size]
    gy = gy_pad[y : y + patch_size, x : x + patch_size]
    magnitude = np.hypot(gx, gy)

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    patch_axis = figure.add_subplot(grid[0, 0])
    gradient_axis = figure.add_subplot(grid[0, 1])
    descriptor_axis = figure.add_subplot(grid[0, 2])
    histogram_axis = figure.add_subplot(grid[1, :])

    patch_axis.imshow(patch, cmap="gray")
    for boundary in range(0, patch_size + 1, patch_size // cells):
        patch_axis.axhline(boundary - 0.5, color="cyan", linewidth=0.7)
        patch_axis.axvline(boundary - 0.5, color="cyan", linewidth=0.7)
    patch_axis.scatter([half], [half], c="red", marker="+")
    patch_axis.set_title(f"patch centered near ({x}, {y})")

    gradient_axis.imshow(magnitude, cmap="magma")
    step = max(1, patch_size // 8)
    yy, xx = np.mgrid[0:patch_size:step, 0:patch_size:step]
    gradient_axis.quiver(
        xx,
        yy,
        gx[::step, ::step],
        gy[::step, ::step],
        color="cyan",
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    gradient_axis.set_title("gradient magnitude and orientation")

    vector = np.asarray(descriptor).reshape(cells, cells, bins)
    descriptor_axis.imshow(np.linalg.norm(vector, axis=2), cmap="viridis")
    descriptor_axis.set_title("cell histogram energy")
    descriptor_axis.set_xticks(range(cells))
    descriptor_axis.set_yticks(range(cells))

    flattened = vector.reshape(cells * cells, bins)
    offsets = np.arange(flattened.shape[0]) * (float(flattened.max()) + 1.0e-4)
    for index, histogram in enumerate(flattened):
        histogram_axis.plot(np.arange(bins), histogram + offsets[index], linewidth=1.2)
    histogram_axis.set_title("orientation histogram for each row-major cell")
    histogram_axis.set_xlabel("orientation bin")
    histogram_axis.set_yticks(offsets)
    histogram_axis.set_yticklabels([f"cell {index}" for index in range(flattened.shape[0])])
    return figure, np.asarray(
        [patch_axis, gradient_axis, descriptor_axis, histogram_axis], dtype=object
    )


def add_brightness(image: np.ndarray, offset: float) -> np.ndarray:
    """Apply a clipped additive brightness offset to a normalized image."""
    return np.clip(np.asarray(image, dtype=np.float32) + float(offset), 0.0, 1.0)


def add_gaussian_noise(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply deterministic Gaussian noise using the caller's generator."""
    if float(sigma) < 0.0:
        raise ValueError("sigma must be nonnegative")
    noise = rng.normal(0.0, float(sigma), size=np.asarray(image).shape)
    return np.clip(np.asarray(image, dtype=np.float32) + noise, 0.0, 1.0).astype(
        np.float32
    )


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur grayscale or RGB normalized data channel by channel."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return gaussian_blur(array, sigma)
    if array.ndim != 3:
        raise ValueError("image must be grayscale or color")
    return np.stack(
        [gaussian_blur(array[..., channel], sigma) for channel in range(array.shape[2])],
        axis=2,
    )


def rotate_and_scale(
    image: np.ndarray,
    angle_degrees: float,
    scale: float,
    border_value: float | Sequence[float] = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a centered similarity warp and return image plus 2x3 transform."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be grayscale or color")
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    height, width = array.shape[:2]
    transform = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        float(angle_degrees),
        float(scale),
    )
    warped = cv2.warpAffine(
        array,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return warped, transform.astype(np.float64)


def filter_keypoints_in_roi(
    keypoints: np.ndarray, roi: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return points and original indices inside ``(x, y, width, height)``."""
    points = np.asarray(keypoints)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape (N, 2)")
    if len(roi) != 4:
        raise ValueError("roi must be (x, y, width, height)")
    x, y, width, height = (float(value) for value in roi)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("roi width and height must be positive")
    mask = (
        (points[:, 0] >= x)
        & (points[:, 0] < x + width)
        & (points[:, 1] >= y)
        & (points[:, 1] < y + height)
    )
    indices = np.flatnonzero(mask)
    return points[indices], indices


def estimate_affine_partial(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    threshold: float = 3.0,
    seed: int = SEED,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Estimate the supplied current-to-reference similarity transform.

    This is the intentionally supplied geometry black box for PA1.
    """
    current = np.asarray(current_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    if current.ndim != 2 or current.shape[1:] != (2,) or reference.shape != current.shape:
        raise ValueError("point arrays must have matching shape (K, 2)")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(reference)):
        raise ValueError("point arrays must be finite")
    if not np.isfinite(threshold) or float(threshold) <= 0.0:
        raise ValueError("threshold must be positive")
    empty_mask = np.zeros((current.shape[0],), dtype=bool)
    if current.shape[0] < 4:
        return None, empty_mask, {"status": "too_few_matches", "inliers": 0, "ratio": 0.0}
    cv2.setRNGSeed(int(seed))
    transform, mask = cv2.estimateAffinePartial2D(
        current,
        reference,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(threshold),
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None or mask is None or not np.all(np.isfinite(transform)):
        return None, empty_mask, {"status": "estimator_failed", "inliers": 0, "ratio": 0.0}
    inliers = mask.ravel().astype(bool)
    count = int(np.count_nonzero(inliers))
    if count < 3:
        return None, inliers, {
            "status": "too_few_inliers",
            "inliers": count,
            "ratio": count / current.shape[0],
        }
    return transform.astype(np.float64), inliers, {
        "status": "ok",
        "inliers": count,
        "ratio": count / current.shape[0],
    }


def smooth_affine_partial(
    previous: np.ndarray | None, current: np.ndarray, current_weight: float = 0.8
) -> np.ndarray:
    """Smooth the four similarity parameters without leaving the model family."""
    if previous is None:
        return np.asarray(current, dtype=np.float64).copy()
    weight = float(current_weight)
    if not 0.0 < weight <= 1.0:
        raise ValueError("current_weight must lie in (0, 1]")
    old = np.asarray(previous, dtype=np.float64)
    new = np.asarray(current, dtype=np.float64)
    if old.shape != (2, 3) or new.shape != (2, 3):
        raise ValueError("affine transforms must have shape (2, 3)")
    old_parameters = np.array([old[0, 0], old[1, 0], old[0, 2], old[1, 2]])
    new_parameters = np.array([new[0, 0], new[1, 0], new[0, 2], new[1, 2]])
    a, b, tx, ty = (1.0 - weight) * old_parameters + weight * new_parameters
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def warp_frame_to_reference(
    frame: np.ndarray, transform: np.ndarray, output_shape: Sequence[int]
) -> np.ndarray:
    """Warp a current frame into reference coordinates using a 2x3 transform."""
    height, width = (int(output_shape[0]), int(output_shape[1]))
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite (2, 3) array")
    return cv2.warpAffine(
        np.asarray(frame),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def annotate_tracking_status(frame: np.ndarray, status: str) -> np.ndarray:
    """Add a visible failure/hold label while leaving valid frames unchanged."""
    output = _display_uint8(frame).copy()
    if status == "ok" or status == "reference":
        return output
    if status == "held_last_transform":
        label = "TRACKING HELD"
        color = (255, 170, 0)
    else:
        label = "TRACKING FAILURE"
        color = (255, 35, 35)
    cv2.rectangle(output, (8, output.shape[0] - 48), (245, output.shape[0] - 8), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (16, output.shape[0] - 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def object_lock_frames(
    frames: Sequence[np.ndarray],
    detect_and_describe: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    matcher: Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    roi: Sequence[float],
    ratio: float = 0.8,
    ransac_threshold: float = 3.0,
    hold_failures: int = 5,
    smoothing_weight: float = 0.8,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Run the supplied first-frame-reference object-locking video harness.

    ``detect_and_describe`` must use the student's detector and descriptor.
    ``matcher`` must be the student's matcher.  Only geometric estimation,
    warping, light smoothing, and failure bookkeeping are supplied here.
    """
    if not frames:
        return [], []
    if int(hold_failures) < 0:
        raise ValueError("hold_failures must be nonnegative")
    reference_frame = np.asarray(frames[0])
    height, width = reference_frame.shape[:2]
    reference_keypoints_all, reference_descriptors_all = detect_and_describe(reference_frame)
    reference_keypoints, roi_indices = filter_keypoints_in_roi(reference_keypoints_all, roi)
    reference_descriptors = np.asarray(reference_descriptors_all)[roi_indices]
    if reference_keypoints.shape[0] < 4:
        raise ValueError("the first-frame ROI must contain at least four described keypoints")

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    last_transform: np.ndarray | None = identity
    failed_run = 0
    outputs = [reference_frame.copy()]
    diagnostics: list[dict[str, Any]] = [
        {
            "frame": 0,
            "status": "reference",
            "matches": 0,
            "inliers": 0,
            "inlier_fraction": 0.0,
            "held": False,
        }
    ]

    for frame_index, frame in enumerate(frames[1:], start=1):
        current_keypoints, current_descriptors = detect_and_describe(np.asarray(frame))
        matches, confidence = matcher(reference_descriptors, current_descriptors, ratio)
        matches = np.asarray(matches, dtype=np.int64)
        current_points = (
            np.asarray(current_keypoints)[matches[:, 1]]
            if matches.size
            else np.empty((0, 2), dtype=np.float64)
        )
        reference_points = (
            reference_keypoints[matches[:, 0]]
            if matches.size
            else np.empty((0, 2), dtype=np.float64)
        )
        transform, inliers, info = estimate_affine_partial(
            current_points,
            reference_points,
            threshold=ransac_threshold,
            seed=SEED + frame_index,
        )
        held = False
        if transform is not None:
            failed_run = 0
            last_transform = smooth_affine_partial(
                last_transform, transform, current_weight=smoothing_weight
            )
            status = "ok"
        else:
            failed_run += 1
            if last_transform is not None and failed_run <= int(hold_failures):
                held = True
                status = "held_last_transform"
            else:
                last_transform = identity
                status = "unwarped_after_failure"
        warped = warp_frame_to_reference(frame, last_transform, (height, width))
        outputs.append(annotate_tracking_status(warped, status))
        diagnostics.append(
            {
                "frame": frame_index,
                "status": status,
                "matches": int(matches.shape[0]),
                "inliers": int(np.count_nonzero(inliers)),
                "inlier_fraction": float(info.get("ratio", 0.0)),
                "best_match_confidence": float(confidence[0]) if len(confidence) else 0.0,
                "held": held,
            }
        )
    return outputs, diagnostics


def read_video(
    path: str | Path,
    target_fps: float = 15.0,
    max_width: int = 640,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Decode an RGB frame list with deterministic temporal sampling/resizing."""
    if not np.isfinite(target_fps) or float(target_fps) <= 0:
        raise ValueError("target_fps must be positive and finite")
    if not isinstance(max_width, (int, np.integer)) or max_width <= 0:
        raise ValueError("max_width must be a positive integer")
    if max_frames is not None and (not isinstance(max_frames, (int, np.integer)) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer or None")
    source = Path(path)
    scratch: Path | None = None
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened() and source.is_file():
        # cv2.VideoCapture cannot open non-ASCII paths on Windows; retry a copy.
        capture.release()
        with tempfile.NamedTemporaryFile(suffix=source.suffix or ".mp4", delete=False) as handle:
            scratch = Path(handle.name)
        shutil.copyfile(source, scratch)
        capture = cv2.VideoCapture(str(scratch))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            source_fps = float(target_fps)
        output_fps = min(source_fps, float(target_fps))
        # Sample the nearest source frame at each output timestamp. Alternating
        # strides handle rates such as 20/24/25 fps without exceeding target_fps.
        source_frames_per_output = source_fps / output_fps
        frames: list[np.ndarray] = []
        source_index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if source_index >= int(np.floor(len(frames) * source_frames_per_output + 0.5)):
                height, width = bgr.shape[:2]
                if width > int(max_width):
                    scale = int(max_width) / width
                    bgr = cv2.resize(
                        bgr,
                        (int(max_width), int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= int(max_frames):
                    break
            source_index += 1
    finally:
        capture.release()
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    if not frames:
        raise ValueError("video contains no decodable frames")
    return frames, output_fps


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float,
) -> tuple[Path, str]:
    """Encode same-sized RGB frames; prefer H.264 and report the used codec."""
    frame_list = [np.asarray(frame) for frame in frames]
    if not frame_list:
        raise ValueError("at least one frame is required")
    height, width = frame_list[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frame_list):
        raise ValueError("all video frames must have the same spatial shape")
    if not np.isfinite(fps) or float(fps) <= 0.0:
        raise ValueError("fps must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workpath = destination
    if not str(destination).isascii():
        # cv2.VideoWriter cannot open non-ASCII paths on Windows; encode to a scratch file.
        with tempfile.NamedTemporaryFile(suffix=destination.suffix or ".mp4", delete=False) as handle:
            workpath = Path(handle.name)
    writer = None
    codec_used = ""
    for codec in ("avc1", "H264", "mp4v"):
        candidate = cv2.VideoWriter(
            str(workpath),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            codec_used = codec
            break
        candidate.release()
    if writer is None:
        if workpath != destination:
            workpath.unlink(missing_ok=True)
        raise OSError("no supported MP4 video encoder is available")
    for frame in frame_list:
        values = frame
        if np.issubdtype(values.dtype, np.floating):
            values = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
        if values.ndim == 2:
            values = np.repeat(values[..., None], 3, axis=2)
        writer.write(cv2.cvtColor(values[..., :3], cv2.COLOR_RGB2BGR))
    writer.release()
    if workpath != destination:
        shutil.move(str(workpath), str(destination))
    return destination, codec_used


def transcode_h264(
    input_path: str | Path,
    output_path: str | Path,
    crf: int = 20,
    max_width: int = 1280,
    max_height: int = 720,
) -> Path:
    """Transcode an intermediate MP4 to H.264/yuv420p using supplied ffmpeg.

    Prefer a system executable and otherwise use the pinned imageio-ffmpeg
    binary. The command strips audio and preserves the input frame rate.
    """
    executable = shutil.which("ffmpeg")
    if executable is None:
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "ffmpeg is required for H.264 export; install imageio-ffmpeg"
            ) from exc
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer in [0, 51]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = (
        f"scale=w='min({int(max_width)},iw)':h='min({int(max_height)},ih)':"
        "force_original_aspect_ratio=decrease,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    command = [
        executable,
        "-y",
        "-i",
        str(source),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    # ffmpeg's console output is not locale-encoded; decode permissively so a
    # Windows code page cannot crash the capture thread.
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if completed.returncode != 0:
        message = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
        raise RuntimeError(f"ffmpeg H.264 export failed: {message}")
    if destination.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("H.264 output exceeds 50 MB; increase CRF or shorten the clip")
    return destination


def side_by_side_frames(
    original: Sequence[np.ndarray], stabilized: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Compose labeled synchronized original/result RGB Hero Result frames."""
    if len(original) != len(stabilized):
        raise ValueError("video frame lists must have equal length")
    result: list[np.ndarray] = []
    for before, after in zip(original, stabilized):
        left = _display_uint8(before).copy()
        right = _display_uint8(after).copy()
        if left.shape[0] != right.shape[0]:
            raise ValueError("paired frames must have the same height")
        for panel, label in ((left, "ORIGINAL"), (right, "OBJECT-LOCKED")):
            cv2.rectangle(panel, (8, 8), (190, 42), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (16, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        result.append(np.concatenate([left, right], axis=1))
    return result

"""Generate deterministic, redistributable PA1 smoke-test assets.

The generated images and video are synthetic and are intentionally not stored
in Git.  They support notebook smoke tests.  The video may satisfy the capture
requirement only when course staff approve an alternative for access, privacy,
safety, or accessibility reasons.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skimage_data


SEED = 16720


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Encode in memory because cv2.imwrite cannot open non-ASCII paths on Windows.
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"could not write {path}")
    encoded.tofile(path)


def make_square(size: int = 128) -> np.ndarray:
    """Return the B2 white-square-on-black sanity image."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    margin = size // 4
    image[margin : size - margin, margin : size - margin] = 255
    return image


def make_texture(height: int = 360, width: int = 480, seed: int = SEED) -> np.ndarray:
    """Create a repeatable, corner-rich RGB scene with a distinctive target."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.empty((height, width, 3), dtype=np.float64)
    base[..., 0] = 40 + 35 * xx / max(width - 1, 1)
    base[..., 1] = 55 + 45 * yy / max(height - 1, 1)
    base[..., 2] = 75 + 25 * (xx + yy) / max(width + height - 2, 1)
    noise = rng.normal(0.0, 8.0, base.shape[:2])[..., None]
    image = np.clip(base + noise, 0, 255).astype(np.uint8)
    for index in range(45):
        x0 = int(rng.integers(0, width - 18))
        y0 = int(rng.integers(0, height - 18))
        x1 = min(width - 1, x0 + int(rng.integers(8, 48)))
        y1 = min(height - 1, y0 + int(rng.integers(8, 48)))
        color = tuple(int(value) for value in rng.integers(30, 235, size=3))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, int(rng.choice((1, 2, -1))))
    target_x0, target_y0 = width // 4, height // 4
    target_x1, target_y1 = 3 * width // 4, 3 * height // 4
    cv2.rectangle(image, (target_x0, target_y0), (target_x1, target_y1), (245, 232, 190), -1)
    cv2.rectangle(image, (target_x0, target_y0), (target_x1, target_y1), (15, 25, 55), 5)
    cell = 18
    for y in range(target_y0 + 8, target_y1 - 8, cell):
        for x in range(target_x0 + 8, target_x1 - 8, cell):
            parity = ((x - target_x0) // cell + (y - target_y0) // cell) % 2
            color = (32, 76, 150) if parity else (225, 88, 55)
            cv2.circle(image, (x, y), 5 if parity else 3, color, -1)
    cv2.putText(
        image,
        "16-720",
        (target_x0 + 24, (target_y0 + target_y1) // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    return image


def make_easy_match_image(reference: np.ndarray) -> np.ndarray:
    """Return a known mild similarity-warped matching image."""
    height, width = reference.shape[:2]
    transform = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 8.0, 0.9)
    transform[:, 2] += np.array([12.0, -7.0])
    warped = cv2.warpAffine(
        reference,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped = np.clip(warped.astype(np.int16) + 14, 0, 255).astype(np.uint8)
    return warped


def make_smoke_video(path: Path, reference: np.ndarray, fps: int = 15) -> None:
    """Write a 12-second synthetic motion clip for infrastructure testing."""
    height, width = reference.shape[:2]
    destination = Path(path)
    workpath = destination
    if not str(destination).isascii():
        # cv2.VideoWriter cannot open non-ASCII paths on Windows; encode to a scratch file.
        with tempfile.NamedTemporaryFile(suffix=destination.suffix or ".mp4", delete=False) as handle:
            workpath = Path(handle.name)
    writer = cv2.VideoWriter(
        str(workpath), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        if workpath != destination:
            workpath.unlink(missing_ok=True)
        raise OSError("the local OpenCV build cannot encode the smoke video")
    for frame_index in range(12 * fps):
        phase = 2.0 * np.pi * frame_index / (12 * fps)
        angle = 6.0 * np.sin(phase)
        scale = 1.0 + 0.07 * np.sin(2.0 * phase)
        transform = cv2.getRotationMatrix2D(
            ((width - 1) / 2, (height - 1) / 2), angle, scale
        )
        transform[:, 2] += np.array([28.0 * np.sin(phase), 16.0 * np.cos(phase)])
        frame = cv2.warpAffine(
            reference,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    if workpath != destination:
        shutil.move(str(workpath), str(destination))


def setup_assets(output_dir: Path) -> list[Path]:
    """Generate and return the paths to all notebook assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    square = make_square()
    flat = np.zeros_like(square)
    texture = make_texture()
    easy = make_easy_match_image(texture)
    images = {
        "square.png": square,
        "flat.png": flat,
        "texture.png": texture,
        "photo_texture.png": skimage_data.coffee(),
        "match_easy.png": easy,
    }
    generated = []
    for filename, image in images.items():
        path = output_dir / filename
        _write_png(path, image)
        generated.append(path)
    video_path = output_dir / "smoke_object_motion.mp4"
    make_smoke_video(video_path, texture)
    generated.append(video_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    arguments = parser.parse_args()
    generated = setup_assets(arguments.output)
    print(f"generated {len(generated)} assets in {arguments.output}")


if __name__ == "__main__":
    main()

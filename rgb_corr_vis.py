#!/usr/bin/env python3
"""
Visualize MMGNet RGB corruptions in one row:

    Raw RGB | Blur | Cutout | Brightness | Contrast

The corruption definitions match inference_multimodal_scene.py:
- Blur kernels:       [7, 13, 21, 33, 45]
- Cutout ratios:      [0.10, 0.20, 0.30, 0.40, 0.50]
- Cutout patch size:  64
- Brightness/contrast ratios:
                      [1.25, 1.50, 1.75, 2.00, 2.50]
  with a deterministic random choice between ratio and its reciprocal.

Example:
    python visualize_rgb_corruptions.py \
        --image /path/to/rgb.png \
        --output rgb_corruptions_s3.png \
        --severity 3 \
        --seed 0
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


BLUR_KERNELS = [7, 13, 21, 33, 45]
CUTOUT_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
ENHANCE_RATIOS = [1.25, 1.50, 1.75, 2.00, 2.50]
CUTOUT_PATCH_SIZE = 64


def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def load_rgb_float01(image_path: str | os.PathLike[str]) -> np.ndarray:
    """Load an image as HxWx3 RGB float32 in [0, 1]."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def defocus_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply the Gaussian defocus blur used by the inference script."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def cutout(
    image: np.ndarray,
    patch_size: int = CUTOUT_PATCH_SIZE,
    mask_ratio: float = 0.1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Randomly remove a fraction of fixed-size image patches."""
    output = np.asarray(image, dtype=np.float32).copy()
    height, width, _ = output.shape

    grid_h = int(np.ceil(height / patch_size))
    grid_w = int(np.ceil(width / patch_size))
    num_patches = grid_h * grid_w
    num_masked = int(np.round(mask_ratio * num_patches))

    if num_masked <= 0:
        return output

    patch_ids = np.random.choice(num_patches, num_masked, replace=False)
    for patch_id in patch_ids:
        row = patch_id // grid_w
        col = patch_id % grid_w
        y0 = row * patch_size
        y1 = min((row + 1) * patch_size, height)
        x0 = col * patch_size
        x1 = min((col + 1) * patch_size, width)
        output[y0:y1, x0:x1, :] = fill_value

    return output


def _to_pil_uint8(image: np.ndarray) -> Image.Image:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_u8, mode="RGB")


def _from_pil_float01(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def apply_rgb_corruption(
    image: np.ndarray,
    corruption: str = "none",
    severity: int = 0,
) -> np.ndarray:
    """
    Apply one RGB corruption.

    Args:
        image: RGB float32 array in [0, 1], shape (H, W, 3).
        corruption: none | blur | cutout | brightness | contrast.
        severity: Integer in [0, 5]; 0 returns the original image.
    """
    corruption = str(corruption).strip().lower()
    if corruption in {"none", "clean", "raw", ""} or severity <= 0:
        return np.asarray(image, dtype=np.float32).copy()

    if severity not in range(1, 6):
        raise ValueError(f"severity must be in [1, 5], got {severity}")

    image = np.asarray(image, dtype=np.float32)
    index = severity - 1

    if corruption == "blur":
        output = defocus_blur(image, BLUR_KERNELS[index])
        return np.clip(output, 0.0, 1.0)

    if corruption == "cutout":
        output = cutout(
            image,
            patch_size=CUTOUT_PATCH_SIZE,
            mask_ratio=CUTOUT_RATIOS[index],
            fill_value=0.0,
        )
        return np.clip(output, 0.0, 1.0)

    ratio = ENHANCE_RATIOS[index]
    factor = ratio if np.random.rand() < 0.5 else 1.0 / ratio
    image_pil = _to_pil_uint8(image)

    if corruption == "brightness":
        output_pil = adjust_brightness(image_pil, factor)
    elif corruption == "contrast":
        output_pil = adjust_contrast(image_pil, max(factor, 0.05))
    else:
        raise ValueError(
            f"Unknown corruption '{corruption}'. "
            "Expected none|blur|cutout|brightness|contrast."
        )

    return np.clip(_from_pil_float01(output_pil), 0.0, 1.0)


def build_comparison(
    image: np.ndarray,
    severity: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Build a deterministic five-panel comparison.

    Independent seed offsets match the top-row convention in the original
    inference visualization code.
    """
    items = [
        ("Raw RGB", "none"),
        ("Blur", "blur"),
        ("Cutout", "cutout"),
        ("Brightness", "brightness"),
        ("Contrast", "contrast"),
    ]

    outputs: Dict[str, np.ndarray] = {}
    for column, (label, corruption) in enumerate(items):
        setup_seed(seed + 10 * column)
        outputs[label] = apply_rgb_corruption(
            image,
            corruption=corruption,
            severity=severity,
        )
    return outputs


def save_comparison_figure(
    images: Dict[str, np.ndarray],
    output_path: str | os.PathLike[str],
    dpi: int = 300,
    font_size: float = 17.0,
) -> None:
    """Save a compact one-row appendix-ready comparison."""
    labels = ["Raw RGB", "Blur", "Cutout", "Brightness", "Contrast"]

    first = images[labels[0]]
    height, width = first.shape[:2]
    aspect = height / max(width, 1)

    panel_width = 3.0
    panel_height = panel_width * aspect
    figure_height = panel_height + 0.48

    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(panel_width * len(labels), figure_height),
        squeeze=False,
    )

    for axis, label in zip(axes[0], labels):
        axis.imshow(np.clip(images[label], 0.0, 1.0))
        axis.axis("off")
        axis.text(
            0.5,
            -0.055,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=font_size,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        top=0.998,
        bottom=0.14,
        wspace=0.015,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[VIS] Saved RGB-corruption comparison to: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Raw RGB | Blur | Cutout | Brightness | Contrast "
            "using the corruption definitions from inference_multimodal_scene.py."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--output",
        default="rgb_corruption_comparison.png",
        help="Output figure path. The extension determines the saved format.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        choices=range(1, 6),
        default=3,
        metavar="{1,2,3,4,5}",
        help="Shared corruption severity; default: 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for cutout and enhancement direction; default: 0.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution; default: 300.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=17.0,
        help="Panel-label font size; default: 17.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_rgb_float01(args.image)
    comparison = build_comparison(
        image=image,
        severity=args.severity,
        seed=args.seed,
    )
    save_comparison_figure(
        images=comparison,
        output_path=args.output,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    print(
        "[CFG] "
        f"severity={args.severity}, "
        f"blur_kernel={BLUR_KERNELS[args.severity - 1]}, "
        f"cutout_ratio={CUTOUT_RATIOS[args.severity - 1]:.2f}, "
        f"enhance_ratio={ENHANCE_RATIOS[args.severity - 1]:.2f}, "
        f"cutout_patch_size={CUTOUT_PATCH_SIZE}, "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()

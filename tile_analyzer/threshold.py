"""Utilities for adaptive binary thresholding and line extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    """Metadata describing a thresholding operation."""

    threshold_value: float
    line_color: str  # "dark" or "light"
    white_ratio: float


def binarize_line(frame_bgr: "cv2.Mat") -> Tuple[np.ndarray, ThresholdResult]:
    """Return a binary mask where the line is 255 and background is 0."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    median_value = float(np.median(gray))
    _, binary = cv2.threshold(gray, median_value, 255, cv2.THRESH_BINARY)

    white_pixels = int(np.count_nonzero(binary))
    total_pixels = binary.size
    white_ratio = white_pixels / total_pixels if total_pixels else 0.0

    # Assume the background occupies most of the frame. If white dominates,
    # the background is likely bright, so the line is dark -> invert.
    if white_ratio >= 0.5:
        line_mask = cv2.bitwise_not(binary)
        line_color = "dark"
    else:
        line_mask = binary
        line_color = "light"

    result = ThresholdResult(
        threshold_value=median_value,
        line_color=line_color,
        white_ratio=white_ratio,
    )
    return line_mask, result

"""Polyline analysis for measuring line coverage density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .geometry import PointMM, PointPX, segment_lengths_mm


@dataclass(frozen=True)
class SegmentReport:
    index: int
    start_px: PointPX
    end_px: PointPX
    length_mm: float
    pixels_on_line: int
    density: float


@dataclass(frozen=True)
class PolylineReport:
    segments: Tuple[SegmentReport, ...]
    average_density: float
    verdict: str


def _draw_segment_mask(shape: Tuple[int, int], start: PointPX, end: PointPX) -> np.ndarray:
    """Return a binary mask with a single-pixel line for the given segment."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, start, end, 255, 1)
    return mask


def analyze_polyline(
    line_mask: np.ndarray,
    points_mm: Sequence[PointMM],
    points_px: Sequence[PointPX],
    min_density: float = 0.5,
) -> PolylineReport:
    """Measure the coverage density for each segment of the reference polyline."""
    if len(points_mm) != len(points_px):
        raise ValueError("points_mm and points_px must contain the same number of points")

    if len(points_px) < 2:
        raise ValueError("At least two points are required to describe a polyline")

    lengths_mm = segment_lengths_mm(points_mm)
    if len(lengths_mm) != len(points_px) - 1:
        raise ValueError("Mismatch between number of segments and computed lengths")

    segment_reports: List[SegmentReport] = []
    height, width = line_mask.shape[:2]
    mask_shape = (height, width)
    weighted_density = 0.0
    total_length = 0.0

    for idx, ((start_px, end_px), length_mm) in enumerate(zip(zip(points_px[:-1], points_px[1:]), lengths_mm)):
        if length_mm <= 0:
            density = float("inf")
            pixels_on_line = 0
        else:
            segment_mask = _draw_segment_mask(mask_shape, start_px, end_px)
            painted = cv2.bitwise_and(line_mask, segment_mask)
            pixels_on_line = int(cv2.countNonZero(painted))
            density = pixels_on_line / length_mm

        segment_reports.append(
            SegmentReport(
                index=idx,
                start_px=start_px,
                end_px=end_px,
                length_mm=length_mm,
                pixels_on_line=pixels_on_line,
                density=density,
            )
        )

        if np.isfinite(density):
            weighted_density += density * length_mm
            total_length += length_mm

    average_density = weighted_density / total_length if total_length > 0 else 0.0
    verdict = "pass" if all(
        (report.density >= min_density) or not np.isfinite(report.density) for report in segment_reports
    ) else "fail"

    return PolylineReport(
        segments=tuple(segment_reports),
        average_density=average_density,
        verdict=verdict,
    )

"""Geometry helpers for converting tile coordinates from millimetres to pixels."""

from __future__ import annotations

from typing import Sequence, Tuple

PointMM = Tuple[float, float]
PointPX = Tuple[int, int]


def compute_px_per_mm(px_tile_width: float, mm_tile_width: float) -> float:
    """Return pixel-to-millimetre scale derived from the tile width."""
    if mm_tile_width <= 0:
        raise ValueError("Tile width in millimetres must be positive")
    return px_tile_width / mm_tile_width


def mm_to_px(
    points_mm: Sequence[PointMM],
    px_per_mm: float,
    origin_px: PointPX,
) -> Tuple[PointPX, ...]:
    """Convert planar coordinates from millimetres to image pixels.

    The origin is expected to be the bottom-left corner of the tile in pixel
    space. Positive Y in millimetres points up, while pixel Y grows down, so we
    subtract the vertical offset.
    """
    ox, oy = origin_px
    result = []
    for x_mm, y_mm in points_mm:
        px = int(round(ox + x_mm * px_per_mm))
        py = int(round(oy - y_mm * px_per_mm))
        result.append((px, py))
    return tuple(result)


def segment_lengths_mm(points_mm: Sequence[PointMM]) -> Tuple[float, ...]:
    """Return the Euclidean length of each consecutive segment in millimetres."""
    if len(points_mm) < 2:
        return ()

    lengths = []
    for (x0, y0), (x1, y1) in zip(points_mm[:-1], points_mm[1:]):
        dx = x1 - x0
        dy = y1 - y0
        lengths.append((dx * dx + dy * dy) ** 0.5)
    return tuple(lengths)

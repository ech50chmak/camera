"""Entry point for the tile polyline analyzer.

Example usage:
    python -m tile_analyzer.main ^
        --origin 120,2340 ^
        --tile-width-mm 120 ^
        --tile-width-px 1485 ^
        --points-mm "[(0, 0), (10, 5), (20, 10)]"

Adjust the arguments to match your setup before running on the Raspberry Pi.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

if __package__ in (None, ""):
    # Allow running the script directly (python main.py) by adding the project root to sys.path.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tile_analyzer.analyzer import PolylineReport, analyze_polyline
    from tile_analyzer.camera import Camera
    from tile_analyzer.geometry import PointMM, compute_px_per_mm, mm_to_px
    from tile_analyzer.threshold import binarize_line
else:
    from .analyzer import PolylineReport, analyze_polyline
    from .camera import Camera
    from .geometry import PointMM, compute_px_per_mm, mm_to_px
    from .threshold import binarize_line


def _parse_point_pairs(text: str) -> List[PointMM]:
    """Parse a JSON or Python-style list of (x, y) tuples."""
    data = json.loads(text.replace("(", "[").replace(")", "]"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of points")
    result: List[PointMM] = []
    for item in data:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(v, (int, float)) for v in item)
        ):
            x, y = float(item[0]), float(item[1])
            result.append((x, y))
        else:
            raise ValueError(f"Invalid point: {item!r}")
    return result


def _parse_origin(text: str) -> Tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("Origin must be formatted as 'x,y'")
    return int(parts[0]), int(parts[1])


def _print_report(report: PolylineReport) -> None:
    for segment in report.segments:
        density_str = "inf" if not math.isfinite(segment.density) else f"{segment.density:.2f}"
        print(
            f"Segment {segment.index}: density={density_str} px/mm, "
            f"length={segment.length_mm:.2f} mm, pixels={segment.pixels_on_line}"
        )
    print(f"Average density: {report.average_density:.2f} px/mm")
    print(f"Verdict: {report.verdict}")


def _report_to_dict(report: PolylineReport) -> dict:
    return {
        "segments": [
            {
                "index": segment.index,
                "start_px": segment.start_px,
                "end_px": segment.end_px,
                "length_mm": segment.length_mm,
                "pixels_on_line": segment.pixels_on_line,
                "density": segment.density,
            }
            for segment in report.segments
        ],
        "average_density": report.average_density,
        "verdict": report.verdict,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the drawn polyline on the tile.")
    parser.add_argument("--origin", default="120,2340", help="Bottom-left tile corner in pixels (x,y).")
    parser.add_argument(
        "--tile-width-mm",
        type=float,
        default=120.0,
        help="Measured tile width in millimetres.",
    )
    parser.add_argument(
        "--tile-width-px",
        type=float,
        default=1485.0,
        help="Measured tile width in pixels taken from the captured frame.",
    )
    parser.add_argument(
        "--points-mm",
        default="[(0, 0), (10, 5), (20, 10)]",
        help="Reference polyline points in millimetres as a JSON list.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to store the analysis results as JSON.",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.5,
        help="Minimum acceptable density in pixels per millimetre for each segment.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    origin_px = _parse_origin(args.origin)
    points_mm = _parse_point_pairs(args.points_mm)

    px_per_mm = compute_px_per_mm(args.tile_width_px, args.tile_width_mm)
    points_px = mm_to_px(points_mm, px_per_mm, origin_px)

    with Camera() as camera:
        frame = camera.capture_frame()

    line_mask, threshold_info = binarize_line(frame)
    report = analyze_polyline(
        line_mask,
        points_mm=points_mm,
        points_px=points_px,
        min_density=args.min_density,
    )

    print(
        f"Threshold={threshold_info.threshold_value:.1f}, "
        f"line_color={threshold_info.line_color}, "
        f"white_ratio={threshold_info.white_ratio:.2f}"
    )
    _print_report(report)

    if args.json_output:
        result_dict = {
            "threshold": {
                "value": threshold_info.threshold_value,
                "line_color": threshold_info.line_color,
                "white_ratio": threshold_info.white_ratio,
            },
            "report": _report_to_dict(report),
            "origin_px": origin_px,
            "px_per_mm": px_per_mm,
        }
        args.json_output.write_text(json.dumps(result_dict, indent=2))
        print(f"Saved detailed report to {args.json_output}")


if __name__ == "__main__":
    main()

"""Camera utilities for capturing frames from Raspberry Pi IMX219 sensor."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2


class Camera:
    """Manage access to the IMX219 camera using OpenCV VideoCapture."""

    def __init__(self, index: int = 0, resolution: Tuple[int, int] = (3280, 2464)) -> None:
        self.index = index
        self.resolution = resolution
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> cv2.VideoCapture:
        """Open the camera if needed and apply the requested resolution."""
        if self._capture is None:
            capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open camera at index {self.index}")

            width, height = self.resolution
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self._capture = capture
        return self._capture

    def capture_frame(self) -> "cv2.Mat":
        """Return a single BGR frame from the camera."""
        capture = self.open()
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self) -> None:
        """Release the camera capture object."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

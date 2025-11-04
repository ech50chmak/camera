"""Camera utilities for capturing frames from Raspberry Pi IMX219 sensor."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from libcamera import Transform, controls
from picamera2 import Picamera2


class Camera:
    """Manage access to the IMX219 camera using libcamera via Picamera2."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (3280, 2464),
        sensor_index: int = 0,
        use_manual_exposure: bool = False,
    ) -> None:
        self.resolution = resolution
        self.sensor_index = sensor_index
        self.use_manual_exposure = use_manual_exposure
        self._picam2: Optional[Picamera2] = None

    def open(self) -> Picamera2:
        """Open the libcamera pipeline and apply the configuration."""
        if self._picam2 is None:
            picam2 = Picamera2(camera_num=self.sensor_index)
            transform = Transform(hflip=True)
            config = picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                transform=transform,
                buffer_count=1,
            )
            picam2.configure(config)

            if self.use_manual_exposure:
                picam2.set_controls({"AeEnable": False, "ExposureTime": 8000, "AnalogueGain": 2.0})
            else:
                picam2.set_controls(
                    {
                        "AeEnable": True,
                        "AwbEnable": True,
                        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
                    }
                )

            picam2.start()
            self._picam2 = picam2
        return self._picam2

    def capture_frame(self) -> "cv2.Mat":
        """Return a single BGR frame from the camera."""
        picam2 = self.open()
        frame_rgb: np.ndarray = picam2.capture_array("main")
        if frame_rgb is None:
            raise RuntimeError("Failed to capture frame from Picamera2")
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        """Stop and close the libcamera pipeline."""
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

"""
Calibration module for EyeCursor.
Maps eye movement range to screen coordinates.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import config


class CalibrationPoint:
    """Represents a calibration point with target and collected samples."""

    def __init__(self, target_normalized: Tuple[float, float]):
        """
        Initialize calibration point.

        Args:
            target_normalized: (x, y) in normalized coordinates (0-1)
        """
        self.target = target_normalized
        self.samples: List[Tuple[float, float]] = []
        self.is_complete = False

    def add_sample(self, gaze_x: float, gaze_y: float):
        """Add a gaze sample."""
        self.samples.append((gaze_x, gaze_y))

    def get_average(self) -> Optional[Tuple[float, float]]:
        """Get average of collected samples."""
        if not self.samples:
            return None
        avg_x = sum(s[0] for s in self.samples) / len(self.samples)
        avg_y = sum(s[1] for s in self.samples) / len(self.samples)
        return (avg_x, avg_y)


class Calibrator:
    """
    Calibrator for eye tracking system.

    Calibration process:
    1. Show target points on screen
    2. User looks at each point for a few seconds
    3. Collect gaze samples for each point
    4. Calculate mapping from gaze space to screen space
    """

    def __init__(self, calibration_points: List[Tuple[float, float]] = None):
        """
        Initialize calibrator.

        Args:
            calibration_points: List of (x, y) normalized points to calibrate
        """
        if calibration_points is None:
            calibration_points = config.CALIBRATION_POINTS

        self.points = [CalibrationPoint(p) for p in calibration_points]
        self.current_point_index = 0
        self.is_active = False
        self.sample_count = 0
        self.required_samples = 30  # Samples to collect per point

        # Results
        self.min_gaze_x = 0.0
        self.max_gaze_x = 1.0
        self.min_gaze_y = 0.0
        self.max_gaze_y = 1.0

    def start(self):
        """Start calibration process."""
        self.is_active = True
        self.current_point_index = 0
        for point in self.points:
            point.samples = []
            point.is_complete = False

    def update(self, gaze_x: Optional[float], gaze_y: Optional[float]) -> bool:
        """
        Update calibration with new gaze data.

        Args:
            gaze_x: Current gaze X or None
            gaze_y: Current gaze Y or None

        Returns:
            bool: True if calibration is complete
        """
        if not self.is_active:
            return False

        if self.current_point_index >= len(self.points):
            self._finalize()
            return True

        current_point = self.points[self.current_point_index]

        if gaze_x is not None and gaze_y is not None:
            current_point.add_sample(gaze_x, gaze_y)

        # Check if current point is complete
        if len(current_point.samples) >= self.required_samples:
            current_point.is_complete = True
            self.current_point_index += 1

        # Check if all points complete
        if self.current_point_index >= len(self.points):
            self._finalize()
            return True

        return False

    def _finalize(self):
        """Finalize calibration and calculate bounds."""
        self.is_active = False

        # Calculate bounds from all samples
        all_x = []
        all_y = []

        for point in self.points:
            for sample in point.samples:
                all_x.append(sample[0])
                all_y.append(sample[1])

        if all_x and all_y:
            # Add padding to bounds (10%)
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)

            self.min_gaze_x = min(all_x) - x_range * 0.1
            self.max_gaze_x = max(all_x) + x_range * 0.1
            self.min_gaze_y = min(all_y) - y_range * 0.1
            self.max_gaze_y = max(all_y) + y_range * 0.1

            # Ensure valid ranges
            if self.min_gaze_x == self.max_gaze_x:
                self.min_gaze_x = 0
                self.max_gaze_x = 1
            if self.min_gaze_y == self.max_gaze_y:
                self.min_gaze_y = 0
                self.max_gaze_y = 1

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw calibration UI on frame.

        Args:
            frame: BGR image

        Returns:
            Frame with calibration UI
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        if not self.is_active:
            return frame

        # Draw progress
        progress = f"Calibrating: {self.current_point_index + 1}/{len(self.points)}"
        cv2.putText(overlay, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_TEXT, 2)

        if self.current_point_index < len(self.points):
            current = self.points[self.current_point_index]
            target_x = int(current.target[0] * w)
            target_y = int(current.target[1] * h)

            # Draw target circle
            cv2.circle(overlay, (target_x, target_y), 20, config.COLOR_CALIBRATION, -1)
            cv2.circle(overlay, (target_x, target_y), 25, config.COLOR_TEXT, 2)
            cv2.circle(overlay, (target_x, target_y), 10, config.COLOR_TEXT, -1)

            # Draw progress arc
            samples = len(current.samples)
            angle = int((samples / self.required_samples) * 360)
            cv2.ellipse(overlay, (target_x, target_y), (30, 30),
                       0, 0, angle, config.COLOR_EYE_LANDMARKS, 4)

            # Instructions
            instruction = "Look at the circle"
            cv2.putText(overlay, instruction, (w//2 - 100, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)

        # Apply semi-transparent overlay
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return not self.is_active and self.current_point_index >= len(self.points)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get calibration bounds.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        return (self.min_gaze_x, self.max_gaze_x,
                self.min_gaze_y, self.max_gaze_y)

    def reset(self):
        """Reset calibration."""
        self.is_active = False
        self.current_point_index = 0
        for point in self.points:
            point.samples = []
            point.is_complete = False

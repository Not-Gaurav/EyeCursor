"""
Cursor controller module for EyeCursor.
Handles smooth cursor movement and mouse actions using PyAutoGUI.
"""

import pyautogui
import numpy as np
from typing import Tuple, Optional
import config


class SmoothCursor:
    """
    Smooth cursor movement using exponential moving average.

    This prevents jitter and creates natural-feeling cursor movement
    by blending current position with target position.
    """

    def __init__(self, smoothing_factor: float = config.SMOOTHING_FACTOR):
        """
        Initialize smooth cursor.

        Args:
            smoothing_factor: 0-1 value, higher = smoother but more lag
        """
        self.smoothing_factor = smoothing_factor
        self.current_x = None
        self.current_y = None
        self.screen_width, self.screen_height = pyautogui.size()

    def update(self, target_x: float, target_y: float) -> Tuple[int, int]:
        """
        Update cursor position with smoothing.

        Args:
            target_x: Target X coordinate (screen coordinates)
            target_y: Target Y coordinate (screen coordinates)

        Returns:
            Tuple of (smoothed_x, smoothed_y) as integers
        """
        if self.current_x is None or self.current_y is None:
            # First update - jump to position
            self.current_x = target_x
            self.current_y = target_y
        else:
            # Apply exponential moving average
            # new_position = old_position * (1 - alpha) + target * alpha
            # But we want more smoothing, so we use:
            # new_position = old_position * alpha + target * (1 - alpha)
            # where alpha is the smoothing factor
            self.current_x = (self.current_x * self.smoothing_factor +
                             target_x * (1 - self.smoothing_factor))
            self.current_y = (self.current_y * self.smoothing_factor +
                             target_y * (1 - self.smoothing_factor))

        return (int(self.current_x), int(self.current_y))

    def reset(self):
        """Reset cursor state."""
        self.current_x = None
        self.current_y = None


class CursorController:
    """
    Main cursor controller that maps eye gaze to screen coordinates.

    Uses calibration data to map normalized eye gaze (0-1 range)
    to actual screen coordinates.
    """

    def __init__(self):
        """Initialize cursor controller."""
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Initialize smooth cursor
        self.smooth_cursor = SmoothCursor(config.SMOOTHING_FACTOR)

        # Calibration data
        self.gaze_min_x = 0.3
        self.gaze_max_x = 0.7
        self.gaze_min_y = 0.3
        self.gaze_max_y = 0.7
        self.is_calibrated = False

        # Movement scaling
        self.scale_x = config.MOVEMENT_SCALE_X
        self.scale_y = config.MOVEMENT_SCALE_Y

        # Eyeroll smoothing
        self.eyeroll_smooth_x = 0.0
        self.eyeroll_smooth_y = 0.0
        self.eyeroll_smoothing = config.EYEROLL_SMOOTHING

        # Center offset (for drift correction)
        self.center_offset_x = 0.0
        self.center_offset_y = 0.0

        # Dead zone (ignore small movements)
        self.dead_zone = 0.02

        # Disable PyAutoGUI fail-safe for demo (be careful!)
        pyautogui.FAILSAFE = False

    def set_calibration(self,
                       gaze_min_x: float, gaze_max_x: float,
                       gaze_min_y: float, gaze_max_y: float):
        """
        Set calibration boundaries from calibration data.

        Args:
            gaze_min_x: Minimum gaze X value observed
            gaze_max_x: Maximum gaze X value observed
            gaze_min_y: Minimum gaze Y value observed
            gaze_max_y: Maximum gaze Y value observed
        """
        self.gaze_min_x = gaze_min_x
        self.gaze_max_x = gaze_max_x
        self.gaze_min_y = gaze_min_y
        self.gaze_max_y = gaze_max_y
        self.is_calibrated = True

    def map_gaze_to_screen(self, gaze_x: float, gaze_y: float,
                           eyeroll_x: float = 0.0, eyeroll_y: float = 0.0) -> Tuple[int, int]:
        """
        Map normalized gaze coordinates to screen coordinates.

        Args:
            gaze_x: Gaze X in normalized coordinates (0-1)
            gaze_y: Gaze Y in normalized coordinates (0-1)
            eyeroll_x: Horizontal eyeroll offset (-1 to 1, negative=left, positive=right)
            eyeroll_y: Vertical eyeroll offset (-1 to 1, negative=up, positive=down)

        Returns:
            Tuple of (screen_x, screen_y)
        """
        # Eyeroll amplification factor - makes eye rolling more responsive
        eyeroll_amp = config.EYEROLL_AMPLIFICATION  # Adjust this to control eyeroll sensitivity

        if not self.is_calibrated:
            # Apply eyeroll to fine-tune gaze position
            adjusted_gaze_x = gaze_x + (eyeroll_x * eyeroll_amp)
            adjusted_gaze_y = gaze_y + (eyeroll_y * eyeroll_amp)

            # Clamp to valid range
            adjusted_gaze_x = np.clip(adjusted_gaze_x, 0, 1)
            adjusted_gaze_y = np.clip(adjusted_gaze_y, 0, 1)

            # Use default mapping
            screen_x = int((1 - adjusted_gaze_x) * self.screen_width)  # Mirror X
            screen_y = int(adjusted_gaze_y * self.screen_height)
        else:
            # Apply eyeroll to fine-tune gaze before calibration mapping
            adjusted_gaze_x = gaze_x + (eyeroll_x * eyeroll_amp)
            adjusted_gaze_y = gaze_y + (eyeroll_y * eyeroll_amp)

            # Map from calibration range to screen
            # Clamp to boundaries
            adjusted_gaze_x = np.clip(adjusted_gaze_x, self.gaze_min_x, self.gaze_max_x)
            adjusted_gaze_y = np.clip(adjusted_gaze_y, self.gaze_min_y, self.gaze_max_y)

            # Normalize to 0-1 within calibration range
            norm_x = (adjusted_gaze_x - self.gaze_min_x) / (self.gaze_max_x - self.gaze_min_x)
            norm_y = (adjusted_gaze_y - self.gaze_min_y) / (self.gaze_max_y - self.gaze_min_y)

            # Apply scaling around center (0.5)
            norm_x = (norm_x - 0.5) * self.scale_x + 0.5
            norm_y = (norm_y - 0.5) * self.scale_y + 0.5

            # Clamp again after scaling
            norm_x = np.clip(norm_x, 0, 1)
            norm_y = np.clip(norm_y, 0, 1)

            # Map to screen (mirror X for natural movement)
            screen_x = int((1 - norm_x) * self.screen_width)
            screen_y = int(norm_y * self.screen_height)

        return (screen_x, screen_y)

    def update_position(self, gaze_x: Optional[float], gaze_y: Optional[float],
                       eyeroll_x: float = 0.0, eyeroll_y: float = 0.0) -> Tuple[Optional[int], Optional[int]]:
        """
        Update cursor position from gaze coordinates.

        Args:
            gaze_x: Gaze X or None if not detected
            gaze_y: Gaze Y or None if not detected
            eyeroll_x: Horizontal eyeroll offset for fine control
            eyeroll_y: Vertical eyeroll offset for fine control

        Returns:
            Tuple of (screen_x, screen_y) or (None, None)
        """
        if gaze_x is None or gaze_y is None:
            return (None, None)

        # Apply smoothing to eyeroll data
        self.eyeroll_smooth_x = (self.eyeroll_smooth_x * self.eyeroll_smoothing +
                                eyeroll_x * (1 - self.eyeroll_smoothing))
        self.eyeroll_smooth_y = (self.eyeroll_smooth_y * self.eyeroll_smoothing +
                                eyeroll_y * (1 - self.eyeroll_smoothing))

        # Map to screen coordinates with eyeroll adjustment
        target_x, target_y = self.map_gaze_to_screen(gaze_x, gaze_y,
                                                      self.eyeroll_smooth_x,
                                                      self.eyeroll_smooth_y)

        # Apply smoothing
        screen_x, screen_y = self.smooth_cursor.update(target_x, target_y)

        # Move cursor
        pyautogui.moveTo(screen_x, screen_y, duration=0)

        return (screen_x, screen_y)

    def click(self, button='left'):
        """
        Perform mouse click.

        Args:
            button: 'left' or 'right'
        """
        pyautogui.click(button=button)

    def right_click(self):
        """Perform right click."""
        pyautogui.rightClick()

    def double_click(self):
        """Perform double click."""
        pyautogui.doubleClick()

    def reset(self):
        """Reset cursor controller."""
        self.smooth_cursor.reset()

    def get_status_text(self) -> str:
        """Get status text for display."""
        if self.is_calibrated:
            return f"Calibrated | Scale: {self.scale_x:.1f}x{self.scale_y:.1f}"
        return "Not Calibrated"

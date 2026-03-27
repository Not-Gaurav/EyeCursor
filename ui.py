"""
UI module for EyeCursor.
Handles visual feedback, status display, and user interface elements.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import config
from enum import Enum


class AppState(Enum):
    """Application states."""
    STARTUP = 0
    CALIBRATING = 1
    RUNNING = 2
    PAUSED = 3


class UI:
    """
    User Interface for EyeCursor application.

    Provides:
    - Visual feedback for eye tracking
    - Status information
    - Instructions and help
    - Performance metrics
    """

    def __init__(self, window_name: str = config.WINDOW_NAME):
        """
        Initialize UI.

        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self.state = AppState.STARTUP
        self.frame_count = 0
        self.fps = 0
        self.last_time = cv2.getTickCount()

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # Status messages
        self.status_message = ""
        self.blink_message = ""
        self.calibration_message = ""

        # Colors for status
        self.status_color = config.COLOR_EYE_LANDMARKS

    def set_state(self, state: AppState):
        """Set application state."""
        self.state = state

    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        current_time = cv2.getTickCount()
        elapsed = (current_time - self.last_time) / cv2.getTickFrequency()

        if elapsed >= 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

    def draw_status_bar(self, frame: np.ndarray,
                       face_detected: bool,
                       cursor_text: str,
                       blink_text: str = "") -> np.ndarray:
        """
        Draw status bar at top of frame.

        Args:
            frame: BGR image
            face_detected: Whether face is detected
            cursor_text: Cursor controller status
            blink_text: Blink detector status

        Returns:
            Frame with status bar
        """
        h, w = frame.shape[:2]
        bar_height = 35

        # Draw semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status indicator
        status = "FACE DETECTED" if face_detected else "NO FACE"
        color = config.COLOR_EYE_LANDMARKS if face_detected else config.COLOR_CURSOR
        cv2.putText(frame, status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {int(self.fps)}", (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)

        return frame

    def draw_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw help/instruction overlay.

        Args:
            frame: BGR image

        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background
        cv2.rectangle(overlay, (10, h - 120), (400, h - 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Instructions
        instructions = [
            "Controls:",
            "  C - Calibrate",
            "  R - Reset",
            "  P - Pause/Resume",
            "  Q - Quit",
            "",
            "Long blink (>300ms) = Left click",
            "Double blink = Right click"
        ]

        y_offset = h - 100
        for i, line in enumerate(instructions):
            color = config.COLOR_TEXT if i == 0 else (200, 200, 200)
            size = 0.6 if i == 0 else 0.5
            cv2.putText(frame, line, (20, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

        return frame

    def draw_calibration_screen(self, frame: np.ndarray,
                               progress: str,
                               current_point: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw calibration screen.

        Args:
            frame: BGR image
            progress: Calibration progress text
            current_point: Current calibration point coordinates

        Returns:
            Frame with calibration UI
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Full screen overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Title
        title = "CALIBRATION"
        cv2.putText(frame, title, (w//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.COLOR_CALIBRATION, 3)

        # Instructions
        instructions = [
            "Follow the circles with your eyes",
            "Keep your head still",
            "Don't move your mouse"
        ]
        y = 100
        for instruction in instructions:
            cv2.putText(frame, instruction, (w//2 - 150, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)
            y += 30

        # Progress
        cv2.putText(frame, progress, (w//2 - 50, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_EYE_LANDMARKS, 2)

        return frame

    def draw_blink_indicator(self, frame: np.ndarray,
                              blink_detected: bool,
                              blink_type: str = "") -> np.ndarray:
        """
        Draw blink detection indicator.

        Args:
            frame: BGR image
            blink_detected: Whether a blink was detected
            blink_type: Type of blink (single/double)

        Returns:
            Frame with indicator
        """
        h, w = frame.shape[:2]

        if blink_detected:
            # Flash indicator
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 150, 40), (w - 10, 90),
                         config.COLOR_CURSOR, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            text = blink_type if blink_type else "BLINK!"
            cv2.putText(frame, text, (w - 140, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)

        return frame

    def draw_pause_screen(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw pause screen overlay.

        Args:
            frame: BGR image

        Returns:
            Frame with pause overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # PAUSED text
        cv2.putText(frame, "PAUSED", (w//2 - 80, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.COLOR_CURSOR, 3)
        cv2.putText(frame, "Press P to resume", (w//2 - 120, h//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)

        return frame

    def render(self, frame: np.ndarray,
              eye_data=None,
              cursor_controller=None,
              blink_detector=None,
              calibrator=None) -> np.ndarray:
        """
        Main render function that assembles the complete UI.

        Args:
            frame: Raw camera frame
            eye_data: EyeData object
            cursor_controller: CursorController object
            blink_detector: BlinkDetector object
            calibrator: Calibrator object

        Returns:
            Fully rendered frame
        """
        rendered = frame.copy()

        # Update FPS
        self.update_fps()

        # Draw eye tracking overlay if available
        if eye_data:
            from eye_tracker import EyeTracker
            # We need to import EyeTracker to use its draw method
            # But to avoid circular import, we'll handle drawing in main
            pass

        # Draw status bar
        face_detected = eye_data.face_detected if eye_data else False
        cursor_text = cursor_controller.get_status_text() if cursor_controller else ""
        blink_text = blink_detector.get_state_text() if blink_detector else ""
        rendered = self.draw_status_bar(rendered, face_detected, cursor_text, blink_text)

        # Draw calibration or help based on state
        if self.state == AppState.CALIBRATING:
            if calibrator:
                progress = f"Point {calibrator.current_point_index + 1}/{len(calibrator.points)}"
                rendered = self.draw_calibration_screen(rendered, progress)
        elif self.state == AppState.PAUSED:
            rendered = self.draw_pause_screen(rendered)
        else:
            # Draw help overlay
            rendered = self.draw_help_overlay(rendered)

        return rendered

    def show(self, frame: np.ndarray):
        """Display the frame."""
        cv2.imshow(self.window_name, frame)

    def get_key(self, delay: int = 1) -> int:
        """Get key press (waitKey wrapper)."""
        return cv2.waitKey(delay) & 0xFF

    def cleanup(self):
        """Clean up UI resources."""
        cv2.destroyAllWindows()

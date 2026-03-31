"""
Eye tracking module using MediaPipe Face Landmarker Task API.
Handles face detection, eye landmark extraction, and gaze estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
from dataclasses import dataclass
import config


@dataclass
class EyeData:
    """Container for eye tracking data."""
    left_eye_center: Optional[Tuple[float, float]] = None
    right_eye_center: Optional[Tuple[float, float]] = None
    left_eye_open: bool = True
    right_eye_open: bool = True
    left_ear: float = 1.0  # Eye Aspect Ratio
    right_ear: float = 1.0
    face_detected: bool = False
    gaze_point: Optional[Tuple[float, float]] = None  # Normalized (x, y)
    landmarks: Optional[List] = None

    # Eyeroll tracking - iris position relative to eye corners
    # Range: -1 to 1, where 0 is center, negative is left/up, positive is right/down
    eyeroll_x: float = 0.0  # Horizontal eyeroll (left/right)
    eyeroll_y: float = 0.0  # Vertical eyeroll (up/down)
    left_iris_offset: Optional[Tuple[float, float]] = None  # (x_offset, y_offset) for left eye
    right_iris_offset: Optional[Tuple[float, float]] = None  # (x_offset, y_offset) for right eye


class EyeTracker:
    """
    Eye tracking using MediaPipe Face Landmarker Task API.

    MediaPipe Face Landmarker provides facial landmarks including
    eye iris tracking. We use specific indices for eye corners
    and edges to calculate eye centers and blink detection.
    """

    def __init__(self):
        """Initialize MediaPipe Face Landmarker."""
        self.base_options = mp.tasks.BaseOptions(
            model_asset_buffer=self._get_model_bytes()
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            num_faces=1,
            min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.FACE_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _get_model_bytes(self) -> bytes:
        """Get the face landmarker model as bytes."""
        import urllib.request
        import os

        model_path = os.path.expanduser("~/.mediapipe/face_landmarker.task")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if not os.path.exists(model_path):
            print("Downloading face landmarker model (first time only)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")

        with open(model_path, 'rb') as f:
            return f.read()

    def calculate_ear(self, landmarks, eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.

        EAR = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)

        When eye is open: EAR is relatively constant (around 0.25-0.35)
        When eye is closed: EAR drops significantly (near 0.1 or less)

        Args:
            landmarks: MediaPipe face landmarks
            eye_indices: List of 6 indices for eye corners [p1, p2, p3, p4, p5, p6]

        Returns:
            float: Eye Aspect Ratio
        """
        if len(eye_indices) != 6:
            return 1.0

        # Get coordinates
        p = [landmarks[i] for i in eye_indices]

        # Calculate distances
        # Vertical distances
        vertical_1 = np.linalg.norm(
            np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y])
        )
        vertical_2 = np.linalg.norm(
            np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y])
        )

        # Horizontal distance
        horizontal = np.linalg.norm(
            np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y])
        )

        if horizontal == 0:
            return 1.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def get_eye_center(self, landmarks, iris_indices: List[int]) -> Tuple[float, float]:
        """
        Calculate eye center using iris landmarks.

        Args:
            landmarks: MediaPipe face landmarks
            iris_indices: List of iris landmark indices

        Returns:
            Tuple of (x, y) normalized coordinates
        """
        iris_points = [(landmarks[i].x, landmarks[i].y)
                      for i in iris_indices]

        center_x = sum(p[0] for p in iris_points) / len(iris_points)
        center_y = sum(p[1] for p in iris_points) / len(iris_points)

        return (center_x, center_y)

    def get_gaze_point(self, left_eye: Tuple[float, float],
                      right_eye: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate gaze point as average of both eye centers.

        Args:
            left_eye: (x, y) of left eye center
            right_eye: (x, y) of right eye center

        Returns:
            Tuple of (x, y) normalized gaze point
        """
        gaze_x = (left_eye[0] + right_eye[0]) / 2
        gaze_y = (left_eye[1] + right_eye[1]) / 2
        return (gaze_x, gaze_y)

    def calculate_eyeroll(self, landmarks, iris_indices: List[int],
                         eye_left_idx: int, eye_right_idx: int,
                         eye_top_idx: int, eye_bottom_idx: int) -> Tuple[float, float]:
        """
        Calculate eyeroll - the position of iris within the eye boundaries.

        This measures how far the iris is from the center of the eye socket,
        which is essential for detecting eye rolling movements.

        Args:
            landmarks: MediaPipe face landmarks
            iris_indices: List of iris landmark indices
            eye_left_idx: Index of left eye corner
            eye_right_idx: Index of right eye corner
            eye_top_idx: Index of top eye point
            eye_bottom_idx: Index of bottom eye point

        Returns:
            Tuple of (x_offset, y_offset) where:
            - x_offset: -1 (iris at left corner) to 1 (iris at right corner), 0 = center
            - y_offset: -1 (iris at top) to 1 (iris at bottom), 0 = center
        """
        # Get iris center
        iris_points = [(landmarks[i].x, landmarks[i].y) for i in iris_indices]
        iris_x = sum(p[0] for p in iris_points) / len(iris_points)
        iris_y = sum(p[1] for p in iris_points) / len(iris_points)

        # Get eye boundaries
        eye_left = landmarks[eye_left_idx]
        eye_right = landmarks[eye_right_idx]
        eye_top = landmarks[eye_top_idx]
        eye_bottom = landmarks[eye_bottom_idx]

        # Calculate eye center
        eye_center_x = (eye_left.x + eye_right.x) / 2
        eye_center_y = (eye_top.y + eye_bottom.y) / 2

        # Calculate eye dimensions
        eye_width = abs(eye_right.x - eye_left.x)
        eye_height = abs(eye_bottom.y - eye_top.y)

        if eye_width == 0 or eye_height == 0:
            return (0.0, 0.0)

        # Calculate iris offset from eye center, normalized by eye size
        # This gives us how far the iris has "rolled" from center
        x_offset = (iris_x - eye_center_x) / (eye_width * 0.5)  # -1 to 1 range
        y_offset = (iris_y - eye_center_y) / (eye_height * 0.5)  # -1 to 1 range

        # Clamp to reasonable range
        x_offset = max(-2.0, min(2.0, x_offset))
        y_offset = max(-2.0, min(2.0, y_offset))

        return (x_offset, y_offset)

    def process_frame(self, frame: np.ndarray) -> EyeData:
        """
        Process a video frame and extract eye tracking data.

        Args:
            frame: BGR image from camera

        Returns:
            EyeData object containing tracking information
        """
        data = EyeData()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        detection_result = self.landmarker.detect(mp_image)

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            data.face_detected = True
            data.landmarks = face_landmarks

            # Calculate EAR for both eyes
            data.left_ear = self.calculate_ear(face_landmarks,
                                               config.LEFT_EYE_INDICES)
            data.right_ear = self.calculate_ear(face_landmarks,
                                                config.RIGHT_EYE_INDICES)

            # Determine if eyes are open (EAR above threshold)
            data.left_eye_open = data.left_ear > config.BLINK_THRESHOLD
            data.right_eye_open = data.right_ear > config.BLINK_THRESHOLD

            # Get eye centers using iris landmarks (more precise)
            try:
                data.left_eye_center = self.get_eye_center(face_landmarks,
                                                          config.LEFT_IRIS)
                data.right_eye_center = self.get_eye_center(face_landmarks,
                                                            config.RIGHT_IRIS)

                # Calculate gaze point
                if data.left_eye_center and data.right_eye_center:
                    data.gaze_point = self.get_gaze_point(data.left_eye_center,
                                                          data.right_eye_center)

                # Calculate eyeroll (iris position relative to eye boundaries)
                data.left_iris_offset = self.calculate_eyeroll(
                    face_landmarks, config.LEFT_IRIS,
                    config.LEFT_EYE_LEFT, config.LEFT_EYE_RIGHT,
                    config.LEFT_EYE_TOP, config.LEFT_EYE_BOTTOM
                )
                data.right_iris_offset = self.calculate_eyeroll(
                    face_landmarks, config.RIGHT_IRIS,
                    config.RIGHT_EYE_LEFT, config.RIGHT_EYE_RIGHT,
                    config.RIGHT_EYE_TOP, config.RIGHT_EYE_BOTTOM
                )

                # Average both eyes for overall eyeroll
                if data.left_iris_offset and data.right_iris_offset:
                    data.eyeroll_x = (data.left_iris_offset[0] + data.right_iris_offset[0]) / 2
                    data.eyeroll_y = (data.left_iris_offset[1] + data.right_iris_offset[1]) / 2

            except (IndexError, AttributeError):
                # Fallback to eye corners if iris not available
                left_indices = config.LEFT_EYE_INDICES
                right_indices = config.RIGHT_EYE_INDICES

                left_eye_points = [(face_landmarks[i].x, face_landmarks[i].y)
                                  for i in left_indices]
                right_eye_points = [(face_landmarks[i].x, face_landmarks[i].y)
                                   for i in right_indices]

                data.left_eye_center = (
                    sum(p[0] for p in left_eye_points) / len(left_eye_points),
                    sum(p[1] for p in left_eye_points) / len(left_eye_points)
                )
                data.right_eye_center = (
                    sum(p[0] for p in right_eye_points) / len(right_eye_points),
                    sum(p[1] for p in right_eye_points) / len(right_eye_points)
                )

                data.gaze_point = self.get_gaze_point(data.left_eye_center,
                                                      data.right_eye_center)

        return data

    def draw_landmarks(self, frame: np.ndarray, data: EyeData) -> np.ndarray:
        """
        Draw eye tracking visualization on frame.

        Args:
            frame: BGR image
            data: EyeData object

        Returns:
            Frame with landmarks drawn
        """
        if not data.face_detected or data.landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Draw eye centers
        if data.left_eye_center:
            lx, ly = int(data.left_eye_center[0] * w), int(data.left_eye_center[1] * h)
            color = config.COLOR_EYE_LANDMARKS if data.left_eye_open else config.COLOR_CURSOR
            cv2.circle(frame, (lx, ly), 5, color, -1)
            cv2.circle(frame, (lx, ly), 10, color, 2)

        if data.right_eye_center:
            rx, ry = int(data.right_eye_center[0] * w), int(data.right_eye_center[1] * h)
            color = config.COLOR_EYE_LANDMARKS if data.right_eye_open else config.COLOR_CURSOR
            cv2.circle(frame, (rx, ry), 5, color, -1)
            cv2.circle(frame, (rx, ry), 10, color, 2)

        # Draw gaze point
        if data.gaze_point:
            gx, gy = int(data.gaze_point[0] * w), int(data.gaze_point[1] * h)
            cv2.circle(frame, (gx, gy), 8, config.COLOR_CURSOR, -1)
            cv2.circle(frame, (gx, gy), 12, (255, 255, 255), 2)

        # Draw EAR values
        cv2.putText(frame, f"L-EAR: {data.left_ear:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)
        cv2.putText(frame, f"R-EAR: {data.right_ear:.3f}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)

        # Draw blink status
        status = "BLINK" if not (data.left_eye_open and data.right_eye_open) else "OPEN"
        color = config.COLOR_CURSOR if status == "BLINK" else config.COLOR_EYE_LANDMARKS
        cv2.putText(frame, f"Eyes: {status}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw eyeroll values
        cv2.putText(frame, f"Eyeroll X: {data.eyeroll_x:+.2f}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)
        cv2.putText(frame, f"Eyeroll Y: {data.eyeroll_y:+.2f}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)

        # Draw eyeroll visualization (crosshair showing roll direction)
        center_x = w // 2
        center_y = h // 2
        roll_line_length = int(min(w, h) * 0.15)

        # Calculate eyeroll indicator position
        indicator_x = center_x + int(data.eyeroll_x * roll_line_length)
        indicator_y = center_y + int(data.eyeroll_y * roll_line_length)

        # Draw crosshair background
        cv2.line(frame, (center_x - roll_line_length, center_y),
                (center_x + roll_line_length, center_y), (50, 50, 50), 2)
        cv2.line(frame, (center_x, center_y - roll_line_length),
                (center_x, center_y + roll_line_length), (50, 50, 50), 2)
        cv2.circle(frame, (center_x, center_y), roll_line_length, (50, 50, 50), 2)

        # Draw eyeroll indicator
        cv2.circle(frame, (indicator_x, indicator_y), 10, config.COLOR_CURSOR, -1)
        cv2.circle(frame, (indicator_x, indicator_y), 12, (255, 255, 255), 2)

        # Draw eyeroll direction arrow
        if abs(data.eyeroll_x) > 0.1 or abs(data.eyeroll_y) > 0.1:
            end_x = center_x + int(data.eyeroll_x * roll_line_length * 1.5)
            end_y = center_y + int(data.eyeroll_y * roll_line_length * 1.5)
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                          config.COLOR_EYE_LANDMARKS, 3, tipLength=0.3)

        return frame

    def release(self):
        """Release resources."""
        self.landmarker.close()

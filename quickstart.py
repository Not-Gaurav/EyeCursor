"""
EyeCursor Quickstart - Minimal setup for testing
================================================

A simplified version of EyeCursor that runs without calibration.
Good for quick testing before using the full version.

Usage:
    python quickstart.py

Controls:
    Q - Quit
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np


def quickstart():
    """Quickstart version with minimal setup."""
    print("EyeCursor Quickstart")
    print("===================")
    print("Move cursor with your eyes!")
    print("Blink slowly to click.")
    print("Press Q to quit.")
    print()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Get screen size
    screen_w, screen_h = pyautogui.size()

    # Smoothing variables
    smooth_x, smooth_y = screen_w // 2, screen_h // 2
    smoothing = 0.8

    # Blink detection
    blink_threshold = 0.25
    last_blink_time = 0
    blink_cooldown = 0.5  # seconds

    # Iris indices
    LEFT_IRIS = [469, 470, 471, 472]
    RIGHT_IRIS = [474, 475, 476, 477]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    pyautogui.FAILSAFE = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Get eye centers
            left_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y)
                          for i in LEFT_IRIS]
            right_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y)
                           for i in RIGHT_IRIS]

            left_center = (sum(p[0] for p in left_points) / 4,
                          sum(p[1] for p in left_points) / 4)
            right_center = (sum(p[0] for p in right_points) / 4,
                           sum(p[1] for p in right_points) / 4)

            # Gaze point
            gaze_x = (left_center[0] + right_center[0]) / 2
            gaze_y = (left_center[1] + right_center[1]) / 2

            # Map to screen (mirrored X)
            target_x = int((1 - gaze_x) * screen_w)
            target_y = int(gaze_y * screen_h)

            # Smooth
            smooth_x = smooth_x * smoothing + target_x * (1 - smoothing)
            smooth_y = smooth_y * smoothing + target_y * (1 - smoothing)

            # Move cursor
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # Calculate EAR for blink detection
            def ear(indices):
                p = [landmarks.landmark[i] for i in indices]
                v1 = np.linalg.norm([p[1].x - p[5].x, p[1].y - p[5].y])
                v2 = np.linalg.norm([p[2].x - p[4].x, p[2].y - p[4].y])
                h = np.linalg.norm([p[0].x - p[3].x, p[0].y - p[3].y])
                return (v1 + v2) / (2 * h) if h else 1

            left_ear = ear(LEFT_EYE)
            right_ear = ear(RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            # Detect blink
            import time
            current_time = time.time()
            if avg_ear < blink_threshold:
                if current_time - last_blink_time > blink_cooldown:
                    pyautogui.click()
                    last_blink_time = current_time
                    print("Click!")

            # Draw visualization
            h, w = frame.shape[:2]
            lx, ly = int(left_center[0] * w), int(left_center[1] * h)
            rx, ry = int(right_center[0] * w), int(right_center[1] * h)
            cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)
            cv2.circle(frame, (rx, ry), 5, (0, 255, 0), -1)

            status = "Open" if avg_ear > blink_threshold else "Closed"
            cv2.putText(frame, f"Eyes: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show
        cv2.imshow("EyeCursor Quickstart", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    quickstart()

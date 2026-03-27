"""
EyeCursor - Eye Tracking Mouse Control
======================================

An accessibility-focused application that allows users to control
the mouse cursor using eye movement and perform clicks using blinking.

Usage:
    python main.py

Controls:
    C - Calibrate
    R - Reset
    P - Pause/Resume
    Q - Quit

Author: EyeCursor Team
License: MIT
"""

import cv2
import numpy as np
import pyautogui
from eye_tracker import EyeTracker, EyeData
from blink_detector import BlinkDetector, IntentionalBlinkDetector, BlinkType
from cursor_controller import CursorController
from calibration import Calibrator
from ui import UI, AppState
import config


def print_instructions():
    """Print startup instructions."""
    print("=" * 60)
    print("  EyeCursor - Eye Tracking Mouse Control")
    print("=" * 60)
    print()
    print("Starting up...")
    print()
    print("Controls:")
    print("  C - Start calibration")
    print("  R - Reset cursor position")
    print("  P - Pause/Resume tracking")
    print("  Q - Quit")
    print()
    print("Mouse Actions:")
    print("  Long blink (>300ms) - Left click")
    print("  Double blink - Right click")
    print()
    print("Tips:")
    print("  - Ensure good lighting on your face")
    print("  - Keep your head relatively still")
    print("  - Calibrate before first use")
    print("  - Practice blinking slowly for clicks")
    print()
    print("=" * 60)


def main():
    """Main application loop."""
    print_instructions()

    # Initialize components
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print("Error: Could not open camera!")
        print("Please check that your webcam is connected.")
        return

    print("Camera initialized successfully.")

    # Initialize modules
    eye_tracker = EyeTracker()
    blink_detector = IntentionalBlinkDetector(
        min_blink_duration_ms=300,
        threshold=config.BLINK_THRESHOLD,
        cooldown_ms=config.BLINK_COOLDOWN_MS,
        double_blink_interval_ms=config.DOUBLE_BLINK_INTERVAL_MS
    )
    cursor_controller = CursorController()
    calibrator = Calibrator()
    ui = UI()

    # Set up blink callbacks
    def on_single_blink():
        if ui.state == AppState.RUNNING:
            cursor_controller.click('left')
            print("Left click!")

    def on_double_blink():
        if ui.state == AppState.RUNNING:
            cursor_controller.right_click()
            print("Right click!")

    blink_detector.on_single_blink = on_single_blink
    blink_detector.on_double_blink = on_double_blink

    # State variables
    running = True
    last_blink_event = None

    print("Starting main loop...")
    print("Press 'Q' to quit, 'C' to calibrate")

    try:
        while running:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame with eye tracker
            eye_data = eye_tracker.process_frame(frame)

            # Draw eye tracking visualization
            if eye_data.face_detected:
                frame = eye_tracker.draw_landmarks(frame, eye_data)

            # Handle calibration
            if ui.state == AppState.CALIBRATING:
                if eye_data.gaze_point:
                    complete = calibrator.update(eye_data.gaze_point[0],
                                                 eye_data.gaze_point[1])
                    if complete:
                        # Calibration complete - update cursor controller
                        bounds = calibrator.get_bounds()
                        cursor_controller.set_calibration(*bounds)
                        ui.set_state(AppState.RUNNING)
                        print("Calibration complete!")
                        print(f"Bounds: X=[{bounds[0]:.3f}, {bounds[1]:.3f}], "
                              f"Y=[{bounds[2]:.3f}, {bounds[3]:.3f}]")

                # Draw calibration UI
                frame = calibrator.draw(frame)

            # Handle running state
            elif ui.state == AppState.RUNNING:
                # Update blink detection
                if eye_data.face_detected:
                    blink_event = blink_detector.update(eye_data.left_ear,
                                                       eye_data.right_ear)

                    # Update cursor position
                    if eye_data.gaze_point:
                        cursor_controller.update_position(eye_data.gaze_point[0],
                                                         eye_data.gaze_point[1])

                    # Draw blink indicator
                    if blink_event:
                        blink_type = "LEFT CLICK" if blink_event.blink_type == BlinkType.SINGLE else "RIGHT CLICK"
                        frame = ui.draw_blink_indicator(frame, True, blink_type)
                else:
                    # No face detected - pause cursor updates
                    pass

            # Draw UI elements
            face_detected = eye_data.face_detected
            cursor_text = cursor_controller.get_status_text()
            blink_text = blink_detector.get_state_text()
            frame = ui.draw_status_bar(frame, face_detected, cursor_text, blink_text)

            if ui.state == AppState.CALIBRATING:
                frame = calibrator.draw(frame)
            elif ui.state == AppState.PAUSED:
                frame = ui.draw_pause_screen(frame)
            else:
                frame = ui.draw_help_overlay(frame)

            # Show frame
            ui.show(frame)

            # Handle key presses
            key = ui.get_key(1)

            if key == ord('q') or key == ord('Q'):
                running = False
            elif key == ord('c') or key == ord('C'):
                print("Starting calibration...")
                calibrator.start()
                ui.set_state(AppState.CALIBRATING)
            elif key == ord('r') or key == ord('R'):
                print("Resetting cursor...")
                cursor_controller.reset()
            elif key == ord('p') or key == ord('P'):
                if ui.state == AppState.RUNNING:
                    ui.set_state(AppState.PAUSED)
                    print("Paused")
                elif ui.state == AppState.PAUSED:
                    ui.set_state(AppState.RUNNING)
                    print("Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print("Shutting down...")
        cap.release()
        eye_tracker.release()
        ui.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()

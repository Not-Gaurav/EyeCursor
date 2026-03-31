"""
Configuration settings for EyeCursor application.
These values can be tuned based on user preference and environment.
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Eye tracking settings
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# Cursor smoothing settings
# Higher = smoother but more lag; Lower = more responsive but jittery
SMOOTHING_FACTOR = 0.3
VELOCITY_THRESHOLD = 0.5  # Minimum movement to register

# Movement scaling
# Adjust these to control cursor speed
MOVEMENT_SCALE_X = 2.5
MOVEMENT_SCALE_Y = 2.0

# Eyeroll settings
# Eyeroll allows fine cursor control by tracking iris position within eye
EYEROLL_AMPLIFICATION = 0.15  # Higher = more sensitive to eye rolling
EYEROLL_SMOOTHING = 0.5  # Smoothing for eyeroll data (0-1)

# Blink detection settings
BLINK_THRESHOLD = 0.25  # EAR threshold for blink detection
BLINK_CONSECUTIVE_FRAMES = 2  # Frames to confirm blink
BLINK_COOLDOWN_MS = 500  # Prevent multiple clicks from one blink
DOUBLE_BLINK_INTERVAL_MS = 400  # Time between blinks for double-click

# Calibration settings
CALIBRATION_POINTS = [
    (0.5, 0.5),   # Center
    (0.2, 0.2),   # Top-left area
    (0.8, 0.2),   # Top-right area
    (0.2, 0.8),   # Bottom-left area
    (0.8, 0.8),   # Bottom-right area
]
CALIBRATION_WAIT_MS = 2000  # Time to wait at each calibration point

# UI settings
WINDOW_NAME = "EyeCursor - Eye Tracking Mouse Control"
FONT_SCALE = 0.7
FONT_COLOR = (0, 255, 0)  # Green
FONT_THICKNESS = 2

# Colors for visualization
COLOR_EYE_LANDMARKS = (0, 255, 0)  # Green
COLOR_FACE_MESH = (0, 255, 255)  # Yellow
COLOR_CURSOR = (0, 0, 255)  # Red
COLOR_CALIBRATION = (255, 0, 0)  # Blue
COLOR_TEXT = (255, 255, 255)  # White
COLOR_TEXT_BG = (0, 0, 0)  # Black

# Eye landmark indices for MediaPipe Face Mesh
# Left eye
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_EYE_LEFT = 362
LEFT_EYE_RIGHT = 263

# Right eye
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133

# Iris landmarks (for more precise tracking)
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

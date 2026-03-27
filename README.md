# EyeCursor - Eye Tracking Mouse Control

An accessibility-focused Python application that enables mouse control through eye movement and clicking through intentional blinking.

![EyeCursor Demo](demo.png)

## Overview

EyeCursor uses computer vision and facial landmark detection to track eye gaze and translate it into cursor movement. It's designed with HCI principles in mind, prioritizing:

- **Accessibility**: Hands-free operation for users with motor disabilities
- **Usability**: Simple calibration and intuitive controls
- **Reliability**: Smooth cursor movement and intentional blink detection
- **Performance**: Real-time operation on standard hardware

## Features

- **Eye Tracking Cursor Control**: Move the cursor by looking where you want to go
- **Blink Detection**: Long blink (>300ms) for left click, double blink for right click
- **Calibration System**: 5-point calibration for accurate screen mapping
- **Smooth Movement**: Exponential moving average eliminates cursor jitter
- **Visual Feedback**: Real-time display of eye tracking status and detection points
- **Safety Features**: Cooldown periods prevent accidental multiple clicks

## How It Works

### Eye Tracking

EyeCursor uses **MediaPipe Face Mesh** to detect 468 facial landmarks in real-time. For eye tracking, it specifically uses:

- **Iris landmarks**: Precise center points of each eye (4 points per iris)
- **Eye aspect ratio (EAR)**: Vertical/horizontal eye ratio for blink detection
- **Gaze point**: Average of left and right eye centers

The gaze point is mapped from normalized camera coordinates (0-1) to screen coordinates using calibration data.

### Blink Detection

Intentional blinks are distinguished from natural blinking using:

1. **Eye Aspect Ratio (EAR)**: Measures eye openness
   - Open eye: EAR ≈ 0.25-0.35
   - Closed eye: EAR < 0.25

2. **Duration filtering**: Intentional blinks last longer (>300ms)
3. **Cooldown periods**: Prevents multiple triggers from single blink
4. **Double-blink pattern**: Two blinks within 400ms = right click

### Cursor Smoothing

Raw gaze data is noisy. EyeCursor applies **exponential moving average**:

```
position = old_position × smoothing_factor + target × (1 - smoothing_factor)
```

This creates smooth, natural cursor movement without lag.

## Installation

### Requirements

- Python 3.8+
- Webcam
- 4GB RAM minimum
- Works on Windows, macOS, and Linux

### Step-by-Step Setup

1. **Clone or download the project:**
   ```bash
   cd eyecursor
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### First-Time Setup

1. Ensure good lighting on your face
2. Position yourself about 50-70cm from the camera
3. Keep your head relatively still during use

### Controls

| Key | Action |
|-----|--------|
| `C` | Start calibration |
| `R` | Reset cursor position |
| `P` | Pause/Resume tracking |
| `Q` | Quit |

### Calibration

1. Press `C` to start calibration
2. Look at each circle that appears (5 points)
3. Keep your gaze steady on each point
4. Calibration completes automatically

### Mouse Actions

| Action | How to Perform |
|--------|----------------|
| Move cursor | Look at target location |
| Left click | Close eyes for >300ms, then open |
| Right click | Blink twice quickly (<400ms between) |

## Tips for Best Accuracy

### Environment

- **Lighting**: Bright, even lighting on your face
- **Position**: Center yourself in the frame
- **Distance**: 50-70cm from camera
- **Background**: Plain background works best

### Usage

- **Head position**: Keep head still, move only eyes
- **Blinking**: Practice long, deliberate blinks for clicks
- **Double-click**: Blink-blink quickly for right click
- **Calibration**: Recalibrate if accuracy degrades

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Cursor jitter | Recalibrate; ensure good lighting |
| False clicks | Blink longer/more deliberately |
| Missed clicks | Blink faster (but still >300ms) |
| Face not detected | Check lighting; move closer to camera |
| Cursor drifts | Recalibrate; check head position |

## Project Structure

```
eyecursor/
├── main.py              # Application entry point
├── config.py            # Configuration settings
├── eye_tracker.py       # Eye detection and tracking
├── blink_detector.py    # Blink detection logic
├── cursor_controller.py # Cursor movement and smoothing
├── calibration.py       # Calibration system
├── ui.py                # User interface
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Technical Details

### Dependencies

- **OpenCV**: Video capture and image processing
- **MediaPipe**: Face mesh detection (468 landmarks)
- **PyAutoGUI**: Mouse cursor control
- **NumPy**: Numerical operations

### Performance

- **Frame rate**: 30 FPS on modern laptops
- **Latency**: ~50-100ms (eye movement to cursor response)
- **CPU usage**: ~15-25% on quad-core processors

### Security Note

PyAutoGUI's failsafe is disabled for this demo. To stop the program if the cursor becomes uncontrollable:
- Press `Alt+Tab` to switch windows
- Press `Q` in the EyeCursor window
- Or use `Ctrl+C` in the terminal

## Customization

Edit `config.py` to adjust:

- `SMOOTHING_FACTOR`: Cursor smoothness (0-1)
- `BLINK_THRESHOLD`: Sensitivity of blink detection
- `MOVEMENT_SCALE_X/Y`: Cursor speed
- `BLINK_COOLDOWN_MS`: Time between clicks

## Future Enhancements

- [ ] Multi-monitor support
- [ ] Adjustable sensitivity settings
- [ ] Click-and-drag gesture
- [ ] Scroll gesture (look up/down at screen edge)
- [ ] Voice command integration
- [ ] Profile system for multiple users

## License

MIT License - See LICENSE file for details

## Acknowledgments

- MediaPipe by Google for face mesh detection
- PyAutoGUI by Al Sweigart for mouse control
- OpenCV team for computer vision tools

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify your webcam is working
4. Try recalibrating in different lighting


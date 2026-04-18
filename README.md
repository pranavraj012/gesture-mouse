[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/mediapipe-blue.svg)](https://github.com/google/mediapipe)
[![OpenCV](https://img.shields.io/badge/opencv-blue.svg)](https://opencv.org)
[![PyAutoGUI](https://img.shields.io/badge/pyautogui-blue.svg)](https://pyautogui.readthedocs.io/)

# gesture-mouse

Control your mouse cursor using hand gestures and a webcam. `hand_track.py` uses OpenCV, the MediaPipe Tasks API (hand landmarker), and PyAutoGUI to map hand landmarks to mouse actions.

## Features
- Absolute hand-to-screen mapping for predictable cursor control
- Click and drag detection using fingertip proximity
- Smooth scrolling (V-pose and thumb-based)
- Real-time, multithreaded processing with adjustable sensitivity
- Optional handwriting mode with a floating drawing panel (always-on-top) and gesture controls; it is disabled by default and can be enabled in `config.toml`

## Supported Gestures

Below is the quick reference for what each gesture does.

- **Cursor move**: Move your hand to move the cursor. The script tracks the ring-finger base and maps it to screen coordinates using absolute mapping.
- **Left click / Drag - Index + Thumb pinch**: Briefly pinch your index fingertip to your thumb to perform a left click. Hold the pinch for ~0.25s to start a drag (mouse down); release to drop (mouse up).
- **Right click / Pinch scroll - Middle + Thumb pinch**: Pinch your middle fingertip to your thumb to enter right-pinch mode. Holding for ~0.18s triggers a right-click; while pinched, vertical movement of the middle finger performs scrolling.
- **V-pose scrolling - Index + Middle extended**: Extend the index and middle fingers while curling the ring and pinky (a "V" shape). Tilt the V up/down relative to the wrist to perform smooth orientation-based scrolling.
- **Back navigation - Thumb + Pinky pinch-hold**: Hold thumb and pinky pinch to trigger app/browser back action.
- **Enter key (normal mode) - Ring + Thumb pinch-hold**: With one hand visible and handwriting mode off, hold ring+thumb pinch to send Enter (useful for Google/YouTube search submit).
- **Pause / Resume - Rock-star pose**: Hold thumb, index, and pinky extended while middle and ring are closed for about 0.4s, then release to toggle pause. Repeat the same hold-and-release to resume.
- **Handwriting mode (optional) - Two-hand ring+thumb hold**: Enable this first in `config.toml` (`handwriting_enabled = true`). With two hands visible, hold ring+thumb pinch on the left-side hand to toggle handwriting mode on/off. In handwriting mode, a floating handwriting window pops up (can stay on top while you are in another app) and shows a live pen cursor/crosshair so writing is easier to track. Use right-side hand index+thumb pinch as pen-down to draw, mapped to the floating panel. Left-side hand gestures: index+thumb hold to submit text, middle+thumb hold to clear, pinky+thumb hold to send backspace.
- **Notes & tuning**: Gesture detection uses adaptive, size-normalized thresholds and hold times (`touch_threshold`, `drag_hold`, `right_click_hold`) from `config.toml`. If gestures are unreliable, improve lighting, keep your hand centered, or tweak these values in the config.

## Requirements
- Python 3.10 or newer (3.10 recommended)
- Install the dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pranavraj012/gesture-mouse.git
cd gesture-mouse
````

2. (Recommended) Create and activate a virtual environment:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Model file

On first run the script will attempt to download the MediaPipe model `hand_landmarker.task` into the repository root if it is not already present. The model file is large and platform-specific and is ignored by the repository (`.gitignore`). If you want offline usage, you may place a downloaded `hand_landmarker.task` next to `hand_track.py`.

## Usage

Run the main script with an active webcam:

```powershell
python hand_track.py
```

## Configuration

* Edit `config.toml` to tweak gesture thresholds, scroll speed, cursor movement responsiveness, zoom behavior, camera size, and timing values.
* `hand_track.py` loads `config.toml` automatically and falls back to built-in defaults if the file is missing.
* Change the camera index in `hand_track.py` (`cv2.VideoCapture`) if you have multiple cameras.

Example values you can tune in `config.toml` include `v_scroll_sensitivity`, `v_orientation_gain`, `cursor_move_alpha_scale`, and `cursor_jitter_threshold`.

For pause gesture tuning, use `pause_hold`, `pause_cooldown`, `pause_open_ratio`, `pause_closed_ratio`, and `pause_thumb_open_ratio`.

## Pause Calibration (Optional)

If your rock-star pause gesture feels inconsistent, calibrate from a reference photo:

```powershell
python calibrate_pause_pose.py --image reference-stop-image.png --out pose_calibration_output.json
```

The script prints measured landmark ratios and recommended threshold values for `config.toml`.

For handwriting feel tuning, use `handwriting_pen_threshold_scale` (higher = easier pen-down), `handwriting_jitter_threshold` (higher = more jitter filtering), `handwriting_move_alpha_floor/ceiling/scale` (higher = more responsive pen movement), and `handwriting_pen_release_grace` (higher = fewer broken strokes during brief pinch loss).

Handwriting is disabled by default (`handwriting_enabled = false`) to avoid accidental trigger for users who only want mouse gestures.

If you enable handwriting mode, install the optional OCR dependencies:

```powershell
python -m pip install pytesseract
```

and install the Tesseract engine on your system. If these are not installed, handwriting mode still works for drawing/clearing/toggling but submit will show an OCR-unavailable message.

## Contributing

Open an issue or submit a pull request.


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

## Supported gestures

- **Cursor move**: Move your hand to move the cursor. The script tracks the ring-finger base and maps it to screen coordinates using absolute mapping.
- **Left click / Drag — Index + Thumb pinch**: Briefly pinch your index fingertip to your thumb to perform a left click. Hold the pinch for ~0.25s to start a drag (mouse down); release to drop (mouse up).
- **Right click / Pinch scroll — Middle + Thumb pinch**: Pinch your middle fingertip to your thumb to enter right-pinch mode. Holding for ~0.18s triggers a right-click; while pinched, vertical movement of the middle finger performs scrolling.
- **V-pose scrolling — Index + Middle extended**: Extend the index and middle fingers while curling the ring and pinky (a "V" shape). Tilt the V up/down relative to the wrist to perform smooth orientation-based scrolling.
- **Notes & tuning**: Gesture detection uses adaptive, size-normalized thresholds and hold times (`touch_threshold`, `drag_hold`, `right_click_hold`) defined in `hand_track.py`. If gestures are unreliable, improve lighting, keep your hand centered, or tweak these values in the script.

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

* Change the camera index in `hand_track.py` (`cv2.VideoCapture`) if you have multiple cameras.
* Tweak thresholds such as `touch_threshold`, detection confidence, and model complexity directly in the script for your setup.

## Contributing

Open an issue or submit a pull request.


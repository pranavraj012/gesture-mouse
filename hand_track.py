"""
hand_track.py — Enhanced hand-tracking mouse controller
═══════════════════════════════════════════════════════
Gestures:
  • Index+Thumb pinch      → Left click (quick) / Drag (hold 0.25s)
  • Double pinch           → Double-click
    • Middle+Thumb pinch-hold→ Right-click
  • Index+Middle+Ring pinch→ Middle-click
  • V-pose (index+middle)  → Scroll: vertical (centroid-Y) + horizontal (centroid-X)
    • 3-finger hold          → Pause/resume tracking toggle
    • Middle+Thumb scroll    → Legacy delta-Y scroll (kept as fallback)

Improvements over v1:
  • Gesture history buffer (10-frame majority vote) — kills false triggers
  • Click cooldown (300 ms) — prevents accidental double-fires
  • Intentional double-click via two quick pinches
  • Scroll direction lock (300 ms) for V-pose
  • Horizontal scroll via hscroll() in V-pose
    • Right-click via middle+thumb hold (with motion guard)
  • Middle-click via three-finger pinch
    • 3-finger hold = pause tracking
  • External YAML-style config dict (edit CONFIG below)
  • Fixed scroll_queue bug → shows accumulator in overlay
    • Two-hand zoom: Ctrl+wheel style zoom mapping
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import urllib.request
import os
import numpy as np
import pyautogui
import time
import threading
from collections import deque

try:
    import pytesseract
except Exception:
    pytesseract = None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — defaults are overridden by config.toml when present
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Camera / performance
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 60,

    # Camera / mapping
    "inner_area_percent": 0.70,   # fraction of camera view mapped to full screen

    # Pinch thresholds
    "touch_threshold": 0.19,      # normalized distance for pinch detection
    "three_finger_threshold": 0.22,  # slightly looser for 3-finger middle-click

    # Drag
    "drag_hold": 0.25,            # seconds of pinch before drag begins
    "drag_release_grace": 0.18,   # grace period for transient drag releases

    # Click
    "click_cooldown": 0.30,       # min seconds between left clicks
    "double_click_window": 0.40,  # max gap between two pinches = double-click

    # Right-click (middle+thumb hold)
    "right_click_hold": 0.18,     # hold time to fire right-click

    # Back navigation gesture (thumb+pinky pinch-hold)
    "back_gesture_threshold": 0.20,
    "back_gesture_hold": 0.20,
    "back_gesture_cooldown": 0.70,
    "back_action": "alt_left",  # alt_left | x1

    # Enter key gesture (normal mode, single-hand ring+thumb pinch-hold)
    "enter_gesture_threshold_scale": 1.0,
    "enter_gesture_hold": 0.22,
    "enter_gesture_cooldown": 0.80,

    # Pause / resume gesture (single-hand 3-finger extension hold)
    "pause_hold": 0.40,
    "pause_cooldown": 0.70,

    # Handwriting mode
    "handwriting_enabled": False,
    "handwriting_toggle_threshold": 0.20,
    "handwriting_toggle_hold": 0.25,
    "handwriting_submit_hold": 0.18,
    "handwriting_clear_hold": 0.18,
    "handwriting_backspace_hold": 0.18,
    "handwriting_backspace_threshold_scale": 1.0,
    "handwriting_action_cooldown": 0.45,
    "handwriting_panel_width_ratio": 0.55,
    "handwriting_panel_height_ratio": 0.30,
    "handwriting_panel_bottom_margin": 16,
    "handwriting_stroke_thickness": 4,
    "handwriting_pen_threshold_scale": 1.16,
    "handwriting_jitter_threshold": 0.0015,
    "handwriting_move_alpha_floor": 0.28,
    "handwriting_move_alpha_ceiling": 0.86,
    "handwriting_move_alpha_scale": 5.4,
    "handwriting_pen_release_grace": 0.08,
    "handwriting_window_width": 720,
    "handwriting_window_height": 300,
    "handwriting_window_x": 580,
    "handwriting_window_y": 740,
    "handwriting_window_topmost": True,

    # Scroll (V-pose orientation)
    "v_orientation_threshold": 0.01,
    "v_orientation_gain": 160.0,   # slightly faster V-pose scroll
    "v_orientation_scroll": 6.0,
    "v_h_gain": 160.0,             # horizontal scroll gain
    "v_h_scroll": 6.0,             # max horizontal scroll per tick
    "scroll_lock_duration": 0.30, # seconds to lock scroll axis after first tick

    # Scroll (middle+thumb legacy)
    "scroll_threshold": 0.001,
    "v_scroll_sensitivity": 360.0,  # slightly faster scrolling
    "max_scroll_step": 32,  # slightly snappier scroll
    "v_scroll_multiplier": 2.35,  # legacy scroll multiplier

    # Gesture interaction guard
    "right_click_suppress_after_middle_click": 0.25,

    # Cursor movement tuning
    "cursor_jitter_threshold": 0.0022,
    "cursor_drag_alpha_floor": 0.25,
    "cursor_drag_alpha_ceiling": 0.85,
    "cursor_drag_alpha_scale": 3.0,
    "cursor_drag_sleep": 0.008,
    "cursor_move_alpha_floor": 0.12,
    "cursor_move_alpha_ceiling": 0.62,
    "cursor_move_alpha_scale": 2.6,
    "cursor_move_sleep": 0.006,

    # Gesture history buffer
    # NOTE: clicks use raw threshold (no buffer) for instant response.
    # Buffer is only used for fist/v-pose/middle-pinch to debounce those.
    "history_frames": 6,          # rolling window size
    "history_min_votes": 4,       # votes needed to confirm slow gestures (fist, V, middle)

    # Stabilization after drag
    "stabilize_duration": 0.20,

    # Finger extension heuristic
    # Lower threshold makes extension detection (pinky/index) easier
    "extension_threshold": 0.16,

    # Fist detection: all tips close to palm
    "fist_threshold": 0.30,       # max ratio of hand_size for each tip-to-wrist dist

    # Two-hand zoom (touch index fingertips, then move vertically)
    "zoom_touch_threshold": 0.22,  # normalized tip-to-tip distance to enter zoom mode
    "zoom_motion_threshold": 0.004,
    "zoom_scroll_gain": 320.0,     # pixels of vertical motion per wheel tick
    "zoom_scroll_cap": 10,
    "zoom_cooldown": 0.05,
}


def load_config(config_path):
    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(config_path):
        return config

    try:
        current_section = None
        with open(config_path, "r", encoding="utf-8") as config_file:
            for raw_line in config_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    continue
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if "#" in value:
                    value = value.split("#", 1)[0].strip()

                if value.startswith('"') and value.endswith('"'):
                    parsed_value = value[1:-1]
                elif value.lower() in ("true", "false"):
                    parsed_value = value.lower() == "true"
                else:
                    try:
                        parsed_value = int(value)
                    except ValueError:
                        try:
                            parsed_value = float(value)
                        except ValueError:
                            parsed_value = value

                if key in config:
                    config[key] = parsed_value
    except Exception as exc:
        print(f"Failed to load config.toml, using defaults: {exc}")

    return config


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
CONFIG = load_config(CONFIG_PATH)


def trigger_back_action():
    action = str(CONFIG.get("back_action", "alt_left")).strip().lower()
    if action == "x1":
        try:
            pyautogui.click(button="x1")
            return
        except Exception:
            pass
    # Fallback and default behavior for broad app support
    pyautogui.hotkey("alt", "left")


def hand_metrics(lms):
    wrist = lms[HandLandmark.WRIST]
    index_tip = lms[HandLandmark.INDEX_FINGER_TIP]
    middle_tip = lms[HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = lms[HandLandmark.RING_FINGER_TIP]
    pinky_tip = lms[HandLandmark.PINKY_TIP]
    thumb_tip = lms[HandLandmark.THUMB_TIP]
    hand_size = lm_dist(wrist, middle_tip)
    adaptive_thr = CONFIG["touch_threshold"] * hand_size
    return {
        "wrist": wrist,
        "index_tip": index_tip,
        "middle_tip": middle_tip,
        "ring_tip": ring_tip,
        "pinky_tip": pinky_tip,
        "thumb_tip": thumb_tip,
        "hand_size": hand_size,
        "adaptive_thr": adaptive_thr,
        "idx_thumb_d": lm_dist(index_tip, thumb_tip),
        "mid_thumb_d": lm_dist(middle_tip, thumb_tip),
        "ring_thumb_d": lm_dist(ring_tip, thumb_tip),
        "pinky_thumb_d": lm_dist(pinky_tip, thumb_tip),
    }


def ensure_canvas(canvas, width, height):
    if canvas is None or canvas.shape[1] != width or canvas.shape[0] != height:
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    return canvas


def render_handwriting_window(canvas, status_text="", cursor_point=None, pen_down=False):
    header_h = 42
    footer_h = 30
    panel_h, panel_w = canvas.shape[:2]
    out = np.ones((panel_h + header_h + footer_h, panel_w, 3), dtype=np.uint8) * 248

    preview = canvas.copy()
    if cursor_point is not None:
        cx, cy = cursor_point
        cx = int(np.clip(cx, 0, panel_w - 1))
        cy = int(np.clip(cy, 0, panel_h - 1))
        color = (20, 90, 235) if pen_down else (55, 175, 70)
        cv2.circle(preview, (cx, cy), 8 if pen_down else 6, color, 2, cv2.LINE_AA)
        cv2.line(preview, (cx - 10, cy), (cx + 10, cy), color, 1, cv2.LINE_AA)
        cv2.line(preview, (cx, cy - 10), (cx, cy + 10), color, 1, cv2.LINE_AA)

    out[header_h:header_h + panel_h] = preview
    cv2.rectangle(out, (0, header_h), (panel_w - 1, header_h + panel_h - 1), (70, 210, 245), 2)

    cv2.putText(
        out,
        "Floating Handwriting",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (40, 40, 40),
        2,
    )
    cv2.putText(
        out,
        "Cmd hand: ring+thumb toggle | index submit | middle clear | pinky backspace",
        (10, panel_h + header_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 60, 60),
        1,
    )

    if status_text:
        cv2.putText(
            out,
            status_text[:72],
            (260, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 120, 30),
            2,
        )

    return out


def show_handwriting_window(canvas, status_text="", cursor_point=None, pen_down=False):
    window_name = "Handwriting Panel"
    display_img = render_handwriting_window(canvas, status_text, cursor_point, pen_down)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        int(CONFIG["handwriting_window_width"]),
        int(CONFIG["handwriting_window_height"]),
    )
    cv2.moveWindow(
        window_name,
        int(CONFIG["handwriting_window_x"]),
        int(CONFIG["handwriting_window_y"]),
    )
    cv2.imshow(window_name, display_img)

    if bool(CONFIG.get("handwriting_window_topmost", True)):
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass


def ocr_canvas(canvas):
    if pytesseract is None:
        return "", "OCR unavailable (install pytesseract + Tesseract)"
    try:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Upscale strokes before OCR; this usually helps handwritten text.
        scale = 2
        up = cv2.resize(blur, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            up,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )

        candidates = [
            (otsu, 7),
            (otsu, 8),
            (adaptive, 7),
            (adaptive, 8),
        ]

        best_text = ""
        best_score = -1e9

        for img, psm in candidates:
            config = f"--oem 1 --psm {psm}"
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

            words = []
            confs = []
            for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
                token = str(raw_text).strip()
                if not token:
                    continue
                words.append(token)
                try:
                    c = float(raw_conf)
                    if c >= 0:
                        confs.append(c)
                except Exception:
                    pass

            if not words:
                continue

            text = " ".join(words).strip()
            mean_conf = float(np.mean(confs)) if confs else 0.0
            length_bonus = min(len(text), 24) * 0.45
            score = mean_conf + length_bonus

            if score > best_score:
                best_score = score
                best_text = text

        normalized = " ".join(best_text.strip().split())
        return normalized, ""
    except Exception as exc:
        return "", f"OCR failed: {exc}"

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe model download / setup
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_FILENAME):
    try:
        print(f"Downloading model to {MODEL_FILENAME}…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
    except Exception as e:
        print("Failed to download model:", e)
        raise

base_options = python.BaseOptions(model_asset_path=MODEL_FILENAME)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                  # enable 2-hand detection for zoom gesture
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.7,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────────────────────────────────────
# Webcam + PyAutoGUI setup
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["camera_fps"])
ret, frame = cap.read()
if not ret:
    print("Failed to capture video")
    exit(1)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_width, screen_height = pyautogui.size()

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def calculate_margins(fw, fh, pct):
    return fw * (1 - pct) / 2, fh * (1 - pct) / 2

def convert_to_screen(x, y, fw, fh, mw, mh):
    x = float(np.clip(np.interp(np.clip(x, mw, fw - mw), (mw, fw - mw), (0, screen_width)), 0, screen_width))
    y = float(np.clip(np.interp(np.clip(y, mh, fh - mh), (mh, fh - mh), (0, screen_height)), 0, screen_height))
    return x, y

def lm_dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

def is_extended(lms, tip_idx, pip_idx, hand_size):
    return lm_dist(lms[tip_idx], lms[pip_idx]) > hand_size * CONFIG["extension_threshold"]

def is_fist(lms, hand_size):
    """All four fingertips are close to the palm (wrist)."""
    tips = [HandLandmark.INDEX_FINGER_TIP, HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.RING_FINGER_TIP, HandLandmark.PINKY_TIP]
    wrist = lms[HandLandmark.WRIST]
    return all(lm_dist(lms[t], wrist) < hand_size * CONFIG["fist_threshold"] for t in tips)

# ─────────────────────────────────────────────────────────────────────────────
# Gesture history buffer — majority-vote smoother
# ─────────────────────────────────────────────────────────────────────────────
class GestureBuffer:
    """
    Keeps a rolling deque of the last N frames' detected gesture label.
    Returns a confirmed gesture only when it has >= min_votes in the window.
    Prevents single-frame jitter from firing actions.
    """
    def __init__(self, size=10, min_votes=7):
        self.buf = deque(maxlen=size)
        self.min_votes = min_votes

    def push(self, label: str):
        self.buf.append(label)

    def confirmed(self, label: str) -> bool:
        return self.buf.count(label) >= self.min_votes

    def dominant(self) -> str:
        if not self.buf:
            return "none"
        return max(set(self.buf), key=self.buf.count)

# ─────────────────────────────────────────────────────────────────────────────
# Cursor movement thread
# ─────────────────────────────────────────────────────────────────────────────
class CursorMovementThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_x, self.current_y = pyautogui.position()
        self.target_x, self.target_y = self.current_x, self.current_y
        self.running = True
        self.active = False
        self.jitter_threshold = CONFIG["cursor_jitter_threshold"]
        self.dragging = False

    def run(self):
        while self.running:
            if self.active:
                dx = self.target_x - self.current_x
                dy = self.target_y - self.current_y
                distance = np.hypot(dx, dy)
                screen_diagonal = np.hypot(screen_width, screen_height)
                jitter_t = self.jitter_threshold * (0.2 if self.dragging else 1.0)
                if distance / screen_diagonal > jitter_t:
                    if self.dragging:
                        alpha = min(CONFIG["cursor_drag_alpha_ceiling"], max(CONFIG["cursor_drag_alpha_floor"], (distance / screen_diagonal) * CONFIG["cursor_drag_alpha_scale"]))
                        sleep_time = CONFIG["cursor_drag_sleep"]
                    else:
                        alpha = min(CONFIG["cursor_move_alpha_ceiling"], max(CONFIG["cursor_move_alpha_floor"], (distance / screen_diagonal) * CONFIG["cursor_move_alpha_scale"]))
                        sleep_time = CONFIG["cursor_move_sleep"]
                    self.current_x += dx * alpha
                    self.current_y += dy * alpha
                    try:
                        pyautogui.moveTo(self.current_x, self.current_y, _pause=False)
                    except Exception:
                        pass
                else:
                    sleep_time = 0.01
                time.sleep(sleep_time)
            else:
                time.sleep(0.05)

    def update_target(self, x, y):
        self.target_x, self.target_y = x, y

    def activate(self):  self.active = True
    def deactivate(self): self.active = False
    def stop(self):      self.running = False

# ─────────────────────────────────────────────────────────────────────────────
# Scroll thread with accumulator + inertia
# ─────────────────────────────────────────────────────────────────────────────
class ScrollThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.lock = threading.Lock()
        self.v_accumulator = 0.0   # vertical
        self.h_accumulator = 0.0   # horizontal
        self.last_input_time = time.time()
        self.running = True
        self.inertia = 0.95
        self.inertia_threshold = 0.01

    def run(self):
        while self.running:
            with self.lock:
                va, ha = self.v_accumulator, self.h_accumulator
            try:
                v_int = int(va)
                if v_int:
                    pyautogui.scroll(v_int)
                    with self.lock:
                        self.v_accumulator -= v_int
                        self.last_input_time = time.time()
                else:
                    if (time.time() - self.last_input_time) > 0.06 and abs(va) > self.inertia_threshold:
                        with self.lock:
                            self.v_accumulator *= self.inertia

                h_int = int(ha)
                if h_int:
                    pyautogui.hscroll(h_int)
                    with self.lock:
                        self.h_accumulator -= h_int
                        self.last_input_time = time.time()
                else:
                    if (time.time() - self.last_input_time) > 0.06 and abs(ha) > self.inertia_threshold:
                        with self.lock:
                            self.h_accumulator *= self.inertia
            except Exception as e:
                print("Scroll thread error:", e)
            time.sleep(0.01)

    def add_scroll(self, v=0.0, h=0.0):
        with self.lock:
            self.h_accumulator += float(h)
            self.v_accumulator += float(v)
            self.last_input_time = time.time()

    def stop(self): self.running = False

# ─────────────────────────────────────────────────────────────────────────────
# Launch background threads
# ─────────────────────────────────────────────────────────────────────────────
movement_thread = CursorMovementThread()
scroll_thread = ScrollThread()
movement_thread.start()
scroll_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# State variables
# ─────────────────────────────────────────────────────────────────────────────
# Left pinch / drag
left_pinch_active = False
left_pinch_start = 0.0
left_is_dragging = False
left_release_time = None

# Double-click tracking
last_click_time = 0.0          # time of last completed click (for cooldown)
last_pinch_release_time = None  # time of last pinch release (for double-click window)
pending_double_click = False    # True after first pinch release, awaiting second

# Right-click (middle+thumb hold)
right_pinch_active = False
right_pinch_start = 0.0
right_clicked = False

# Back gesture (thumb+pinky pinch-hold)
back_pinch_active = False
back_pinch_start = 0.0
back_triggered = False
back_cooldown_until = 0.0

# Enter gesture (ring+thumb pinch-hold)
enter_pinch_active = False
enter_pinch_start = 0.0
enter_cooldown_until = 0.0
enter_status_until = 0.0

# Middle-click (3-finger pinch)
mid_click_cooldown = 0.0
right_click_suppress_until = 0.0

# V-pose scroll state
v_scroll_lock_axis = None      # None | "v" | "h"
v_scroll_lock_until = 0.0

previous_y_scroll = None

# Pause / resume
tracking_paused = False
pause_hold_active = False
pause_hold_start = 0.0
pause_cooldown_until = 0.0

# Two-hand zoom
zoom_active = False
zoom_prev_mid_y = None
zoom_last_action = 0.0
zoom_last_msg = ""

# Handwriting mode
handwriting_mode = False
handwriting_toggle_active = False
handwriting_toggle_start = 0.0
handwriting_toggle_latch = False
handwriting_canvas = None
handwriting_prev_point = None
handwriting_cursor_point = None
handwriting_pen_is_down = False
handwriting_pen_release_time = None
handwriting_status = ""
handwriting_status_until = 0.0
handwriting_submit_start = 0.0
handwriting_submit_active = False
handwriting_clear_start = 0.0
handwriting_clear_active = False
handwriting_backspace_start = 0.0
handwriting_backspace_active = False
handwriting_action_cooldown_until = 0.0

# Stabilization after drag
stabilize_until = 0.0
stabilized_target_x = None
stabilized_target_y = None

# Gesture history buffer (one per hand, we'll use the primary hand)
gesture_buf = GestureBuffer(CONFIG["history_frames"], CONFIG["history_min_votes"])

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        now = time.time()

        # ── Single-hand processing ─────────────────────────────────────────
        if result and result.hand_landmarks:
            lms = result.hand_landmarks[0]   # primary hand
            second_hand = result.hand_landmarks[1] if len(result.hand_landmarks) > 1 else None

            wrist        = lms[HandLandmark.WRIST]
            index_tip    = lms[HandLandmark.INDEX_FINGER_TIP]
            middle_tip   = lms[HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip     = lms[HandLandmark.RING_FINGER_TIP]
            pinky_tip    = lms[HandLandmark.PINKY_TIP]
            thumb_tip    = lms[HandLandmark.THUMB_TIP]
            ring_mcp     = lms[HandLandmark.RING_FINGER_MCP]

            hand_size = lm_dist(wrist, middle_tip)
            adaptive_thr = CONFIG["touch_threshold"] * hand_size
            three_thr    = CONFIG["three_finger_threshold"] * hand_size

            first_hand_data = hand_metrics(lms)
            second_hand_data = hand_metrics(second_hand) if second_hand is not None else None
            command_hand_data = None
            write_hand_data = None
            handwriting_feature_enabled = bool(CONFIG.get("handwriting_enabled", False))

            if handwriting_feature_enabled and second_hand_data is not None:
                # Use screen-space left hand for commands and right hand for writing.
                if first_hand_data["wrist"].x <= second_hand_data["wrist"].x:
                    command_hand_data = first_hand_data
                    write_hand_data = second_hand_data
                else:
                    command_hand_data = second_hand_data
                    write_hand_data = first_hand_data

                toggle_raw = command_hand_data["ring_thumb_d"] < (
                    CONFIG["handwriting_toggle_threshold"] * command_hand_data["hand_size"]
                )
                if toggle_raw and not handwriting_toggle_latch:
                    if not handwriting_toggle_active:
                        handwriting_toggle_active = True
                        handwriting_toggle_start = now
                    elif (now - handwriting_toggle_start) >= CONFIG["handwriting_toggle_hold"]:
                        handwriting_mode = not handwriting_mode
                        handwriting_toggle_latch = True
                        handwriting_toggle_active = False
                        handwriting_submit_active = False
                        handwriting_clear_active = False
                        handwriting_backspace_active = False
                        handwriting_prev_point = None
                        handwriting_cursor_point = None
                        handwriting_pen_is_down = False
                        handwriting_pen_release_time = None
                        handwriting_status = "Handwriting ON" if handwriting_mode else "Handwriting OFF"
                        handwriting_status_until = now + 1.2
                elif not toggle_raw:
                    handwriting_toggle_active = False
                    handwriting_toggle_latch = False
            else:
                handwriting_toggle_active = False
                handwriting_toggle_latch = False
                if not handwriting_feature_enabled and handwriting_mode:
                    handwriting_mode = False
                    handwriting_prev_point = None
                    handwriting_cursor_point = None
                    handwriting_pen_is_down = False
                    handwriting_pen_release_time = None
                    handwriting_submit_active = False
                    handwriting_clear_active = False
                    handwriting_backspace_active = False

            if handwriting_feature_enabled and handwriting_mode:
                movement_thread.deactivate()
                panel_w = int(CONFIG["handwriting_window_width"])
                panel_h = max(160, int(CONFIG["handwriting_window_height"]) - 72)
                handwriting_canvas = ensure_canvas(handwriting_canvas, panel_w, panel_h)

                # Command-hand gestures: submit (index+thumb), clear (middle+thumb), backspace (pinky+thumb).
                if command_hand_data is not None:
                    cmd_idx_pinch = command_hand_data["idx_thumb_d"] < command_hand_data["adaptive_thr"]
                    cmd_mid_pinch = command_hand_data["mid_thumb_d"] < command_hand_data["adaptive_thr"]
                    cmd_pinky_pinch = command_hand_data["pinky_thumb_d"] < (
                        command_hand_data["adaptive_thr"] * float(CONFIG["handwriting_backspace_threshold_scale"])
                    )

                    if now >= handwriting_action_cooldown_until:
                        if cmd_idx_pinch:
                            if not handwriting_submit_active:
                                handwriting_submit_active = True
                                handwriting_submit_start = now
                            elif (now - handwriting_submit_start) >= CONFIG["handwriting_submit_hold"]:
                                text, err = ocr_canvas(handwriting_canvas)
                                if text:
                                    pyautogui.write(text)
                                    handwriting_status = f"Inserted: {text[:32]}"
                                else:
                                    handwriting_status = err or "No text recognized"
                                handwriting_status_until = now + 1.8
                                handwriting_action_cooldown_until = now + CONFIG["handwriting_action_cooldown"]
                                handwriting_submit_active = False
                        else:
                            handwriting_submit_active = False

                        if cmd_mid_pinch:
                            if not handwriting_clear_active:
                                handwriting_clear_active = True
                                handwriting_clear_start = now
                            elif (now - handwriting_clear_start) >= CONFIG["handwriting_clear_hold"]:
                                handwriting_canvas[:] = 255
                                handwriting_prev_point = None
                                handwriting_status = "Canvas cleared"
                                handwriting_status_until = now + 1.2
                                handwriting_action_cooldown_until = now + CONFIG["handwriting_action_cooldown"]
                                handwriting_clear_active = False
                        else:
                            handwriting_clear_active = False

                        if cmd_pinky_pinch:
                            if not handwriting_backspace_active:
                                handwriting_backspace_active = True
                                handwriting_backspace_start = now
                            elif (now - handwriting_backspace_start) >= CONFIG["handwriting_backspace_hold"]:
                                pyautogui.press("backspace")
                                handwriting_status = "Backspace"
                                handwriting_status_until = now + 1.0
                                handwriting_action_cooldown_until = now + CONFIG["handwriting_action_cooldown"]
                                handwriting_backspace_active = False
                        else:
                            handwriting_backspace_active = False
                else:
                    handwriting_submit_active = False
                    handwriting_clear_active = False
                    handwriting_backspace_active = False

                # Write-hand drawing in panel.
                draw_active = False
                if write_hand_data is not None:
                    tip = write_hand_data["index_tip"]
                    raw_px = float(np.clip(tip.x, 0.0, 1.0) * (panel_w - 1))
                    raw_py = float(np.clip(tip.y, 0.0, 1.0) * (panel_h - 1))

                    if handwriting_cursor_point is None:
                        sx, sy = raw_px, raw_py
                    else:
                        dx = raw_px - float(handwriting_cursor_point[0])
                        dy = raw_py - float(handwriting_cursor_point[1])
                        distance = np.hypot(dx, dy)
                        panel_diag = max(1.0, np.hypot(panel_w, panel_h))
                        move_ratio = distance / panel_diag

                        if move_ratio <= float(CONFIG["handwriting_jitter_threshold"]):
                            sx, sy = handwriting_cursor_point
                        else:
                            alpha = min(
                                float(CONFIG["handwriting_move_alpha_ceiling"]),
                                max(
                                    float(CONFIG["handwriting_move_alpha_floor"]),
                                    move_ratio * float(CONFIG["handwriting_move_alpha_scale"]),
                                ),
                            )
                            sx = float(handwriting_cursor_point[0]) + (dx * alpha)
                            sy = float(handwriting_cursor_point[1]) + (dy * alpha)
                    handwriting_cursor_point = (int(sx), int(sy))

                    px, py = handwriting_cursor_point
                    pen_raw = write_hand_data["idx_thumb_d"] < (
                        write_hand_data["adaptive_thr"] * float(CONFIG["handwriting_pen_threshold_scale"])
                    )
                    if pen_raw:
                        handwriting_pen_is_down = True
                        handwriting_pen_release_time = None
                    elif handwriting_pen_is_down:
                        if handwriting_pen_release_time is None:
                            handwriting_pen_release_time = now
                        if (now - handwriting_pen_release_time) > float(CONFIG["handwriting_pen_release_grace"]):
                            handwriting_pen_is_down = False
                            handwriting_pen_release_time = None

                    draw_active = handwriting_pen_is_down
                    if draw_active:
                        point = (px, py)
                        if handwriting_prev_point is None:
                            handwriting_prev_point = point
                        cv2.line(
                            handwriting_canvas,
                            handwriting_prev_point,
                            point,
                            (10, 10, 10),
                            int(CONFIG["handwriting_stroke_thickness"]),
                            cv2.LINE_AA,
                        )
                        handwriting_prev_point = point
                else:
                    handwriting_cursor_point = None
                    handwriting_pen_is_down = False
                    handwriting_pen_release_time = None

                if not draw_active:
                    handwriting_prev_point = None

                cv2.putText(
                    frame,
                    "HANDWRITING MODE | Floating panel active (top-most)",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (60, 220, 250),
                    1,
                )
                if now < handwriting_status_until and handwriting_status:
                    cv2.putText(frame, handwriting_status, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2)

                show_handwriting_window(
                    handwriting_canvas,
                    handwriting_status if now < handwriting_status_until else "",
                    handwriting_cursor_point,
                    draw_active,
                )

                cv2.imshow("HandTrack", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # ── Zoom mode: both index fingertips touching, then move up/down ─
            zoom_msg = ""
            if second_hand is not None:
                other_index_tip = second_hand[HandLandmark.INDEX_FINGER_TIP]
                other_wrist = second_hand[HandLandmark.WRIST]
                other_middle_tip = second_hand[HandLandmark.MIDDLE_FINGER_TIP]
                other_hand_size = lm_dist(other_wrist, other_middle_tip)
                zoom_touch_thr = CONFIG["zoom_touch_threshold"] * max(hand_size, other_hand_size)
                zoom_tip_dist = lm_dist(index_tip, other_index_tip)
                zoom_mid_y = (index_tip.y + other_index_tip.y) / 2.0

                if zoom_tip_dist <= zoom_touch_thr:
                    if not zoom_active:
                        zoom_active = True
                        zoom_prev_mid_y = zoom_mid_y
                        zoom_last_msg = "ZOOM MODE"
                    else:
                        if zoom_prev_mid_y is not None:
                            delta_y = zoom_mid_y - zoom_prev_mid_y
                            if abs(delta_y) > CONFIG["zoom_motion_threshold"] and (now - zoom_last_action) >= CONFIG["zoom_cooldown"]:
                                wheel = int(np.clip((-delta_y) * CONFIG["zoom_scroll_gain"], -CONFIG["zoom_scroll_cap"], CONFIG["zoom_scroll_cap"]))
                                if wheel != 0:
                                    pyautogui.keyDown('ctrl')
                                    pyautogui.scroll(wheel)
                                    pyautogui.keyUp('ctrl')
                                    zoom_last_action = now
                                    zoom_last_msg = "ZOOM IN" if wheel > 0 else "ZOOM OUT"
                        zoom_prev_mid_y = zoom_mid_y

                    cv2.putText(frame, f"{zoom_last_msg} | tips touching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    cv2.putText(frame, f"Tip gap: {zoom_tip_dist:.3f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
                    cv2.putText(frame, "Move up = zoom in | move down = zoom out", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                    cv2.imshow("HandTrack", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                else:
                    zoom_active = False
                    zoom_prev_mid_y = None
                    zoom_last_msg = ""
            else:
                zoom_active = False
                zoom_prev_mid_y = None
                zoom_last_msg = ""

            # ── Finger extension flags ─────────────────────────────────────
            idx_ext   = is_extended(lms, HandLandmark.INDEX_FINGER_TIP,  HandLandmark.INDEX_FINGER_PIP,  hand_size)
            mid_ext   = is_extended(lms, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP, hand_size)
            ring_ext  = is_extended(lms, HandLandmark.RING_FINGER_TIP,   HandLandmark.RING_FINGER_PIP,   hand_size)
            pinky_ext = is_extended(lms, HandLandmark.PINKY_TIP,         HandLandmark.PINKY_PIP,         hand_size)

            # ── Gesture label for this frame ───────────────────────────────
            idx_thr_d  = lm_dist(index_tip, thumb_tip)
            mid_thr_d  = lm_dist(middle_tip, thumb_tip)
            ring_thr_d = lm_dist(ring_tip, thumb_tip)
            pinky_thr_d = lm_dist(pinky_tip, thumb_tip)

            left_pinch_raw  = idx_thr_d < adaptive_thr
            mid_pinch_raw   = mid_thr_d < adaptive_thr and not left_pinch_raw
            three_pinch_raw = (idx_thr_d < three_thr and mid_thr_d < three_thr and ring_thr_d < three_thr)
            back_pinch_raw = pinky_thr_d < (CONFIG["back_gesture_threshold"] * hand_size)
            v_pose_raw      = idx_ext and mid_ext and not ring_ext and not pinky_ext
            pause_raw       = idx_ext and mid_ext and ring_ext and not pinky_ext

            if pause_raw:
                frame_label = "pause_hold"
            elif three_pinch_raw:
                frame_label = "three_pinch"
            elif left_pinch_raw:
                frame_label = "left_pinch"
            elif v_pose_raw:
                frame_label = "v_pose"
            elif mid_pinch_raw:
                frame_label = "mid_pinch"
            else:
                frame_label = "none"

            gesture_buf.push(frame_label)

            # ── Pause toggle (deliberate 3-finger hold) ───────────────────
            if pause_raw and now >= pause_cooldown_until:
                if not pause_hold_active:
                    pause_hold_active = True
                    pause_hold_start = now
                elif (now - pause_hold_start) >= CONFIG["pause_hold"]:
                    tracking_paused = not tracking_paused
                    pause_cooldown_until = now + CONFIG["pause_cooldown"]
                    pause_hold_active = False
                    print(f"[handTrack] tracking {'PAUSED' if tracking_paused else 'RESUMED'}")
            else:
                pause_hold_active = False

            if tracking_paused:
                cv2.putText(frame, "PAUSED (3-finger hold to resume)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow("HandTrack", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            movement_thread.activate()

            # ── Cursor position (ring MCP anchor) ─────────────────────────
            fw, fh = frame.shape[1], frame.shape[0]
            mcp_x = int(ring_mcp.x * fw)
            mcp_y = int(ring_mcp.y * fh)
            mw, mh = calculate_margins(fw, fh, CONFIG["inner_area_percent"])
            target_x, target_y = convert_to_screen(mcp_x, mcp_y, fw, fh, mw, mh)

            if now < stabilize_until and stabilized_target_x is not None:
                movement_thread.update_target(stabilized_target_x, stabilized_target_y)
            else:
                movement_thread.update_target(target_x, target_y)
                stabilized_target_x, stabilized_target_y = target_x, target_y

            # ─────────────────────────────────────────────────────────────
            # CLICK / DRAG / RIGHT-CLICK all use RAW distance thresholds
            # (same as the original code) so they feel instant with zero lag.
            # Only fist + V-pose use the gesture buffer to debounce.
            # ─────────────────────────────────────────────────────────────

            # ── Three-finger middle-click (raw) ────────────────────────────
            if three_pinch_raw and now > mid_click_cooldown:
                pyautogui.click(button='middle')
                mid_click_cooldown = now + 0.5
                right_click_suppress_until = now + CONFIG["right_click_suppress_after_middle_click"]
                right_pinch_active = False
                right_clicked = False
                right_pinch_start = 0.0
                print("[handTrack] middle-click")

            # ── Left pinch: click / double-click / drag (raw) ─────────────
            elif left_pinch_raw:
                left_release_time = None   # re-pinched during grace → reset
                if not left_pinch_active:
                    left_pinch_active = True
                    left_pinch_start = now
                else:
                    if not left_is_dragging and (now - left_pinch_start) >= CONFIG["drag_hold"]:
                        try:
                            pyautogui.moveTo(int(target_x), int(target_y), _pause=False)
                        except Exception:
                            pass
                        pyautogui.mouseDown()
                        left_is_dragging = True
                        movement_thread.dragging = True
                        print(f"[handTrack] drag start ({int(target_x)},{int(target_y)})")
            else:
                if left_pinch_active:
                    if left_is_dragging:
                        if left_release_time is None:
                            left_release_time = now
                        elif (now - left_release_time) >= CONFIG["drag_release_grace"]:
                            pyautogui.mouseUp()
                            left_is_dragging = False
                            movement_thread.dragging = False
                            stabilize_until = now + CONFIG["stabilize_duration"]
                            stabilized_target_x, stabilized_target_y = target_x, target_y
                            left_pinch_active = False
                            left_pinch_start = 0.0
                            left_release_time = None
                            print(f"[handTrack] drag end ({int(target_x)},{int(target_y)})")
                    else:
                        # Click on release — cooldown + double-click logic
                        if now - last_click_time >= CONFIG["click_cooldown"]:
                            if (last_pinch_release_time is not None and
                                    now - last_pinch_release_time <= CONFIG["double_click_window"]):
                                pyautogui.doubleClick()
                                print("[handTrack] double-click")
                                last_pinch_release_time = None
                            else:
                                pyautogui.click()
                                print("[handTrack] click")
                                last_pinch_release_time = now
                            last_click_time = now
                        left_pinch_active = False
                        left_pinch_start = 0.0

            # ── V-pose → scroll (buffer-debounced, with axis lock + horiz) ─
            if gesture_buf.confirmed("v_pose") and not left_pinch_active:
                centroid_y = (index_tip.y + middle_tip.y) / 2.0
                centroid_x = (index_tip.x + middle_tip.x) / 2.0
                v_offset = wrist.y - centroid_y   # +ve = fingers up
                h_offset = wrist.x - centroid_x   # +ve = fingers left

                # Determine dominant axis if not locked
                if now > v_scroll_lock_until:
                    v_scroll_lock_axis = None

                if v_scroll_lock_axis is None:
                    if abs(v_offset) > abs(h_offset) and abs(v_offset) > CONFIG["v_orientation_threshold"]:
                        v_scroll_lock_axis = "v"
                        v_scroll_lock_until = now + CONFIG["scroll_lock_duration"]
                    elif abs(h_offset) > abs(v_offset) and abs(h_offset) > CONFIG["v_orientation_threshold"]:
                        v_scroll_lock_axis = "h"
                        v_scroll_lock_until = now + CONFIG["scroll_lock_duration"]

                if v_scroll_lock_axis == "v" and abs(v_offset) > CONFIG["v_orientation_threshold"]:
                    raw = -v_offset * CONFIG["v_orientation_gain"]
                    raw = max(-CONFIG["v_orientation_scroll"], min(CONFIG["v_orientation_scroll"], raw))
                    scroll_thread.add_scroll(v=raw * 2.2)  # boost V-pose scroll
                elif v_scroll_lock_axis == "h" and abs(h_offset) > CONFIG["v_orientation_threshold"]:
                    raw = -h_offset * CONFIG["v_h_gain"]
                    raw = max(-CONFIG["v_h_scroll"], min(CONFIG["v_h_scroll"], raw))
                    scroll_thread.add_scroll(h=raw * 2.2)  # boost horizontal scroll

                previous_y_scroll = None

            # ── Middle+thumb: right-click hold + fallback scroll ───────────
            elif mid_pinch_raw and not left_pinch_active and now >= right_click_suppress_until:
                if not right_pinch_active:
                    right_pinch_active = True
                    right_pinch_start = now
                    right_clicked = False

                # Allow scrolling while the user moves the pinched middle thumb (legacy behavior)
                if not right_clicked:
                    if previous_y_scroll is not None:
                        delta_y = middle_tip.y - previous_y_scroll
                        if abs(delta_y) > CONFIG["scroll_threshold"]:
                            raw = -delta_y * CONFIG["v_scroll_sensitivity"]
                            raw = max(-CONFIG["max_scroll_step"], min(CONFIG["max_scroll_step"], raw))
                            scroll_thread.add_scroll(v=raw)
                    previous_y_scroll = middle_tip.y
                else:
                    previous_y_scroll = None

                # Trigger right-click if held long enough
                if not right_clicked and (now - right_pinch_start) >= CONFIG["right_click_hold"]:
                    pyautogui.click(button='right')
                    right_clicked = True
                    previous_y_scroll = None
                    print("[handTrack] right-click (middle+thumb)")
            # If user is finishing a drag selection and immediately does middle+thumb,
            # release drag grace early so right-click can start without waiting.
            elif mid_pinch_raw and left_is_dragging and left_release_time is not None and now >= right_click_suppress_until:
                pyautogui.mouseUp()
                left_is_dragging = False
                movement_thread.dragging = False
                left_pinch_active = False
                left_pinch_start = 0.0
                left_release_time = None
                stabilize_until = now + CONFIG["stabilize_duration"]
                stabilized_target_x, stabilized_target_y = target_x, target_y
                right_pinch_active = True
                right_pinch_start = now
                right_clicked = False
                previous_y_scroll = middle_tip.y
            else:
                right_pinch_active = False
                right_clicked = False
                right_pinch_start = 0.0
                previous_y_scroll = None
                v_scroll_lock_axis = None

            # ── Back navigation: thumb+pinky pinch-hold ───────────────────
            back_allowed = (not left_pinch_raw) and (not mid_pinch_raw) and (not three_pinch_raw) and (not v_pose_raw)
            if back_pinch_raw and back_allowed and now >= back_cooldown_until:
                if not back_pinch_active:
                    back_pinch_active = True
                    back_pinch_start = now
                    back_triggered = False
                elif (not back_triggered) and ((now - back_pinch_start) >= CONFIG["back_gesture_hold"]):
                    try:
                        trigger_back_action()
                        print("[handTrack] back action")
                    except Exception as exc:
                        print(f"[handTrack] back action failed: {exc}")
                    back_triggered = True
                    back_cooldown_until = now + CONFIG["back_gesture_cooldown"]
            else:
                back_pinch_active = False
                back_triggered = False

            # ── Enter key: single-hand ring+thumb pinch-hold ─────────────
            enter_pinch_raw = (
                second_hand is None
                and (ring_thr_d < (adaptive_thr * float(CONFIG["enter_gesture_threshold_scale"])))
            )
            enter_allowed = (
                (not left_pinch_raw)
                and (not mid_pinch_raw)
                and (not three_pinch_raw)
                and (not back_pinch_raw)
                and (not v_pose_raw)
                and (not fist_raw)
            )

            if enter_pinch_raw and enter_allowed and now >= enter_cooldown_until:
                if not enter_pinch_active:
                    enter_pinch_active = True
                    enter_pinch_start = now
                elif (now - enter_pinch_start) >= CONFIG["enter_gesture_hold"]:
                    pyautogui.press("enter")
                    enter_cooldown_until = now + CONFIG["enter_gesture_cooldown"]
                    enter_status_until = now + 1.0
                    enter_pinch_active = False
                    print("[handTrack] enter key")
            else:
                enter_pinch_active = False

            # ── Debug overlay ──────────────────────────────────────────────
            dominant = gesture_buf.dominant()
            status = (
                f"Gesture:{dominant}  "
                f"Drag:{'Y' if left_is_dragging else 'N'}  "
                f"Paused:{'Y' if tracking_paused else 'N'}  "
                f"Back:{'Y' if back_pinch_active else 'N'}  "
                f"Enter:{'Y' if enter_pinch_active else 'N'}"
            )
            v_acc = scroll_thread.v_accumulator
            h_acc = scroll_thread.h_accumulator
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"V-acc:{v_acc:.2f}  H-acc:{h_acc:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
            cv2.putText(frame, f"Axis lock:{v_scroll_lock_axis}  Zoom:2-hand", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
            if now < enter_status_until:
                cv2.putText(frame, "ENTER", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 220, 255), 2)

        else:
            # No hands → clean up state
            if left_pinch_active:
                if left_is_dragging:
                    pyautogui.mouseUp()
                    left_is_dragging = False
                    movement_thread.dragging = False
                left_pinch_active = False
                left_pinch_start = 0.0
            if right_pinch_active:
                right_pinch_active = False
                right_clicked = False
                right_pinch_start = 0.0
            back_pinch_active = False
            back_triggered = False
            enter_pinch_active = False
            previous_y_scroll = None
            fist_prev = False
            gesture_buf.buf.clear()
            movement_thread.deactivate()

        if not handwriting_mode:
            try:
                cv2.destroyWindow("Handwriting Panel")
            except Exception:
                pass

        try:
            cv2.imshow("HandTrack", frame)
        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    movement_thread.stop()
    scroll_thread.stop()
    if left_is_dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("[handTrack] exited cleanly")
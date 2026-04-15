"""
hand_track.py — Enhanced hand-tracking mouse controller
═══════════════════════════════════════════════════════
Gestures:
  • Index+Thumb pinch      → Left click (quick) / Drag (hold 0.25s)
  • Double pinch           → Double-click
  • Pinky extended         → Right-click  (replaces middle+thumb)
  • Index+Middle+Ring pinch→ Middle-click
  • V-pose (index+middle)  → Scroll: vertical (centroid-Y) + horizontal (centroid-X)
  • Fist                   → Pause/resume tracking toggle
  • Middle+Thumb scroll    → Legacy delta-Y scroll (kept as fallback)

Improvements over v1:
  • Gesture history buffer (10-frame majority vote) — kills false triggers
  • Click cooldown (300 ms) — prevents accidental double-fires
  • Intentional double-click via two quick pinches
  • Scroll direction lock (300 ms) for V-pose
  • Horizontal scroll via hscroll() in V-pose
  • Right-click via extended pinky (natural, no conflict)
  • Middle-click via three-finger pinch
  • Fist = pause tracking
  • External YAML-style config dict (edit CONFIG below)
  • Fixed scroll_queue bug → shows accumulator in overlay
  • Two-hand zoom: Ctrl+scroll maps to browser zoom
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit values here instead of hunting through the code
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
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

    # Right-click (pinky gesture)
    # Lowered so pinky extension is easier to trigger on small hands
    "pinky_hold": 0.12,           # hold time to fire right-click

    # Scroll (V-pose orientation)
    "v_orientation_threshold": 0.01,
    "v_orientation_gain": 60.0,
    "v_orientation_scroll": 2.0,
    "v_h_gain": 60.0,             # horizontal scroll gain
    "v_h_scroll": 2.0,            # max horizontal scroll per tick
    "scroll_lock_duration": 0.30, # seconds to lock scroll axis after first tick

    # Scroll (middle+thumb legacy)
    "scroll_threshold": 0.001,
    "v_scroll_sensitivity": 60.0,
    "max_scroll_step": 6,

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

    # Two-hand zoom (Ctrl+scroll) - distance delta sensitivity
    "zoom_sensitivity": 200.0,
}

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
        self.jitter_threshold = 0.003
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
                        alpha = min(0.85, max(0.25, (distance / screen_diagonal) * 3.0))
                        sleep_time = 0.008
                    else:
                        # Snappier: increased floor and ceiling so movement is more responsive
                        alpha = min(0.90, max(0.25, (distance / screen_diagonal) * 4.0))
                        sleep_time = 0.006
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
# Smoothed cursor target
smoothed_target_x, smoothed_target_y = pyautogui.position()

# Left pinch / drag
left_pinch_active = False
left_pinch_start = 0.0
left_is_dragging = False
left_release_time = None

# Double-click tracking
last_click_time = 0.0          # time of last completed click (for cooldown)
last_pinch_release_time = None  # time of last pinch release (for double-click window)
pending_double_click = False    # True after first pinch release, awaiting second

# Right-click (pinky gesture)
pinky_right_active = False
pinky_right_start = 0.0
pinky_right_fired = False

# Middle-click (3-finger pinch)
mid_click_cooldown = 0.0

# V-pose scroll state
v_scroll_lock_axis = None      # None | "v" | "h"
v_scroll_lock_until = 0.0

# Legacy middle+thumb scroll
middle_pinch_active = False
previous_y_scroll = None

# Fist / pause
tracking_paused = False
fist_prev = False              # debounce fist toggle

# Two-hand zoom
zoom_prev_dist = None
zoom_ctrl_held = False

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

        # ── Two-hand zoom ──────────────────────────────────────────────────
        if result and len(result.hand_landmarks) == 2:
            lms0, lms1 = result.hand_landmarks[0], result.hand_landmarks[1]
            tip0 = lms0[HandLandmark.INDEX_FINGER_TIP]
            tip1 = lms1[HandLandmark.INDEX_FINGER_TIP]
            # Convert to pixel space for stable distance
            fw, fh = frame.shape[1], frame.shape[0]
            dx = (tip0.x - tip1.x) * fw
            dy = (tip0.y - tip1.y) * fh
            dist = np.hypot(dx, dy)
            if zoom_prev_dist is not None:
                delta = dist - zoom_prev_dist
                if abs(delta) > 2.0:  # pixel threshold to ignore noise
                    zoom_steps = delta / CONFIG["zoom_sensitivity"]
                    zoom_steps = max(-3, min(3, zoom_steps))
                    if abs(zoom_steps) > 0.1:
                        pyautogui.hotkey('ctrl', 'equal') if zoom_steps > 0 else pyautogui.hotkey('ctrl', 'minus')
            zoom_prev_dist = dist
            # Skip single-hand processing when two hands are present
            cv2.putText(frame, "ZOOM MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("HandTrack", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        else:
            zoom_prev_dist = None

        # ── Single-hand processing ─────────────────────────────────────────
        if result and result.hand_landmarks:
            lms = result.hand_landmarks[0]   # primary hand

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

            # ── Finger extension flags ─────────────────────────────────────
            idx_ext   = is_extended(lms, HandLandmark.INDEX_FINGER_TIP,  HandLandmark.INDEX_FINGER_PIP,  hand_size)
            mid_ext   = is_extended(lms, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP, hand_size)
            ring_ext  = is_extended(lms, HandLandmark.RING_FINGER_TIP,   HandLandmark.RING_FINGER_PIP,   hand_size)
            pinky_ext = is_extended(lms, HandLandmark.PINKY_TIP,         HandLandmark.PINKY_PIP,         hand_size)

            # ── Gesture label for this frame ───────────────────────────────
            idx_thr_d  = lm_dist(index_tip, thumb_tip)
            mid_thr_d  = lm_dist(middle_tip, thumb_tip)
            ring_thr_d = lm_dist(ring_tip, thumb_tip)

            left_pinch_raw  = idx_thr_d < adaptive_thr
            mid_pinch_raw   = mid_thr_d < adaptive_thr and not left_pinch_raw
            three_pinch_raw = (idx_thr_d < three_thr and mid_thr_d < three_thr and ring_thr_d < three_thr)
            v_pose_raw      = idx_ext and mid_ext and not ring_ext and not pinky_ext
            pinky_pose_raw  = pinky_ext and not idx_ext and not mid_ext and not ring_ext
            fist_raw        = is_fist(lms, hand_size)

            if fist_raw:
                frame_label = "fist"
            elif three_pinch_raw:
                frame_label = "three_pinch"
            elif left_pinch_raw:
                frame_label = "left_pinch"
            elif pinky_pose_raw:
                frame_label = "pinky"
            elif v_pose_raw:
                frame_label = "v_pose"
            elif mid_pinch_raw:
                frame_label = "mid_pinch"
            else:
                frame_label = "none"

            gesture_buf.push(frame_label)

            # ── Fist toggle (debounced) ────────────────────────────────────
            fist_confirmed = gesture_buf.confirmed("fist")
            if fist_confirmed and not fist_prev:
                tracking_paused = not tracking_paused
                print(f"[handTrack] tracking {'PAUSED' if tracking_paused else 'RESUMED'}")
            fist_prev = fist_confirmed

            if tracking_paused:
                cv2.putText(frame, "PAUSED (fist to resume)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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
                smoothed_target_x = float(stabilized_target_x)
                smoothed_target_y = float(stabilized_target_y)
                movement_thread.update_target(smoothed_target_x, smoothed_target_y)
            else:
                screen_diag = np.hypot(screen_width, screen_height)
                rel_move = np.hypot(target_x - smoothed_target_x, target_y - smoothed_target_y) / (screen_diag + 1e-9)
                if left_is_dragging:
                    smooth_alpha = 0.92
                elif rel_move > 0.02:
                    # aggressive catch-up for larger moves
                    smooth_alpha = 0.90
                elif rel_move > 0.005:
                    smooth_alpha = 0.70
                else:
                    # baseline smoothing when almost still
                    smooth_alpha = 0.50
                smoothed_target_x += (target_x - smoothed_target_x) * smooth_alpha
                smoothed_target_y += (target_y - smoothed_target_y) * smooth_alpha
                movement_thread.update_target(smoothed_target_x, smoothed_target_y)
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

            # ── Pinky gesture → right-click (raw, with short hold) ────────
            # Uses pinky_pose_raw directly — no buffer delay
            if pinky_pose_raw and not left_pinch_active:
                if not pinky_right_active:
                    pinky_right_active = True
                    pinky_right_start = now
                    pinky_right_fired = False
                elif not pinky_right_fired and (now - pinky_right_start) >= CONFIG["pinky_hold"]:
                    pyautogui.click(button='right')
                    pinky_right_fired = True
                    print("[handTrack] right-click (pinky)")
            else:
                pinky_right_active = False
                pinky_right_fired = False

            # Visual debugging helper: show when pinky raw pose is detected
            if pinky_pose_raw:
                try:
                    cv2.putText(frame, "PINKY_RAW", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2)
                except Exception:
                    pass

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
                    scroll_thread.add_scroll(v=raw)
                elif v_scroll_lock_axis == "h" and abs(h_offset) > CONFIG["v_orientation_threshold"]:
                    raw = -h_offset * CONFIG["v_h_gain"]
                    raw = max(-CONFIG["v_h_scroll"], min(CONFIG["v_h_scroll"], raw))
                    scroll_thread.add_scroll(h=raw)

                previous_y_scroll = None

            # ── Middle+thumb legacy scroll (raw, fallback when not V-pose) ─
            elif mid_pinch_raw and not left_pinch_active:
                if not middle_pinch_active:
                    middle_pinch_active = True
                if previous_y_scroll is not None:
                    delta_y = middle_tip.y - previous_y_scroll
                    if abs(delta_y) > CONFIG["scroll_threshold"]:
                        raw = -delta_y * CONFIG["v_scroll_sensitivity"]
                        raw = max(-CONFIG["max_scroll_step"], min(CONFIG["max_scroll_step"], raw))
                        scroll_thread.add_scroll(v=raw)
                previous_y_scroll = middle_tip.y
            else:
                middle_pinch_active = False
                previous_y_scroll = None
                v_scroll_lock_axis = None

            # ── Debug overlay ──────────────────────────────────────────────
            dominant = gesture_buf.dominant()
            status = (
                f"Gesture:{dominant}  "
                f"Drag:{'Y' if left_is_dragging else 'N'}  "
                f"Paused:{'Y' if tracking_paused else 'N'}"
            )
            v_acc = scroll_thread.v_accumulator
            h_acc = scroll_thread.h_accumulator
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"V-acc:{v_acc:.2f}  H-acc:{h_acc:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
            cv2.putText(frame, f"Axis lock:{v_scroll_lock_axis}  Zoom:2-hand", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        else:
            # No hands → clean up state
            if left_pinch_active:
                if left_is_dragging:
                    pyautogui.mouseUp()
                    left_is_dragging = False
                    movement_thread.dragging = False
                left_pinch_active = False
                left_pinch_start = 0.0
            if pinky_right_active:
                pinky_right_active = False
                pinky_right_fired = False
            middle_pinch_active = False
            previous_y_scroll = None
            fist_prev = False
            zoom_prev_dist = None
            gesture_buf.buf.clear()
            movement_thread.deactivate()

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
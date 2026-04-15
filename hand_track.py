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

# Ensure hand landmarker model is available (downloads if missing)
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_FILENAME):
    try:
        print(f"Downloading model to {MODEL_FILENAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
    except Exception as e:
        print("Failed to download model:", e)
        raise

# Initialize MediaPipe HandLandmarker (Tasks API)
base_options = python.BaseOptions(model_asset_path=MODEL_FILENAME)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.7,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# Set up the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to capture video")
    exit(1)

# Configure PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Get screen size
screen_width, screen_height = pyautogui.size()

# Define the portion of the camera view to map to the full screen (70% here)
inner_area_percent = 0.7

# Calculate the margins around the inner area
def calculate_margins(frame_width, frame_height, inner_area_percent):
    margin_width = frame_width * (1 - inner_area_percent) / 2
    margin_height = frame_height * (1 - inner_area_percent) / 2
    return margin_width, margin_height

# Convert video coordinates to screen coordinates
def convert_to_screen_coordinates(x, y, frame_width, frame_height, margin_width, margin_height):
    screen_x = np.interp(x, (margin_width, frame_width - margin_width), (0, screen_width))
    screen_y = np.interp(y, (margin_height, frame_height - margin_height), (0, screen_height))
    return screen_x, screen_y

# Function to get distance between two landmarks
def get_landmark_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# Heuristic: determine whether a finger is extended by comparing tip-to-pip distance
def is_finger_extended(hand_landmarks, tip_idx, pip_idx, hand_size, threshold=0.22):
    tip = hand_landmarks[tip_idx]
    pip = hand_landmarks[pip_idx]
    return get_landmark_distance(tip, pip) > (hand_size * threshold)

# Movement Thread for smoother cursor movement
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
                # Adjust responsiveness when the user is actively dragging
                jitter_threshold_local = self.jitter_threshold * (0.2 if self.dragging else 1.0)
                if distance / screen_diagonal > jitter_threshold_local:
                    if self.dragging:
                        alpha = min(0.85, max(0.25, (distance / screen_diagonal) * 3.0))
                        sleep_time = 0.008
                    else:
                        alpha = min(0.45, max(0.06, (distance / screen_diagonal) * 2.0))
                        sleep_time = 0.01

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

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def stop(self):
        self.running = False

# Scrolling Thread for smooth scrolling with inertia
class ScrollThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.scroll_lock = threading.Lock()
        self.accumulator = 0.0
        self.last_input_time = time.time()
        self.running = True
        self.inertia = 0.95  # Slower reduction for rolling stop effect
        self.scroll_step = 0.01  # Smaller step for smoother scroll
        self.inertia_threshold = 0.01  # Minimum inertia scroll amount

    def run(self):
        while self.running:
            # Read accumulator and emit integer wheel ticks when possible
            with self.scroll_lock:
                acc = self.accumulator
            try:
                int_part = int(acc)  # truncates toward zero
                if int_part != 0:
                    pyautogui.scroll(int_part)
                    with self.scroll_lock:
                        self.accumulator -= int_part
                        self.last_input_time = time.time()
                else:
                    # apply gentle decay when idle to let small residuals die out
                    if (time.time() - self.last_input_time) > 0.06 and abs(acc) > self.inertia_threshold:
                        with self.scroll_lock:
                            self.accumulator *= self.inertia
            except Exception as e:
                print("Scroll thread error:", e)
            time.sleep(0.01)

    def add_scroll(self, scroll_amount):
        with self.scroll_lock:
            # Accumulate fractional scroll amounts; we emit integer ticks when accumulated
            self.accumulator += float(scroll_amount)
            self.last_input_time = time.time()

    def stop(self):
        self.running = False

# Initialize the movement and scroll threads
movement_thread = CursorMovementThread()
scroll_thread = ScrollThread()
movement_thread.start()
scroll_thread.start()

# Initialize control variables
# Left (index+thumb) pinch states for click vs drag
left_pinch_active = False
left_pinch_start = 0.0
left_is_dragging = False
drag_hold = 0.25  # seconds to hold before starting drag

# Right (middle+thumb) pinch states for right-click
right_pinch_active = False
right_pinch_start = 0.0
right_clicked = False
right_click_hold = 0.18  # seconds hold to trigger right-click

# Ambiguity margin between pinch pairs (normalized coords)
ambiguous_margin = 0.03

touch_threshold = 0.19
scroll_threshold = 0.001  # movement threshold (normalized coordinates)
scroll_sensitivity = 0.05  # legacy sensitivity for thumb-based scroll
v_scroll_sensitivity = 60.0  # sensitivity for V-pose scrolling (moderate)
max_scroll_step = 6  # cap scroll steps per tick to avoid large jumps

# Stabilization window to keep cursor still after drag release (seconds)
stabilize_until = 0.0
stabilized_target_x = None
stabilized_target_y = None
v_orientation_threshold = 0.01  # how far centroid must be from wrist to count as up/down V
v_orientation_gain = 60.0  # multiplier to convert orientation offset into scroll amount
v_orientation_scroll = 2.0  # max base scroll per tick (will be clamped)

try:
    previous_y = None
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a natural selfie-view, and convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image and run video-mode detection
        mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Convert the frame color back so it can be displayed
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Check for the presence of hands
        if detection_result and detection_result.hand_landmarks:
            movement_thread.activate()
            for hand_landmarks in detection_result.hand_landmarks:
                # Use the base of the ring finger (RING_FINGER_MCP) for tracking
                ring_finger_mcp = hand_landmarks[HandLandmark.RING_FINGER_MCP]
                mcp_x = int(ring_finger_mcp.x * frame.shape[1])
                mcp_y = int(ring_finger_mcp.y * frame.shape[0])

                # Calculate margins based on the current frame size
                margin_width, margin_height = calculate_margins(frame.shape[1], frame.shape[0], inner_area_percent)

                # Convert video coordinates to screen coordinates
                target_x, target_y = convert_to_screen_coordinates(mcp_x, mcp_y, frame.shape[1], frame.shape[0], margin_width, margin_height)

                # Update target position in movement thread, but respect any short
                # stabilization window after drag release so the cursor doesn't shift
                # away from a freshly selected area before a right-click.
                now = time.time()
                if now < stabilize_until and (stabilized_target_x is not None):
                    movement_thread.update_target(stabilized_target_x, stabilized_target_y)
                else:
                    movement_thread.update_target(target_x, target_y)
                    stabilized_target_x, stabilized_target_y = target_x, target_y

                # Calculate the adaptive touch threshold based on the average length of fingers
                wrist = hand_landmarks[HandLandmark.WRIST]
                middle_finger_tip = hand_landmarks[HandLandmark.MIDDLE_FINGER_TIP]
                hand_size = get_landmark_distance(wrist, middle_finger_tip)
                adaptive_threshold = touch_threshold * hand_size

                # Check if index finger and thumb are touching (for clicking / dragging)
                index_tip = hand_landmarks[HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks[HandLandmark.THUMB_TIP]
                index_thumb_distance = get_landmark_distance(index_tip, thumb_tip)

                # LEFT: index+thumb pinch handling (click vs drag)
                if index_thumb_distance < adaptive_threshold:
                    if not left_pinch_active:
                        left_pinch_active = True
                        left_pinch_start = time.time()
                    else:
                        # start drag if held long enough
                        if not left_is_dragging and (time.time() - left_pinch_start) >= drag_hold:
                            # snap cursor to the current target immediately then press down
                            try:
                                pyautogui.moveTo(int(target_x), int(target_y), _pause=False)
                            except Exception:
                                pass
                            pyautogui.mouseDown()
                            left_is_dragging = True
                            movement_thread.dragging = True
                            print(f"[handTrack] drag start at ({int(target_x)},{int(target_y)})")
                else:
                    if left_pinch_active:
                        if left_is_dragging:
                            pyautogui.mouseUp()
                            left_is_dragging = False
                            movement_thread.dragging = False
                            # brief stabilization so right-clicks land on the selected area
                            stabilize_until = time.time() + 0.20
                            stabilized_target_x, stabilized_target_y = target_x, target_y
                            print(f"[handTrack] drag end at ({int(target_x)},{int(target_y)})")
                        else:
                            pyautogui.click()
                        left_pinch_active = False
                        left_pinch_start = 0.0

                # Determine V-pose (index+middle extended, ring+pinky curled) for scrolling
                middle_tip = hand_landmarks[HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks[HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks[HandLandmark.PINKY_TIP]
                middle_thumb_distance = get_landmark_distance(middle_tip, thumb_tip)

                index_extended = is_finger_extended(hand_landmarks, HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_PIP, hand_size)
                middle_extended = is_finger_extended(hand_landmarks, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP, hand_size)
                ring_extended = is_finger_extended(hand_landmarks, HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_PIP, hand_size)
                pinky_extended = is_finger_extended(hand_landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP, hand_size)

                v_pose = index_extended and middle_extended and (not ring_extended) and (not pinky_extended)

                # If left pinch is active, give it priority and skip other actions
                if left_pinch_active:
                    previous_y = None
                else:
                    if v_pose:
                        # Orientation-based V-scroll: use centroid vs wrist to determine direction
                        centroid_y = (index_tip.y + middle_tip.y) / 2.0
                        # positive offset -> centroid above wrist (fingers pointing up)
                        orientation_offset = wrist.y - centroid_y
                        if abs(orientation_offset) > v_orientation_threshold:
                            # scale scroll proportionally to offset and clamp
                            raw = -orientation_offset * v_orientation_gain
                            raw = max(-v_orientation_scroll, min(v_orientation_scroll, raw))
                            scroll_thread.add_scroll(raw)
                        # clear previous_y to avoid mixing with other scroll modes
                        previous_y = None
                        # reset any right-pinch tracking while two-finger V active
                        if right_pinch_active:
                            right_pinch_active = False
                            right_clicked = False
                            right_pinch_start = 0.0
                    elif middle_thumb_distance < adaptive_threshold:
                        # begin right-pinch tracking if needed
                        if not right_pinch_active:
                            right_pinch_active = True
                            right_pinch_start = time.time()
                            right_clicked = False

                        # Allow scrolling while the user moves the pinched middle thumb (legacy behavior)
                        if not right_clicked:
                            if previous_y is not None:
                                delta_y = middle_tip.y - previous_y
                                if abs(delta_y) > scroll_threshold:
                                    raw = -delta_y * v_scroll_sensitivity
                                    raw = max(-max_scroll_step, min(max_scroll_step, raw))
                                    scroll_thread.add_scroll(raw)
                            previous_y = middle_tip.y
                        else:
                            previous_y = None

                        # Trigger right-click if held long enough
                        if not right_clicked and (time.time() - right_pinch_start) >= right_click_hold:
                            pyautogui.click(button='right')
                            right_clicked = True
                    else:
                        # reset right-pinch state and scrolling anchor
                        if right_pinch_active:
                            right_pinch_active = False
                            right_clicked = False
                            right_pinch_start = 0.0
                        previous_y = None

                    # draw a minimal status overlay on the frame for debugging
                    try:
                        status_text = f"Drag:{'Y' if left_is_dragging else 'N'} V:{'Y' if v_pose else 'N'} Right:{'Y' if right_pinch_active else 'N'}"
                        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        qlen = len(scroll_thread.scroll_queue)
                        cv2.putText(frame, f"SQ:{qlen}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                    except Exception:
                        pass
        else:
            # No hands detected
            if left_pinch_active:
                if left_is_dragging:
                    pyautogui.mouseUp()
                    left_is_dragging = False
                left_pinch_active = False
                left_pinch_start = 0.0
            if right_pinch_active:
                right_pinch_active = False
                right_clicked = False
                right_pinch_start = 0.0
            previous_y = None
            movement_thread.deactivate()

        # show the debug frame
        try:
            cv2.imshow("HandTrack", frame)
        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    movement_thread.stop()
    scroll_thread.stop()
    cap.release()
    cv2.destroyAllWindows()

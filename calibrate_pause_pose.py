"""
Calibrate pause gesture thresholds from a reference hand image.

Usage:
  python calibrate_pause_pose.py --image reference-stop-image.png
"""

import argparse
import json
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark


DEFAULTS = {
    "pause_open_ratio": 1.12,
    "pause_closed_ratio": 1.03,
    "pause_thumb_open_ratio": 1.05,
    "back_gesture_threshold": 0.20,
}


def load_config(config_path):
    cfg = DEFAULTS.copy()
    if not os.path.exists(config_path):
        return cfg

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.split("#", 1)[0].strip()
                if key not in cfg:
                    continue
                try:
                    cfg[key] = float(val)
                except ValueError:
                    pass
    except Exception:
        pass

    return cfg


def lm_dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def analyze(image_path, model_path, cfg):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Validate image can be loaded.
    if cv2.imread(image_path) is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    mp_img = mp.Image.create_from_file(image_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        raise RuntimeError("No hand detected in reference image.")

    lms = result.hand_landmarks[0]

    wrist = lms[HandLandmark.WRIST]
    thumb_tip = lms[HandLandmark.THUMB_TIP]
    index_tip = lms[HandLandmark.INDEX_FINGER_TIP]
    index_pip = lms[HandLandmark.INDEX_FINGER_PIP]
    middle_tip = lms[HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = lms[HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = lms[HandLandmark.RING_FINGER_TIP]
    ring_pip = lms[HandLandmark.RING_FINGER_PIP]
    pinky_tip = lms[HandLandmark.PINKY_TIP]
    pinky_pip = lms[HandLandmark.PINKY_PIP]

    hand_size = lm_dist(wrist, middle_tip)
    if hand_size <= 1e-6:
        raise RuntimeError("Invalid hand size from landmarks.")

    pause_open_ratio = float(cfg["pause_open_ratio"])
    pause_closed_ratio = float(cfg["pause_closed_ratio"])
    pause_thumb_open_ratio = float(cfg["pause_thumb_open_ratio"])
    back_thr = float(cfg["back_gesture_threshold"])

    thumb_ip = lms[HandLandmark.THUMB_IP]
    index_pip = lms[HandLandmark.INDEX_FINGER_PIP]
    middle_pip = lms[HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = lms[HandLandmark.RING_FINGER_PIP]
    pinky_pip = lms[HandLandmark.PINKY_PIP]

    thumb_tip_w = lm_dist(thumb_tip, wrist)
    thumb_ip_w = max(lm_dist(thumb_ip, wrist), 1e-6)
    index_tip_w = lm_dist(index_tip, wrist)
    index_pip_w = max(lm_dist(index_pip, wrist), 1e-6)
    middle_tip_w = lm_dist(middle_tip, wrist)
    middle_pip_w = max(lm_dist(middle_pip, wrist), 1e-6)
    ring_tip_w = lm_dist(ring_tip, wrist)
    ring_pip_w = max(lm_dist(ring_pip, wrist), 1e-6)
    pinky_tip_w = lm_dist(pinky_tip, wrist)
    pinky_pip_w = max(lm_dist(pinky_pip, wrist), 1e-6)

    ratios = {
        "thumb_tip_vs_ip_to_wrist": thumb_tip_w / thumb_ip_w,
        "index_tip_vs_pip_to_wrist": index_tip_w / index_pip_w,
        "middle_tip_vs_pip_to_wrist": middle_tip_w / middle_pip_w,
        "ring_tip_vs_pip_to_wrist": ring_tip_w / ring_pip_w,
        "pinky_tip_vs_pip_to_wrist": pinky_tip_w / pinky_pip_w,
        "pinky_thumb_ratio": lm_dist(pinky_tip, thumb_tip) / hand_size,
    }

    flags = {
        "thumb_open": ratios["thumb_tip_vs_ip_to_wrist"] > pause_thumb_open_ratio,
        "idx_open": ratios["index_tip_vs_pip_to_wrist"] > pause_open_ratio,
        "mid_closed": ratios["middle_tip_vs_pip_to_wrist"] < pause_closed_ratio,
        "ring_closed": ratios["ring_tip_vs_pip_to_wrist"] < pause_closed_ratio,
        "pinky_open": ratios["pinky_tip_vs_pip_to_wrist"] > pause_open_ratio,
        "back_pinch_raw": ratios["pinky_thumb_ratio"] < back_thr,
    }

    pause_match = (
        flags["thumb_open"]
        and flags["idx_open"]
        and flags["pinky_open"]
        and flags["mid_closed"]
        and flags["ring_closed"]
        and (not flags["back_pinch_raw"])
    )

    open_min = min(ratios["index_tip_vs_pip_to_wrist"], ratios["pinky_tip_vs_pip_to_wrist"])
    closed_max = max(ratios["middle_tip_vs_pip_to_wrist"], ratios["ring_tip_vs_pip_to_wrist"])

    if open_min > closed_max:
        rec_open = clamp(open_min * 0.92, 1.02, 1.35)
        rec_closed = clamp(closed_max * 1.08, 0.90, 1.25)
        if rec_closed >= rec_open:
            rec_closed = max(0.90, rec_open - 0.05)
    else:
        rec_open = pause_open_ratio
        rec_closed = pause_closed_ratio

    rec_thumb = clamp(ratios["thumb_tip_vs_ip_to_wrist"] * 0.92, 1.01, 1.35)
    rec_back = clamp(min(back_thr, ratios["pinky_thumb_ratio"] - 0.03), 0.12, 0.24)

    out = {
        "image": image_path,
        "model": model_path,
        "current_thresholds": {
            "pause_open_ratio": pause_open_ratio,
            "pause_closed_ratio": pause_closed_ratio,
            "pause_thumb_open_ratio": pause_thumb_open_ratio,
            "back_gesture_threshold": back_thr,
        },
        "ratios": {k: round(v, 5) for k, v in ratios.items()},
        "flags": flags,
        "pause_pose_match_current_config": pause_match,
        "recommended_thresholds": {
            "pause_open_ratio": round(rec_open, 4),
            "pause_closed_ratio": round(rec_closed, 4),
            "pause_thumb_open_ratio": round(rec_thumb, 4),
            "back_gesture_threshold": round(rec_back, 4),
        },
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Calibrate rock-star pause pose from a reference image")
    parser.add_argument("--image", default="reference-stop-image.png", help="Path to reference image")
    parser.add_argument("--model", default="hand_landmarker.task", help="Path to MediaPipe hand model")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--out", default="", help="Optional output JSON file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = analyze(args.image, args.model, cfg)

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved calibration report to: {args.out}")


if __name__ == "__main__":
    main()

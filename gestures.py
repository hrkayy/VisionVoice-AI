import cv2
import os
import urllib.request
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# --------------------------------------------------
# DOWNLOAD PRE-TRAINED HAND LANDMARKER MODEL
# --------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model file...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")

# --------------------------------------------------
# INITIALIZE THE DETECTOR IN VIDEO MODE
# --------------------------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def get_dist(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5

def detect_gesture(frame):
    """
    Processes the frame with MediaPipe Tasks HandLandmarker in Video Tracking mode,
    draws skeleton connectors, and detects signs using robust collinearity ratios.
    """
    h, w, _ = frame.shape
    timestamp_ms = int(time.time() * 1000)
    
    # Convert OpenCV image (BGR) to MediaPipe Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run Inference
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    gesture_text = "No Hand Detected"
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        lm = hand_landmarks
        
        # 1. Establish hand scale reference (palm size: Wrist to Middle Finger MCP Knuckle)
        palm_size = get_dist(lm[0], lm[9])
        if palm_size < 0.001:
            palm_size = 0.001
            
        # Draw skeletal connections (light pink-indigo lines)
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            p1 = hand_landmarks[start_idx]
            p2 = hand_landmarks[end_idx]
            pt1 = (int(p1.x * w), int(p1.y * h))
            pt2 = (int(p2.x * w), int(p2.y * h))
            cv2.line(frame, pt1, pt2, (248, 140, 129), 2)
            
        # Draw joints (bright blue circles)
        for l in hand_landmarks:
            pt = (int(l.x * w), int(l.y * h))
            cv2.circle(frame, pt, 5, (56, 189, 248), -1)
            
        # 2. Collinearity-based finger open detection:
        # Compares straight-line length (Tip-to-MCP) against total segment path (Tip-to-PIP + PIP-to-MCP).
        # Straight finger ratio is close to 1.0 (e.g. > 0.80). Curled finger ratio drops to < 0.65.
        # This is 100% rotation-independent.
        
        # Index: 8 (Tip), 6 (PIP), 5 (MCP)
        index_path = get_dist(lm[8], lm[6]) + get_dist(lm[6], lm[5])
        index_open = (get_dist(lm[8], lm[5]) / index_path > 0.80) if index_path > 0 else False
        
        # Middle: 12 (Tip), 10 (PIP), 9 (MCP)
        middle_path = get_dist(lm[12], lm[10]) + get_dist(lm[10], lm[9])
        middle_open = (get_dist(lm[12], lm[9]) / middle_path > 0.80) if middle_path > 0 else False
        
        # Ring: 16 (Tip), 14 (PIP), 13 (MCP)
        ring_path = get_dist(lm[16], lm[14]) + get_dist(lm[14], lm[13])
        ring_open = (get_dist(lm[16], lm[13]) / ring_path > 0.80) if ring_path > 0 else False
        
        # Pinky: 20 (Tip), 18 (PIP), 17 (MCP)
        pinky_path = get_dist(lm[20], lm[18]) + get_dist(lm[18], lm[17])
        pinky_open = (get_dist(lm[20], lm[17]) / pinky_path > 0.80) if pinky_path > 0 else False
        
        # Thumb: Open if the tip (4) is far enough from the index knuckle (5) relative to palm size
        thumb_open = get_dist(lm[4], lm[5]) > palm_size * 0.50
        
        # OK sign circle: Index Tip (8) touching Thumb Tip (4)
        ok_circle = get_dist(lm[4], lm[8]) < palm_size * 0.28
        
        # 3. Classify gesture / translation
        if ok_circle and middle_open and ring_open and pinky_open:
            gesture_text = "👌 OK (Perfect / Understood)"
        elif thumb_open and index_open and pinky_open and not middle_open and not ring_open:
            gesture_text = "🤟 I Love You (Friendship)"
        elif thumb_open and pinky_open and not index_open and not middle_open and not ring_open:
            gesture_text = "🤙 Call Me (Contact)"
        elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
            # Check if thumb tip is pointing up or down relative to thumb base joint
            if lm[4].y < lm[2].y:
                gesture_text = "👍 Thumbs Up (Yes / OK)"
            else:
                gesture_text = "👎 Thumbs Down (No / Not OK)"
        elif not index_open and not middle_open and not ring_open and not pinky_open and not thumb_open:
            gesture_text = "✊ Fist (Stop / Emergency)"
        elif index_open and middle_open and ring_open and pinky_open and thumb_open:
            gesture_text = "✋ Open Hand (Hello / Help)"
        elif index_open and middle_open and not ring_open and not pinky_open:
            gesture_text = "✌️ Victory Sign (Success)"
        elif index_open and not middle_open and not ring_open and not pinky_open and not thumb_open:
            gesture_text = "☝️ Pointing (Look / Attention)"
        else:
            gesture_text = "Hand detected"
            
    return frame, gesture_text

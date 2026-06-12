import cv2
from ultralytics import YOLO

# -----------------------------
# LOAD MODEL (optimized nano version for real-time CPU speed)
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# IMPORTANT OBJECTS ONLY
# (assistive AI focus)
# -----------------------------
IMPORTANT_OBJECTS = [
    "person", "chair", "bottle", "cell phone",
    "car", "door", "table", "stairs", "bench"
]

# -----------------------------
# BAD / NOISE DETECTIONS
# -----------------------------
IGNORE = [
    "vase", "toilet", "sink", "tv", "dining table"
]

# -----------------------------
# BONUS: RISK ANALYSIS
# -----------------------------
def get_risk(objects):
    if "person" in objects:
        return "🔴 HIGH RISK"

    if len(objects) >= 3:
        return "🟠 MEDIUM RISK"

    return "🟢 LOW RISK"

# -----------------------------
# MAIN DETECTION FUNCTION
# -----------------------------
def detect_objects(frame, conf=0.5):

    # -------------------------
    # OPTIMIZED DOWN-SCALING
    # -------------------------
    frame = cv2.resize(frame, (640, 480))

    # -------------------------
    # YOLO INFERENCE
    # -------------------------
    results = model(frame, conf=conf, iou=0.45, verbose=False)

    # annotated output
    annotated_frame = results[0].plot()

    objects = []
    positions = []

    # -------------------------
    # FILTER OBJECTS & TRACK COORDINATES
    # -------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf_val = float(box.conf[0])

            # filter logic
            if conf_val >= conf:
                if label in IMPORTANT_OBJECTS and label not in IGNORE:
                    objects.append(label)
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Calculate center X and height relative to image dimensions
                    x_center = float((x1 + x2) / 2.0) / 640.0
                    y_height = float(y2 - y1) / 480.0
                    positions.append({
                        "label": label,
                        "x_center": x_center,
                        "y_height": y_height
                    })

    # -------------------------
    # REMOVE DUPLICATES
    # -------------------------
    objects = list(set(objects))

    # -------------------------
    # RISK LEVEL
    # -------------------------
    risk = get_risk(objects)

    return annotated_frame, objects, risk, positions
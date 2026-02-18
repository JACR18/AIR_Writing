import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ────────────────────────────────────────────────
# Globals
canvas = None
prev_x, prev_y = 0, 0
latest_result = None
fullscreen = False          # Track fullscreen state
window_name = "Air Writing + Idle Erase"

# ────────────────────────────────────────────────
# Callback: receives async results
def on_result(result: vision.HandLandmarkerResult,
              output_image: mp.Image,
              timestamp_ms: int):
    global latest_result
    latest_result = result

# ────────────────────────────────────────────────
# Setup HandLandmarker (LIVE_STREAM mode)
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = "hand_landmarker.task"  # Ensure this file is in the same folder

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    result_callback=on_result
)

landmarker = HandLandmarker.create_from_options(options)

# ────────────────────────────────────────────────
# Create window that supports fullscreen
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # Mirror
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    # Init canvas on first frame
    if canvas is None:
        canvas = np.zeros_like(frame)

    drawing = False
    erasing = False

    if latest_result is not None and latest_result.hand_landmarks:
        hand_landmarks = latest_result.hand_landmarks[0]  # First detected hand

        # Key landmarks
        h, w = frame.shape[:2]

        idx_tip = hand_landmarks[8]
        idx_x = int(idx_tip.x * w)
        idx_y = int(idx_tip.y * h)

        mid_tip = hand_landmarks[12]
        mid_y = int(mid_tip.y * h)

        idx_pip_y = int(hand_landmarks[6].y * h)
        mid_pip_y = int(hand_landmarks[10].y * h)

        # ── Gesture logic ───────────────────────────────────────
        # Draw: Index extended, middle NOT extended
        if idx_y < idx_pip_y - 20 and mid_y > mid_pip_y - 10:
            drawing = True

        # Erase: Both index + middle extended
        elif idx_y < idx_pip_y - 20 and mid_y < mid_pip_y - 20:
            erasing = True

        else:
            prev_x, prev_y = 0, 0

        current_x, current_y = idx_x, idx_y

        # ── Apply action ────────────────────────────────────────
        if drawing and prev_x != 0 and prev_y != 0:
            cv2.line(canvas, (prev_x, prev_y), (current_x, current_y),
                     (0, 255, 0), thickness=8)   # Green draw

        elif erasing and prev_x != 0 and prev_y != 0:
            cv2.line(canvas, (prev_x, prev_y), (current_x, current_y),
                     (0, 0, 0), thickness=35)    # Thick black eraser

        if drawing or erasing:
            prev_x, prev_y = current_x, current_y

        # Visual feedback
        color = (0, 255, 0) if drawing else (0, 0, 255) if erasing else (255, 255, 0)
        cv2.circle(frame, (idx_x, idx_y), 12, color, 3)

        # Optional landmarks
        for lm in hand_landmarks:
            lx = int(lm.x * w)
            ly = int(lm.y * h)
            cv2.circle(frame, (lx, ly), 3, (200, 200, 200), -1)

    # Combine frame + canvas
    combined = cv2.addWeighted(frame, 0.65, canvas, 0.35, 0)

    # On-screen instructions (updated with fullscreen key)
    cv2.putText(combined, "Draw: Index up | Erase: Index+Middle up | 'f' = fullscreen toggle", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "'c' = clear canvas | 'q' = quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)   # Clear
    elif key == ord('f'):               # ← Fullscreen toggle
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
landmarker.close()
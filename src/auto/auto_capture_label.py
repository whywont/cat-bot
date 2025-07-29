import os
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time
from libcamera import Transform
from tensorflow.keras.models import load_model
import serial
import random
import sys
sys.path.append('../data')
from logger import log_behavior


cnn_model = load_model("../classifer/cat_classifier.h5")
CLASS_NAMES = ["Doja", "Harlow"] 
# --- Configuration ---
REFERENCE_DIR = "../cat_profiles"  # cat1_Doja/, cat2_Harlow/

MODEL_PATH = "../yolov8n.pt"

model = YOLO(MODEL_PATH)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    transform=Transform(hflip=True, vflip=True)
)
config["main"]["format"] = "RGB888"
picam2.configure(config)
picam2.start()
time.sleep(0.5)

def compute_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def load_reference_histograms():
    hists = {}
    for folder in os.listdir(REFERENCE_DIR):
        full_path = os.path.join(REFERENCE_DIR, folder)
        if not os.path.isdir(full_path):
            continue
        class_id = 0 if "cat1" in folder else 1
        samples = []
        for file in os.listdir(full_path):
            img = cv2.imread(os.path.join(full_path, file))
            if img is not None:
                samples.append(compute_histogram(img))
        if samples:
            hists[class_id] = np.mean(samples, axis=0)
    return hists


esp = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)

def read_sensor_data():
    try:
        line = esp.readline().decode().strip()
        if "DIST:" in line:
            parts = dict(item.split(":") for item in line.split(","))
            dist = float(parts.get("DIST", 0))
            return dist
    except Exception as e:
        print("Parse error:", e)
    return None

#Old version
# def run_heuristic_probe(state_vector, avg_offset):
#     box_area = state_vector[2]
#     velocity_mag = state_vector[11]
#     distance = state_vector[12]

#     if distance is None or distance < 0:
#         distance = 100.0  # fallback

#     if box_area > 0.15 and abs(avg_offset) < 0.1:
#         context = "very_close"
#         valid_actions = [0, 3]
#     elif box_area < 0.03 or distance > 100:
#         context = "very_far"
#         valid_actions = [0, 1, 2, 4]
#     elif avg_offset < -0.3:
#         context = "off_center"
#         valid_actions = [1]  # turn right
#     elif avg_offset > 0.3:
#         context = "off_center"
#         valid_actions = [2]  # turn left

#     elif velocity_mag < 0.01:
#         context = "stationary"
#         valid_actions = [4, 0]
#     else:
#         context = "default"
#         valid_actions = [0, 4, 1, 2]

#     action = random.choice(valid_actions)

    # Heuristic-derived dense reward
    # reward = {
    #     "very_close": 1.0 if action == 0 else -0.2,
    #     "very_far": 0.5 if action in [1, 2, 4] else -0.1,
    #     "off_center": 0.7 if action in [1, 2] else -0.3,
    #     "stationary": 0.6 if action == 4 else -0.2,
    #     "default": 0.2 if action in [4, 1, 2] else 0.0
    # }[context]

    # return action, context, reward

def run_heuristic_probe(state_vector, avg_offset):
    box_area = state_vector[2]
    velocity_mag = state_vector[11]
    distance = state_vector[12]

    if distance is None or distance < 0:
        distance = 100.0  # fallback

    # Not neccessary for transformer
    # if box_area > 0.15 and abs(avg_offset) < 0.1:
    #     context = "very_close"
    # elif box_area < 0.03 or distance > 100:
    #     context = "very_far"
    # elif abs(avg_offset) > 0.3:
    #     context = "off_center"
    # elif velocity_mag < 0.01:
    #     context = "stationary"
    # else:
    #     context = "default"
    context = "Rule-based"
    action = random.choice([0, 1, 2, 3, 4])  # Let model learn which is best

    # Reward shaping
    visibility = state_vector[10]
    centeredness = 1 - abs(avg_offset)
    proximity_bonus = 0.5 if 0.03 < box_area < 0.15 else 0.0
    still_penalty = -0.1 if velocity_mag < 0.01 else 0.0

    reward = visibility + 0.3 * centeredness + proximity_bonus + still_penalty
    reward = round(float(reward), 4)  # clean and capped

    return action, context, reward

def handle_search_mode(search_index, last_valid_offset, obstacle_detected=False):
    search_actions = [1, 4, 2, 4]  # R, F, L, F (repeated pattern)

    if obstacle_detected:
        action_id = 3  # B (back up)
    elif last_valid_offset is not None:
        if last_valid_offset < -0.3:
            action_id = 1  # R
        elif last_valid_offset > 0.3:
            action_id = 2  # L
        else:
            action_id = 4  # F
    else:
        action_id = search_actions[search_index % len(search_actions)]

    send_action(action_id)
    return action_id, (search_index + 1) % len(search_actions)


def send_action(action_id):
    command_map = {
        0: "S\n",
        1: "R\n",
        2: "L\n",
        3: "B\n",
        4: "F\n"
    }
    cmd = command_map.get(action_id, "S\n")
    esp.write(cmd.encode())

#reference_hists = load_reference_histograms()

# --- Capture loop ---
prev_center = None
prev_area = None
prev_time = time.time()
last_action_id = 0
prev_frame = None

state_buffer = []
action_buffer = []
return_buffer = []
frame_index = 0

offset_history = []
search_index = 0
in_search_mode = False
last_valid_offset = None
while True:
    frame = picam2.capture_array()
    h, w = frame.shape[:2]

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        diff = cv2.absdiff(frame_gray, prev_frame)
        frame_entropy = np.mean(diff)
    else:
        frame_entropy = 0.0
    prev_frame = frame_gray

    motion_detected = frame_entropy > 0.1

    if not motion_detected:
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    results = model(frame, verbose=False)[0]
    label_lines = []

    search_actions = [1, 4, 2, 4]  # R, F, L, F
    search_index = 0
    in_search_mode = False
    action_taken_this_frame = False


    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or int(cls) != 15:
            time.sleep(0.5)
            distance = read_sensor_data()
            obstacle = distance is not None and 0 < distance < 20

            if not in_search_mode:
                in_search_mode = True

            action_id, search_index = handle_search_mode(search_index, last_valid_offset, obstacle)

            # Optional: state_vector when cat is missing
            dummy_state = [
                -1.0,   # best_class
                0.0,    # confidence
                0.0,    # box_area
                0.0,    # dx
                0.0,    # dy
                0.0,    # darea
                0.0,    # elapsed
                float(last_action_id),
                last_valid_offset if last_valid_offset is not None else 0.0,  # normalized_offset
                0.0,    # aspect_ratio
                0.0,    # visibility_score
                0.0,    # velocity_mag
                distance if distance is not None else -1.0,  # distance
                last_valid_offset if last_valid_offset is not None else 0.0,  # last_valid_offset
                1.0     # in_search_mode
            ]


            state_buffer.append(dummy_state)
            action_buffer.append(last_action_id)
            return_buffer.append(0.0)  # No reward when lost

            log_behavior(
                source="heuristic",
                goal_id="searching",
                goal_status="active",
                selected_action=action_id,
                reward=0.0,
                success=False,
                state_vector=dummy_state,
                in_search_mode=True
            )
            last_action_id = action_id
            action_taken_this_frame = True

            continue



        resized = cv2.resize(crop, (128, 128))
        input_tensor = np.expand_dims(resized / 255.0, axis=0)
        prediction = cnn_model.predict(input_tensor, verbose=0)[0]
        best_class = int(np.argmax(prediction))
        confidence = float(prediction[best_class])


        # Bounding box metrics
        x_center_px = (x1 + x2) / 2
        y_center_px = (y1 + y2) / 2
        x_center_norm = x_center_px / w
        y_center_norm = y_center_px / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h
        box_area = box_w * box_h
        aspect_ratio = box_w / box_h if box_h != 0 else 0

        # Offset from center
        frame_center = w / 2
        normalized_offset = (x_center_px - frame_center) / frame_center
        normalized_offset *= -1
        offset_history.append(normalized_offset)
        if len(offset_history) > 5:
            offset_history.pop(0)
        avg_offset = sum(offset_history) / len(offset_history)


        # Dynamic threshold
        min_area, max_area = 0.01, 0.2
        min_thresh, max_thresh = 0.15, 0.4
        clipped_area = max(min_area, min(box_area, max_area))
        dynamic_thresh = min_thresh + (clipped_area - min_area) * (max_thresh - min_thresh) / (max_area - min_area)

        if confidence < dynamic_thresh:
            continue

        current_time = time.time()
        elapsed = current_time - prev_time

        # Motion features
        if prev_center is not None:
            dx = x_center_norm - prev_center[0]
            dy = y_center_norm - prev_center[1]
            velocity_mag = np.sqrt(dx**2 + dy**2)
        else:
            dx = dy = velocity_mag = 0.0

        darea = box_area - prev_area if prev_area is not None else 0.0
        visibility_score = confidence * box_area * (1 - abs(normalized_offset))

        prev_center = (x_center_norm, y_center_norm)
        prev_area = box_area
        prev_time = current_time

        # UI overlay
        label = f"{'Doja' if best_class == 0 else 'Harlow'} ({confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        distance = read_sensor_data()
        
        if distance is None:
            distance = -1.0  # fallback or missing signal
        in_search_mode = 0

        # Final model input vector
        state_vector = [
            float(best_class),
            confidence,
            box_area,
            dx,
            dy,
            darea,
            elapsed,
            float(last_action_id),
            normalized_offset,
            aspect_ratio,
            visibility_score,
            velocity_mag,
            distance,
            last_valid_offset,
            float(in_search_mode)
        ]
        action_id, context, reward = run_heuristic_probe(state_vector, avg_offset)

        state_buffer.append(state_vector)
        action_buffer.append(last_action_id)
        return_buffer.append(reward)


        label_lines.append(f"{best_class} {x_center_norm:.6f} {y_center_norm:.6f} {box_w:.6f} {box_h:.6f}")

        send_action(action_id) 
        log_behavior(
            source="heuristic",
            goal_id=context,
            goal_status="active",
            selected_action=action_id,
            reward=reward,
            success=False,
            state_vector=state_vector,
            in_search_mode=in_search_mode
        )


        last_action_id = action_id
        last_valid_offset = normalized_offset

        action_taken_this_frame = True


        print(state_vector)
    
    if not action_taken_this_frame:
            in_search_mode = True

            action_id, search_index = handle_search_mode(search_index, last_valid_offset, obstacle)

            # Optional: state_vector when cat is missing
            dummy_state = [
                -1.0,   # best_class
                0.0,    # confidence
                0.0,    # box_area
                0.0,    # dx
                0.0,    # dy
                0.0,    # darea
                0.0,    # elapsed
                float(last_action_id),
                last_valid_offset if last_valid_offset is not None else 0.0,  # normalized_offset
                0.0,    # aspect_ratio
                0.0,    # visibility_score
                0.0,    # velocity_mag
                distance if distance is not None else -1.0,  # distance
                last_valid_offset if last_valid_offset is not None else 0.0,  # last_valid_offset
                1.0     # in_search_mode
            ]


            state_buffer.append(dummy_state)
            action_buffer.append(last_action_id)
            return_buffer.append(0.0)  # No reward when lost

            log_behavior(
                source="heuristic",
                goal_id="searching",
                goal_status="active",
                selected_action=action_id,
                reward=0.0,
                success=False,
                state_vector=dummy_state,
                in_search_mode=True
            )
            last_action_id = action_id
            continue   
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()

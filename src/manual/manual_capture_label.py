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
from threading import Thread
from flask_manual import start_flask, get_manual_action, clear_manual_action
import sys
sys.path.append('../data')
from logger import log_behavior

cnn_model = load_model("../classifier/cat_classifier.h5")
CLASS_NAMES = ["Doja", "Harlow"]

REFERENCE_DIR = "../cat_profiles"
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

esp = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)

def read_sensor_data():
    try:
        line = esp.readline().decode().strip()
        if "DIST:" in line:
            parts = dict(item.split(":") for item in line.split(","))
            dist = float(parts.get("DIST", 0))
            return dist
    except:
        pass
    return None

def send_action(action_id):
    command_map = {
        0: "S\n",
        1: "R\n",
        2: "L\n",
        3: "B\n",
        4: "F\n"
    }
    cmd = command_map.get(action_id, "S\n")
    print(f"[SEND] {cmd.strip()}")
    esp.write(cmd.encode())

action_map = {
    "F": 4,
    "B": 3,
    "L": 2,
    "R": 1,
    "S": 0
}

flask_thread = Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()

prev_center = None
prev_area = None
prev_time = time.time()
last_action_id = 0
prev_frame = None
state_buffer = []
action_buffer = []
return_buffer = []
offset_history = []
search_index = 0
in_search_mode = False
last_valid_offset = None



while True:
    action_taken_this_frame = False
    frame = picam2.capture_array()
    h, w = frame.shape[:2]

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        diff = cv2.absdiff(frame_gray, prev_frame)
        frame_entropy = np.mean(diff)
    else:
        frame_entropy = 0.0
    prev_frame = frame_gray

    results = model(frame, verbose=False)[0]
    label_lines = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or int(cls) != 15:
            manual_id = get_manual_action()
            if manual_id is not None:
                action_id = action_map.get(manual_id)
                if action_id is not None:
                    send_action(action_id)
                    distance = read_sensor_data()
                    in_search_mode = True

                    dummy_state = [
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        float(last_action_id),
                        last_valid_offset if last_valid_offset is not None else 0.0,
                        0.0, 0.0, 0.0,
                        distance if distance is not None else -1.0,
                        last_valid_offset if last_valid_offset is not None else 0.0,
                        float(in_search_mode)
                    ]

                    reward = 1.0
                    context = "manual_search"

                    log_behavior(
                        source="manual",
                        goal_id=context,
                        goal_status="active",
                        selected_action=action_id,
                        reward=reward,
                        success=True,
                        state_vector=dummy_state,
                        action_probs=None,
                        model_name=None
                    )
                    action_taken_this_frame = True


                    last_action_id = action_id
                    clear_manual_action()
            continue

        resized = cv2.resize(crop, (128, 128))
        input_tensor = np.expand_dims(resized / 255.0, axis=0)
        prediction = cnn_model.predict(input_tensor, verbose=0)[0]
        best_class = int(np.argmax(prediction))
        confidence = float(prediction[best_class])

        x_center_px = (x1 + x2) / 2
        y_center_px = (y1 + y2) / 2
        x_center_norm = x_center_px / w
        y_center_norm = y_center_px / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h
        box_area = box_w * box_h
        aspect_ratio = box_w / box_h if box_h != 0 else 0

        frame_center = w / 2
        normalized_offset = (x_center_px - frame_center) / frame_center * -1
        offset_history.append(normalized_offset)
        if len(offset_history) > 5:
            offset_history.pop(0)
        avg_offset = sum(offset_history) / len(offset_history)

        min_area, max_area = 0.01, 0.2
        min_thresh, max_thresh = 0.10, 0.4
        clipped_area = max(min_area, min(box_area, max_area))
        dynamic_thresh = min_thresh + (clipped_area - min_area) * (max_thresh - min_thresh) / (max_area - min_area)

        # if confidence < dynamic_thresh:
        #     continues

        current_time = time.time()
        elapsed = current_time - prev_time

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

        label = f"{'Doja' if best_class == 0 else 'Harlow'} ({confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        cat_name = 'Doja' if best_class == 0 else 'Harlow'
        filename = f"detected_{cat_name}_{timestamp}.jpg"
        save_path = os.path.join("saved_detections", filename)

        # Create directory if it doesn't exist
        # os.makedirs("saved_detections", exist_ok=True)
        # scv2.imwrite(save_path, frame)

        distance = read_sensor_data()
        if distance is None:
            distance = -1.0
        in_search_mode = 0

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
            last_valid_offset if last_valid_offset is not None else 0.0,
            float(in_search_mode)
        ]

        manual_id = get_manual_action()
        action_id = -1  # Default to "no action"
        reward = 1.0  

        manual_id = get_manual_action()
        if manual_id is not None:
            action_id = action_map.get(manual_id)
            if action_id is not None:
                send_action(action_id)

                reward = 1.0
                

                state_buffer.append(state_vector)
                action_buffer.append(action_id)
                return_buffer.append(reward)

                last_action_id = action_id
                last_valid_offset = normalized_offset
                action_taken_this_frame = True
                clear_manual_action()

        context = "manual_tracking"
        
        log_behavior(
            source="manual",
            goal_id=context,
            goal_status="active",
            selected_action=action_id,
            reward=reward,
            success=True,
            state_vector=state_vector,

                        action_probs=None,
                        model_name=None
        )


        # print(state_vector)
   


    manual_id = get_manual_action()
    if manual_id is not None and not action_taken_this_frame:
        action_id = action_map.get(manual_id)
        if action_id is not None:
                    send_action(action_id)
                    distance = read_sensor_data()
                    in_search_mode = True

                    dummy_state = [
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        float(last_action_id),
                        last_valid_offset if last_valid_offset is not None else 0.0,
                        0.0, 0.0, 0.0,
                        distance if distance is not None else -1.0,
                        last_valid_offset if last_valid_offset is not None else 0.0,
                        float(in_search_mode)
                    ]

                    reward = 1.0
                    context = "manual_search"

                    log_behavior(
                        source="manual",
                        goal_id=context,
                        goal_status="active",
                        selected_action=action_id,
                        reward=reward,
                        success=True,
                        state_vector=dummy_state,
                        action_probs=None,
                        model_name=None
                    )

                    last_action_id = action_id
                    clear_manual_action()

    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
# cv2.destroyAllWindows()

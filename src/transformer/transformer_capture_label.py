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
import torch
from decision_transformer import ImprovedDecisionTransformer
import random
import sys
sys.path.append('../data')
from logger import log_behavior


# Model loading
cnn_model = load_model("../classifier/cat_classifier.h5")
CLASS_NAMES = ["Doja", "Harlow"] 

# Configuration
REFERENCE_DIR = "../cat_profiles"
MODEL_PATH = "../yolov8n.pt"


# Transformer setup
context_len = 3      
state_dim = 14     
action_dim = 5
hidden_size = 64     


transformer_model = ImprovedDecisionTransformer(
    state_dim=state_dim,
    action_dim=action_dim,
    context_len=context_len,
    hidden_size=hidden_size,
    n_layers=3  # Add this parameter
)
transformer_model.load_state_dict(torch.load("expert_demo_transformer.pt", map_location=torch.device("cpu")))
transformer_model.eval()


# YOLO model
yolo_model = YOLO(MODEL_PATH)

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    transform=Transform(hflip=True, vflip=True)
)
config["main"]["format"] = "RGB888"
picam2.configure(config)
picam2.start()
time.sleep(0.5)

# Serial connection
esp = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.1)

def read_sensor_data():
    """Read distance sensor data from ESP32"""
    try:
        line = esp.readline().decode().strip()
        if "DIST:" in line:
            parts = dict(item.split(":") for item in line.split(","))
            dist = float(parts.get("DIST", 0))
            return dist
    except Exception as e:
        print("Parse error:", e)
    return None

def send_action(action_id):
    """Send action command to ESP32"""
    command_map = {
        0: "S\n",  # Stop
        1: "R\n",  # Right
        2: "L\n",  # Left
        3: "B\n",  # Back
        4: "F\n"   # Forward
    }
    cmd = command_map.get(action_id, "S\n")
    esp.write(cmd.encode())

def select_action_with_transformer(state_buffer, action_buffer, return_buffer, model, last_action_id):
    if len(state_buffer) < 3:  # Changed from 2 to 3
        return 0, None, 0.0, "warmup"

    input_states = torch.tensor([state_buffer[-3:]], dtype=torch.float32)  # Last 3 frames
    input_actions = torch.tensor([action_buffer[-3:]], dtype=torch.long)   # Last 3 actions
    input_returns = torch.tensor([[[1.0], [1.0], [1.0]]], dtype=torch.float32)  # 3 returns

    with torch.no_grad():
        logits = model(input_states, input_actions, input_returns)
        last_logits = logits[:, -1, :]  # Take last timestep prediction
        probs = torch.softmax(last_logits, dim=-1)
        action_id = torch.argmax(last_logits, dim=-1).item()
        action_probs = probs.squeeze().numpy().tolist()

    return action_id, action_probs, 1.0, "expert_transformer"

def create_dummy_state(distance, last_action_id, last_valid_offset, in_search_mode):
    """Create dummy state vector when no cat is detected"""
    return [
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

    ]


def handle_search_mode(search_index, last_valid_offset, obstacle_detected=False):
    """Handle search behavior when cat is not detected"""
    search_actions = [1, 4, 2, 4, 0]  # R, F, L, F (repeated pattern)

    if obstacle_detected:
        action_id = 3  # Back up
    elif last_valid_offset is not None:
        if last_valid_offset < -0.3:
            action_id = 1  # Turn right
        elif last_valid_offset > 0.3:
            action_id = 2  # Turn left
        else:
            action_id = 4  # Forward
    else:
        action_id = search_actions[search_index % len(search_actions)]

    return action_id, (search_index + 1) % len(search_actions)

def process_cat_detection(frame, box, cls, cnn_model, prev_center, prev_area, prev_time, w, h, offset_history):
    """Process detected cat and extract features"""
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None, None, None, None, None
    
    # Cat classification
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

    # Dynamic confidence threshold
    min_area, max_area = 0.01, 0.2
    min_thresh, max_thresh = 0.10, 0.4
    clipped_area = max(min_area, min(box_area, max_area))
    dynamic_thresh = min_thresh + (clipped_area - min_area) * (max_thresh - min_thresh) / (max_area - min_area)

    if confidence < dynamic_thresh:
        return None, None, None, None, None

    # Motion calculations
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

    detection_data = {
        'best_class': best_class,
        'confidence': confidence,
        'box_area': box_area,
        'dx': dx,
        'dy': dy,
        'darea': darea,
        'elapsed': elapsed,
        'normalized_offset': normalized_offset,
        'aspect_ratio': aspect_ratio,
        'visibility_score': visibility_score,
        'velocity_mag': velocity_mag,
        'bbox': (x1, y1, x2, y2),
        'new_center': (x_center_norm, y_center_norm),
        'new_area': box_area,
        'new_time': current_time
    }
    
    return detection_data, x1, y1, x2, y2

def main():
    # Initialize tracking variables
    prev_center = None
    prev_area = None
    prev_time = time.time()
    last_action_id = 0
    prev_frame = None
    
    # Buffers for transformer
    state_buffer = []
    action_buffer = []
    return_buffer = []
    
    # Search mode variables
    offset_history = []
    search_index = 0
    last_valid_offset = None

    last_search_time = 0
    search_delay = random.uniform(0.5, 5.0)
    cat_lost_time = None  
    
    print("Starting cat tracking system...")
    
    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]
        
        # Motion detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(frame_gray, prev_frame)
            frame_entropy = np.mean(diff)
        else:
            frame_entropy = 0.0
        prev_frame = frame_gray

        # Skip if no motion detected
        if frame_entropy <= 0.1:
            cv2.imshow("Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # YOLO detection
        results = yolo_model(frame, verbose=False)[0]
        distance = read_sensor_data()
        
        action_taken = False
        
        # Process detections
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) != 15:  # Not a cat
                continue
                
            detection_data, x1, y1, x2, y2 = process_cat_detection(
                frame, box, cls, cnn_model, prev_center, prev_area, prev_time, w, h, offset_history
            )
            
            if detection_data is None:  # Low confidence detection
                continue
            
            # Update tracking variables
            prev_center = detection_data['new_center']
            prev_area = detection_data['new_area']
            prev_time = detection_data['new_time']
            last_valid_offset = detection_data['normalized_offset']
            
            # Create state vector for transformer
            state_vector = [
                float(detection_data['best_class']),
                detection_data['confidence'],
                detection_data['box_area'],
                detection_data['dx'],
                detection_data['dy'],
                detection_data['darea'],
                detection_data['elapsed'],
                float(last_action_id),
                detection_data['normalized_offset'],
                detection_data['aspect_ratio'],
                detection_data['visibility_score'],
                detection_data['velocity_mag'],
                distance if distance is not None else -1.0,
                last_valid_offset if last_valid_offset is not None else 0.0,
            ]
            
            # Update buffers
            state_buffer.append(state_vector)
            action_buffer.append(last_action_id)
            return_buffer.append(1.0 if detection_data['best_class'] in [0, 1] else 0.0)

            if len(state_buffer) < 3:
                action_id = 0  # Stop during warmup
                reward = 0.0
                source = "warmup"
                action_probs = None # Researc/debugging
            else:
                
                action_id, action_probs, reward, source = select_action_with_transformer(
                    state_buffer, action_buffer, return_buffer, transformer_model, last_action_id
                )
            
          
            
            # Send action and log
            send_action(action_id)
            log_behavior(
                source=source,
                goal_id='tracking',
                goal_status="active",
                selected_action=action_id,
                reward=reward,
                success=True,
                state_vector=state_vector,
              
            )
            
            # Draw bounding box and label
            cat_name = 'Doja' if detection_data['best_class'] == 0 else 'Harlow'
            label = f"{cat_name} ({detection_data['confidence']:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            last_action_id = action_id
            action_taken = True
            
            print(f"Cat detected: {state_vector}")
            cat_lost_time = None
            break  # Only process first valid cat detection
        
        # Handle case when no cat is detected
        # In your main loop (no cat detected):
        if not action_taken:
            current_time = time.time()
            
            if cat_lost_time is None:
                cat_lost_time = current_time
            
            time_lost = current_time - cat_lost_time
            
            if time_lost < 1.0:  # First 1 second - use transformer
                dummy_state = create_dummy_state(distance, last_action_id, last_valid_offset, True)
                state_buffer.append(dummy_state)
                action_buffer.append(last_action_id)
                
                if len(state_buffer) >= 3:
                    action_id, _, reward, source = select_action_with_transformer(
                        state_buffer, action_buffer, return_buffer, transformer_model, last_action_id
                    )

                    log_behavior(
                        source=source,
                        goal_id="reframing",
                        goal_status="active",
                        selected_action=action_id,
                        reward=0.0,
                        success=True,
                        state_vector=dummy_state,
            
                    )
                else:
                    action_id = 0
                    source = "warmup"
                
                goal_id = "reframing"
                
            else:  # After 1 second - use heuristic search
                if current_time - last_search_time > search_delay:
                    obstacle_detected = distance is not None and 0 < distance < 20
                    action_id, search_index = handle_search_mode(search_index, last_valid_offset, obstacle_detected)
                    last_search_time = current_time
                    search_delay = random.uniform(0.5, 5.0)
                else:
                    action_id = last_action_id
                
                source = "search_heuristic"
                goal_id = "searching"
            send_action(action_id)
            # No Learning no logging
            # log_behavior(
            #     source=source,
            #     goal_id="searching",
            #     goal_status="active",
            #     selected_action=action_id,
            #     reward=0.0,
            #     success=False,
            #     state_vector=dummy_state,
         
            # )
            
            last_action_id = action_id
            # print(f"No cat detected: {dummy_state}")
        
        # Maintain buffer size
        if len(state_buffer) > 100:
            state_buffer.pop(0)
            action_buffer.pop(0)
            return_buffer.pop(0)
        
        # Display frame
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    esp.close()

if __name__ == "__main__":
    
    main()
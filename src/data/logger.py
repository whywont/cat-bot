import pandas as pd
import os
import time

LOG_PATH = "behavior_log.csv"

# Define state vector column names based on your state_vector structure
state_columns = [
    "cat_class",           # float(best_class)
    "confidence",          # confidence
    "box_area",           # box_area
    "dx",                 # dx
    "dy",                 # dy
    "darea",              # darea
    "elapsed",            # elapsed
    "last_action_id",     # float(last_action_id)
    "normalized_offset",  # normalized_offset
    "aspect_ratio",       # aspect_ratio
    "visibility_score",   # visibility_score
    "velocity_mag",       # velocity_mag
    "distance",           # distance
    "last_valid_offset"   # last_valid_offset
]

columns = [
    "timestamp",          # UNIX time
    "source",             # "heuristic", "model", "script", etc.
    "goal_id",            # e.g., "track_cat", "avoid_obstacle"
    "goal_status",        # e.g., "active", "achieved", "failed"
    "selected_action",    # "F", "B", "L", "R", "S"
    "reward",             # float
    "success",            # bool
    "action_probs",       # optional: for model phase
    "model_name"          # optional: for model phase
] + state_columns

# Initialize log file if needed
if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=columns)
    df.to_csv(LOG_PATH, index=False)

def log_behavior(source, goal_id, goal_status, selected_action, reward, success, state_vector, action_probs=None, model_name=None):
    # Create base entry
    entry = {
        "timestamp": int(time.time()),
        "source": source,
        "goal_id": goal_id,
        "goal_status": goal_status,
        "selected_action": selected_action,
        "reward": reward,
        "success": success,
        "action_probs": action_probs,
        "model_name": model_name
    }
    
    # Add state vector as individual columns
    if state_vector and len(state_vector) >= len(state_columns):
        for i, col_name in enumerate(state_columns):
            entry[col_name] = state_vector[i]
    else:
        # Fill with None if state_vector is missing or wrong length
        for col_name in state_columns:
            entry[col_name] = None
    
    df = pd.DataFrame([entry])
    df.to_csv(LOG_PATH, mode="a", header=False, index=False)
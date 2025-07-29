# Cognitive Enrichment Robot for Pet Interaction

This project implements a cognitive robotics system that simulates action control and decision making through heuristic filtering, expert demonstration, and supervised learning. The system is embodied in a mobile robot and designed to autonomously interact with pets—particularly cats, using real-time computer vision and action selection.

---

## Overview

The goal of this project is to examine how cognitive mechanisms like attention and action selection can be instantiated in a physical agent operating under uncertainty. The system uses a Raspberry Pi 5 as the main controller, with an onboard camera for visual input, an ultrasonic sensor for depth, and a mecanum-wheeled chassis for mobility.

The robot is capable of:
- Detecting cats using YOLOv8
- Identifying individual cats via a custom CNN
- Extracting spatial and contextual features into 14D state vectors
- Selecting actions via a Decision Transformer trained through behavioral cloning
- Logging all perceptual and motor data for analysis

---

## Use Case

The system is designed to enrich pet environments by acting as an interactive, responsive robotic toy. Rather than relying on repetitive preprogrammed behaviors, it adapts based on perceptual input and learned experience, mimicking cognitive mechanisms like attention, memory, and decision-making.

---

## Cognitive Architecture

| Module               | Description |
|----------------------|-------------|
| **Rule-Based System** | Heuristics based on spatial metrics (e.g., bounding box size) to shape behavior or assign rewards. |
| **Expert Demonstration** | Manual control through a Flask web interface to generate high-quality training data. |
| **Decision Transformer** | Transformer model trained on expert traces to learn sequential decision-making. |

A state vector is computed per frame, combining:
- Object detection results
- CNN-based identity classification
- Velocity and aspect ratio
- Depth sensing via ultrasonic input
- Previous state and last action taken

---

## Hardware Setup

| Component           | Model                      | Cost |
|--------------------|----------------------------|------|
| Camera             | IMX708 Pi Camera Module    | $18  |
| Compute Unit       | Raspberry Pi 5             | $95  |
| Robot Chassis      | LewanSoul Chassis          | $40  |
| Motor Drivers      | BTS7960 x2                 | $15  |
| Microcontroller    | ESP32 DevKitC              | $12  |
| Distance Sensor    | HC-SR04                    | $5   |
| Power Supply       | 6V Battery + Buck Converter| $24  |

---

## Behavior Pipeline

1. **Detection**: YOLOv8 identifies cat presence.
2. **Classification**: A custom CNN distinguishes between individual cats.
3. **State Vector**: Frame is encoded into a 14-dimensional vector.
4. **Action Selection**:
    - If detection is confident, the Decision Transformer chooses the next action.
    - In ambiguous states (e.g., no detection), a rule-based controller is used.
5. **Execution**: Action commands are sent to motor drivers for movement.
6. **Logging**: Each frame’s state, action, and confidence are logged for analysis.

---

## Model Details

### Custom CNN
- 3 Conv layers: 16, 32, 64 filters
- GlobalAvgPooling, Dense(64), Dropout(30%)
- Input: Cropped 128x128 image patches

### Decision Transformer
- Input: [batch, 2, 14] state vector sequence
- Output: Next action (stop, forward, back, left, right)
- Trained with behavioral cloning on expert demonstration
- Reward-agnostic generalization (learns to favor cat-present states)

---

## Decision Transformer Controlled Agent Demonstration

![Demo](demo/decision_transformer_demo.gif)

---

## Screenshots


![Frame Detection](demo/Live%20Detection_screenshot_11.07.2025.png)
_Bounding box and label shown during real-time inference._


![Frame Detection](demo/Live%20Detection_screenshot_12.07.2025.png)
_Bounding box and label shown during real-time inference._

---




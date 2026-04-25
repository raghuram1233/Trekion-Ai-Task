# Trekion.ai Robotics Perception Pipeline

This repository contains a production-grade multi-sensor perception and telemetry pipeline developed for the Trekion.ai technical assignment. The project implements a complete robotics stack for data synchronization, depth estimation, and scene understanding.

## Overview

The pipeline is divided into three core modules:

1.  **Task 1: Multi-Sensor Synchronization**
    *   Parses proprietary binary data (`.imu` and `.vts`).
    *   Synchronizes high-frequency IMU data with video frames using binary search.
    *   Generates a telemetry-overlaid video with 3D scrolling waterfall plots and a real-time HUD.
2.  **Task 2: Monocular Depth Estimation**
    *   Utilizes the **MiDaS (DPT_Large)** Transformer-based model for dense depth maps.
    *   Implements temporal EMA smoothing and percentile-based normalization.
    *   Produces a side-by-side comparison video using the `TURBO` colormap.
3.  **Task 3: Object Detection & Scene Segmentation**
    *   Deploys **YOLOv8m-seg** for real-time instance segmentation and bounding box detection.
    *   Integrates **MediaPipe Hands** for 21-landmark 3D hand tracking and handedness classification.
    *   Generates a detailed perception overlay with confidence scores and object statistics.

## Project Structure

```text
Trekion.ai/
├── Data/                       # Input recording data
│   ├── recording2.mp4          # 1080p wide-angle video
│   ├── recording2.imu          # Binary IMU telemetry
│   └── recording2.vts          # Binary frame-to-timestamp mapping
├── Task1/                      # Multi-Sensor Sync
│   └── main.py                 # Core sync and HUD pipeline
├── Task2/                      # Depth Estimation
│   └── depth_pipeline.py       # MiDaS implementation
├── Task3/                      # Scene Understanding
│   └── detect_pipeline.py      # YOLO + MediaPipe pipeline
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── yolov8m-seg.pt              # Pre-trained model weights
```

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: A GPU is recommended for running the Depth (Task 2) and Detection (Task 3) pipelines at interactive speeds.*

## Execution Instructions

Each task is self-contained and configured with hardcoded paths pointing to the `Data/` directory.

### Task 1: Telemetry & Sync
```bash
python Task1/main.py
```
*Output: Task1/recording2_telemetry.mp4*

### Task 2: Depth Estimation
```bash
python Task2/depth_pipeline.py
```
*Output: Task2/depth_output.mp4*

### Task 3: Scene Understanding
```bash
python Task3/detect_pipeline.py
```
*Output: Task3/detect_output.mp4*

### Output Files
   Can be Found on https://drive.google.com/drive/folders/1m8vXuIEDgreiBRkwQW54GbP_ZJTk1icA?usp=sharing

## Technical Implementation Details

*   **Binary Parsing**: IMU records are parsed using explicit `struct` unpacking (int64 timestamp + 10x float32 sensors + 32-byte padding).
*   **Synchronization**: Employs `numpy.searchsorted` for O(log N) lookup efficiency, ensuring the telemetry HUD is precisely aligned with visual frames.
*   **Visualization**: Uses a custom HUD rendering engine designed to prevent GPU driver contention and resource locking during video encoding.
*   **Hand Tracking**: Implemented using MediaPipe's video-mode configuration to leverage temporal consistency for improved landmark stability.

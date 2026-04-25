# Trekion.ai Robotics Perception Pipeline: Technical Writeup

## Approach to Parsing Binary Formats

**There is a major issue with output of task 1 even though the data has been parsed the visualization part is not working properly as the graphs keep fluctuating/Flickering after every frame is proccessed.**

The project required parsing two proprietary binary formats: `.imu` files containing IMU telemetry data and `.vts` files containing video frame-to-timestamp mappings. For the IMU data, we implemented a structured approach using Python's `struct` module to unpack binary records. Each IMU record consists of 80 bytes with a specific format: a 64-bit timestamp followed by 10 float values (accelerometer, gyroscope, magnetometer, and temperature) and 32 padding bytes.

The parsing logic involved:

- Skipping the first 64 bytes of header data
- Unpacking each record using the format string `'<q10f32x'` (little-endian: long long, 10 floats, 32 bytes padding)
- Converting timestamps from microseconds to seconds for consistency
- Validating data integrity through statistical analysis (e.g., accelerometer magnitude calculations)

For VTS files, we used NumPy's structured arrays for efficient parsing:

- Skipping 32 bytes of header
- Defining a dtype with frame index, global timestamp, camera index, and camera timestamp
- Converting timestamps to seconds for synchronization

This approach ensured robust parsing with minimal memory overhead and provided clear error handling for malformed files.

## Model Choices and Reasoning

### Depth Estimation (Task 2)

We selected the MiDaS DPT_Large model for monocular depth estimation due to its state-of-the-art performance on diverse datasets and transformer-based architecture. The DPT_Large variant was chosen over smaller models for its superior accuracy, despite higher computational requirements. This decision was motivated by the need for high-quality depth maps in robotics applications where precision is critical.

The implementation includes:

- Temporal exponential moving average (EMA) smoothing to reduce flickering
- Percentile-based normalization (2nd-98th percentile) for robust colormap scaling
- Bicubic interpolation for maintaining depth map resolution

### Object Detection and Segmentation (Task 3)

For scene understanding, we deployed YOLOv8m-seg, the medium-sized segmentation variant of the YOLOv8 family. This choice balanced accuracy and inference speed, making it suitable for real-time applications. The segmentation capability was crucial for providing detailed object boundaries beyond simple bounding boxes.

Additionally, we integrated MediaPipe Hands for 3D hand tracking and handedness classification. This lightweight solution complemented YOLO's object detection by providing specialized hand pose estimation, which is valuable in human-robot interaction scenarios.

## Challenges Faced

### Binary Format Parsing

The main Challenge was to visualize the data in real-time, and generating the output side by side.

Another challenge was reverse-engineering the binary formats without official specifications. We relied on trial-and-error with struct formats and validation through statistical analysis of the parsed data. Ensuring timestamp accuracy was critical for proper sensor synchronization.

### Sensor Synchronization

Synchronizing high-frequency IMU data (200+ Hz) with video frames (30 FPS) required efficient algorithms. We implemented binary search for timestamp matching, but handling edge cases like missing data points and ensuring sub-millisecond accuracy proved challenging.

### Performance Optimization

Running multiple AI models (MiDaS, YOLO, MediaPipe) on video streams demanded careful resource management. GPU utilization was essential for maintaining real-time performance, and we had to implement frame skipping strategies for slower hardware configurations.

### Visualization Complexity

Creating the 3D scrolling telemetry plots required integrating Matplotlib with OpenCV video processing. Managing memory efficiently for real-time rendering while maintaining visual quality was non-trivial.

## Ideas for Improvement

### Enhanced Binary Parsing

- Implement more robust header detection and validation
- Add support for compressed or variable-length records
- Create a configuration file for different binary format variants

### Advanced Synchronization

- Implement Kalman filtering for better IMU-video fusion
- Add interpolation for missing IMU samples
- Support for multi-camera timestamp synchronization

### Model Optimization

- Explore quantized models (INT8) for faster inference on edge devices
- Implement model ensemble approaches for improved accuracy
- Add domain adaptation techniques for specific robotics environments

### Pipeline Enhancements

- Integrate SLAM (Simultaneous Localization and Mapping) for pose estimation
- Add temporal tracking across frames for object persistence
- Implement confidence-based filtering for noisy detections

### Performance Improvements

- GPU acceleration for all processing stages
- Multi-threading for parallel model inference
- Streaming processing for reduced memory usage

### User Experience

- Add configuration files for easy parameter tuning
- Implement logging and performance monitoring
- Create a web-based visualization interface

This pipeline demonstrates a comprehensive approach to multi-sensor robotics perception, balancing accuracy, performance, and maintainability. The modular design allows for easy extension and optimization for specific use cases.

import cv2
import numpy as np
import time
import sys
import os
from collections import defaultdict

# Generate a visually distinct color palette for object classes
def generate_color_palette(num_classes: int = 80) -> list:
    palette = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        color_hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append(tuple(int(c) for c in color_bgr))
    return palette

COLOR_PALETTE = generate_color_palette(80)

# Load the pre-trained YOLO segmentation model
def load_yolo_model(model_path: str = "yolov8m-seg.pt"):
    from ultralytics import YOLO
    print(f"[Model] Loading YOLO '{model_path}'...")
    model = YOLO(model_path)
    device_name = "CUDA" if hasattr(model, 'device') and str(model.device) != 'cpu' else "CPU"
    print(f"[Model] Loaded successfully. Device: {device_name}")
    return model

# Initialize the MediaPipe Hands detector
def load_hand_detector():
    try:
        import mediapipe as mp  #type:ignore
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[Hands] MediaPipe Hands loaded successfully.")
        return hands, mp_hands, mp_drawing, mp_drawing_styles
    except ImportError:
        print("[Hands] MediaPipe not installed. Hand detection disabled.")
        return None, None, None, None

# Run YOLO inference on a single BGR frame
def process_frame(frame_bgr: np.ndarray, model, conf: float = 0.35) -> list:
    results = model(frame_bgr, conf=conf, verbose=False)
    return results

# Draw bounding boxes, labels, and segmentation masks on the frame
def draw_detections(frame: np.ndarray, results, model, mask_alpha: float = 0.4) -> np.ndarray:
    result = results[0]
    boxes = result.boxes
    names = model.names

    if boxes is None or len(boxes) == 0:
        return frame

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    if hasattr(result, 'masks') and result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for i, mask in enumerate(masks_data):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_resized > 0.5
            color = COLOR_PALETTE[cls_ids[i] % len(COLOR_PALETTE)]
            overlay[mask_bool] = color
        cv2.addWeighted(overlay, mask_alpha, frame, 1.0 - mask_alpha, 0, frame)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        cls_id = cls_ids[i]
        conf = confs[i]
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        class_name = names.get(cls_id, f"cls_{cls_id}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

# Detect and draw hand landmarks using MediaPipe
def draw_hands(frame: np.ndarray, hands_detector, mp_hands, mp_drawing, mp_drawing_styles) -> np.ndarray:
    if hands_detector is None:
        return frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Hand"
            if results.multi_handedness and hand_idx < len(results.multi_handedness):
                handedness = results.multi_handedness[hand_idx].classification[0].label

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            h, w = frame.shape[:2]
            wrist = hand_landmarks.landmark[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f"[{handedness}]", (wx - 20, wy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

# Main pipeline to process video for object detection and hand tracking
def generate_video(input_path: str, output_path: str, model_path: str = "yolov8m-seg.pt", conf: float = 0.35, enable_hands: bool = True):
    model = load_yolo_model(model_path)
    hands_detector, mp_hands, mp_drawing, mp_drawing_styles = (None, None, None, None)
    if enable_hands:
        hands_detector, mp_hands, mp_drawing, mp_drawing_styles = load_hand_detector()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Video] Input: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[Error] Cannot create output video: {output_path}")
        cap.release()
        sys.exit(1)

    frame_idx = 0
    total_time = 0
    detection_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()
        results = process_frame(frame, model, conf=conf)
        frame = draw_detections(frame, results, model)
        if hands_detector is not None:
            frame = draw_hands(frame, hands_detector, mp_hands, mp_drawing, mp_drawing_styles)

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        total_time += elapsed
        current_fps = 1.0 / elapsed if elapsed > 0 else 0

        result = results[0]
        if result.boxes is not None:
            for cls_id in result.boxes.cls.cpu().numpy().astype(int):
                detection_counts[model.names[cls_id]] += 1

        fps_text = f"FPS: {current_fps:.1f} | Frame: {frame_idx}/{total_frames}"
        cv2.putText(frame, fps_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        det_count = len(result.boxes) if result.boxes is not None else 0
        cv2.putText(frame, f"Objects: {det_count}", (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            print(f"  [{frame_idx:>5}/{total_frames}] Avg FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    if hands_detector is not None:
        hands_detector.close()
    print(f"\nPipeline Complete. Saved to: {output_path}")

if __name__ == "__main__":
    input_video = "C:/TRR/TRR/Trekion.ai/Data/recording2.mp4"
    output_video = "C:/TRR/TRR/Trekion.ai/Task3/detect_output.mp4"
    model_path = "yolov8m-seg.pt"
    confidence = 0.35
    enable_hands = True
    generate_video(input_video, output_video, model_path, confidence, enable_hands)

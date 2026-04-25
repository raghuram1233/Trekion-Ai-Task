import cv2
import numpy as np
import torch
import time
import sys
import os

# Load MiDaS model and the appropriate transform
def load_model(model_type: str = "DPT_Large", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Loading MiDaS '{model_type}' on {device}...")

    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    print(f"[Model] Loaded successfully. Device: {device}")
    return model, transform, device

# Convert BGR frame to RGB and run MiDaS depth inference
def process_frame(frame_bgr: np.ndarray, model, transform, device) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
 
    depth_map = prediction.cpu().numpy().astype(np.float32)
    return depth_map

# Apply temporal smoothing and colormap to the raw depth map
def apply_colormap(depth: np.ndarray, prev_depth: np.ndarray = None, alpha: float = 0.8) -> tuple:
    if prev_depth is not None:
        smoothed = alpha * depth + (1.0 - alpha) * prev_depth
    else:
        smoothed = depth.copy()

    p_low = np.percentile(smoothed, 2)
    p_high = np.percentile(smoothed, 98)

    if p_high - p_low < 1e-6:
        normalized = np.zeros_like(smoothed, dtype=np.uint8)
    else:
        normalized = np.clip((smoothed - p_low) / (p_high - p_low), 0.0, 1.0)
        normalized = (normalized * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return depth_colored, smoothed

# Process input video through MiDaS and save a side-by-side output video
def generate_video(input_path: str, output_path: str, model_type: str = "DPT_Large"):
    model, transform, device = load_model(model_type)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Video] Input: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    out_width = width * 2
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    if not out.isOpened():
        print(f"[Error] Cannot create output video: {output_path}")
        cap.release()
        sys.exit(1)

    prev_depth = None
    frame_idx = 0
    total_inference_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()
        depth_map = process_frame(frame, model, transform, device)
        depth_colored, prev_depth = apply_colormap(depth_map, prev_depth, alpha=0.8)

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        total_inference_time += elapsed
        current_fps = 1.0 / elapsed if elapsed > 0 else 0

        fps_text = f"FPS: {current_fps:.1f} | Frame: {frame_idx}/{total_frames}"
        cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(depth_colored, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        combined = np.hstack((frame, depth_colored))
        out.write(combined)

        frame_idx += 1
        if frame_idx % 30 == 0:
            avg_fps = frame_idx / total_inference_time if total_inference_time > 0 else 0
            print(f"  [{frame_idx:>5}/{total_frames}] Avg FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    print(f"\nPipeline Complete. Saved to: {output_path}")

if __name__ == "__main__":
    input_video = "C:/TRR/TRR/Trekion.ai/Data/recording2.mp4"
    output_video = "C:/TRR/TRR/Trekion.ai/Task2/depth_output.mp4"
    model_type = "DPT_Large"
    generate_video(input_video, output_video, model_type)

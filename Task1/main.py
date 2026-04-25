import cv2
import numpy as np
import struct
import io
import time
import sys
import os
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Parse VTS binary file to extract frame-to-timestamp mappings
def parse_vts(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        f.read(32)
        data = f.read()

    record_size = 24
    num_records = len(data) // record_size

    dtype = np.dtype([
        ('frame_idx', '<i4'),
        ('ts_global', '<i8'),
        ('cam_idx',   '<i4'),
        ('ts_cam',    '<i8'),
    ])
    raw = np.frombuffer(data[:num_records * record_size], dtype=dtype)

    vts = np.zeros((num_records, 2), dtype=np.float64)
    vts[:, 0] = raw['frame_idx']
    vts[:, 1] = raw['ts_global'].astype(np.float64) / 1e6

    print(f"[VTS] Parsed {num_records} frame mappings")
    print(f"[VTS] Time range: {vts[0,1]:.3f}s — {vts[-1,1]:.3f}s")
    return vts

# Parse IMU binary file to extract multi-axis sensor data
def parse_imu(file_path: str) -> Tuple[np.ndarray, float]:
    with open(file_path, 'rb') as f:
        f.read(64)
        data = f.read()

    record_size = 80
    num_records = len(data) // record_size
    fmt = '<q10f32x'
    
    imu_data = np.zeros((num_records, 11), dtype=np.float64)

    for i in range(num_records):
        offset = i * record_size
        values = struct.unpack(fmt, data[offset:offset + record_size])
        imu_data[i, 0] = values[0] / 1e6
        imu_data[i, 1:] = values[1:]

    dt = np.diff(imu_data[:, 0])
    sample_rate = 1.0 / np.median(dt) if len(dt) > 0 else 0.0

    acc_mag = np.sqrt(imu_data[:, 1]**2 + imu_data[:, 2]**2 + imu_data[:, 3]**2)
    print(f"[IMU] Parsed {num_records} records at {sample_rate:.1f} Hz")
    print(f"[IMU] Accel magnitude: mean={np.mean(acc_mag):.2f}")
    return imu_data, sample_rate

# Synchronize IMU data with video frames using binary search
class SensorSynchronizer:
    def __init__(self, imu_data: np.ndarray, vts_data: np.ndarray):
        self.imu_data = imu_data
        self.vts_data = vts_data
        self.imu_timestamps = imu_data[:, 0]
        self.frame_timestamps = vts_data[:, 1]

        indices = np.searchsorted(self.imu_timestamps, self.frame_timestamps)
        indices = np.clip(indices, 0, len(self.imu_timestamps) - 1)
        self.sync_indices = indices
        synced_imu_ts = self.imu_timestamps[self.sync_indices]
        self.sync_delays = np.abs(synced_imu_ts - self.frame_timestamps)

    # Return synchronized IMU index for each video frame
    def get_sync_indices(self) -> np.ndarray:
        return self.sync_indices

    # Return synchronization delay statistics
    def get_sync_stats(self) -> dict:
        return {
            "mean":   np.mean(self.sync_delays),
            "median": np.median(self.sync_delays),
            "max":    np.max(self.sync_delays),
        }

    # Extract sliding time window of IMU data for plotting
    def get_window(self, frame_idx: int, window_sec: float = 0.5) -> np.ndarray:
        if frame_idx >= len(self.frame_timestamps):
            frame_idx = len(self.frame_timestamps) - 1

        center_ts = self.frame_timestamps[frame_idx]
        t_start = center_ts - window_sec
        t_end = center_ts + window_sec

        idx_start = np.searchsorted(self.imu_timestamps, t_start, side='left')
        idx_end = np.searchsorted(self.imu_timestamps, t_end, side='right')

        if idx_end - idx_start < 2:
            idx_start = max(0, idx_start - 1)
            idx_end = min(len(self.imu_timestamps), idx_end + 1)

        return self.imu_data[idx_start:idx_end]

# Render real-time scrolling 3D telemetry plots and HUD overlays
class TelemetryVisualizer:
    def __init__(self, plot_width: int = 800, plot_height: int = 1080):
        self.plot_width = plot_width
        self.plot_height = plot_height

        self.fig = plt.figure(figsize=(8, 10.8), dpi=100)
        self.fig.patch.set_facecolor('#121212')
        self.fig.tight_layout(pad=3.0)

        self.ax_acc = self.fig.add_subplot(311, projection='3d')
        self.ax_gyr = self.fig.add_subplot(312, projection='3d')
        self.ax_mag = self.fig.add_subplot(313, projection='3d')
        self.axes = [self.ax_acc, self.ax_gyr, self.ax_mag]

    # Render 3D waterfall plots into a BGR numpy array
    def render_plots(self, window_data: np.ndarray, current_ts: float) -> np.ndarray:
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor('#1e1e1e')
            ax.xaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
            ax.yaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
            ax.zaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
            ax.grid(color='#444444', linestyle='--', linewidth=0.5)
            ax.tick_params(colors='white', labelsize=7)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')

        ts = window_data[:, 0] - current_ts

        self.ax_acc.plot(ts, np.full_like(ts, 1), window_data[:, 1], color='#ff4444', linewidth=1)
        self.ax_acc.plot(ts, np.full_like(ts, 2), window_data[:, 2], color='#44ff44', linewidth=1)
        self.ax_acc.plot(ts, np.full_like(ts, 3), window_data[:, 3], color='#4488ff', linewidth=1)
        self.ax_acc.set_title('Accelerometer (m/s²)', color='white', fontsize=10)
        self.ax_acc.set_xlim3d(-0.5, 0.5)
        self.ax_acc.set_ylim3d(0.5, 3.5)
        self.ax_acc.set_zlim3d(-15, 15)
        self.ax_acc.set_xlabel('Time (s)', fontsize=8)
        self.ax_acc.set_yticks([1, 2, 3])
        self.ax_acc.set_yticklabels(['X', 'Y', 'Z'], color='white', fontsize=8)

        self.ax_gyr.plot(ts, np.full_like(ts, 1), window_data[:, 4], color='#ff4444', linewidth=1)
        self.ax_gyr.plot(ts, np.full_like(ts, 2), window_data[:, 5], color='#44ff44', linewidth=1)
        self.ax_gyr.plot(ts, np.full_like(ts, 3), window_data[:, 6], color='#4488ff', linewidth=1)
        self.ax_gyr.set_title('Gyroscope (rad/s)', color='white', fontsize=10)
        self.ax_gyr.set_xlim3d(-0.5, 0.5)
        self.ax_gyr.set_ylim3d(0.5, 3.5)
        self.ax_gyr.set_zlim3d(-4, 4)
        self.ax_gyr.set_xlabel('Time (s)', fontsize=8)
        self.ax_gyr.set_yticks([1, 2, 3])
        self.ax_gyr.set_yticklabels(['X', 'Y', 'Z'], color='white', fontsize=8)

        self.ax_mag.plot(ts, np.full_like(ts, 1), window_data[:, 7], color='#ff4444', linewidth=1)
        self.ax_mag.plot(ts, np.full_like(ts, 2), window_data[:, 8], color='#44ff44', linewidth=1)
        self.ax_mag.plot(ts, np.full_like(ts, 3), window_data[:, 9], color='#4488ff', linewidth=1)
        self.ax_mag.set_title('Magnetometer (µT)', color='white', fontsize=10)
        self.ax_mag.set_xlim3d(-0.5, 0.5)
        self.ax_mag.set_ylim3d(0.5, 3.5)
        self.ax_mag.set_zlim3d(-150, 150)
        self.ax_mag.set_xlabel('Time (s)', fontsize=8)
        self.ax_mag.set_yticks([1, 2, 3])
        self.ax_mag.set_yticklabels(['X', 'Y', 'Z'], color='white', fontsize=8)

        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', facecolor=self.fig.get_facecolor())
        buf.seek(0)
        img = Image.open(buf)
        img_np = np.array(img.convert('RGB'))
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Draw telemetry HUD overlay on the video frame
    def draw_hud(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (460, 340), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        color = (0, 255, 0)
        thick = 1

        lines = [
            f"FRAME: {stats.get('frame_idx', 0)}",
            f"TIME:  {stats.get('timestamp', 0):.3f} s",
            f"TEMP:  {stats.get('temp', 0):.1f} C",
            f"IMU RATE: {stats.get('imu_rate', 0):.1f} Hz",
            "",
            f"ACC (X,Y,Z): {stats.get('acc_x',0):>7.2f}, {stats.get('acc_y',0):>7.2f}, {stats.get('acc_z',0):>7.2f}",
            f"GYR (X,Y,Z): {stats.get('gyr_x',0):>7.3f}, {stats.get('gyr_y',0):>7.3f}, {stats.get('gyr_z',0):>7.3f}",
            f"MAG (X,Y,Z): {stats.get('mag_x',0):>7.1f}, {stats.get('mag_y',0):>7.1f}, {stats.get('mag_z',0):>7.1f}",
            "",
            f"SYNC DELAY:",
            f"  Mean:   {stats.get('sync_mean', 0)*1000:.2f} ms",
            f"  Median: {stats.get('sync_median', 0)*1000:.2f} ms",
            f"  Max:    {stats.get('sync_max', 0)*1000:.2f} ms",
        ]

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (35, 55 + i * 24), font, scale, color, thick, cv2.LINE_AA)
        return frame

    # Combine video frame and plot panel side-by-side
    def combine(self, video_frame: np.ndarray, plot_frame: np.ndarray) -> np.ndarray:
        target_h = video_frame.shape[0]
        if plot_frame.shape[0] != target_h or plot_frame.shape[1] != 800:
            plot_frame = cv2.resize(plot_frame, (800, target_h))
        return np.hstack((video_frame, plot_frame))

# Process binary data, synchronize, and render the output video
def generate_video(video_path: str, imu_path: str, vts_path: str, output_path: str):
    cv2.ocl.setUseOpenCL(False)
    print("=" * 60)
    print("  Multi-Sensor Synchronization Pipeline")
    print("=" * 60)

    print("\n[Step 1] Parsing binary sensor data...")
    vts_data = parse_vts(vts_path)
    imu_data, imu_rate = parse_imu(imu_path)

    print("\n[Step 2] Building synchronization engine...")
    sync = SensorSynchronizer(imu_data, vts_data)
    sync_stats = sync.get_sync_stats()
    sync_indices = sync.get_sync_indices()

    print(f"[Sync] Delay — Mean: {sync_stats['mean']*1000:.2f} ms")

    print("\n[Step 3] Setting up video pipeline...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 30 or total_frames < len(vts_data):
        total_frames = len(vts_data)

    out_width = width + 800
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    if not out.isOpened():
        print(f"[Error] Cannot create output video: {output_path}")
        cap.release()
        sys.exit(1)

    print(f"\n[Step 4] Processing {total_frames} frames...")
    viz = TelemetryVisualizer()
    t_total_start = time.perf_counter()

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret or i >= len(vts_data):
            break

        imu_idx = sync_indices[i]
        current_imu = imu_data[imu_idx]
        current_ts = vts_data[i, 1]
        window = sync.get_window(i, window_sec=0.5)

        stats = {
            "frame_idx": i,
            "timestamp": current_ts,
            "temp":      current_imu[10],
            "imu_rate":  imu_rate,
            "acc_x": current_imu[1], "acc_y": current_imu[2], "acc_z": current_imu[3],
            "gyr_x": current_imu[4], "gyr_y": current_imu[5], "gyr_z": current_imu[6],
            "mag_x": current_imu[7], "mag_y": current_imu[8], "mag_z": current_imu[9],
            "sync_mean":   sync_stats["mean"],
            "sync_median": sync_stats["median"],
            "sync_max":    sync_stats["max"],
        }

        plot_frame = viz.render_plots(window, current_ts)
        frame_with_hud = viz.draw_hud(frame, stats)
        combined = viz.combine(frame_with_hud, plot_frame)

        out.write(combined)

        if i % 50 == 0:
            elapsed = time.perf_counter() - t_total_start
            est_fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i:>5}/{total_frames}] Processing FPS: {est_fps:.1f}")

    cap.release()
    out.release()
    print(f"\nPipeline Complete. Saved to: {output_path}")

if __name__ == "__main__":
    video_path  = "C:/TRR/TRR/Trekion.ai/Data/recording2.mp4"
    imu_path    = "C:/TRR/TRR/Trekion.ai/Data/recording2.imu"
    vts_path    = "C:/TRR/TRR/Trekion.ai/Data/recording2.vts"
    output_path = "C:/TRR/TRR/Trekion.ai/Task1/recording2_telemetry.mp4"
    generate_video(video_path, imu_path, vts_path, output_path)

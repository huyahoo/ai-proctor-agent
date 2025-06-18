import os
import json
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from core.config import Config
from core.utils import load_video_capture, assign_yolo_pids, assign_pose_pids
from cv.yolo_detector import YOLODetector
from cv.pose_estimator import PoseEstimator
from cv.gaze_tracker import GazeTracker
from ai.anomaly_detector import AnomalyDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process a video and export detection/anomaly data to JSON with progress bar.")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--output', '-o', type=str, default=None, help="Path for the output JSON file (default: ./data/output/<video_basename>.json)")
    parser.add_argument('--export_dir', '-e', type=str, default=None, help="Directory to export visualization images (optional)")
    return parser.parse_args()


def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def process_video(video_path: str, config: Config, export_dir: str = None) -> dict:
    """
    Processes the video and returns a dict ready to be dumped as JSON.
    Displays a tqdm progress bar. Optionally exports frame visualizations.
    """
    # Prepare export directory
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        print(f"Exporting images to {export_dir}")
    else:
        print("No export directory provided; skipping image export.")

    cap = load_video_capture(video_path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = config.FRAME_SKIP
    yolo = YOLODetector(config)
    pose = PoseEstimator(config)
    gaze = GazeTracker(config)
    anomaly = AnomalyDetector(config)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    result_data = []

    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            continue

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Perform detections
        yolo_dets = yolo.detect(frame)
        pose_ests = pose.detect(frame)
        gaze_ests = gaze.detect(frame)
        # Assign PIDs
        yolo_dets = assign_yolo_pids(yolo_dets, gaze_ests)
        pose_ests = assign_pose_pids(pose_ests, gaze_ests)
        # Detect anomalies
        frame_info = {
            'yolo_detections': yolo_dets,
            'pose_estimations': pose_ests,
            'gaze_estimations': gaze_ests,
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0]
        }
        anomalies = anomaly.detect_anomalies(frame_info, timestamp_sec)

        result_data.append({
            'frame_idx': frame_idx,
            'yolo_detections': yolo_dets,
            'pose_estimations': pose_ests,
            'gaze_estimations': gaze_ests,
            'anomalies': anomalies
        })

        if export_dir:
            yolo_frame = yolo.draw_results(frame.copy(), yolo_dets)
            pose_frame = pose.draw_results(frame.copy(), pose_ests)
            gaze_frame = gaze.draw_results(frame.copy(), gaze_ests)
            cv2.imwrite(os.path.join(export_dir, f"{base_name}_frame_{frame_idx:05d}_orig.jpg"), frame)
            cv2.imwrite(os.path.join(export_dir, f"{base_name}_frame_{frame_idx:05d}_yolo.jpg"), yolo_frame)
            cv2.imwrite(os.path.join(export_dir, f"{base_name}_frame_{frame_idx:05d}_pose.jpg"), pose_frame)
            cv2.imwrite(os.path.join(export_dir, f"{base_name}_frame_{frame_idx:05d}_gaze.jpg"), gaze_frame)

    cap.release()

    return {
        'video_path': video_path,
        'FRAME_SKIP': skip,
        'data': result_data
    }


def main():
    args = parse_args()
    config = Config()

    try:
        output = process_video(args.input, config, args.export_dir)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        # default to data/output/<basename>.json
        os.makedirs('data/output', exist_ok=True)
        out_path = os.path.join('data/output', f"{os.path.splitext(os.path.basename(args.input))[0]}.json")

    # Convert types and dump
    output = convert_to_python_types(output)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Exported JSON to {out_path}")

if __name__ == '__main__':
    main()
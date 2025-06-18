import os
import json
import argparse
import cv2

import numpy as np
from core.config import Config
from core.utils import load_video_capture, assign_yolo_pids, assign_pose_pids
from cv.yolo_detector import YOLODetector
from cv.pose_estimator import PoseEstimator
from cv.gaze_tracker import GazeTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Replay and visualize detections from JSON exports.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('-j', '--json', type=str, required=True, help='Path to JSON predictions file')
    parser.add_argument('-e', '--export_dir', type=str, default=None, help='Directory to write comparison images')
    return parser.parse_args()


def load_predictions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frame_map = {int(item['frame_idx']): item for item in data.get('data', [])}
    return frame_map, data.get('FRAME_SKIP', 1)


def main():
    args = parse_args()
    os.makedirs(args.export_dir, exist_ok=True)
    config = Config()

    # detectors for visualization
    yolo = YOLODetector(config)
    pose = PoseEstimator(config)
    gaze = GazeTracker(config)

    preds, frame_skip = load_predictions(args.json)
    cap = load_video_capture(args.input)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in preds:
            entry = preds[frame_idx]
            # extract prediction lists
            yolo_dets = entry['yolo_detections']
            pose_ests = entry['pose_estimations']
            gaze_ests = entry['gaze_estimations']

            # draw results
            orig_viz = frame.copy()
            yolo_viz = yolo.draw_results(frame.copy(), yolo_dets)
            pose_viz = pose.draw_results(frame.copy(), pose_ests)
            gaze_viz = gaze.draw_results(frame.copy(), gaze_ests)

            # save comparison images
            cv2.imwrite(
                os.path.join(args.export_dir, f"{base_name}_frame_{frame_idx:05d}_orig_comparison.jpg"),
                orig_viz
            )
            cv2.imwrite(
                os.path.join(args.export_dir, f"{base_name}_frame_{frame_idx:05d}_yolo_comparison.jpg"),
                yolo_viz
            )
            cv2.imwrite(
                os.path.join(args.export_dir, f"{base_name}_frame_{frame_idx:05d}_pose_comparison.jpg"),
                pose_viz
            )
            cv2.imwrite(
                os.path.join(args.export_dir, f"{base_name}_frame_{frame_idx:05d}_gaze_comparison.jpg"),
                gaze_viz
            )

        frame_idx += 1
        # skip frames
        for _ in range(frame_skip - 1):
            cap.grab()
            frame_idx += 1

    cap.release()
    print(f"Exported comparison images to {args.export_dir}")

if __name__ == '__main__':
    main()
import os
import cv2
import argparse
from tqdm import tqdm
import json
import sys
from pathlib import Path

# Get the project root directory and src directory
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"

# Add src directory to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ai.anomaly_detector import AnomalyDetector
from core.config import Config

def get_person_map(frame_data):
    person_map = {}  # { pid: {'bbox': ..., 'pose': ..., 'gaze': ...} }

    # Build PID-based map from YOLO
    person_map = {
        det['pid']: {'bbox': det['bbox'], 'pose': None, 'gaze': None}
        for det in frame_data["yolo_detections"]["person"]
        if det['label'] == 'person'
    }

    for det in frame_data["yolo_detections"]["exam_paper"]:
        if det["label"] == "paper" and det["pid"] != -1: 
            person_map[det["pid"]]["exam_paper"] = det["bbox"]

    # Associate pose keypoints by pid
    for p in frame_data.get('pose_estimations', []):
        pid = p['pid']
        if pid in person_map:
            person_map[pid]['pose'] = p['keypoints']

    # Associate gaze data by pid
    for g in frame_data.get('gaze_estimations', []):
        pid = g['pid']
        if pid in person_map:
            person_map[pid]['gaze'] = {
                'bbox': g['bbox'],
                'point': [g['gaze_point'][0] * frame_data["frame_width"], g['gaze_point'][1] * frame_data["frame_height"]],
                'vector': g['gaze_vector'],
                'score': g['inout_score']
            }

    return person_map

def main():
    anomaly_detector = AnomalyDetector(Config())
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    test_dir = os.path.join(parent_dir, "src", "test")
    with open(os.path.join(test_dir, "gaze_estimations.json"), "r") as f:
        gaze_info = json.load(f)
    with open(os.path.join(test_dir, "pose_estimations.json"), "r") as f:
        pose_info = json.load(f)
    with open(os.path.join(test_dir, "yolo_detections.json"), "r") as f:
        yolo_info = json.load(f)
    
    gaze_estimations = gaze_info["gaze_estimations"]
    pose_estimations = pose_info["pose_estimations"]
    yolo_detections = yolo_info["yolo_detections"]
    frame_width = yolo_info["frame_width"]
    frame_height = yolo_info["frame_height"]

    frame_data = {
        'yolo_detections': yolo_detections,
        'pose_estimations': pose_estimations,
        'gaze_estimations': gaze_estimations,
        'frame_width': frame_width,
        'frame_height': frame_height
    }

    person_map = get_person_map(frame_data)
    time_stamp = 0
    print("Person Map:", person_map)

    arm_anomaly = anomaly_detector.check_suspicious_arm_angle(person_map, time_stamp)
    # print("Arm Anomaly Detected:", arm_anomaly)
    head_anomaly = anomaly_detector.check_looking_away(person_map, time_stamp)
    # print("Head Anomaly Detected:", head_anomaly)
    missing_wrist_anomaly = anomaly_detector.check_missing_wrists(person_map, time_stamp)
    print("Missing Wrist Anomaly Detected:", missing_wrist_anomaly)

if __name__ == "__main__":
    main()
    
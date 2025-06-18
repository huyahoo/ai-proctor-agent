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
        for det in frame_data.get('yolo_detections', [])
        if det['label'] == 'person'
    }

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
    dummy_file = '/Users/ducky/Downloads/proctor-agent/dummy.json'
    with open(dummy_file, 'r') as f:
        frame_data = json.load(f)
    
    gaze_estimations = [{'bbox': [1202.5040283203125, 720.3736572265625, 1428.29150390625, 1026.041748046875], 'gaze_point': [0.34375, 0.5], 'gaze_vector': [0.19035454094409943, 0.9817154407501221], 'inout_score': 0.9985774755477905, 'pid': 1}, {'bbox': [2880.576171875, 928.5159912109375, 3132.960693359375, 1240.22607421875], 'gaze_point': [0.75, 0.515625], 'gaze_vector': [-0.10129646956920624, 0.9948562383651733], 'inout_score': 0.9992438554763794, 'pid': 0}]
    pose_estimations = [{'keypoints': [[1335, 959, 0.99655986], [1371, 913, 0.98335433], [1300, 916, 0.9909903], [1404, 879, 0.895593], [1228, 881, 0.9756579], [1488, 1046, 0.9043865], [1118, 1036, 0.9619447], [1569, 1277, 0.90428007], [1044, 1267, 0.9067887], [1513, 1350, 0.89524233], [1153, 1378, 0.89112836], [1516, 1430, 0.69332856], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]], 'pid': 1}, {'keypoints': [[2980, 1198, 0.96752983], [3034, 1167, 0.9980983], [2955, 1150, 0.9757154], [0, 0, 0.0], [2895, 1068, 0.9083508], [3108, 1177, 0.5522549], [2754, 1130, 0.90443397], [3220, 1332, 0.53227305], [2573, 1340, 0.8817467], [3162, 1408, 0.67375934], [2773, 1419, 0.8958108], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]], 'pid': 0}]
    yolo_detections = [{'bbox': [989.7490234375, 697.2697143554688, 1612.502685546875, 1457.103515625], 'label': 'person', 'confidence': 0.9137017726898193, 'pid': 1}, {'bbox': [2521.2197265625, 921.1282958984375, 3280.34619140625, 1528.0670166015625], 'label': 'person', 'confidence': 0.866401731967926, 'pid': 0}]

    frame_data = {
        'yolo_detections': yolo_detections,
        'pose_estimations': pose_estimations,
        'gaze_estimations': gaze_estimations,
        'frame_width': 4032,
        'frame_height': 3024
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
    
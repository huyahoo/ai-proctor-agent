import os
import cv2
import argparse
from tqdm import tqdm

from ai.anomaly_detector import AnomalyDetector
from core.config import Config
import json

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
                'point': g['gaze_point'],
                'vector': g['gaze_vector'],
                'score': g['inout_score']
            }

    return person_map

def main():
    anomaly_detector = AnomalyDetector(Config())
    dummy_file = '/Users/ducky/Downloads/proctor-agent/dummy.json'
    with open(dummy_file, 'r') as f:
        frame_data = json.load(f)

    person_map = get_person_map(frame_data)
    time_stamp = 0
    # print("Person Map:", person_map)

    arm_anomaly = anomaly_detector.check_suspicious_arm_angle(person_map, time_stamp)
    print("Arm Anomaly Detected:", arm_anomaly)

if __name__ == "__main__":
    main()
    
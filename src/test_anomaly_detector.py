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
    
    gaze_estimations = [{'bbox': [2747.9052734375, 1191.8621826171875, 2980.486572265625, 1501.100341796875], 'gaze_point': [0.703125, 0.609375], 'gaze_vector': [-0.196572944521904, 0.9804891347885132], 'inout_score': 0.9973465204238892, 'pid': 1}, {'bbox': [1141.3909912109375, 998.2349243164062, 1410.483642578125, 1278.93115234375], 'gaze_point': [0.703125, 0.484375], 'gaze_vector': [0.9483205676078796, 0.31731361150741577], 'inout_score': 0.9280946850776672, 'pid': 0}]
    pose_estimations = [{'keypoints': [[1389, 1181, 0.9973079], [1383, 1143, 0.96620274], [1350, 1153, 0.9870641], [0, 0, 0.0], [1243, 1187, 0.9821736], [1445, 1349, 0.8597006], [1064, 1368, 0.909312], [1486, 1563, 0.8154559], [943, 1595, 0.86335737], [1382, 1687, 0.84963006], [1086, 1680, 0.86447686], [1387, 1715, 0.59989476], [1107, 1772, 0.5771159], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]], 'pid': 0}, {'keypoints': [[2835, 1443, 0.98832923], [2887, 1414, 0.99854004], [2809, 1397, 0.9868587], [0, 0, 0.0], [2760, 1321, 0.9575415], [2971, 1452, 0.69686383], [2624, 1407, 0.9403583], [3073, 1621, 0.6775232], [2445, 1611, 0.8947286], [3028, 1701, 0.76224923], [2601, 1704, 0.9189014], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]], 'pid': 1}]
    yolo_detections = [{'bbox': [881.458740234375, 995.4932250976562, 1535.9293212890625, 1817.1707763671875], 'label': 'person', 'confidence': 0.8941870331764221, 'pid': 0}, {'bbox': [2384.013916015625, 1187.575927734375, 3139.385498046875, 1818.9384765625], 'label': 'person', 'confidence': 0.8844392895698547, 'pid': 1}]

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
    print("Head Anomaly Detected:", head_anomaly)

if __name__ == "__main__":
    main()
    
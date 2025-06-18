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
    
    gaze_estimations = [{'bbox': [1051.534912109375, 427.95166015625, 1151.1109619140625, 558.1237182617188], 'gaze_point': [0.609375, 0.625], 'gaze_vector': [0.08420028537511826, 0.996448814868927], 'inout_score': 0.9989210367202759, 'pid': 3}, {'bbox': [734.8357543945312, 342.3213806152344, 812.5286865234375, 446.6407775878906], 'gaze_point': [0.421875, 0.546875], 'gaze_vector': [0.06804251670837402, 0.9976824522018433], 'inout_score': 0.9987419247627258, 'pid': 2}, {'bbox': [1281.304443359375, 329.313720703125, 1361.4693603515625, 423.94091796875], 'gaze_point': [0.5625, 0.609375], 'gaze_vector': [-0.7742968201637268, 0.6328225135803223], 'inout_score': 0.9274418950080872, 'pid': 1}, {'bbox': [367.7159729003906, 396.76495361328125, 458.3671875, 523.9398193359375], 'gaze_point': [0.234375, 0.609375], 'gaze_vector': [0.0857202410697937, 0.9963192939758301], 'inout_score': 0.9791598320007324, 'pid': 0}]
    pose_estimations = [{'keypoints': [[1294, 391, 0.99857455], [1311, 384, 0.986624], [1290, 378, 0.9870455], [1342, 388, 0.9746195], [0, -4, 0.0], [1370, 433, 0.9728371], [1260, 432, 0.9709997], [1401, 497, 0.97289217], [1243, 511, 0.9640115], [1415, 534, 0.96717894], [1305, 534, 0.9660097], [1349, 555, 0.94283086], [1281, 570, 0.90656483], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0]], 'pid': 1}, {'keypoints': [[1100, 537, 0.99069816], [1119, 524, 0.98965025], [1086, 520, 0.99873734], [0, -4, 0.0], [1059, 491, 0.98339444], [1155, 529, 0.93961084], [1005, 520, 0.9751733], [1209, 604, 0.9201762], [930, 590, 0.9558292], [1198, 638, 0.9034582], [972, 620, 0.9565518], [1091, 661, 0.79893064], [1014, 642, 0.82578695], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0]], 'pid': 3}, {'keypoints': [[412, 490, 0.5870105], [426, 476, 0.6513884], [398, 481, 0.47528052], [452, 456, 0.93959916], [373, 464, 0.714725], [493, 511, 0.96829075], [352, 516, 0.99457127], [537, 598, 0.9614979], [324, 595, 0.98057795], [479, 626, 0.9438305], [332, 625, 0.9527373], [471, 678, 0.90092915], [382, 670, 0.9359374], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0]], 'pid': 0}, {'keypoints': [[778, 425, 0.8527327], [789, 413, 0.6983382], [768, 413, 0.8413501], [803, 401, 0.6924112], [749, 402, 0.9081199], [834, 448, 0.9952777], [710, 446, 0.9811418], [847, 530, 0.6921933], [667, 506, 0.96803045], [0, -4, 0.0], [692, 532, 0.9632253], [811, 596, 0.7027991], [733, 598, 0.8813605], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0], [0, -4, 0.0]], 'pid': 2}]
    yolo_detections = [{'bbox': [649.0118408203125, 337.1012878417969, 870.5059204101562, 574.90234375], 'label': 'person', 'confidence': 0.8863111734390259, 'pid': 2}, {'bbox': [908.0943603515625, 426.1865234375, 1233.1512451171875, 683.4783325195312], 'label': 'person', 'confidence': 0.8753053545951843, 'pid': 3}, {'bbox': [1224.1759033203125, 325.8657531738281, 1437.0565185546875, 579.1185302734375], 'label': 'person', 'confidence': 0.8493860363960266, 'pid': 1}, {'bbox': [295.9591979980469, 395.33837890625, 561.0631713867188, 648.8636474609375], 'label': 'person', 'confidence': 0.8054350018501282, 'pid': 0}]

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
    
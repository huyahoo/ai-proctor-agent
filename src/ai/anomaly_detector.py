import math
from core.config import Config
import numpy as np

class AnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.gaze_history = {} # {person_id: [(timestamp, yaw, pitch), ...]}
        self.MAX_HISTORY_LENGTH = 30 # For last few seconds of gaze history

    def _get_person_center(self, bbox):
        """Calculates the center of a person's bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(self, p1_center, p2_center):
        """Calculates Euclidean distance between two points."""
        return math.sqrt((p1_center[0] - p2_center[0])**2 + (p1_center[1] - p2_center[1])**2)

    def detect_anomalies(self, frame_data, current_timestamp):
        """
        Detects anomalies based on YOLO, Pose, and Gaze data.
        frame_data = {
            'yolo_detections': [{'bbox': [x1,y1,x2,y2], 'label': 'person', 'confidence':0.99}, ...],
            'pose_estimations': [{'keypoints': [...], 'bbox': [...], 'person_id': 'unique_id'}, ...],
            'gaze_estimations': [{'bbox': [...], 'head_pose': [nose_x, nose_y, pitch, yaw, roll], 'person_id': 'unique_id'}, ...]
        }
        Assigns temporary unique IDs to persons for tracking across detections.
        """
        anomalies = []

        persons_yolo = [d for d in frame_data['yolo_detections'] if d['label'] == 'person']

        # Simple person ID assignment (can be improved with persistent tracking)
        person_map = {} # Map person_id from pose/gaze to bbox from YOLO
        for i, p_yolo in enumerate(persons_yolo):
            # For simplicity, just use YOLO bbox as primary identifier, map others
            # In a real system, you'd need more robust tracking (e.g., DeepSORT)
            person_id = f"Person_{i}"
            person_map[person_id] = {
                'bbox': p_yolo['bbox'],
                'pose': None,
                'gaze': None
            }
            # Match pose and gaze to YOLO person bbox based on overlap
            for p_pose in frame_data['pose_estimations']:
                if self._check_overlap(p_yolo['bbox'], p_pose['bbox'], threshold=0.5):
                    person_map[person_id]['pose'] = p_pose['keypoints']
            for p_gaze in frame_data['gaze_estimations']:
                 if self._check_overlap(p_yolo['bbox'], p_gaze['bbox'], threshold=0.5):
                    person_map[person_id]['gaze'] = p_gaze['head_pose']

            # Update gaze history
            if person_map[person_id]['gaze']:
                gaze_info = person_map[person_id]['gaze']
                self.gaze_history.setdefault(person_id, []).append((current_timestamp, gaze_info[3], gaze_info[2])) # (timestamp, yaw, pitch)
                self.gaze_history[person_id] = self.gaze_history[person_id][-self.MAX_HISTORY_LENGTH:] # Keep recent history


        # 1. Individual Cheating: Unauthorized Material
        for obj in frame_data['yolo_detections']:
            if obj['label'] not in ['person'] and obj['confidence'] > self.config.OBJECT_MIN_CONFIDENCE:
                # Check if object is close to a person's hand or lap area
                for p_id, p_data in person_map.items():
                    if p_data['pose']:
                        # Simplified check: is object bbox overlapping with person's lower torso/hand area?
                        # This needs specific keypoint analysis for robust detection
                        # For hackathon: check general overlap with person bbox
                        obj_center = self._get_person_center(obj['bbox'])
                        person_center = self._get_person_center(p_data['bbox'])
                        if self._calculate_distance(obj_center, person_center) < self.config.COLLABORATIVE_DISTANCE_THRESHOLD * frame_data['frame_width']: # Use frame width for scaling
                            anomalies.append({
                                'type': 'individual_unauthorized_material',
                                'person_id': p_id,
                                'object_label': obj['label'],
                                'confidence': obj['confidence'],
                                'bbox': obj['bbox'],
                                'timestamp': current_timestamp,
                                'description': f"{p_id} detected with potential unauthorized {obj['label']}."
                            })

        # 2. Collaborative Cheating: Gaze Correlation
        for i, (p1_id, p1_data) in enumerate(person_map.items()):
            if not p1_data['gaze'] or not p1_data['bbox']: continue
            p1_center = self._get_person_center(p1_data['bbox'])

            for j, (p2_id, p2_data) in enumerate(person_map.items()):
                if i >= j or not p2_data['gaze'] or not p2_data['bbox']: continue
                p2_center = self._get_person_center(p2_data['bbox'])

                # Check if persons are close enough for interaction
                if self._calculate_distance(p1_center, p2_center) < self.config.COLLABORATIVE_DISTANCE_THRESHOLD * frame_data['frame_width']:
                    # Simplified gaze correlation: if P1 looks towards P2, and P2 looks towards P1
                    # This is highly simplified and needs calibration
                    p1_yaw = p1_data['gaze'][3] # Yaw angle
                    p2_yaw = p2_data['gaze'][3]

                    # Rough check: if p1 is looking towards p2's general direction
                    # (assuming positive yaw is right, negative is left from center)
                    # This logic needs careful tuning with real data
                    is_p1_looking_at_p2 = (p1_yaw < 0 and p1_center[0] > p2_center[0]) or \
                                          (p1_yaw > 0 and p1_center[0] < p2_center[0])
                    is_p2_looking_at_p1 = (p2_yaw < 0 and p2_center[0] > p1_center[0]) or \
                                          (p2_yaw > 0 and p2_center[0] < p1_center[0])

                    if is_p1_looking_at_p2 and is_p2_looking_at_p1:
                         # More robust check: check gaze history for sustained correlation
                        p1_recent_gaze = [g[1] for g in self.gaze_history.get(p1_id, []) if current_timestamp - g[0] < 2] # Last 2 seconds
                        p2_recent_gaze = [g[1] for g in self.gaze_history.get(p2_id, []) if current_timestamp - g[0] < 2]

                        if len(p1_recent_gaze) >= self.config.GAZE_CONSECUTIVE_FRAMES and \
                           len(p2_recent_gaze) >= self.config.GAZE_CONSECUTIVE_FRAMES:
                            # If their average yaw difference is small and they are looking at each other
                            avg_p1_yaw = np.mean(p1_recent_gaze)
                            avg_p2_yaw = np.mean(p2_recent_gaze)
                            if abs(avg_p1_yaw - avg_p2_yaw) < 20: # Example threshold
                                anomalies.append({
                                    'type': 'collaborative_gaze_correlation',
                                    'person_ids': [p1_id, p2_id],
                                    'confidence': 0.8, # Placeholder
                                    'timestamp': current_timestamp,
                                    'description': f"Correlated gaze between {p1_id} and {p2_id} detected."
                                })

        # 3. Collaborative Cheating: Hand Gestures / Object Passing
        for i, (p1_id, p1_data) in enumerate(person_map.items()):
            if not p1_data['pose'] or not p1_data['bbox']: continue
            p1_center = self._get_person_center(p1_data['bbox'])

            for j, (p2_id, p2_data) in enumerate(person_map.items()):
                if i >= j or not p2_data['pose'] or not p2_data['bbox']: continue
                p2_center = self._get_person_center(p2_data['bbox'])

                if self._calculate_distance(p1_center, p2_center) < self.config.COLLABORATIVE_DISTANCE_THRESHOLD * frame_data['frame_width']:
                    # Check for hands close to each other
                    # For MediaPipe, keypoints 19 and 20 are left and right wrist
                    p1_left_wrist = p1_data['pose'][19] if len(p1_data['pose']) > 19 else None
                    p1_right_wrist = p1_data['pose'][20] if len(p1_data['pose']) > 20 else None
                    p2_left_wrist = p2_data['pose'][19] if len(p2_data['pose']) > 19 else None
                    p2_right_wrist = p2_data['pose'][20] if len(p2_data['pose']) > 20 else None

                    if p1_left_wrist and p2_right_wrist and \
                       self._calculate_distance((p1_left_wrist[0], p1_left_wrist[1]), (p2_right_wrist[0], p2_right_wrist[1])) < 50: # pixel distance
                        anomalies.append({
                            'type': 'collaborative_hand_gesture_proximity',
                            'person_ids': [p1_id, p2_id],
                            'confidence': 0.7,
                            'timestamp': current_timestamp,
                            'description': f"Hands of {p1_id} and {p2_id} in close proximity, possibly exchanging object."
                        })
                    # Add similar checks for other hand combinations
        return anomalies

    def _check_overlap(self, bbox1, bbox2, threshold=0.5):
        """Checks if two bounding boxes overlap by a certain IoU threshold."""
        # Intersection over Union (IoU) calculation
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xA = max(x1_1, x1_2)
        yA = max(y1_1, y1_2)
        xB = min(x2_1, x2_2)
        yB = min(y2_1, y2_2)

        inter_width = xB - xA
        inter_height = yB - yA

        if inter_width < 0 or inter_height < 0:
            return 0.0

        inter_area = inter_width * inter_height
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = float(box1_area + box2_area - inter_area)

        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou >= threshold



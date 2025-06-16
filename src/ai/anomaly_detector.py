import math
import numpy as np
from core.config import Config
from core.utils import get_angle_between_keypoints

class AnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        # Stores recent gaze history for each detected person ID
        # Format: {person_id: [(timestamp, yaw, pitch), ...]}
        self.gaze_history = {}

    def _get_person_center(self, bbox: list) -> tuple:
        """Calculates the center of a person's bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(self, p1_center: tuple, p2_center: tuple) -> float:
        """Calculates Euclidean distance between two points."""
        return math.sqrt((p1_center[0] - p2_center[0])**2 + (p1_center[1] - p2_center[1])**2)

    def _check_overlap(self, bbox1: list, bbox2: list, threshold: float = 0.5) -> bool:
        """Checks if two bounding boxes overlap by a certain IoU threshold."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xA = max(x1_1, x1_2)
        yA = max(y1_1, y1_2)
        xB = min(x2_1, x2_2)
        yB = min(y2_1, y2_2)

        inter_width = xB - xA
        inter_height = yB - yA

        if inter_width < 0 or inter_height < 0:
            return False # No overlap

        inter_area = inter_width * inter_height
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = float(box1_area + box2_area - inter_area)

        if union_area == 0:
            return False

        iou = inter_area / union_area
        return iou >= threshold
    
    def _check_arm_angle(self, keypoints: list, threshold: int = 160) -> bool:
        """
        Checks if the angle between the arms (shoulders to wrists) exceeds a threshold.
        This can indicate suspicious behavior like passing objects.
        """
        arm_angle = get_angle_between_keypoints(keypoints[0], keypoints[1], keypoints[2])  # Left arm
        if arm_angle is None:
            return False
        return arm_angle > threshold

    def detect_anomalies(self, frame_data: dict, current_timestamp: float) -> list:
        """
        Detects anomalies based on YOLO, Pose, and Gaze data for the current frame.
        Args:
            frame_data (dict): Contains raw CV detection data for the frame.
                {
                    'yolo_detections': list of YOLO detection dicts,
                    'pose_estimations': list of pose estimation dicts,
                    'gaze_estimations': list of gaze estimation dicts,
                    'frame_width': int,
                    'frame_height': int
                }
            current_timestamp (float): Timestamp of the current frame in seconds.
        Returns:
            list: A list of detected anomaly event dictionaries.
        """
        anomalies = []

        # Map CV detections to unique person IDs based on YOLO's 'person' detections
        # In a robust system, this would involve a multi-object tracker (e.g., DeepSORT)
        person_map = {} # {person_id: {'bbox': [], 'pose': [], 'gaze': []}}

        persons_yolo = [d for d in frame_data['yolo_detections'] if d['label'] == 'person']

        for i, p_yolo in enumerate(persons_yolo):
            person_id = f"Person_{i}" # Simple ID for this frame (not persistent across frames yet)
            person_map[person_id] = {'bbox': p_yolo['bbox'], 'pose': None, 'gaze': None}

            # Attempt to match pose and gaze detections to this YOLO person bbox
            # for p_pose in frame_data['pose_estimations']:
            #     if p_pose['bbox'] and self._check_overlap(p_yolo['bbox'], p_pose['bbox'], threshold=0.6): # Higher overlap needed
            #         person_map[person_id]['pose'] = p_pose['keypoints']
            for p_gaze in frame_data['gaze_estimations']:
                if self._check_overlap(p_yolo['bbox'], p_gaze['bbox'], threshold=0.6):
                    person_map[person_id]['gaze'] = p_gaze['head_pose']

            # Update gaze history for this person
            if person_map[person_id]['gaze'] and len(person_map[person_id]['gaze']) >= 4:
                # gaze_info format: [nose_x, nose_y, pitch_deg, yaw_deg, roll_deg]
                gaze_info = person_map[person_id]['gaze']
                self.gaze_history.setdefault(person_id, []).append((current_timestamp, gaze_info[3], gaze_info[2])) # (timestamp, yaw_deg, pitch_deg)
                # Keep only history within MAX_HISTORY_LENGTH_SECONDS
                self.gaze_history[person_id] = [g for g in self.gaze_history[person_id]
                                                if current_timestamp - g[0] < self.config.MAX_HISTORY_LENGTH_SECONDS]

        # --- Cheating Detection Logic ---

        # 1. Individual Cheating: Unauthorized Material
        for obj in frame_data['yolo_detections']:
            if obj['label'] == 'person': continue # Skip persons

            # Check if object is close to a person's hand or lap area
            for p_id, p_data in person_map.items():
                if p_data['bbox'] and self._check_overlap(obj['bbox'], p_data['bbox'], threshold=0.1): # Some overlap
                    # Heuristic: check if object is in the lower half of the person's bbox (lap/desk area)
                    person_y_center = (p_data['bbox'][1] + p_data['bbox'][3]) / 2
                    obj_y_center = (obj['bbox'][1] + obj['bbox'][3]) / 2

                    if obj_y_center > person_y_center: # Object is roughly in the lower half of person's bounding box
                         anomalies.append({
                            'type': 'individual_unauthorized_material',
                            'person_ids': [p_id],
                            'object_label': obj['label'],
                            'confidence': obj['confidence'],
                            'bbox': obj['bbox'],
                            'timestamp': current_timestamp,
                            'description': f"{p_id} detected with potential unauthorized '{obj['label']}' in lap/desk area."
                        })

        # 2. Individual Cheating: Suspicious Gaze (looking away from exam/screen)
        for p_id, p_data in person_map.items():
            if p_data['gaze']:
                _, _, pitch, yaw, _ = p_data['gaze']

                # Check for extreme yaw (looking far left/right) or extreme pitch (looking up/down)
                # These thresholds need tuning!
                if abs(yaw) > 45 or abs(pitch) > 30: # Example: >45 deg side, >30 deg up/down
                     anomalies.append({
                        'type': 'individual_suspicious_gaze',
                        'person_ids': [p_id],
                        'confidence': 0.7,
                        'timestamp': current_timestamp,
                        'description': f"{p_id} exhibiting extreme gaze (yaw: {yaw:.1f}deg, pitch: {pitch:.1f}deg), possibly looking away from exam."
                    })

        # 3. Collaborative Cheating: Gaze Correlation (looking at each other's paper/screen)
        for i, (p1_id, p1_data) in enumerate(person_map.items()):
            if not p1_data['gaze'] or not p1_data['bbox']: continue
            p1_center = self._get_person_center(p1_data['bbox'])

            for j, (p2_id, p2_data) in enumerate(person_map.items()):
                if i >= j or not p2_data['gaze'] or not p2_data['bbox']: continue
                p2_center = self._get_person_center(p2_data['bbox'])

                # Check if persons are close enough for interaction (normalized by frame width)
                normalized_distance = self._calculate_distance(p1_center, p2_center) / frame_data['frame_width']
                if normalized_distance < self.config.COLLABORATIVE_DISTANCE_THRESHOLD:

                    # Get recent gaze data for both
                    p1_recent_gaze = [g for g in self.gaze_history.get(p1_id, [])]
                    p2_recent_gaze = [g for g in self.gaze_history.get(p2_id, [])]

                    # Check for sustained, correlated gaze over few frames
                    if len(p1_recent_gaze) >= self.config.GAZE_CONSECUTIVE_FRAMES and \
                       len(p2_recent_gaze) >= self.config.GAZE_CONSECUTIVE_FRAMES:

                        # Simplified: check if both are looking towards each other's general direction
                        # A more robust check would involve calculating a vector from head to other person's paper bbox

                        # Get average yaw/pitch over the last few frames
                        avg_p1_yaw = np.mean([g[1] for g in p1_recent_gaze])
                        avg_p1_pitch = np.mean([g[2] for g in p1_recent_gaze])
                        avg_p2_yaw = np.mean([g[1] for g in p2_recent_gaze])
                        avg_p2_pitch = np.mean([g[2] for g in p2_recent_gaze])

                        # Heuristic: if p1 is left of p2 and p1 looks right (positive yaw)
                        # AND p2 is right of p1 and p2 looks left (negative yaw)
                        p1_looks_at_p2_approx = (p1_center[0] < p2_center[0] and avg_p1_yaw > self.config.GAZE_YAW_THRESHOLD) or \
                                                (p1_center[0] > p2_center[0] and avg_p1_yaw < -self.config.GAZE_YAW_THRESHOLD)
                        p2_looks_at_p1_approx = (p2_center[0] < p1_center[0] and avg_p2_yaw > self.config.GAZE_YAW_THRESHOLD) or \
                                                (p2_center[0] > p1_center[0] and avg_p2_yaw < -self.config.GAZE_YAW_THRESHOLD)

                        if p1_looks_at_p2_approx and p2_looks_at_p1_approx:
                             anomalies.append({
                                'type': 'collaborative_gaze_correlation',
                                'person_ids': [p1_id, p2_id],
                                'confidence': 0.8,
                                'timestamp': current_timestamp,
                                'description': f"Sustained, correlated side-gaze between {p1_id} and {p2_id} detected. Both students looking towards each other's exam area."
                            })

        # 4. Collaborative Cheating: Hand Gestures / Object Passing
        for i, (p1_id, p1_data) in enumerate(person_map.items()):
            if not p1_data['pose'] or not p1_data['bbox']: continue
            p1_pose_keypoints = p1_data['pose']

            for j, (p2_id, p2_data) in enumerate(person_map.items()):
                if i >= j or not p2_data['pose'] or not p2_data['bbox']: continue
                p2_pose_keypoints = p2_data['pose']

                normalized_distance = self._calculate_distance(self._get_person_center(p1_data['bbox']), self._get_person_center(p2_data['bbox'])) / frame_data['frame_width']
                if normalized_distance < self.config.COLLABORATIVE_DISTANCE_THRESHOLD:

                    # MediaPipe keypoints for wrists: 15 (left wrist), 16 (right wrist)
                    # Use a small pixel threshold for hand proximity
                    hand_proximity_pixel_threshold = 0.05 * frame_data['frame_width'] # 5% of frame width

                    # Helper to check if two keypoints are close and confident
                    def are_keypoints_close(kp1, kp2):
                        if kp1 and kp2 and \
                           (len(kp1) < 3 or kp1[2] > self.config.POSE_MIN_CONFIDENCE) and \
                           (len(kp2) < 3 or kp2[2] > self.config.POSE_MIN_CONFIDENCE): # Check visibility confidence if present
                            return self._calculate_distance((kp1[0], kp1[1]), (kp2[0], kp2[1])) < hand_proximity_pixel_threshold
                        return False

                    # Check various hand combinations for close proximity
                    if (p1_pose_keypoints and p2_pose_keypoints):
                        if (are_keypoints_close(p1_pose_keypoints[15], p2_pose_keypoints[16]) or # P1 Left Wrist, P2 Right Wrist
                            are_keypoints_close(p1_pose_keypoints[16], p2_pose_keypoints[15]) or # P1 Right Wrist, P2 Left Wrist
                            are_keypoints_close(p1_pose_keypoints[15], p2_pose_keypoints[15]) or # P1 Left Wrist, P2 Left Wrist (e.g., passing over desk)
                            are_keypoints_close(p1_pose_keypoints[16], p2_pose_keypoints[16])): # P1 Right Wrist, P2 Right Wrist

                            anomalies.append({
                                'type': 'collaborative_hand_gesture_proximity',
                                'person_ids': [p1_id, p2_id],
                                'confidence': 0.7,
                                'timestamp': current_timestamp,
                                'description': f"Hands of {p1_id} and {p2_id} in close proximity, possibly exchanging object or signaling."
                            })
        
        # 5. Suspicious Arm Angles (e.g., passing objects)
        for p_id, p_data in frame_data['pose_estimations']["keypoints"].items():
            right_arm_kpts_info = [2, 3, 4] # Right shoulder, elbow, wrist
            left_arm_kpts_info = [5, 6, 7]  # Left shoulder, elbow, wrist
            right_arm = []
            left_arm = []
            for kpt_data in p_data:
                if kpt_data[-1] in right_arm_kpts_info:
                    right_arm.append([kpt_data[0], kpt_data[1]])
                if kpt_data[-1] in left_arm_kpts_info:
                    left_arm.append([kpt_data[0], kpt_data[1]])

            if len(right_arm) < 3 or len(left_arm) < 3:
                continue
            # Check if arms are at suspicious angles (e.g., passing objects)
            if self._check_arm_angle(right_arm) or self._check_arm_angle(left_arm):
                anomalies.append({
                    'type': 'suspicious_arm_angle',
                    'person_ids': [p_id],
                    'confidence': 0.6,
                    'timestamp': current_timestamp,
                    'description': f"{p_id} detected with suspicious arm angle, possibly passing an object."
                })
                print("Suspicious arm angle detected for", anomalies)

        return anomalies


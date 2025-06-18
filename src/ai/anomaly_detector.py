import math
import numpy as np
from core.config import Config

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
    
    def _calculate_angle(self, arm_keypoints: list) -> float:
        shoulder, elbow, wrist = arm_keypoints
        
        # Vector shoulder → elbow và wrist → elbow
        vec1 = np.array(shoulder) - np.array(elbow)
        vec2 = np.array(wrist) - np.array(elbow)

        # Tính cos của góc giữa 2 vector
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _is_point_in_bbox(self, point: list, bbox: list) -> bool:
        """
        Check if a point (x, y) is inside a bounding box defined by [x1, y1, x2, y2].
        """
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    # Passing material: Suspicious Arm Angles
    def check_suspicious_arm_angle(self, person_map: dict, timestamp: float) -> dict | None:
        """
        Check if this person’s left or right arm is nearly straight (>160°).
        Returns one anomaly dict or None.
        """
        anomaly = []
        for pid, data in person_map.items():
            pose_data = data.get('pose')
            if not pose_data:
                return None

            # keypoint indices: (shoulder, elbow, wrist)
            RIGHT_IDS = [6, 8, 10]
            LEFT_IDS  = [5, 7,  9]

            # Remove confidence score from keypoints
            try:
                right = [pose_data[i][:2] for i in RIGHT_IDS]
                left  = [pose_data[i][:2] for i in LEFT_IDS]
            except (TypeError, IndexError):
                return None  # missing or malformed keypoints

            print(f"Person {pid}: Right arm: {right}, Left arm: {left}")
            
            # compute elbow angles
            right_angle = self._calculate_angle(right)
            left_angle  = self._calculate_angle(left)
            if right_angle is None or left_angle is None:
                return None

            print(f"Person {pid}: Right arm angle: {right_angle:.1f}°, Left arm angle: {left_angle:.1f}°")
            # if either arm is nearly straight, flag it
            if right_angle > 160 or left_angle > 160:
                anomaly.append({
                    'type': 'suspicious_arm_angle',
                    'person_ids': [pid],
                    'timestamp': timestamp,
                    'reason': (
                        f"PID {pid} arm angles R={right_angle:.1f}°, "
                        f"L={left_angle:.1f}° — possibly passing an object."
                    )
                })

        return anomaly if anomaly else None
    
    def check_looking_away(self, person_map: dict, timestamp: float) -> list:
        """
        Check if people are looking at each other.
        Returns a list of anomaly dictionaries.
        """
        anomalies = []
        
        # Check each pair of people
        for pid1, data1 in person_map.items():
            if not data1.get('gaze') or not data1.get('bbox'):
                continue
                
            gaze1 = data1['gaze']
            bbox1 = data1['bbox']
            
            # Get gaze point for person 1
            gaze_point1 = gaze1['point']
            gaze_score1 = gaze1['score']
            
            # Check against all other people
            for pid2, data2 in person_map.items():
                if pid1 == pid2 or not data2.get('bbox'):
                    continue
                    
                # Check if person 1's gaze point is inside person 2's bbox
                bbox2 = data2['bbox']
                if gaze_score1 > 0.7 and self._is_point_in_bbox(gaze_point1, bbox2):
                    # Person 1 is looking at person 2
                    # Check if person 2 is also looking at person 1
                    if data2.get('gaze'):
                        gaze2 = data2['gaze']
                        gaze_point2 = gaze2['point']
                        gaze_score2 = gaze2['score']
                        
                        if gaze_score2 > 0.7 and self._is_point_in_bbox(gaze_point2, bbox1):
                            # Both people are looking at each other
                            anomalies.append({
                                'type': 'mutual_gaze',
                                'person_ids': [pid1, pid2],
                                # 'confidence': min(gaze_score1, gaze_score2),
                                'timestamp': timestamp,
                                'description': f"People {pid1} and {pid2} are looking at each other"
                            })
                        else:
                            # Only person 1 is looking at person 2
                            anomalies.append({
                                'type': 'one_way_gaze',
                                'person_ids': [pid1],
                                # 'confidence': gaze_score1,
                                'timestamp': timestamp,
                                'description': f"Person {pid1} is looking at person {pid2}"
                            })

        return anomalies if anomalies else None
    
    def check_missing_wrists(self, person_map: dict, timestamp: float) -> list:
        """
        Check if any person has missing or low confidence wrist keypoints.
        
        Args:
            person_map (dict): Dictionary containing person detections and their keypoints
            timestamp (float): Current frame timestamp
        
        Returns:
            list: List of anomaly dictionaries for people with missing wrists
        """
        anomalies = []
        # Wrist indices in pose keypoints
        LEFT_WRIST_IDX = 9   # Index for left wrist
        RIGHT_WRIST_IDX = 10  # Index for right wrist
        CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence threshold
        
        for pid, data in person_map.items():
            if not data.get('pose'):
                continue
                
            pose_data = data['pose']
            missing_wrists = []
            
            # Check left wrist
            try:
                left_wrist = pose_data[LEFT_WRIST_IDX]
                if left_wrist[2] < CONFIDENCE_THRESHOLD or (left_wrist[0] == 0 and left_wrist[1] == 0):
                    missing_wrists.append('left')
            except (IndexError, TypeError):
                missing_wrists.append('left')
                
            # Check right wrist
            try:
                right_wrist = pose_data[RIGHT_WRIST_IDX]
                if right_wrist[2] < CONFIDENCE_THRESHOLD or (right_wrist[0] == 0 and right_wrist[1] == 0):
                    missing_wrists.append('right')
            except (IndexError, TypeError):
                missing_wrists.append('right')
                
            # Create anomaly if any wrists are missing
            if missing_wrists:
                anomalies.append({
                    'type': 'missing_wrists',
                    'person_ids': [pid],
                    'timestamp': timestamp,
                    'missing': missing_wrists,
                    'description': f"Person {pid} has high probability of using the phone under the table."
                })
        
        return anomalies if anomalies else None
    
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

        # Map CV detections to unique person IDs
        # In a robust system, this would involve a multi-object tracker (e.g., DeepSORT)
            # Build a map of person_id → their combined data

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

        # print(person_map)
        # person_map = {id: {'bbox': [x1,y1,x2,y2], 'pose': [[x,y,conf],...], 'gaze': {'bbox': [x1,y1,x2,y2], 'point': [x,y], 'vector': [x,y], 'score': 0.0}}}

        # --- Cheating Detection Logic ---
        anomalies = []

        # 1. Passing material: Suspicious Arm Angles
        arm_anomaly = self.check_suspicious_arm_angle(person_map, current_timestamp)

        if arm_anomaly:
            anomalies.extend(arm_anomaly)

        # 2. Looking away: Mutual gaze detection
        gaze_anomalies = self.check_looking_away(person_map, current_timestamp)
        if gaze_anomalies:
            anomalies.extend(gaze_anomalies)
            
        # # 1. Individual Cheating: Unauthorized Material
        # for obj in frame_data['yolo_detections']:
        #     if obj['label'] == 'person': continue # Skip persons

        #     # Check if object is close to a person's hand or lap area
        #     for p_id, p_data in person_map.items():
        #         if p_data['bbox'] and self._check_overlap(obj['bbox'], p_data['bbox'], threshold=0.1): # Some overlap
        #             # Heuristic: check if object is in the lower half of the person's bbox (lap/desk area)
        #             person_y_center = (p_data['bbox'][1] + p_data['bbox'][3]) / 2
        #             obj_y_center = (obj['bbox'][1] + obj['bbox'][3]) / 2

        #             if obj_y_center > person_y_center: # Object is roughly in the lower half of person's bounding box
        #                  anomalies.append({
        #                     'type': 'individual_unauthorized_material',
        #                     'person_ids': [p_id],
        #                     'object_label': obj['label'],
        #                     'confidence': obj['confidence'],
        #                     'bbox': obj['bbox'],
        #                     'timestamp': current_timestamp,
        #                     'description': f"{p_id} detected with potential unauthorized '{obj['label']}' in lap/desk area."
        #                 })

        # # 2. Individual Cheating: Suspicious Gaze (looking away from exam/screen)
        # for p_id, p_data in person_map.items():
        #     if p_data['gaze']:
        #         _, _, pitch, yaw, _ = p_data['gaze']

        #         # Check for extreme yaw (looking far left/right) or extreme pitch (looking up/down)
        #         # These thresholds need tuning!
        #         if abs(yaw) > 45 or abs(pitch) > 30: # Example: >45 deg side, >30 deg up/down
        #              anomalies.append({
        #                 'type': 'individual_suspicious_gaze',
        #                 'person_ids': [p_id],
        #                 'confidence': 0.7,
        #                 'timestamp': current_timestamp,
        #                 'description': f"{p_id} exhibiting extreme gaze (yaw: {yaw:.1f}deg, pitch: {pitch:.1f}deg), possibly looking away from exam."
        #             })

        # # 3. Collaborative Cheating: Gaze Correlation (looking at each other's paper/screen)
        # for i, (p1_id, p1_data) in enumerate(person_map.items()):
        #     if not p1_data['gaze'] or not p1_data['bbox']: continue
        #     p1_center = self._get_person_center(p1_data['bbox'])

        #     for j, (p2_id, p2_data) in enumerate(person_map.items()):
        #         if i >= j or not p2_data['gaze'] or not p2_data['bbox']: continue
        #         p2_center = self._get_person_center(p2_data['bbox'])

        #         # Check if persons are close enough for interaction (normalized by frame width)
        #         normalized_distance = self._calculate_distance(p1_center, p2_center) / frame_data['frame_width']
        #         if normalized_distance < self.config.COLLABORATIVE_DISTANCE_THRESHOLD:

        #             # Get recent gaze data for both
        #             p1_recent_gaze = [g for g in self.gaze_history.get(p1_id, [])]
        #             p2_recent_gaze = [g for g in self.gaze_history.get(p2_id, [])]

        #             # Check for sustained, correlated gaze over few frames
        #             if len(p1_recent_gaze) >= self.config.GAZE_CONSECUTIVE_FRAMES and \
        #                len(p2_recent_gaze) >= self.config.GAZE_CONSECUTIVE_FRAMES:

        #                 # Simplified: check if both are looking towards each other's general direction
        #                 # A more robust check would involve calculating a vector from head to other person's paper bbox

        #                 # Get average yaw/pitch over the last few frames
        #                 avg_p1_yaw = np.mean([g[1] for g in p1_recent_gaze])
        #                 avg_p1_pitch = np.mean([g[2] for g in p1_recent_gaze])
        #                 avg_p2_yaw = np.mean([g[1] for g in p2_recent_gaze])
        #                 avg_p2_pitch = np.mean([g[2] for g in p2_recent_gaze])

        #                 # Heuristic: if p1 is left of p2 and p1 looks right (positive yaw)
        #                 # AND p2 is right of p1 and p2 looks left (negative yaw)
        #                 p1_looks_at_p2_approx = (p1_center[0] < p2_center[0] and avg_p1_yaw > self.config.GAZE_YAW_THRESHOLD) or \
        #                                         (p1_center[0] > p2_center[0] and avg_p1_yaw < -self.config.GAZE_YAW_THRESHOLD)
        #                 p2_looks_at_p1_approx = (p2_center[0] < p1_center[0] and avg_p2_yaw > self.config.GAZE_YAW_THRESHOLD) or \
        #                                         (p2_center[0] > p1_center[0] and avg_p2_yaw < -self.config.GAZE_YAW_THRESHOLD)

        #                 if p1_looks_at_p2_approx and p2_looks_at_p1_approx:
        #                      anomalies.append({
        #                         'type': 'collaborative_gaze_correlation',
        #                         'person_ids': [p1_id, p2_id],
        #                         'confidence': 0.8,
        #                         'timestamp': current_timestamp,
        #                         'description': f"Sustained, correlated side-gaze between {p1_id} and {p2_id} detected. Both students looking towards each other's exam area."
        #                     })

        # # 4. Collaborative Cheating: Hand Gestures / Object Passing
        # for i, (p1_id, p1_data) in enumerate(person_map.items()):
        #     if not p1_data['pose'] or not p1_data['bbox']: continue
        #     p1_pose_keypoints = p1_data['pose']

        #     for j, (p2_id, p2_data) in enumerate(person_map.items()):
        #         if i >= j or not p2_data['pose'] or not p2_data['bbox']: continue
        #         p2_pose_keypoints = p2_data['pose']

        #         normalized_distance = self._calculate_distance(self._get_person_center(p1_data['bbox']), self._get_person_center(p2_data['bbox'])) / frame_data['frame_width']
        #         if normalized_distance < self.config.COLLABORATIVE_DISTANCE_THRESHOLD:

        #             # MediaPipe keypoints for wrists: 15 (left wrist), 16 (right wrist)
        #             # Use a small pixel threshold for hand proximity
        #             hand_proximity_pixel_threshold = 0.05 * frame_data['frame_width'] # 5% of frame width

        #             # Helper to check if two keypoints are close and confident
        #             def are_keypoints_close(kp1, kp2):
        #                 if kp1 and kp2 and \
        #                    (len(kp1) < 3 or kp1[2] > self.config.POSE_MIN_CONFIDENCE) and \
        #                    (len(kp2) < 3 or kp2[2] > self.config.POSE_MIN_CONFIDENCE): # Check visibility confidence if present
        #                     return self._calculate_distance((kp1[0], kp1[1]), (kp2[0], kp2[1])) < hand_proximity_pixel_threshold
        #                 return False

        #             # Check various hand combinations for close proximity
        #             if (p1_pose_keypoints and p2_pose_keypoints):
        #                 if (are_keypoints_close(p1_pose_keypoints[15], p2_pose_keypoints[16]) or # P1 Left Wrist, P2 Right Wrist
        #                     are_keypoints_close(p1_pose_keypoints[16], p2_pose_keypoints[15]) or # P1 Right Wrist, P2 Left Wrist
        #                     are_keypoints_close(p1_pose_keypoints[15], p2_pose_keypoints[15]) or # P1 Left Wrist, P2 Left Wrist (e.g., passing over desk)
        #                     are_keypoints_close(p1_pose_keypoints[16], p2_pose_keypoints[16])): # P1 Right Wrist, P2 Right Wrist

        #                     anomalies.append({
        #                         'type': 'collaborative_hand_gesture_proximity',
        #                         'person_ids': [p1_id, p2_id],
        #                         'confidence': 0.7,
        #                         'timestamp': current_timestamp,
        #                         'description': f"Hands of {p1_id} and {p2_id} in close proximity, possibly exchanging object or signaling."
        #                     })

        # # 5. Passing material: Suspicious Arm Angles
        # for p_id, p_data in frame_data['pose_estimations'].items():
        #     right_arm_kpts_info = [6, 8, 10] # Right shoulder, elbow, wrist
        #     left_arm_kpts_info = [5, 7, 9]  # Left shoulder, elbow, wrist

        #     right_arm = [p_data[i] for i in right_arm_kpts_info]
        #     left_arm = [p_data[i] for i in left_arm_kpts_info]

        #     if len(right_arm) < 3 or len(left_arm) < 3:
        #         continue

        #     right_arm_angle = self._calculate_angle(right_arm)
        #     left_arm_angle = self._calculate_angle(left_arm)
        #     # print(f"Person {p_id}: Right arm angle: {right_arm_angle}, Left arm angle: {left_arm_angle}")

        #     # Check if arms are at suspicious angles (e.g., passing objects)
        #     if right_arm_angle is None or left_arm_angle is None:
        #         continue
        #     if right_arm_angle > 160 or left_arm_angle > 160:
        #         anomalies.append({
        #             'type': 'suspicious_arm_angle',
        #             'person_ids': [p_id],
        #             'timestamp': current_timestamp,
        #             'description': f"{p_id} detected with suspicious arm angle, possibly passing an object."
        #         })
        #         print(f"Anomaly detected for {p_id}: suspicious arm angle.")

        return anomalies


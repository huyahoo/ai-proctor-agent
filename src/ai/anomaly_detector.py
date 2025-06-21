import math
import numpy as np
from core.config import Config
from core.logger import logger

class AnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        # Stores recent gaze history for each detected person ID
        # Format: {person_id: [(timestamp, yaw, pitch), ...]}
        self.gaze_history = {}
        self.temp_anomalies_history = []

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
        if not bbox: return False
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    # Passing material: Suspicious Arm Angles
    def check_suspicious_arm_angle(self, person_map: dict, timestamp: float) -> list | None:
        """
        Check if this person's left or right arm is nearly straight (>160°).
        Returns a list of anomaly dicts or None.
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

            # logger.info(f"Person {pid}: Right arm: {right}, Left arm: {left}")
            
            # compute elbow angles
            right_angle = self._calculate_angle(right)
            left_angle  = self._calculate_angle(left)
            if right_angle is None or left_angle is None:
                return None

            # logger.info(f"Person {pid}: Right arm angle: {right_angle:.1f}°, Left arm angle: {left_angle:.1f}°")
            # if either arm is nearly straight, flag it
            if right_angle > 170 or left_angle > 170:
                anomaly.append({
                    'type': 'suspicious_arm_angle',
                    'person_ids': [pid],
                    'timestamp': timestamp,
                    'description': (
                        f"Student ID {pid}'s who has visual feature desbribe as {person_map[pid]['person_feature']} "
                        f"has suspicious arm angles right={right_angle:.1f}°, left={left_angle:.1f}° "
                        f"— possibly holding or passing an unauthorized object."
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
                                'description': f"Student ID {pid1} who has visual feature desbribe as {person_map[pid1]['person_feature']} "
                                f"and student ID {pid2} who has visual feature desbribe as {person_map[pid2]['person_feature']} "
                                f"are looking at each other - possibly collaborating cheating actions."
                            })
                        else:
                            # Only person 1 is looking at person 2
                            anomalies.append({
                                'type': 'one_way_gaze',
                                'person_ids': [pid1],
                                # 'confidence': gaze_score1,
                                'timestamp': timestamp,
                                'description': f"Student ID {pid1} who has visual feature desbribe as {person_map[pid1]['person_feature']} "
                                f"is looking at student ID {pid2} who has visual feature desbribe as {person_map[pid2]['person_feature']} "
                                f"— possibly Student ID {pid1} looking at exam paper of Student ID {pid2}."
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
        anomaly_pid = []
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
                anomaly_pid.append(pid)
        
        return anomaly_pid if anomaly_pid else None
    
    def check_gaze_on_exam_paper(self, pid: int,  person_map: dict, timestamp: float) -> list:
        anomaly_person_data = person_map[pid]
        gaze_point = anomaly_person_data["gaze"]["point"]
        exam_paper_bbox = anomaly_person_data["exam_paper"]

        # Check if student is looking at their exam paper
        if not self._is_point_in_bbox(gaze_point, exam_paper_bbox):
            # Create anomaly for looking away from paper
            return {
                'type': 'suspicious_under_table',
                'person_ids': [pid],
                'timestamp': timestamp,
                'description': f"Student ID {pid} who has visual feature desbribe as {person_map[pid]['person_feature']} "
                f"is not looking at his/her exam paper and do something under table - possibly use unauthorized material such as phone, cheat sheet, smartwatch, etc."
            }
        
        return None
    
    def check_suspicious_under_table(self, person_map: dict, timestamp: float) -> list:
        wrist_anomalies = []
        wrist_anomaly_pid = self.check_missing_wrists(person_map, timestamp)
        if wrist_anomaly_pid:
                for pid in wrist_anomaly_pid:
                    anomaly = self.check_gaze_on_exam_paper(pid, person_map, timestamp)
                    if anomaly: wrist_anomalies.append(anomaly)
        
        return wrist_anomalies

    def check_looking_others_paper(self, person_map: dict, timestamp: float) -> list:
        anomalies = []
        for pid in person_map:
            is_not_looking_at_own_paper = self.check_gaze_on_exam_paper(pid, person_map, timestamp)
            pid_gaze_point = person_map[pid]["gaze"]["point"]
            if is_not_looking_at_own_paper:
                for student_id in person_map:
                    if student_id == pid: continue
                    exam_paper_bbox = person_map[student_id]["exam_paper"]
                    if self._is_point_in_bbox(pid_gaze_point, exam_paper_bbox):
                        anomalies.append({
                            'type': 'copying_others_answer',
                            'person_ids': [pid],
                            'timestamp': timestamp,
                            'description': f"Student ID {pid} who has visual feature desbribe as {person_map[pid]['person_feature']} "
                            f"is looking at student ID {student_id} who has visual feature desbribe as {person_map[student_id]['person_feature']} "
                            f"exam paper and copying student ID {student_id}'s answer - possibly cheating."
                        })
        return anomalies if anomalies else None

    def _add_anomaly_if_not_duplicate(self, new_anomaly: dict, anomalies: list):
        is_duplicate = False
        for history_anomaly in self.temp_anomalies_history:
            # Check for malformed entries in history (which could be lists)
            if not isinstance(history_anomaly, dict):
                continue
            
            if (history_anomaly.get('type') == new_anomaly['type'] and
                    history_anomaly.get('person_ids') == new_anomaly['person_ids'] and
                    abs(history_anomaly.get('timestamp', float('inf')) - new_anomaly['timestamp']) < 2):
                
                logger.debug(f"Skipping adding {new_anomaly['type']} anomaly for {new_anomaly['person_ids']} to history at {new_anomaly['timestamp']:.2f}s (near previous at {history_anomaly['timestamp']:.2f}s)")
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.temp_anomalies_history.append(new_anomaly)
            logger.debug(f"Appending {new_anomaly['type']} anomaly for {new_anomaly['person_ids']} to history at {new_anomaly['timestamp']:.2f}s")
            anomalies.append(new_anomaly)
    
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

        person_map = {}  # { pid: {'bbox': ..., 'pose': ..., 'gaze': ..., 'exam_paper': ...} }

        # Build PID-based map from YOLO
        person_map = {
            det['pid']: {'bbox': det['bbox'], 'pose': None, 'gaze': None}
            for det in frame_data.get("yolo_detections", [])
            if det['label'] == 'person'
        }

        # Associate exam paper to pid
        for det in frame_data["yolo_detections"]:
            if det["label"] == "paper" and det["pid"] != -1: 
                person_map[det["pid"]]["exam_paper"] = det["bbox"]
        # Handle exam paper not detected case
        for pid in person_map:
            if "exam_paper" not in person_map[pid]:
                person_map[pid]["exam_paper"] = []

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
        
        # Associate person_feature to pid
        for det in frame_data["yolo_detections"]:
            if det["label"] == "person" and det["pid"] != -1: 
                person_map[det["pid"]]["person_feature"] = det["person_feature"]

        # logger.debug(f"Person map: {person_map}")
        # logger.debug(f"frame_data: {frame_data}")

        # --- Cheating Detection Logic ---
        anomalies = []


        # 1. Passing material: Suspicious Arm Angles
        arm_anomalies = self.check_suspicious_arm_angle(person_map, current_timestamp)

        if arm_anomalies:
            for new_anomaly in arm_anomalies:
                self._add_anomaly_if_not_duplicate(new_anomaly, anomalies)


        # 2. Looking away: Mutual gaze detection
        gaze_anomalies = self.check_looking_away(person_map, current_timestamp)
        if gaze_anomalies:
            for new_anomaly in gaze_anomalies:
                self._add_anomaly_if_not_duplicate(new_anomaly, anomalies)

    
        # 3. Missing wrists: Check for missing or low confidence wrist keypoints
        under_table_anomalies = self.check_suspicious_under_table(person_map, current_timestamp)
        if under_table_anomalies:
            for new_anomaly in under_table_anomalies:
                self._add_anomaly_if_not_duplicate(new_anomaly, anomalies)

        # 4. Looking at others exam paper for copying answer
        copying_answer_anomalies = self.check_looking_others_paper(person_map, current_timestamp)
        if copying_answer_anomalies:
            for new_anomaly in copying_answer_anomalies:
                self._add_anomaly_if_not_duplicate(new_anomaly, anomalies)

        return anomalies


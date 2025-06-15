import mediapipe as mp
import cv2
from cv.base_detector import BaseDetector
from core.config import Config

class PoseEstimator(BaseDetector):
    def __init__(self, config: Config):
        super().__init__(config)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=self.config.POSE_MIN_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        """
        Estimates poses in a frame using MediaPipe BlazePose.
        Returns a list of dictionaries, one per detected person:
        [{'keypoints': [[x, y, visibility], ...], 'bbox': [x1, y1, x2, y2]}, ...]
        Note: MediaPipe detects one pose per image by default. For multi-person,
        you'd typically crop detected persons from YOLO and run pose for each,
        or use a multi-person pose model like OpenPose directly (more complex setup).
        For simplicity in hackathon, we'll assume a main person or simplify multi-person.
        If YOLO provides person bboxes, we can iterate.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        poses_data = []
        if results.pose_landmarks:
            h, w, c = frame.shape
            keypoints = []
            x_coords = []
            y_coords = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                keypoints.append([cx, cy, lm.visibility])
                x_coords.append(cx)
                y_coords.append(cy)

            # Simple bbox from keypoints
            if x_coords and y_coords:
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = [0,0,0,0] # Fallback

            poses_data.append({
                'keypoints': keypoints,
                'bbox': bbox # Bounding box based on keypoints
            })
        return poses_data

    def draw_landmarks(self, frame, results):
        """Draws pose landmarks on the frame (for visualization)."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame



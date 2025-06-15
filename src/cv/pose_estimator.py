import mediapipe as mp
import cv2
import numpy as np
from cv.base_detector import BaseDetector
from core.config import Config
from core.utils import draw_keypoints
from core.constants import POSE_CONNECTIONS_INDICES # Use the pre-converted indices
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'OpenPoseNet'))

from util.decode_pose import decode_pose
from util.openpose_net import OpenPoseNet

class PoseEstimator(BaseDetector):
    # def __init__(self, config: Config):
    #     super().__init__(config)
    #     self.mp_pose = mp.solutions.pose
    #     self.pose = self.mp_pose.Pose(
    #         min_detection_confidence=self.config.POSE_MIN_CONFIDENCE,
    #         min_tracking_confidence=self.config.POSE_MIN_CONFIDENCE
    #     )
    #     self.mp_drawing = mp.solutions.drawing_utils

    def __init__(self):
        self.net = OpenPoseNet()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        weight_file_path = os.path.join(self.current_dir, 'weights', 'pose_model_scratch.pth')
        # Update torch.load with weights_only=True
        self.net_weights = torch.load(
            weight_file_path, 
            map_location={'cuda:0': 'cpu'},
            weights_only=True  # Add this parameter
        )
        keys = list(self.net_weights.keys())
        # print(keys)
        weights_load = {}

        for i in range(len(keys)):
            weights_load[list(self.net.state_dict().keys())[i]
                        ] = self.net_weights[list(keys)[i]]

        state = self.net.state_dict()
        state.update(weights_load)
        self.net.load_state_dict(state)

        print('Pose Estimation Model Loaded')

    def detect(self, frame: np.ndarray) -> list:
        """
        Estimates poses in a frame using MediaPipe BlazePose.
        Note: MediaPipe's default Pose solution primarily detects one dominant person.
        For robust multi-person, you might need to combine with YOLO's person detections
        and run pose for each cropped person region, or use a truly multi-person pose model.
        For this hackathon, it will process the whole frame and return detected poses.
        Returns a list of dictionaries:
        [{'keypoints': [[x, y, visibility], ...], 'bbox': [x1, y1, x2, y2]}, ...]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        poses_data = []
        if results.pose_landmarks:
            h, w, _ = frame.shape
            keypoints = []
            x_coords = []
            y_coords = []
            for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                keypoints.append([cx, cy, lm.visibility])
                x_coords.append(cx)
                y_coords.append(cy)

            # Create a simple bbox from keypoints
            if x_coords and y_coords:
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                # Add a small buffer to the bbox
                buffer = 20
                x_min = max(0, x_min - buffer)
                y_min = max(0, y_min - buffer)
                x_max = min(w, x_max + buffer)
                y_max = min(h, y_max + buffer)
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = [0,0,0,0] # Fallback

            poses_data.append({
                'keypoints': keypoints,
                'bbox': bbox
            })
        return poses_data

    def draw_results(self, original_frame: np.ndarray, poses_data: list) -> np.ndarray:
        """
        Draws pose estimations (keypoints and connections) on a copy of the original frame.
        Args:
            original_frame (np.ndarray): The frame to draw on.
            poses_data (list): Output from self.detect method.
        Returns:
            np.ndarray: A new frame with pose detections drawn.
        """
        display_frame = original_frame.copy() # Draw on a copy of the original frame
        for pose_data in poses_data:
            if pose_data['keypoints']:
                draw_keypoints(display_frame, pose_data['keypoints'], connections=POSE_CONNECTIONS_INDICES, color=(0, 255, 255)) # Cyan color
        return display_frame
import openpifpaf
import cv2
import numpy as np
from cv.base_detector import BaseDetector
from core.config import Config
from core.utils import draw_keypoints

class PoseEstimator(BaseDetector):
    def __init__(self, config: Config):
        super().__init__(config)
        # Initialize OpenPifPaf predictor with default settings
        self.predictor = openpifpaf.Predictor()
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Estimates poses in a frame using OpenPifPaf.
        Supports multi-person pose estimation out of the box.
        Returns a list of keypoint arrays in format [[x, y, confidence], ...]
        """
        # Convert frame to RGB (OpenPifPaf expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        predictions, _, _ = self.predictor.numpy_image(frame_rgb)
        
        poses_data = {}
        
        for id, pred in enumerate(predictions):
            # Reshape keypoints to match the format in the provided code
            keypoints = pred.data.reshape(-1, 3)
            
            # Convert keypoints to the expected format
            keypoints_list = []
            
            for x, y, conf in keypoints:
                cx, cy = int(x), int(y)
                keypoints_list.append([cx, cy, conf])
            
            poses_data[id] = keypoints_list
            
        return poses_data

    def draw_results(self, original_frame: np.ndarray, poses_data: list) -> np.ndarray:
        """
        Draws pose estimations (keypoints and connections) on a copy of the original frame.
        Uses OpenPifPaf style drawing with blue keypoints and cyan connections.
        Args:
            original_frame (np.ndarray): The frame to draw on.
            poses_data (list): List of keypoint arrays from self.detect method.
        Returns:
            np.ndarray: A new frame with pose detections drawn.
        """
        display_frame = original_frame.copy()

        keypoints_list = []
        for id in poses_data:
            if poses_data[id]:
                keypoints_list.append(poses_data[id])
        
        for keypoint in keypoints_list:
            draw_keypoints(display_frame, keypoint)

        return display_frame
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'))

from OpenPoseNet.util.decode_pose import decode_pose
from OpenPoseNet.util.openpose_net import OpenPoseNet

class PoseEstimator(BaseDetector):
    # def __init__(self, config: Config):
    #     super().__init__(config)
    #     self.mp_pose = mp.solutions.pose
    #     self.pose = self.mp_pose.Pose(
    #         min_detection_confidence=self.config.POSE_MIN_CONFIDENCE,
    #         min_tracking_confidence=self.config.POSE_MIN_CONFIDENCE
    #     )
    #     self.mp_drawing = mp.solutions.drawing_utils

    def __init__(self, config: Config):
        super().__init__(config)
        self.net = OpenPoseNet()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Update path to weights file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Get src directory
        weight_file_path = os.path.join(
            base_dir, 
            'models', 
            'pose_model_scratch.pth'
        )

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

    # def detect(self, frame: np.ndarray) -> list:
    #     """
    #     Estimates poses in a frame using MediaPipe BlazePose.
    #     Note: MediaPipe's default Pose solution primarily detects one dominant person.
    #     For robust multi-person, you might need to combine with YOLO's person detections
    #     and run pose for each cropped person region, or use a truly multi-person pose model.
    #     For this hackathon, it will process the whole frame and return detected poses.
    #     Returns a list of dictionaries:
    #     [{'keypoints': [[x, y, visibility], ...], 'bbox': [x1, y1, x2, y2]}, ...]
    #     """
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.pose.process(frame_rgb)

    #     poses_data = []
    #     if results.pose_landmarks:
    #         h, w, _ = frame.shape
    #         keypoints = []
    #         x_coords = []
    #         y_coords = []
    #         for lm_id, lm in enumerate(results.pose_landmarks.landmark):
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             keypoints.append([cx, cy, lm.visibility])
    #             x_coords.append(cx)
    #             y_coords.append(cy)

    #         # Create a simple bbox from keypoints
    #         if x_coords and y_coords:
    #             x_min, y_min = min(x_coords), min(y_coords)
    #             x_max, y_max = max(x_coords), max(y_coords)
    #             # Add a small buffer to the bbox
    #             buffer = 20
    #             x_min = max(0, x_min - buffer)
    #             y_min = max(0, y_min - buffer)
    #             x_max = min(w, x_max + buffer)
    #             y_max = min(h, y_max + buffer)
    #             bbox = [x_min, y_min, x_max, y_max]
    #         else:
    #             bbox = [0,0,0,0] # Fallback

    #         poses_data.append({
    #             'keypoints': keypoints,
    #             'bbox': bbox
    #         })
    #     return poses_data

    def detect(self, frame: np.ndarray) -> list:
        """
        Estimates poses in a frame using OpenPoseNet.
        Returns a list of dictionaries:
        [{'keypoints': [[x, y, visibility], ...], 'bbox': [x1, y1, x2, y2]}, ...]
        """
        oriImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Resize the image to the input size of the model
        size = (368, 368)
        img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)

        # Preprocess the image
        img = img.astype(np.float32) / 255.

        # Standardization of color information
        color_mean = [0.485, 0.456, 0.406]
        color_std = [0.229, 0.224, 0.225]

        # Color channels in wrong order
        preprocessed_img = img.copy()  # RGB

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        # (H, W, C) -> (C, H, W)
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

        # Convert to PyTorch tensor
        img = torch.from_numpy(img)

        # Add batch dimension
        x = img.unsqueeze(0)

        # Calculate heatmaps and PAFs with OpenPose
        self.net.eval()
        predicted_outputs, _ = self.net(x)

        # Resize outputs to original image size
        pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
        heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

        pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

        pafs = cv2.resize(
            pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(
            heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Decode poses
        _, result_img, joint_list, person_to_joint_assoc = decode_pose(oriImg, heatmaps, pafs)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        poses_data = {
            'image': result_img,  # The image with poses drawn
        }

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
        # for pose_data in poses_data:
        #     if pose_data['keypoints']:
        #         draw_keypoints(display_frame, pose_data['keypoints'], connections=POSE_CONNECTIONS_INDICES, color=(0, 255, 255)) # Cyan color
        return poses_data['image'] if 'image' in poses_data else display_frame

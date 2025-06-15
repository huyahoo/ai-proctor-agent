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
        # Preprocess image
        h, w, _ = frame.shape
        input_w, input_h = 368, 368  # Standard input size for OpenPose
        
        # Resize and normalize image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scaled_img = cv2.resize(frame_rgb, (input_w, input_h)) / 255.0
        scaled_img = scaled_img.transpose(2, 0, 1)[None]  # NHWC -> NCHW
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(scaled_img)
        
        # Get network prediction
        with torch.no_grad():
            predicted_outputs = self.net(img_tensor)
        
        # Decode poses
        poses = decode_pose(predicted_outputs)
        
        poses_data = []
        for pose in poses:
            keypoints = []
            x_coords = []
            y_coords = []
            
            # Convert normalized coordinates back to original image scale
            for kp in pose:
                if kp is not None:
                    cx = int(kp[0] * w / input_w)
                    cy = int(kp[1] * h / input_h)
                    conf = float(kp[2]) if len(kp) > 2 else 1.0
                    
                    keypoints.append([cx, cy, conf])
                    x_coords.append(cx)
                    y_coords.append(cy)
            
            if x_coords and y_coords:
                # Create bounding box
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                
                # Add buffer to bbox
                buffer = 20
                x_min = max(0, x_min - buffer)
                y_min = max(0, y_min - buffer)
                x_max = min(w, x_max + buffer)
                y_max = min(h, y_max + buffer)
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = [0, 0, 0, 0]  # Fallback
                
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
    

if __name__ == "__main__":
    # Example usage
    pose_estimator = PoseEstimator()
    # cap = cv2.VideoCapture(0)  # Use webcam or replace with video file path
    video_path = '/home/dh11255z/Documents/proctor_agent_base/test.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Kiểm tra video có mở thành công không
    if not cap.isOpened():
        print("Error: Could not open video")
        exit()
        
    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Tạo VideoWriter để lưu output video (nếu cần)
    # out = cv2.VideoWriter('output.mp4', 
    #                      cv2.VideoWriter_fourcc(*'mp4v'), 
    #                      fps, 
    #                      (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Detect poses
            poses_data = pose_estimator.detect(frame)
            
            # Draw results
            output_frame = pose_estimator.draw_results(frame, poses_data)
            
            # Hiển thị frame
            cv2.imshow('OpenPose Detection', output_frame)
            
            # Lưu video output (nếu cần)
            # out.write(output_frame)
            
            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

    # Giải phóng resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
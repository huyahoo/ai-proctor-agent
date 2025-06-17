import cv2
import numpy as np
import os
import json
from PIL import Image # For VLM input processing
from core.logger import logger
from core.constants import COCO_PERSON_SKELETON_INDICES

def load_video_capture(video_path):
    """Loads a video file and returns a cv2.VideoCapture object."""
    if not os.path.exists(video_path):
        logger.error(f"Error: Video file not found at {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file at {video_path}")
        return None
    return cap

def create_blank_frame(width, height, color=(0, 0, 0)):
    """Creates a black frame of specified dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8) + np.array(color, dtype=np.uint8)

def draw_bbox(frame, bbox, label=None, color=(0, 255, 0)):
    """Draws a bounding box on the frame."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_keypoints(frame, keypoints, color=(0, 255, 0), connections=None):
    """
    Draws keypoints and connections on the frame using OpenPifPaf style.
    Args:
        frame: The frame to draw on
        keypoints: List of keypoints in format [[x, y, confidence], ...]
        color: Color for keypoints and connections (BGR format)
        connections: List of connections between keypoints. If None, uses COCO skeleton
    """
    if not keypoints:
        return frame

    keypoints = np.array(keypoints)
    
    # Use COCO skeleton by default if no connections provided
    if connections is None:
        connections = COCO_PERSON_SKELETON_INDICES
        # Use blue for keypoints and cyan for connections in OpenPifPaf style
        keypoint_color = (255, 0, 0)  # Blue
        connection_color = (0, 255, 255)  # Cyan
    else:
        keypoint_color = color
        connection_color = color

    # Draw keypoints
    for x, y, conf in keypoints:
        if conf > 0.2:  # Only draw keypoints with confidence > 0.2
            cv2.circle(frame, (int(x), int(y)), 3, keypoint_color, -1)

    # Draw connections
    for j1, j2 in connections:
        if (j1 < len(keypoints) and j2 < len(keypoints) and 
            keypoints[j1][2] > 0.2 and keypoints[j2][2] > 0.2):
            pt1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
            pt2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
            cv2.line(frame, pt1, pt2, connection_color, 2)

    return frame

def draw_gaze(frame, head_pose, color=(255, 0, 0)):
    """Draws a simplified gaze vector based on head pose."""
    # head_pose: (nose_x, nose_y, pitch, yaw, roll) from GazeTracker
    if head_pose is None or len(head_pose) < 5: # Ensure all 5 elements are present
        return frame

    nose_x, nose_y, pitch_deg, yaw_deg, _ = head_pose # pitch, yaw are in degrees from GazeTracker
    line_length = 50

    # Convert angles from degrees to radians
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    # Calculate end point of the gaze line in 2D projection
    # Horizontal movement (yaw) and vertical movement (pitch)
    end_x = int(nose_x + line_length * np.sin(yaw_rad) * np.cos(pitch_rad))
    end_y = int(nose_y - line_length * np.sin(pitch_rad)) # Negative for looking up (positive pitch)

    cv2.line(frame, (int(nose_x), int(nose_y)), (end_x, end_y), color, 2)
    cv2.circle(frame, (int(nose_x), int(nose_y)), 3, color, -1) # Draw nose point
    return frame

def cv2_to_pil(cv2_image):
    """Converts an OpenCV image (BGR) to a PIL Image (RGB)."""
    if cv2_image is None:
        return None
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    """Converts a PIL Image (RGB) to an OpenCV image (BGR)."""
    if pil_image is None:
        return None
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



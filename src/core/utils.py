import cv2
import numpy as np
import os
import json
from PIL import Image # For VLM input processing
import mediapipe as mp # For pose drawing connections if needed for utility

def load_video_capture(video_path):
    """Loads a video file and returns a cv2.VideoCapture object."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
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
    """Draws keypoints and connections on the frame."""
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 3, color, -1)
    if connections:
        for connection in connections:
            # Ensure keypoints have visibility/confidence for drawing
            start_kp = keypoints[connection[0]] if len(keypoints) > connection[0] else None
            end_kp = keypoints[connection[1]] if len(keypoints) > connection[1] else None

            if start_kp and end_kp and \
               (len(start_kp) < 3 or start_kp[2] > 0.5) and \
               (len(end_kp) < 3 or end_kp[2] > 0.5): # Check visibility confidence if available
                cv2.line(frame, (int(start_kp[0]), int(start_kp[1])),
                         (int(end_kp[0]), int(end_kp[1])), color, 2)

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



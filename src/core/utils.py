import cv2
import numpy as np
import os
import json

def load_video(video_path):
    """Loads a video file and returns a cv2.VideoCapture object."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None
    return cap

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
            start_kp = keypoints[connection[0]]
            end_kp = keypoints[connection[1]]
            if start_kp[2] > 0.5 and end_kp[2] > 0.5: # Check visibility confidence
                cv2.line(frame, (int(start_kp[0]), int(start_kp[1])),
                         (int(end_kp[0]), int(end_kp[1])), color, 2)

def draw_gaze(frame, head_pose, color=(255, 0, 0)):
    """Draws a simplified gaze vector based on head pose."""
    # head_pose: (nose_x, nose_y, pitch, yaw, roll) - simplified for demo
    if head_pose is None:
        return frame
    nose_x, nose_y, pitch, yaw = head_pose[0], head_pose[1], head_pose[2], head_pose[3]
    # Simple line to indicate direction from nose
    line_length = 50
    # Adjust yaw for direction
    end_x = int(nose_x + line_length * np.sin(yaw))
    end_y = int(nose_y - line_length * np.cos(pitch)) # Simplified pitch effect
    cv2.line(frame, (int(nose_x), int(nose_y)), (end_x, end_y), color, 2)
    return frame

def save_feedback_annotation(image_path, event_data, vlm_decision, human_feedback, vlm_explanation):
    """Appends feedback to the annotations file."""
    os.makedirs(Config.FEEDBACK_DATA_DIR, exist_ok=True)
    annotation_data = {
        "image_path": image_path,
        "event_data": event_data,
        "vlm_decision": vlm_decision,
        "human_feedback": human_feedback, # e.g., "confirmed_cheating", "false_positive"
        "vlm_explanation": vlm_explanation,
        "timestamp": cv2.getTickCount() / cv2.getTickFrequency() # More precise timestamp
    }
    with open(Config.FEEDBACK_ANNOTATIONS_FILE, 'a') as f:
        f.write(json.dumps(annotation_data) + '\n')
    print(f"Feedback saved for {image_path}")

def get_video_frame_at_timestamp(video_path, timestamp_sec):
    """Retrieves a specific frame from a video at a given timestamp (seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None



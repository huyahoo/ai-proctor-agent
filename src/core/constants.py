import mediapipe as mp

# Constants for keypoint indexing (MediaPipe BlazePose) and connections for drawing
# (Simplified to common connections for visualization)
POSE_CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EYE),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.RIGHT_EYE),
    (mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.LEFT_EAR),
    (mp.solutions.pose.PoseLandmark.RIGHT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EAR),
    (mp.solutions.pose.PoseLandmark.MOUTH_LEFT, mp.solutions.pose.PoseLandmark.MOUTH_RIGHT),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
]
# Convert to integer indices for easier use with list of keypoints
POSE_CONNECTIONS_INDICES = [(lm1.value, lm2.value) for lm1, lm2 in POSE_CONNECTIONS]


# Object classes for YOLO (expand if needed for specific items)
# These are common COCO dataset classes. You might need to fine-tune YOLO for specific "cheating notes" etc.
YOLO_CLASSES_OF_INTEREST = ['person', 'cell phone', 'book', 'laptop', 'bottle', 'backpack', 'keyboard', 'mouse', 'tv', 'remote', 'clock', 'scissors', 'pen', 'pencil', 'tablet', 'calculator']
# Added more common items that could be relevant in an exam, e.g., calculator, pen.



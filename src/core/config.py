import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Add OpenAI key if needed for future integration
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Paths
    YOLO_MODEL_PATH = "models/yolov8n.pt" # Ensure this path is correct

    # Video Processing
    FPS = 30 # Target frames per second for processing and display
    FRAME_SKIP = 1 # Process every Nth frame (1 = process every frame)

    # Anomaly Detection Thresholds (adjust these during testing)
    GAZE_CONSECUTIVE_FRAMES = 5 # How many frames must gaze be detected towards a target
    GAZE_YAW_THRESHOLD = 15 # Degrees. How much yaw deviation from straight is considered "looking sideways"
    POSE_MIN_CONFIDENCE = 0.5 # Min confidence for pose detection keypoints
    OBJECT_MIN_CONFIDENCE = 0.6 # Min confidence for detected objects (non-person)
    COLLABORATIVE_DISTANCE_THRESHOLD = 0.3 # Max normalized distance (0-1, relative to frame width) between student centers for "close" interaction

    # LLM/VLM Settings
    LLM_MODEL_NAME = "gemini-1.5-flash-latest"
    VLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
    VLM_ANALYSIS_CLIP_SECONDS = 2 # How many seconds of video to send to VLM (centered on anomaly)

    # Feedback Learning
    FEEDBACK_DATA_DIR = "data/feedback"
    FEEDBACK_ANNOTATIONS_FILE = os.path.join(FEEDBACK_DATA_DIR, "annotations.jsonl")

    def __str__(self):
        return f"Proctor Agent Configuration:\n" \
               f"  YOLO Model: {self.YOLO_MODEL_PATH}\n" \
               f"  Target FPS: {self.FPS}, Frame Skip: {self.FRAME_SKIP}\n" \
               f"  LLM Model: {self.LLM_MODEL_NAME}\n" \
               f"  VLM Model: {self.VLM_MODEL_NAME}\n" \
               f"  Feedback Dir: {self.FEEDBACK_DATA_DIR}\n"


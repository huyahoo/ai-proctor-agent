import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Only if you decide to use OpenAI

    # Model Paths
    YOLO_MODEL_PATH = "models/yolov8n.pt" # Ensure this path is correct after downloading/placing the model

    # Video Processing
    FPS = 30 # Frames per second for processing
    FRAME_SKIP = 1 # Process every Nth frame (1 = process every frame)

    # Anomaly Detection Thresholds (adjust these during testing)
    GAZE_CONSECUTIVE_FRAMES = 5 # How many frames must gaze be detected towards a target
    POSE_MIN_CONFIDENCE = 0.5 # Min confidence for pose detection
    OBJECT_MIN_CONFIDENCE = 0.6 # Min confidence for object detection
    COLLABORATIVE_DISTANCE_THRESHOLD = 0.3 # Max normalized distance between students for "close" interaction

    # LLM/VLM Settings
    LLM_MODEL_NAME = "gemini-2.0-flash" # Or "gpt-4o"
    VLM_MODEL_NAME = "gemini-2.0-flash" # Or "gpt-4o" for VLM capabilities

    # Feedback Learning
    FEEDBACK_DATA_DIR = "data/feedback"
    FEEDBACK_ANNOTATIONS_FILE = os.path.join(FEEDBACK_DATA_DIR, "annotations.jsonl")

    def __str__(self):
        return f"Proctor Agent Configuration:\n" \
               f"  YOLO Model: {self.YOLO_MODEL_PATH}\n" \
               f"  FPS: {self.FPS}, Frame Skip: {self.FRAME_SKIP}\n" \
               f"  LLM Model: {self.LLM_MODEL_NAME}\n" \
               f"  VLM Model: {self.VLM_MODEL_NAME}\n"


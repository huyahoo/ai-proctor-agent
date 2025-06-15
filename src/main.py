import sys
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from ui.main_window import ProctorAgentApp
from core.config import Config # Import Config to print it out

# Ensure necessary folders exist
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/feedback/images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Placeholder for YOLO model download (run this manually first time)
# from ultralytics import YOLO
# try:
#     YOLO("yolov8n.pt") # This will download if not present
# except Exception as e:
#     print(f"Could not automatically download YOLOv8n.pt: {e}. Please ensure it's in the 'models/' folder.")


if __name__ == "__main__":
    # Print config to console for verification
    print(Config())

    app = QApplication(sys.argv)
    window = ProctorAgentApp()
    window.show()
    sys.exit(app.exec())



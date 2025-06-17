import sys
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from ui.main_window import ProctorAgentApp
from core.config import Config
from core.logger import logger
from core.utils import setup_warning_filters

# Setup warning filters
setup_warning_filters()

# Ensure necessary folders exist
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/feedback/images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Important: Ensure `yolov8n.pt` is in the `models/` directory.
# This script does NOT automatically download it. You must place it there.
# You can get it from: https://github.com/ultralytics/ultralytics/releases
# For example, download `yolov8n.pt` and place it into `proctor_agent/models/`.

if __name__ == "__main__":
    # Print config to console for verification
    logger.step("Starting Proctor Agent...")
    logger.info(Config())

    app = QApplication(sys.argv)
    window = ProctorAgentApp()
    window.show()
    sys.exit(app.exec())



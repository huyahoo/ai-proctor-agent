from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path

# Get the project root directory and src directory
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"

# Add src directory to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
from cv.yolo_detector import YOLODetector
from core.config import Config
from core.logger import logger
from core.utils import assign_yolo_pids

config = Config()
yolo_detector = YOLODetector(config)

def test_yolo_with_image():
    """
    Test YOLO object detection on a sample image.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(parent_dir, "src", "test", "gaze_estimations.json"), "r") as f:
        gaze_info = json.load(f)
    input_image = gaze_info["image_path"] 
    image = cv2.imread(input_image)
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    if image is None:
        logger.error(f"Could not read image from {input_image}")
        return
    
    gaze_estimations = gaze_info["gaze_estimations"]
    yolo_detections = yolo_detector.detect(image)
    yolo_detections = assign_yolo_pids(yolo_detections, gaze_estimations)
    yolo_info = {
        "image_path": input_image,
        "yolo_detections": yolo_detections,
        "frame_width": width,
        "frame_height": height
    }
    with open(os.path.join(parent_dir, "src", "test", "yolo_detections.json"), "w") as f:
        json.dump(yolo_info, f, indent=4)
        print(f"Results saved to gaze_estimations.json")
    print(f"Detected {len(yolo_detections)} YOLO objects in the image: ", yolo_detections)

    
    yolo_viz_frame = yolo_detector.draw_results(image, yolo_detections)

    output_image = os.path.join(parent_dir, "data", "output", "test_yolo_output.jpg")
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    cv2.imwrite(output_image, yolo_viz_frame)
    logger.success(f"Annotated image saved to: {output_image}")

def main():
    test_yolo_with_image()

if __name__ == "__main__":
    main()
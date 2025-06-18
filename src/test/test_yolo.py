from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
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
from core.utils import assign_yolo_pids, assign_pose_pids

config = Config()
yolo_detector = YOLODetector(config)

def test_yolo_with_image():
    """
    Test YOLO object detection on a sample image.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_image = os.path.join(parent_dir, "data", "images", "IMG_4741.jpg") 
    image = cv2.imread(input_image)
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    if image is None:
        logger.error(f"Could not read image from {input_image}")
        return
    
    gaze_estimations = [{'bbox': [1202.5040283203125, 720.3736572265625, 1428.29150390625, 1026.041748046875], 'gaze_point': [0.34375, 0.5], 'gaze_vector': [0.19035454094409943, 0.9817154407501221], 'inout_score': 0.9985774755477905, 'pid': 1}, {'bbox': [2880.576171875, 928.5159912109375, 3132.960693359375, 1240.22607421875], 'gaze_point': [0.75, 0.515625], 'gaze_vector': [-0.10129646956920624, 0.9948562383651733], 'inout_score': 0.9992438554763794, 'pid': 0}]
    yolo_detections = yolo_detector.detect(image)
    yolo_detections = assign_yolo_pids(yolo_detections, gaze_estimations)
    print(f"Detected {len(yolo_detections)} YOLO objects in the image: ", yolo_detections)

    
    yolo_viz_frame = yolo_detector.draw_results(image, yolo_detections)

    output_image = "results/test_yolo_output.jpg"
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    cv2.imwrite(output_image, yolo_viz_frame)
    logger.success(f"Annotated image saved to: {output_image}")

def main():
    test_yolo_with_image()

if __name__ == "__main__":
    main()
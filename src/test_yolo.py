from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from core.utils import assign_yolo_pids, assign_pose_pids

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import numpy as np
from cv.yolo_detector import YOLODetector
from core.config import Config
from core.logger import logger

config = Config()
yolo_detector = YOLODetector(config)

def test_yolo_with_image():
    """
    Test YOLO object detection on a sample image.
    """
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    input_image = os.path.join(parent_dir, "data", "images", "IMG_4732.jpg") 
    image = cv2.imread(input_image)
    
    if image is None:
        logger.error(f"Could not read image from {input_image}")
        return
    
    gaze_estimations = [{'bbox': [2747.9052734375, 1191.8621826171875, 2980.486572265625, 1501.100341796875], 'gaze_point': [0.703125, 0.609375], 'gaze_vector': [-0.196572944521904, 0.9804891347885132], 'inout_score': 0.9973465204238892, 'pid': 1}, {'bbox': [1141.3909912109375, 998.2349243164062, 1410.483642578125, 1278.93115234375], 'gaze_point': [0.703125, 0.484375], 'gaze_vector': [0.9483205676078796, 0.31731361150741577], 'inout_score': 0.9280946850776672, 'pid': 0}]
    yolo_detections = yolo_detector.detect(image)
    yolo_detections = assign_yolo_pids(yolo_detections, gaze_estimations)
    # print(f"Detected {len(yolo_detections)} YOLO objects in the image: ", yolo_detections)
    
    yolo_viz_frame = yolo_detector.draw_results(image, yolo_detections)

    output_image = "results/test_yolo_output.jpg"
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    cv2.imwrite(output_image, yolo_viz_frame)
    logger.success(f"Annotated image saved to: {output_image}")

def main():
    test_yolo_with_image()

if __name__ == "__main__":
    main()
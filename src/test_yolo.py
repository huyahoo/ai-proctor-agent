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
    input_image = os.path.join(parent_dir, "data", "images", "162.jpg") 
    image = cv2.imread(input_image)
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    if image is None:
        logger.error(f"Could not read image from {input_image}")
        return
    
    gaze_estimations = [{'bbox': [1051.534912109375, 427.95166015625, 1151.1109619140625, 558.1237182617188], 'gaze_point': [0.609375, 0.625], 'gaze_vector': [0.08420028537511826, 0.996448814868927], 'inout_score': 0.9989210367202759, 'pid': 3}, {'bbox': [734.8357543945312, 342.3213806152344, 812.5286865234375, 446.6407775878906], 'gaze_point': [0.421875, 0.546875], 'gaze_vector': [0.06804251670837402, 0.9976824522018433], 'inout_score': 0.9987419247627258, 'pid': 2}, {'bbox': [1281.304443359375, 329.313720703125, 1361.4693603515625, 423.94091796875], 'gaze_point': [0.5625, 0.609375], 'gaze_vector': [-0.7742968201637268, 0.6328225135803223], 'inout_score': 0.9274418950080872, 'pid': 1}, {'bbox': [367.7159729003906, 396.76495361328125, 458.3671875, 523.9398193359375], 'gaze_point': [0.234375, 0.609375], 'gaze_vector': [0.0857202410697937, 0.9963192939758301], 'inout_score': 0.9791598320007324, 'pid': 0}]
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
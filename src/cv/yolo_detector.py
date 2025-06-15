from ultralytics import YOLO
import cv2
import numpy as np
from cv.base_detector import BaseDetector
from core.config import Config
from core.constants import YOLO_CLASSES_OF_INTEREST
from core.utils import draw_bbox
from core.logger import logger  

class YOLODetector(BaseDetector):
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            self.model = YOLO(self.config.YOLO_MODEL_PATH)
            logger.info(f"YOLO model loaded from {self.config.YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading YOLO model from {self.config.YOLO_MODEL_PATH}: {e}. Make sure the model file exists and path is correct.")
            self.model = None

    def detect(self, frame: np.ndarray) -> list:
        """
        Detects objects in a frame using YOLOv8.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'label': 'person', 'confidence': 0.95}, ...]
        """
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                confidence = box.conf.item()
                class_id = box.cls.item()
                label = self.model.names[int(class_id)]

                if confidence > self.config.OBJECT_MIN_CONFIDENCE and label in YOLO_CLASSES_OF_INTEREST:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': label,
                        'confidence': confidence
                    })
        return detections

    def draw_results(self, original_frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws YOLO bounding boxes and labels on a copy of the original frame.
        Args:
            original_frame (np.ndarray): The frame to draw on.
            detections (list): Output from self.detect method.
        Returns:
            np.ndarray: A new frame with YOLO detections drawn.
        """
        display_frame = original_frame.copy() # Draw on a copy of the original frame
        for det in detections:
            color = (0, 255, 0) # Green for persons
            if det['label'] in ['cell phone', 'book', 'note', 'earbud', 'smartwatch', 'calculator']: # Unauthorized items in red
                color = (0, 0, 255) 
            elif det['label'] != 'person': # Other objects in yellow
                color = (255, 255, 0) 
            
            draw_bbox(display_frame, det['bbox'], f"{det['label']}: {det['confidence']:.2f}", color)
        return display_frame
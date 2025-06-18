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
            logger.success(f"YOLO model loaded from {self.config.YOLO_MODEL_PATH}")
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
        Draws YOLO detections on a copy of the original frame,
        syncing person colors by pid and overriding colors for other labels.
        """
        frame = original_frame.copy()

        for det in detections:
            label = det.get('label', '')
            bbox  = det['bbox']
            conf  = det.get('confidence', 0.0)
            pid   = det.get('pid', None)

            # Build the text you'll draw
            text = f"{label}: {conf:.2f}"
            draw_bbox(frame, bbox, label, text, pid)

        return frame
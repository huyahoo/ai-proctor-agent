from ultralytics import YOLO
import cv2
from cv.base_detector import BaseDetector
from core.config import Config
from core.constants import YOLO_CLASSES_OF_INTEREST

class YOLODetector(BaseDetector):
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            self.model = YOLO(self.config.YOLO_MODEL_PATH)
            print(f"YOLO model loaded from {self.config.YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Make sure the model file exists.")
            self.model = None

    def detect(self, frame):
        """
        Detects objects in a frame using YOLOv8.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'label': 'person', 'confidence': 0.95}, ...]
        """
        if self.model is None:
            return []

        results = self.model(frame, verbose=False) # Run inference
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



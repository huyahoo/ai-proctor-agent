from ultralytics import YOLO
import cv2
import numpy as np
from cv.base_detector import BaseDetector
from core.config import Config
from core.constants import YOLO_CLASSES_OF_INTEREST, UNAUTHORIZED_ITEMS
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

        try:
            self.cheatsheet_model = YOLO(self.config.CHEATSHEET_MODEL_PATH)
            self.earpods_model = YOLO(self.config.EARPODS_MODEL_PATH)
            self.electronics_model = YOLO(self.config.ELECTRONICS_MODEL_PATH)
            logger.success(f"YOLO model loaded from {self.config.CHEATSHEET_MODEL_PATH}, {self.config.EARPODS_MODEL_PATH}, {self.config.ELECTRONICS_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading YOLO model for authorized objects: {e}. Make sure the model file exists and path is correct.")
            self.model = None

    def _detect_unauthorized_items(self, frame: np.ndarray) -> list:
        """
        Detects unauthorized items in a frame using YOLO.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'label': 'cell phone', 'confidence': 0.95}, ...]
        """
        cheatsheet_results = self.cheatsheet_model(frame, verbose=False)
        earpods_results = self.earpods_model(frame, verbose=False)
        electronics_results = self.electronics_model(frame, verbose=False)
        unauthorized_detections = []
        # print(f"Cheatsheet results: {cheatsheet_results}")
        # print(f"Earpods results: {earpods_results}")
        # print(f"Electronics results: {electronics_results}")
        for results in [cheatsheet_results, earpods_results, electronics_results]:
            for r in results:
                for box in r.boxes:
                    confidence = box.conf.item()
                    class_id = box.cls.item()
                    label = self.model.names[int(class_id)]

                    print("HÂHHAHAHAHAHAHAHAHAHA: ", label)

                    if confidence > self.config.OBJECT_MIN_CONFIDENCE and label in UNAUTHORIZED_ITEMS:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        unauthorized_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': label,
                            'confidence': confidence
                        })
                        
        return unauthorized_detections

    def detect(self, frame: np.ndarray) -> list:
        """
        Detects objects in a frame using YOLOv8.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'label': 'person', 'confidence': 0.95}, ...]
        """
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)

        yolo_data = {}
        unauthorized_object = self._detect_unauthorized_items(frame)
        yolo_data['unauthorized_objects'] = unauthorized_object

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
        
        yolo_data['detections'] = detections

        return yolo_data

    def draw_results(self, original_frame: np.ndarray, yolo_data: dict) -> np.ndarray:
        """
        Draws YOLO bounding boxes and labels on a copy of the original frame.
        Args:
            original_frame (np.ndarray): The frame to draw on.
            detections (list): Output from self.detect method.
        Returns:
            np.ndarray: A new frame with YOLO detections drawn.
        """
        display_frame = original_frame.copy() # Draw on a copy of the original frame
        for det in yolo_data["detections"]:
            color = (0, 255, 0) # Green for persons
            if det['label'] != 'person': # Other objects in yellow
                color = (255, 255, 0)             
            draw_bbox(display_frame, det['bbox'], f"{det['label']}: {det['confidence']:.2f}", color)

        for det in yolo_data["unauthorized_objects"]:
            color = (0, 0, 255)
            draw_bbox(display_frame, det['bbox'], f"{det['label']}: {det['confidence']:.2f}", color)

        return display_frame
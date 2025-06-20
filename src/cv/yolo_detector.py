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

        try:
            self.exam_paper_model = YOLO(self.config.EXAM_PAPER_MODEL_PATH)
            logger.success(f"Exam paper YOLO model loaded from {self.config.EXAM_PAPER_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading exam paper YOLO model from {self.config.EXAM_PAPER_MODEL_PATH}: {e}. Make sure the model file exists and path is correct.")
            self.exam_paper_model = None
    
    def _post_process_detections(self, model, results):
        detections = []
        for r in results:
            for box in r.boxes:
                confidence = box.conf.item()
                class_id = box.cls.item()
                label = model.names[int(class_id)]

                if confidence > self.config.OBJECT_MIN_CONFIDENCE and label in YOLO_CLASSES_OF_INTEREST:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': label,
                        'confidence': confidence
                    })
        return detections
    
    def _draw_frame(self, frame, detection_results):
        for det in detection_results:
            pid = det.get('pid', None)
            if pid == -1: continue
            label = det.get('label', '')
            bbox = det['bbox']
            conf = det.get('confidence', 0.0)

            # Build the text you'll draw
            text = f"{label}: {conf:.2f}"
            draw_bbox(frame, bbox, label, text, pid)

    def detect(self, frame: np.ndarray) -> list:
        """
        Detects objects in a frame using YOLOv8.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'label': 'person', 'confidence': 0.95}, ...]
        """
        if self.model is None:
            return []

        person_results = self.model(frame, verbose=False)
        exam_paper_results = self.exam_paper_model(frame, verbose=False) if self.exam_paper_model else []

        detections = self._post_process_detections(self.model, person_results)
        exam_paper_detections = self._post_process_detections(self.exam_paper_model, exam_paper_results) if self.exam_paper_model else []
        detections.extend(exam_paper_detections)

        # if is_first_frame:
        #     for detection in detections:
        #         if detection["label"] != "person": continue
        #         feature = self.extract_person_feature(frame, detection)

        return detections

    def draw_results(self, original_frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws YOLO detections on a copy of the original frame,
        syncing person colors by pid and overriding colors for other labels.
        """
        frame = original_frame.copy()

        self._draw_frame(frame, detections)
        # self._draw_frame(frame, detections["exam_paper"])
        # self._draw_frame(frame, detections["unauthorized_objects"])

        return frame
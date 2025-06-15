from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """Abstract base class for all CV detectors."""
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def detect(self, frame: np.ndarray):
        """
        Processes a single frame to detect relevant features.
        Returns raw data (e.g., list of bboxes, keypoints).
        """
        pass

    @abstractmethod
    def draw_results(self, frame_shape: tuple, data):
        """
        Draws the detected features onto a blank frame for visualization.
        Args:
            frame_shape (tuple): (height, width, channels) of the original frame.
            data: The raw data output from the detect method.
        Returns:
            np.ndarray: A new frame with only the drawn detections.
        """
        pass


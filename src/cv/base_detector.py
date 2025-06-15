from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Abstract base class for all CV detectors."""
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def detect(self, frame):
        """Processes a single frame to detect relevant features."""
        pass


import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QSize

class VideoOutputWidget(QWidget):
    """
    A widget to display a single video frame (NumPy array) with a title.
    Designed to be used for Original, YOLO, Pose, and Gaze feeds.
    """
    def __init__(self, title="Video Feed", parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(5,5,5,5) # Small margins for separation
        self.layout().setSpacing(5)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            background-color: #2c3e50; /* Dark blue/gray */
            color: white;
            padding: 5px;
            border-radius: 5px;
        """)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 240) # Smaller default size for 4 views, will scale

        self.layout().addWidget(self.title_label)
        self.layout().addWidget(self.image_label)

        self.current_frame = None

    def display_frame(self, frame_np_array: np.ndarray):
        """
        Receives a numpy array frame and displays it.
        """
        if frame_np_array is None:
            self.image_label.clear()
            self.current_frame = None
            return

        self.current_frame = frame_np_array # Store for potential external use (e.g. feedback frames)

        # Convert NumPy array (BGR) to QPixmap (RGB)
        # Assuming input frame_np_array is BGR from OpenCV for consistent handling
        rgb_image = cv2.cvtColor(frame_np_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale pixmap to fit label, maintaining aspect ratio
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSlider
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

class VideoPlayer(QWidget):
    # Signals for communicating with main window
    frame_ready = pyqtSignal(np.ndarray, float) # current frame (np array) and timestamp

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.frame_rate = 30 # Default frame rate
        self.current_frame_idx = 0
        self.total_frames = 0
        self.is_playing = False

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480) # Default size, will scale

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.sliderReleased.connect(self._slider_released)
        self.seeking = False

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.overlays = {
            'yolo': [], # List of {'bbox': [], 'label': '', 'confidence': 0.0}
            'pose': [], # List of {'keypoints': [], 'bbox': []}
            'gaze': []  # List of {'bbox': [], 'head_pose': []}
        }
        self.show_yolo_overlay = False
        self.show_pose_overlay = False
        self.show_gaze_overlay = False

        self.person_bboxes = [] # To pass person bboxes for multi-person pose estimation if needed

    def load_video(self, video_path):
        self.stop()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return False

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.total_frames - 1)
        self.current_frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        self.play() # Auto-play on load
        return True

    def _next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        if not self.seeking: # Only advance frame if not seeking with slider
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return

            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if not self.seeking: # Update slider only if not being moved by user
                self.slider.setValue(self.current_frame_idx)

            timestamp_sec = self.current_frame_idx / self.frame_rate
            self.frame_ready.emit(frame.copy(), timestamp_sec) # Emit signal with frame and timestamp

            self.display_frame(frame)

    def display_frame(self, frame):
        """Displays the frame with active overlays."""
        display_frame = frame.copy() # Work on a copy

        # Draw YOLO detections
        if self.show_yolo_overlay:
            for det in self.overlays['yolo']:
                if det['label'] == 'person':
                    color = (0, 255, 0) # Green for persons
                elif det['label'] in ['cell phone', 'book', 'note']: # Example unauthorized
                    color = (0, 0, 255) # Red for suspicious objects
                else:
                    color = (255, 255, 0) # Yellow for others
                self._draw_bbox_on_frame(display_frame, det['bbox'], f"{det['label']}: {det['confidence']:.2f}", color)

        # Draw Pose estimations
        if self.show_pose_overlay:
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            for pose_data in self.overlays['pose']:
                if pose_data['keypoints']:
                    h, w, c = display_frame.shape
                    
                    # Recreate MediaPipe landmark list from keypoints
                    landmark_list = landmark_pb2.NormalizedLandmarkList()
                    for kp in pose_data['keypoints']:
                        landmark = landmark_list.landmark.add()
                        landmark.x = kp[0] / w
                        landmark.y = kp[1] / h
                        if len(kp) > 2:
                            landmark.visibility = kp[2]
                    
                    mp_drawing.draw_landmarks(
                        display_frame, landmark_list, mp_pose.POSE_CONNECTIONS
                    )

        # Draw Gaze
        if self.show_gaze_overlay:
            for gaze_data in self.overlays['gaze']:
                if gaze_data['head_pose']:
                    self._draw_gaze_on_frame(display_frame, gaze_data['head_pose'], color=(255,0,0)) # Blue for gaze

        # Convert to QPixmap and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _draw_bbox_on_frame(self, frame, bbox, label=None, color=(0, 255, 0)):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_gaze_on_frame(self, frame, head_pose, color=(255, 0, 0)):
        # head_pose: [nose_x, nose_y, pitch, yaw, roll]
        nose_x, nose_y, pitch, yaw = head_pose[0], head_pose[1], head_pose[3], head_pose[4] # Using yaw from head_pose[3]
        line_length = 50
        # Convert yaw and pitch (in degrees) to radians for trigonometric functions
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)

        # Calculate endpoint based on yaw (horizontal direction) and pitch (vertical direction)
        # Simplified projection for drawing a line
        end_x = int(nose_x + line_length * np.sin(yaw_rad))
        end_y = int(nose_y - line_length * np.sin(pitch_rad)) # Positive pitch is looking up, so subtract
        # For a truly 3D projection, you'd need the rotation matrix and translation vector

        cv2.line(frame, (int(nose_x), int(nose_y)), (end_x, end_y), color, 2)
        cv2.circle(frame, (int(nose_x), int(nose_y)), 3, color, -1)


    def set_overlays(self, yolo_data, pose_data, gaze_data):
        self.overlays['yolo'] = yolo_data
        self.overlays['pose'] = pose_data
        self.overlays['gaze'] = gaze_data

    def toggle_yolo_overlay(self, state):
        self.show_yolo_overlay = state
        if self.cap and self.cap.isOpened(): # Redraw if video is loaded
            self.display_frame(self.cap.read()[1])

    def toggle_pose_overlay(self, state):
        self.show_pose_overlay = state
        if self.cap and self.cap.isOpened():
            self.display_frame(self.cap.read()[1])

    def toggle_gaze_overlay(self, state):
        self.show_gaze_overlay = state
        if self.cap and self.cap.isOpened():
            self.display_frame(self.cap.read()[1])

    def play(self):
        if self.cap and self.cap.isOpened():
            self.is_playing = True
            self.timer.start(int(1000 / self.frame_rate)) # Milliseconds per frame

    def pause(self):
        self.is_playing = False
        self.timer.stop()

    def stop(self):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.image_label.clear()
        self.slider.setValue(0)
        self.slider.setMaximum(0)
        self.current_frame_idx = 0
        self.overlays = {'yolo': [], 'pose': [], 'gaze': []}

    def set_position(self, position):
        if self.cap and self.cap.isOpened():
            self.seeking = True
            self.current_frame_idx = position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                timestamp_sec = self.current_frame_idx / self.frame_rate
                self.frame_ready.emit(frame.copy(), timestamp_sec)
                self.display_frame(frame)
            self.seeking = False # Reset seeking flag after update

    def _slider_released(self):
        self.seeking = False # Ensure seeking flag is reset when user releases slider


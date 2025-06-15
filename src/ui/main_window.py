import sys
import os
import time
import queue
import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QFrame, QSlider, QMessageBox, QSizePolicy
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer # QTimer for video playback synchronization
from PyQt6.QtGui import QIcon

from ui.video_output_widget import VideoOutputWidget
from ui.alert_panel import AlertPanel
from cv.yolo_detector import YOLODetector
from cv.pose_estimator import PoseEstimator
from cv.gaze_tracker import GazeTracker
from ai.anomaly_detector import AnomalyDetector
from ai.llm_constraint_generator import LLMConstraintGenerator
from ai.vlm_analyzer import VLMAnalyzer
from ai.feedback_learner import FeedbackLearner
from core.config import Config
from core.utils import load_video_capture, cv2_to_pil # Renamed to load_video_capture
from core.logger import logger

# Thread for processing video frames, running CV models, and emitting frames/anomalies
class VideoProcessingThread(QThread):
    # Signals to send processed frames and anomaly events to the main thread
    frame_update = pyqtSignal(dict) # Contains original, yolo, pose, gaze frames & timestamp
    anomaly_detected = pyqtSignal(dict) # Anomaly event data
    processing_finished = pyqtSignal()

    def __init__(self, video_path: str, config: Config):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.stop_flag = False
        self.pause_flag = False # New flag for pausing processing
        self.current_frame_idx = 0
        self.cap = None

        # Initialize CV detectors
        self.yolo_detector = YOLODetector(config)
        self.pose_estimator = PoseEstimator(config)
        self.gaze_tracker = GazeTracker(config)
        self.anomaly_detector = AnomalyDetector(config)

    def run(self):
        self.cap = load_video_capture(self.video_path)
        if not self.cap:
            logger.error("Failed to open video in processing thread.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx) # Set initial position

        while not self.stop_flag:
            if self.pause_flag:
                self.msleep(100) # Sleep briefly if paused
                continue

            ret, frame = self.cap.read()
            if not ret:
                break # End of video

            current_timestamp_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if self.current_frame_idx % self.config.FRAME_SKIP == 0:
                frame_height, frame_width, _ = frame.shape

                # Perform CV detections (get raw data)
                yolo_detections = self.yolo_detector.detect(frame)
                pose_estimations = self.pose_estimator.detect(frame)
                gaze_estimations = self.gaze_tracker.detect(frame)

                # Generate CV visualization frames (on black background)
                yolo_viz_frame = self.yolo_detector.draw_results(frame.shape, yolo_detections)
                pose_viz_frame = self.pose_estimator.draw_results(frame.shape, pose_estimations)
                gaze_viz_frame = self.gaze_tracker.draw_results(frame.shape, gaze_estimations)

                # Emit all frames and current timestamp for UI display
                self.frame_update.emit({
                    'original': frame.copy(),
                    'yolo': yolo_viz_frame,
                    'pose': pose_viz_frame,
                    'gaze': gaze_viz_frame,
                    'timestamp': current_timestamp_sec,
                    'frame_idx': self.current_frame_idx
                })

                # Detect anomalies using raw data
                frame_info_for_anomaly = {
                    'yolo_detections': yolo_detections,
                    'pose_estimations': pose_estimations,
                    'gaze_estimations': gaze_estimations,
                    'frame_width': frame_width,
                    'frame_height': frame_height
                }
                anomalies = self.anomaly_detector.detect_anomalies(frame_info_for_anomaly, current_timestamp_sec)
                
                for anomaly in anomalies:
                    anomaly['video_path'] = self.video_path
                    # Assign a unique ID for each event instance
                    anomaly['event_id'] = f"{anomaly['type']}_{int(current_timestamp_sec*1000)}_{np.random.randint(1000, 9999)}"
                    self.anomaly_detected.emit(anomaly) # Trigger LLM/VLM process

            time.sleep(1 / (self.config.FPS / self.config.FRAME_SKIP)) # Control processing speed

        self.cap.release()
        self.processing_finished.emit()

    def stop_processing(self):
        self.stop_flag = True

    def pause_processing(self):
        self.pause_flag = True

    def resume_processing(self):
        self.pause_flag = False

    def set_frame_position(self, frame_idx: int):
        """Sets the video capture position to a specific frame index."""
        if self.cap and self.cap.isOpened():
            self.current_frame_idx = frame_idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


# Thread for consuming anomaly events and running LLM/VLM
class AnomalyConsumer(QThread):
    llm_constraint_generated = pyqtSignal(str) # Emits constraint string
    vlm_analysis_complete = pyqtSignal(dict) # Contains event_data, llm_constraint, vlm_result

    def __init__(self, event_queue: queue.Queue, llm_generator: LLMConstraintGenerator, vlm_analyzer: VLMAnalyzer, config: Config):
        super().__init__()
        self.event_queue = event_queue
        self.llm_generator = llm_generator
        self.vlm_analyzer = vlm_analyzer
        self.config = config
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                # Use a short timeout to allow the thread to gracefully exit if stop_flag is set
                event_data = self.event_queue.get(timeout=0.1)
                logger.info(f"Consumer: Processing event {event_data.get('type')} at {event_data.get('timestamp'):.2f}s")

                # Step 1: LLM generates constraints
                llm_constraint = self.llm_generator.generate_constraints(event_data)
                self.llm_constraint_generated.emit(llm_constraint)
                logger.info(f"Consumer: LLM generated constraint: {llm_constraint}")

                # Step 2: VLM analyzes frames with constraints
                # Retrieve a short sequence of original frames around the timestamp for VLM
                frames_for_vlm = []
                video_path = event_data['video_path']
                timestamp_sec = event_data['timestamp']

                cap_temp = load_video_capture(video_path)
                if cap_temp:
                    fps = cap_temp.get(cv2.CAP_PROP_FPS)
                    # Grab a clip of VLM_ANALYSIS_CLIP_SECONDS centered on event
                    half_clip_frames = int((self.config.VLM_ANALYSIS_CLIP_SECONDS / 2) * fps)
                    start_frame_idx = max(0, int(timestamp_sec * fps) - half_clip_frames)
                    end_frame_idx = int(timestamp_sec * fps) + half_clip_frames

                    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                    for _ in range(start_frame_idx, end_frame_idx):
                        ret, frame = cap_temp.read()
                        if not ret: break
                        frames_for_vlm.append(cv2_to_pil(frame)) # Convert to PIL Image for VLM
                    cap_temp.release()
                else:
                    logger.error(f"Consumer: Failed to open video {video_path} for VLM frame extraction.")

                vlm_result = self.vlm_analyzer.analyze_and_explain(frames_for_vlm, llm_constraint)
                logger.info(f"Consumer: VLM analysis complete: {vlm_result}")

                # Emit full data for UI update
                self.vlm_analysis_complete.emit({'event_data': event_data, 'llm_constraint': llm_constraint, 'vlm_result': vlm_result})

                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Consumer Thread Error processing event: {e}")
                import traceback
                traceback.print_exc()
                self.event_queue.task_done()

    def stop_consuming(self):
        self.stop_flag = True

class ProctorAgentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.llm_generator = LLMConstraintGenerator(self.config)
        self.vlm_analyzer = VLMAnalyzer(self.config)
        self.feedback_learner = FeedbackLearner(self.config)
        self.processing_thread = None
        self.anomaly_event_queue = queue.Queue()
        self.current_video_path = None # Store current video path for re-processing/seeking

        self.setWindowTitle("Proctor Agent - AI Cheating Detection")
        self.setGeometry(100, 100, 1600, 900)

        self._init_ui()
        self._init_connections()

        self.anomaly_consumer_thread = QThread()
        self.anomaly_consumer = AnomalyConsumer(self.anomaly_event_queue, self.llm_generator, self.vlm_analyzer, self.config)
        self.anomaly_consumer.moveToThread(self.anomaly_consumer_thread)
        self.anomaly_consumer_thread.started.connect(self.anomaly_consumer.run)
        self.anomaly_consumer.llm_constraint_generated.connect(self.alert_panel.update_llm_constraint)
        self.anomaly_consumer.vlm_analysis_complete.connect(self.on_vlm_analysis_complete)
        self.anomaly_consumer_thread.start()
        logger.success("\nAnomaly consumer thread started.")


    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel: 4 Video Players and Controls
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setLayout(QVBoxLayout())
        left_panel.setMinimumWidth(960) # Sufficient width for two 480px wide videos

        # Video Display Grid (2x2)
        video_grid_layout = QVBoxLayout()
        top_row_layout = QHBoxLayout()
        bottom_row_layout = QHBoxLayout()

        self.original_video_widget = VideoOutputWidget(title="Original Input")
        self.yolo_video_widget = VideoOutputWidget(title="YOLO Detections")
        self.pose_video_widget = VideoOutputWidget(title="Pose Estimation")
        self.gaze_video_widget = VideoOutputWidget(title="Gaze Tracking")

        # Set size policy for videos to expand equally
        for widget in [self.original_video_widget, self.yolo_video_widget, self.pose_video_widget, self.gaze_video_widget]:
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        top_row_layout.addWidget(self.original_video_widget)
        top_row_layout.addWidget(self.yolo_video_widget)
        bottom_row_layout.addWidget(self.pose_video_widget)
        bottom_row_layout.addWidget(self.gaze_video_widget)

        video_grid_layout.addLayout(top_row_layout)
        video_grid_layout.addLayout(bottom_row_layout)
        left_panel.layout().addLayout(video_grid_layout)

        # Video Controls (Shared for all players)
        controls_layout = QHBoxLayout()
        self.load_video_btn = QPushButton("📂 Load Video")
        self.load_video_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #3498db; color: white; border-radius: 5px;")
        self.select_example_btn = QPushButton("📚 Example Video")
        self.select_example_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #2ecc71; color: white; border-radius: 5px;")
        self.play_pause_btn = QPushButton("▶️ Play")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.setChecked(False) # Start paused, wait for user to hit play
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
                background-color: #f39c12; /* Orange */
                color: white;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #e67e22; /* Darker Orange when checked (playing) */
            }
        """)
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #e74c3c; color: white; border-radius: 5px;")

        controls_layout.addWidget(self.load_video_btn)
        controls_layout.addWidget(self.select_example_btn)
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.stop_btn)

        left_panel.layout().addLayout(controls_layout)

        self.video_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_position_slider.setMinimum(0)
        self.video_position_slider.setMaximum(100) # Placeholder, set on video load
        self.video_position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #eee;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 18px;
                height: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
        """)
        self.video_position_slider.sliderMoved.connect(self._on_slider_moved)
        self.video_position_slider.sliderReleased.connect(self._on_slider_released)
        self.seeking = False # Flag to prevent simultaneous updates from thread and user

        left_panel.layout().addWidget(self.video_position_slider)

        main_layout.addWidget(left_panel)

        # Right Panel: Alert and AI Reasoning
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setLayout(QVBoxLayout())
        right_panel.setMinimumWidth(500)
        right_panel.setStyleSheet("background-color: #fcfcfc; border-radius: 10px; border: 1px solid #e0e0e0;")
        self.alert_panel = AlertPanel()
        right_panel.layout().addWidget(self.alert_panel)
        main_layout.addWidget(right_panel)

    def _init_connections(self):
        self.load_video_btn.clicked.connect(self.load_video_file)
        self.select_example_btn.clicked.connect(self.load_example_video)
        self.play_pause_btn.toggled.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop_all_processing)

        self.alert_panel.feedback_provided.connect(self.on_feedback_provided)

    def _start_video_and_processing(self, video_path: str):
        self.stop_all_processing() # Ensure previous processes are stopped
        self.current_video_path = video_path # Store for potential re-play or seeking

        cap_for_info = load_video_capture(video_path)
        if not cap_for_info:
            QMessageBox.critical(self, "Error Loading Video", f"Could not load video from {video_path}")
            return

        total_frames = int(cap_for_info.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_position_slider.setMaximum(total_frames - 1)
        cap_for_info.release()

        self.processing_thread = VideoProcessingThread(video_path, self.config)

        # Connect CV frame signals to respective video display widgets
        self.processing_thread.frame_update.connect(self.on_frame_update)
        # Connect anomaly signal to the main window's handler (which adds to consumer queue)
        self.processing_thread.anomaly_detected.connect(self.on_anomaly_detected)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)

        self.processing_thread.start()
        self.play_pause_btn.setChecked(True) # Set to play state
        self.play_pause_btn.setText("⏸️ Pause")
        logger.step(f"Started processing for: {video_path}")

    def load_video_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self._start_video_and_processing(selected_file)

    def load_example_video(self):
        example_path = "data/videos/example_exam.mp4"
        if not os.path.exists(example_path):
            QMessageBox.warning(self, "Missing File", f"Example video not found at: {example_path}\nPlease place a video there or use 'Load Video'.")
            return
        self._start_video_and_processing(example_path)

    def toggle_play_pause(self, checked: bool):
        if not self.current_video_path:
            QMessageBox.information(self, "No Video", "Please load a video first.")
            self.play_pause_btn.setChecked(False) # Ensure button state reflects no video
            return

        if checked: # If button is checked (means play is active)
            if self.processing_thread and not self.processing_thread.isRunning():
                # If thread was stopped (e.g., after seeking or explicit pause)
                self._start_video_and_processing(self.current_video_path) # Restart from current position
            elif self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.resume_processing() # Resume if paused
            self.play_pause_btn.setText("⏸️ Pause")
        else: # If button is unchecked (means pause is active)
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.pause_processing() # Pause the processing thread
            self.play_pause_btn.setText("▶️ Play")

    def stop_all_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            self.processing_thread.wait() # Wait for thread to finish gracefully
        self.processing_thread = None # Clear reference
        self.current_video_path = None

        # Clear all video displays
        self.original_video_widget.display_frame(None)
        self.yolo_video_widget.display_frame(None)
        self.pose_video_widget.display_frame(None)
        self.gaze_video_widget.display_frame(None)

        self.video_position_slider.setValue(0)
        self.video_position_slider.setMaximum(0) # Reset slider max

        self.play_pause_btn.setChecked(False)
        self.play_pause_btn.setText("▶️ Play")

        # Optionally clear anomaly events or reset alert panel
        # self.alert_panel.clear_all_events()
        logger.step("Stopped all video processing and cleared displays.")

    def _on_slider_moved(self, position: int):
        self.seeking = True
        # For performance, only update displays without processing if user is actively sliding
        # No processing will happen until sliderReleased

    def _on_slider_released(self):
        self.seeking = False
        if self.processing_thread and self.processing_thread.isRunning():
            # Stop, seek, and restart the processing thread
            self.processing_thread.pause_processing() # Pause current processing
            self.processing_thread.set_frame_position(self.video_position_slider.value()) # Set new position
            self.processing_thread.resume_processing() # Resume processing from new position
        elif self.current_video_path: # If video loaded but not playing, just seek visuals
            temp_cap = load_video_capture(self.current_video_path)
            if temp_cap:
                temp_cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_position_slider.value())
                ret, frame = temp_cap.read()
                if ret:
                    self.original_video_widget.display_frame(frame)
                    # For other CV displays, would need to re-process this single frame
                    # For simplicity in this case, only original video seeks visually without full reprocessing
                temp_cap.release()

    def on_frame_update(self, frame_data: dict):
        """Receives all updated frames from VideoProcessingThread and updates widgets."""
        if not self.seeking: # Only update displays if not actively seeking
            self.original_video_widget.display_frame(frame_data['original'])
            self.yolo_video_widget.display_frame(frame_data['yolo'])
            self.pose_video_widget.display_frame(frame_data['pose'])
            self.gaze_video_widget.display_frame(frame_data['gaze'])
            self.video_position_slider.setValue(frame_data['frame_idx'])

    def on_anomaly_detected(self, event_data: dict):
        """Called when AnomalyDetector flags an event. Adds it to queue."""
        self.anomaly_event_queue.put(event_data)
        logger.info(f"Anomaly detected and added to queue: {event_data.get('type')} at {event_data.get('timestamp'):.2f}s (ID: {event_data.get('event_id')})")
        self.alert_panel.add_event(event_data) # Display initial event in UI list

    def on_vlm_analysis_complete(self, data: dict):
        """Called when VLM analysis is complete by the consumer thread."""
        event_data = data['event_data']
        llm_constraint = data['llm_constraint']
        vlm_result = data['vlm_result']

        # Update the UI with the full AI reasoning process
        self.alert_panel.update_vlm_result(data) # Pass the whole dict for update

    def on_feedback_provided(self, event_data: dict, feedback_type: str):
        """Handles feedback from the UI and passes it to the FeedbackLearner."""
        logger.info(f"Received feedback: {feedback_type} for event ID: {event_data.get('event_id')}")

        # Retrieve the relevant frames for saving for RL
        frames_to_save = []
        cap_temp = load_video_capture(event_data['video_path'])
        if cap_temp:
            fps = cap_temp.get(cv2.CAP_PROP_FPS)
            # Retrieve the specific clip of frames used for VLM analysis
            clip_start_timestamp = max(0, event_data['timestamp'] - self.config.VLM_ANALYSIS_CLIP_SECONDS / 2)
            clip_end_timestamp = event_data['timestamp'] + self.config.VLM_ANALYSIS_CLIP_SECONDS / 2

            start_frame_idx = int(clip_start_timestamp * fps)
            end_frame_idx = int(clip_end_timestamp * fps)

            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

            for i in range(start_frame_idx, end_frame_idx):
                ret, frame = cap_temp.read()
                if not ret: break
                frames_to_save.append(cv2_to_pil(frame)) # Convert to PIL Image
            cap_temp.release()

        if frames_to_save:
            self.feedback_learner.save_feedback(
                event_data=event_data,
                frame_sequence=frames_to_save,
                vlm_decision=event_data['vlm_decision_text'], # Using text from the VLM display
                human_feedback=feedback_type,
                vlm_explanation=event_data['vlm_explanation_text'] # Using text from the VLM display
            )
        else:
            logger.error("Could not retrieve frames for feedback saving.")

    def on_processing_finished(self):
        logger.info("Video processing thread finished.")
        self.play_pause_btn.setChecked(False)
        self.play_pause_btn.setText("▶️ Play")
        # Ensure slider is at end if video finished naturally
        if self.processing_thread and self.processing_thread.cap:
             self.video_position_slider.setValue(int(self.processing_thread.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))

    def closeEvent(self, event):
        self.stop_all_processing()
        self.anomaly_consumer.stop_consuming()
        self.anomaly_consumer_thread.quit()
        self.anomaly_consumer_thread.wait()
        super().closeEvent(event)



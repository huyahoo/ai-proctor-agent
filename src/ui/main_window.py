import sys
import os
import time
import queue
import cv2
import numpy as np
from PIL import Image
import json

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QFrame, QSlider, QMessageBox, QSizePolicy, QGridLayout, QStyle, QScrollArea, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer # QTimer for video playback synchronization
from PyQt6.QtGui import QIcon, QColor

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
from core.utils import assign_yolo_pids, assign_pose_pids


# Thread for processing video frames, running CV models, and emitting frames/anomalies
class VideoProcessingThread(QThread):
    # Signals to send processed frames and anomaly events to the main thread
    frame_update = pyqtSignal(dict) # Contains original, yolo, pose, gaze frames & timestamp
    anomaly_detected = pyqtSignal(dict) # Anomaly event data
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str) # Signal for critical errors

    def __init__(self, video_path: str, config: Config, is_example: bool = False):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.is_example = is_example
        self.stop_flag = False
        self.pause_flag = False
        self.current_frame_idx = 0
        self.cap = None

        # Initialize CV detectors
        self.yolo_detector = YOLODetector(config)
        self.pose_estimator = PoseEstimator(config)
        self.gaze_tracker = GazeTracker(config)
        self.anomaly_detector = AnomalyDetector(config)

    def run(self):
        """Dispatches to the correct run method based on the mode."""
        try:
            if self.is_example:
                self.run_from_json()
            else:
                self.run_live_detection()
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Critical error in video processing thread: {e}\n{error_trace}")
            self.processing_error.emit(f"A critical error occurred in the video processing thread:\n\n{e}\n\nPlease check the logs for the full traceback.")

    def run_from_json(self):
        """Runs video processing by reading detection data from a JSON file."""
        logger.step(f"Running in example mode from: {self.config.EXAMPLE_DETECTIONS_PATH}")
        try:
            with open(self.config.EXAMPLE_DETECTIONS_PATH, 'r') as f:
                detection_data = json.load(f)
            # Create a dictionary for quick lookup by frame_idx
            detections_by_frame = {item['frame_idx']: item for item in detection_data.get('data', [])}
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse example detections JSON: {e}")
            self.processing_finished.emit()
            return

        self.cap = load_video_capture(self.video_path)
        if not self.cap:
            logger.error("Failed to open video in processing thread for example mode.")
            self.processing_finished.emit()
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        while not self.stop_flag:
            if self.pause_flag:
                self.msleep(100)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break # End of video

            current_timestamp_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Get pre-computed results from the JSON data
            frame_detections = detections_by_frame.get(self.current_frame_idx)

            if frame_detections:
                # Use pre-computed data instead of running live detectors
                yolo_detections = frame_detections.get('yolo_detections', [])
                pose_estimations = frame_detections.get('pose_estimations', [])
                gaze_estimations = frame_detections.get('gaze_estimations', [])
                anomalies = frame_detections.get('anomalies', [])
                
                # Generate visualization frames
                yolo_viz_frame = self.yolo_detector.draw_results(frame.copy(), yolo_detections)
                pose_viz_frame = self.pose_estimator.draw_results(frame.copy(), pose_estimations)
                gaze_viz_frame = self.gaze_tracker.draw_results(frame.copy(), gaze_estimations)

                self.frame_update.emit({
                    'original': frame.copy(), 'yolo': yolo_viz_frame, 'pose': pose_viz_frame,
                    'gaze': gaze_viz_frame, 'timestamp': current_timestamp_sec, 'frame_idx': self.current_frame_idx
                })

                # Emit any anomalies found in the JSON for this frame
                for anomaly in anomalies:
                    anomaly['video_path'] = self.video_path
                    anomaly['event_id'] = f"{anomaly['type']}_{int(current_timestamp_sec*1000)}_{np.random.randint(1000, 9999)}"
                    self.anomaly_detected.emit(anomaly)

            # Control playback speed
            time.sleep(1 / self.config.FPS)

        self.cap.release()
        self.processing_finished.emit()

    def run_live_detection(self):
        """Runs video processing by performing live detection on each frame."""
        logger.step("Running in live detection mode.")
        self.cap = load_video_capture(self.video_path)
        if not self.cap:
            logger.error("Failed to open video in processing thread.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx) # Set initial position

        while not self.stop_flag:
            if self.pause_flag:
                self.msleep(100)
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

                # Assign pids to yolo detections and pose estimations
                yolo_detections = assign_yolo_pids(yolo_detections, gaze_estimations)
                pose_estimations = assign_pose_pids(pose_estimations, gaze_estimations)

                # Generate CV visualization frames by drawing on a COPY of the original frame
                # Pass the original frame to the draw_results methods
                yolo_viz_frame = self.yolo_detector.draw_results(frame.copy(), yolo_detections)
                pose_viz_frame = self.pose_estimator.draw_results(frame.copy(), pose_estimations)
                gaze_viz_frame = self.gaze_tracker.draw_results(frame.copy(), gaze_estimations)

                # Emit all frames and current timestamp for UI display
                self.frame_update.emit({
                    'original': frame.copy(), # Send a copy for the original display
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
                    self.anomaly_detected.emit(anomaly) 

            # Control processing speed to match desired FPS / FRAME_SKIP
            # This sleep is meant to slow down processing if it's faster than target FPS / FRAME_SKIP.
            # If processing is *slower*, this sleep will effectively be 0 or negative (and capped at 0).
            # The actual FPS will be limited by the slowest part of the pipeline.
            time.sleep(1 / (self.config.FPS / self.config.FRAME_SKIP)) 

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
                logger.info(f"AnomalyConsumer: Processing event {event_data.get('type')} at {event_data.get('timestamp'):.2f}s")

                # Step 1: LLM generates constraints
                llm_constraint = self.llm_generator.generate_constraints(event_data)
                self.llm_constraint_generated.emit(llm_constraint)
                logger.success(f"\nLLM generated constraint: {llm_constraint}")

                # Step 2: VLM analyzes frames with constraints
                # Retrieve a short sequence of original frames around the timestamp for VLM
                frames_for_vlm = []
                video_path = event_data['video_path']
                timestamp_sec = event_data['timestamp']

                cap_temp = load_video_capture(video_path)
                if cap_temp:
                    fps = cap_temp.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        # As per request: clip starts 2s before event, lasts for VLM_ANALYSIS_CLIP_SECONDS,
                        # and samples 2 frames per second.
                        seconds_before = 2.0
                        vlm_clip_duration = self.config.VLM_ANALYSIS_CLIP_SECONDS
                        seconds_after = vlm_clip_duration - seconds_before
                        vlm_frames_per_sec = 2

                        if seconds_after < 0:
                            logger.warning(f"VLM_ANALYSIS_CLIP_SECONDS ({vlm_clip_duration}s) is less than 2s. Adjusting clip to be only before the event.")
                            seconds_after = 0

                        # Calculate start and end frame indices for the entire clip
                        start_frame_idx = max(0, int((timestamp_sec - seconds_before) * fps))
                        end_frame_idx = int((timestamp_sec + seconds_after) * fps)

                        # Calculate how many frames to skip to achieve the desired sampling rate
                        frame_step = int(fps / vlm_frames_per_sec)
                        if frame_step < 1:
                            frame_step = 1 # Ensure we always advance, even for low FPS videos

                        # Seek to and read each sampled frame
                        for frame_to_grab in range(start_frame_idx, end_frame_idx, frame_step):
                            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_to_grab)
                            ret, frame = cap_temp.read()
                            if ret:
                                frames_for_vlm.append(cv2_to_pil(frame))
                            else:
                                break # Reached end of video
                    else:
                        logger.error("Could not determine FPS for VLM frame extraction.")

                    cap_temp.release()
                else:
                    logger.error(f"VLM: Failed to open video {video_path} for VLM frame extraction.")

                vlm_result = self.vlm_analyzer.analyze_and_explain(frames_for_vlm, llm_constraint)
                logger.success(f"VLM Response: {vlm_result}")

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
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = Config()
        self.video_widgets: dict[str, VideoOutputWidget] = {}
        self.processing_thread: VideoProcessingThread | None = None
        self.current_video_path: str | None = None
        self.seeking = False
        self.is_example_mode = False

        self._init_logic_components()
        self._init_ui()
        self._init_connections()
        self._start_anomaly_consumer()

    def _init_ui(self) -> None:
        self._setup_main_window()
        self._init_widgets()
        self._init_layouts()

    def _setup_main_window(self) -> None:
        self.setWindowTitle("Proctor Agent")
        self.setGeometry(100, 100, 1800, 1000)
        self.setStyleSheet("""
            QMainWindow { background-color: #d7d7d7; }
            QWidget { color: #d7d7d7; font-family: Montserrat, sans-serif; }
            
            QWidget#MainView {
                background-color: #FFFFFF;
                border-radius: 8px;
            }

            QPushButton {
                background-color: #4CC764; border: none; color: white;
                padding: 10px 24px; font-size: 14px; margin: 4px 2px;
                border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #06C755; }
            QPushButton:disabled { background-color: #d7d7d7; color: #888; }
            
            QSlider::groove:horizontal {
                border-radius: 4px;
                height: 8px;
                margin: 0px;
                background-color: #d7d7d7;
            }
            QSlider::handle:horizontal {
                background-color: #06C755;
                border: none;
                height: 16px;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                border-radius: 4px;
                background-color: #06C755;
            }
        """)

    def _init_widgets(self) -> None:
        self.video_widgets = {
            "Original": VideoOutputWidget("Original Input"),
            "YOLO": VideoOutputWidget("YOLO Detections"),
            "Pose": VideoOutputWidget("Pose Estimation"),
            "Gaze": VideoOutputWidget("Gaze Tracking")
        }
        self.alert_panel = AlertPanel()
        self.play_pause_btn = QPushButton(" Play")
        self.play_pause_btn.setCheckable(True)
        self.stop_btn = QPushButton(" Stop")
        self.load_video_btn = QPushButton(" Load Video")
        self.select_example_btn = QPushButton(" Example Video")
        self.video_position_slider = QSlider(Qt.Orientation.Horizontal)
        self._setup_button_icons()

    def _init_layouts(self) -> None:
        # Component 1: Video Grid
        video_grid = self._create_video_grid()
        video_grid_container = QWidget()
        video_grid_container.setObjectName("MainView")  # Use same style as before
        video_grid_layout = QVBoxLayout(video_grid_container)
        video_grid_layout.setContentsMargins(15, 15, 15, 15)
        video_grid_layout.addWidget(video_grid)
        self._apply_shadow(video_grid_container)

        # Component 3: Controls Block
        controls = self._create_controls_layout()
        controls_block = QWidget()
        controls_block.setObjectName("MainView")  # Use same style for a consistent block look
        controls_block_layout = QVBoxLayout(controls_block)
        controls_block_layout.setContentsMargins(15, 15, 15, 15)
        controls_block_layout.addWidget(self.video_position_slider)
        controls_block_layout.addLayout(controls)
        self._apply_shadow(controls_block)

        # Left side panel containing video and controls
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(10)
        left_panel_layout.addWidget(video_grid_container, 1)  # Video grid takes most space
        left_panel_layout.addWidget(controls_block)

        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # External margin
        main_layout.setSpacing(10)  # Spacing between panels

        # Add the left panel and the alert panel (Component 2)
        main_layout.addWidget(left_panel_widget, 2)
        main_layout.addWidget(self.alert_panel, 1)

    def _apply_shadow(self, widget: QWidget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(5)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(215,215,215, 80))
        widget.setGraphicsEffect(shadow)

    def _init_logic_components(self):
        self.llm_generator = LLMConstraintGenerator(self.config)
        self.vlm_analyzer = VLMAnalyzer(self.config)
        self.feedback_learner = FeedbackLearner(self.config)
        self.anomaly_event_queue = queue.Queue()

    def _init_connections(self):
        self.load_video_btn.clicked.connect(self.load_video_file)
        self.select_example_btn.clicked.connect(self.load_example_video)
        self.play_pause_btn.toggled.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop_all_processing)
        self.alert_panel.feedback_provided.connect(self.on_feedback_provided)
        self.alert_panel.analysis_requested.connect(self._request_ai_analysis)
        self.video_position_slider.sliderMoved.connect(self._on_slider_moved)
        self.video_position_slider.sliderReleased.connect(self._on_slider_released)

    def _start_anomaly_consumer(self):
        self.anomaly_consumer_thread = QThread()
        self.anomaly_consumer = AnomalyConsumer(self.anomaly_event_queue, self.llm_generator, self.vlm_analyzer, self.config)
        self.anomaly_consumer.moveToThread(self.anomaly_consumer_thread)
        self.anomaly_consumer_thread.started.connect(self.anomaly_consumer.run)
        self.anomaly_consumer.llm_constraint_generated.connect(self.alert_panel.update_llm_constraint)
        self.anomaly_consumer.vlm_analysis_complete.connect(self.on_vlm_analysis_complete)
        self.anomaly_consumer_thread.start()
        logger.success("Anomaly consumer thread started.")

    def _request_ai_analysis(self, event_data: dict):
        """Queues a specific event for AI analysis and seeks video if in example mode."""
        logger.info(f"User requested AI analysis for event: {event_data.get('event_id')}")
        self.anomaly_event_queue.put(event_data)

        # TODO: Remove if application is untable when seeking to the event's timestamp
        # If in example mode, also seek the video to the event's timestamp
        if self.is_example_mode and self.processing_thread:
            timestamp = event_data.get('timestamp')
            if timestamp is not None and self.current_video_path:
                cap = load_video_capture(self.current_video_path)
                if cap:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        frame_idx = int(timestamp * fps)
                        logger.info(f"Seeking to frame {frame_idx} for event at {timestamp:.2f}s.")
                        self.processing_thread.set_frame_position(frame_idx)
                        # If paused, resume playback to see the event in motion
                        if not self.play_pause_btn.isChecked():
                            self.play_pause_btn.setChecked(True)
                    cap.release()

    def _start_video_and_processing(self, video_path: str, is_example: bool = False):
        self.is_example_mode = is_example
        self.stop_all_processing()
        self.current_video_path = video_path
        cap = load_video_capture(video_path)
        if not cap:
            QMessageBox.critical(self, "Error", f"Could not load video: {video_path}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_position_slider.setMaximum(total_frames - 1)
        cap.release()

        self.processing_thread = VideoProcessingThread(video_path, self.config, is_example=is_example)
        self.processing_thread.frame_update.connect(self.on_frame_update)
        self.processing_thread.anomaly_detected.connect(self.on_anomaly_detected)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.start()
        self.play_pause_btn.setChecked(True)
        logger.step(f"Started processing for: {video_path}")

    def stop_all_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            self.processing_thread.wait()
        self.processing_thread = None
        self.current_video_path = None
        for widget in self.video_widgets.values():
            widget.clear()
        self.alert_panel.clear_all_events()
        self.video_position_slider.setValue(0)
        self.play_pause_btn.setChecked(False)
        logger.step("Stopped all processing and reset UI.")

    # --- UI Creation Helpers ---
    def _create_video_grid(self) -> QWidget:
        grid_widget = QWidget()
        layout = QGridLayout(grid_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.video_widgets["Original"], 0, 0)
        layout.addWidget(self.video_widgets["YOLO"], 0, 1)
        layout.addWidget(self.video_widgets["Pose"], 1, 0)
        layout.addWidget(self.video_widgets["Gaze"], 1, 1)
        return grid_widget

    def _create_controls_layout(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.load_video_btn)
        layout.addWidget(self.select_example_btn)
        layout.addStretch(1)
        layout.addWidget(self.play_pause_btn)
        layout.addWidget(self.stop_btn)
        return layout

    def _setup_button_icons(self):
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.load_video_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon))
        self.select_example_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon))

    # --- Event Handlers / Slots ---
    def load_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if path: self._start_video_and_processing(path, is_example=False)

    def load_example_video(self):
        path = self.config.EXAMPLE_VIDEO_PATH
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"Example video not found: {path}")
            return
        self._start_video_and_processing(path, is_example=True)

    def toggle_play_pause(self, checked: bool):
        if not self.current_video_path:
            self.play_pause_btn.setChecked(False)
            return
        if self.processing_thread:
            if checked:
                self.play_pause_btn.setText(" Pause")
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
                self.processing_thread.resume_processing()
            else:
                self.play_pause_btn.setText(" Play")
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                self.processing_thread.pause_processing()

    def on_frame_update(self, frame_data: dict):
        if self.seeking: return
        self.video_widgets["Original"].display_frame(frame_data['original'])
        self.video_widgets["YOLO"].display_frame(frame_data['yolo'])
        self.video_widgets["Pose"].display_frame(frame_data['pose'])
        self.video_widgets["Gaze"].display_frame(frame_data['gaze'])
        self.video_position_slider.setValue(frame_data['frame_idx'])

    def on_anomaly_detected(self, event_data: dict):
        """Adds a detected anomaly to the alert panel list without processing it."""
        logger.info(f"Anomaly detected: {event_data.get('type')} at {event_data.get('timestamp'):.2f}s")
        self.alert_panel.add_event(event_data)

    def on_vlm_analysis_complete(self, data: dict):
        self.alert_panel.update_vlm_result(data)

    def on_processing_error(self, error_message: str):
        """Shows a critical error message and closes the application."""
        QMessageBox.critical(self, "Processing Error", error_message)
        self.close()

    def on_feedback_provided(self, event_data: dict, feedback_type: str):
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
        logger.info("Video processing finished.")
        self.play_pause_btn.setChecked(False)
        self.play_pause_btn.setText(" Play")
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def _on_slider_moved(self, position: int):
        self.seeking = True

    def _on_slider_released(self):
        if self.current_video_path:
            pos = self.video_position_slider.value()
            if self.processing_thread:
                self.processing_thread.set_frame_position(pos)
            else: # If paused, manually update the original view
                cap = load_video_capture(self.current_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret: self.video_widgets["Original"].display_frame(frame)
                cap.release()
        self.seeking = False

    def closeEvent(self, event):
        self.stop_all_processing()
        if hasattr(self, 'anomaly_consumer'): self.anomaly_consumer.stop_consuming()
        if hasattr(self, 'anomaly_consumer_thread'): self.anomaly_consumer_thread.quit(); self.anomaly_consumer_thread.wait()
        super().closeEvent(event)



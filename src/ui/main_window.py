from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QCheckBox, QLabel, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
# from PyQt6.QtGui import QIcon # This seems unused

from ui.video_player import VideoPlayer
from ui.alert_panel import AlertPanel
from cv.yolo_detector import YOLODetector
from cv.pose_estimator import PoseEstimator
from cv.gaze_tracker import GazeTracker
from ai.anomaly_detector import AnomalyDetector
from ai.llm_constraint_generator import LLMConstraintGenerator
from ai.vlm_analyzer import VLMAnalyzer
from ai.feedback_learner import FeedbackLearner
from core.config import Config
# from core.utils import get_video_frame_at_timestamp # This seems unused
import cv2
from PIL import Image
import time
import queue  # For thread-safe event handling
import os


class VideoProcessingThread(QThread):
    # Signals to update UI
    cv_results_ready = pyqtSignal(dict)  # YOLO, Pose, Gaze data for current frame
    anomaly_detected = pyqtSignal(dict)  # Anomaly event data
    processing_finished = pyqtSignal()

    def __init__(self, video_path, config: Config):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.stop_flag = False
        self.yolo_detector = YOLODetector(config)
        self.pose_estimator = PoseEstimator(config)
        self.gaze_tracker = GazeTracker(config)
        self.anomaly_detector = AnomalyDetector(config)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Failed to open video in processing thread.")
            self.processing_finished.emit()
            return

        frame_count = 0
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.config.FRAME_SKIP == 0:
                current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Perform CV detections
                yolo_detections = self.yolo_detector.detect(frame)
                pose_estimations = self.pose_estimator.detect(frame)
                gaze_estimations = self.gaze_tracker.detect(frame)

                cv_data = {
                    'yolo': yolo_detections,
                    'pose': pose_estimations,
                    'gaze': gaze_estimations
                }
                self.cv_results_ready.emit(cv_data)  # Send CV data to main thread for overlays

                # Detect anomalies
                frame_info_for_anomaly = {
                    'yolo_detections': yolo_detections,
                    'pose_estimations': pose_estimations,
                    'gaze_estimations': gaze_estimations,
                    'frame_width': frame.shape[1],
                    'frame_height': frame.shape[0]
                }
                anomalies = self.anomaly_detector.detect_anomalies(frame_info_for_anomaly, current_timestamp_sec)

                for anomaly in anomalies:
                    anomaly['video_path'] = self.video_path  # Add video path for later VLM analysis
                    self.anomaly_detected.emit(anomaly)  # Trigger LLM/VLM process
                self.stop_flag = True
            frame_count += 1
            # Small delay to prevent burning CPU (adjust based on needs)
            time.sleep(1 / self.config.FPS / self.config.FRAME_SKIP)

        cap.release()
        self.processing_finished.emit()

    def stop_processing(self):
        self.stop_flag = True

class ProctorAgentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.llm_generator = LLMConstraintGenerator(self.config)
        self.vlm_analyzer = VLMAnalyzer(self.config)
        self.feedback_learner = FeedbackLearner(self.config)
        self.processing_thread = None
        self.anomaly_event_queue = queue.Queue()  # Thread-safe queue for events

        self.setWindowTitle("Proctor Agent - AI Cheating Detection")
        self.setGeometry(100, 100, 1400, 800)  # Wider window for layout

        self._init_ui()
        self._init_connections()

        # Start a QThread for consuming anomaly events from queue
        self.anomaly_consumer_thread = QThread()
        self.anomaly_consumer = AnomalyConsumer(self.anomaly_event_queue, self.llm_generator, self.vlm_analyzer)
        self.anomaly_consumer.moveToThread(self.anomaly_consumer_thread)
        self.anomaly_consumer.llm_constraint_generated.connect(self.alert_panel.update_llm_constraint)
        self.anomaly_consumer.vlm_analysis_complete.connect(self.on_vlm_analysis_complete)
        self.anomaly_consumer_thread.started.connect(self.anomaly_consumer.run)
        self.anomaly_consumer_thread.start()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel: Video Player and Controls
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setLayout(QVBoxLayout())
        left_panel.setMinimumWidth(800)  # Give more space to video

        # Video Player
        self.video_player = VideoPlayer()
        left_panel.layout().addWidget(self.video_player)

        # Video Controls
        controls_layout = QHBoxLayout()
        self.load_video_btn = QPushButton("📂 Load Video")
        self.select_example_btn = QPushButton("📚 Example Video")
        self.play_pause_btn = QPushButton("▶️ Play")
        self.play_pause_btn.setCheckable(True)  # Toggle button for play/pause
        self.play_pause_btn.setChecked(True)  # Start in play mode initially
        self.stop_btn = QPushButton("⏹️ Stop")

        controls_layout.addWidget(self.load_video_btn)
        controls_layout.addWidget(self.select_example_btn)
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.stop_btn)
        left_panel.layout().addLayout(controls_layout)

        # Overlay Checkboxes
        overlay_layout = QHBoxLayout()
        self.yolo_checkbox = QCheckBox("Show YOLO")
        self.pose_checkbox = QCheckBox("Show Pose")
        self.gaze_checkbox = QCheckBox("Show Gaze")

        overlay_layout.addWidget(self.yolo_checkbox)
        overlay_layout.addWidget(self.pose_checkbox)
        overlay_layout.addWidget(self.gaze_checkbox)
        left_panel.layout().addLayout(overlay_layout)

        main_layout.addWidget(left_panel)

        # Right Panel: Alert and AI Reasoning
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setLayout(QVBoxLayout())
        self.alert_panel = AlertPanel()
        right_panel.layout().addWidget(self.alert_panel)
        main_layout.addWidget(right_panel)

    def _init_connections(self):
        self.load_video_btn.clicked.connect(self.load_video_file)
        self.select_example_btn.clicked.connect(self.load_example_video)
        self.play_pause_btn.toggled.connect(self.toggle_play_pause)
        self.stop_btn.clicked.connect(self.stop_all_processing)

        self.yolo_checkbox.toggled.connect(self.video_player.toggle_yolo_overlay)
        self.pose_checkbox.toggled.connect(self.video_player.toggle_pose_overlay)
        self.gaze_checkbox.toggled.connect(self.video_player.toggle_gaze_overlay)

        self.video_player.frame_ready.connect(self.on_frame_ready_for_cv_processing)

        self.alert_panel.feedback_provided.connect(self.on_feedback_provided)

    def on_cv_results_ready(self, cv_data: dict):
        """Unpacks CV data and passes it to the video player overlays."""
        self.video_player.set_overlays(
            yolo_data=cv_data.get('yolo', []),
            pose_data=cv_data.get('pose', []),
            gaze_data=cv_data.get('gaze', [])
        )

    def load_video_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov)")
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self._start_video_and_processing(selected_file)

    def load_example_video(self):
        # Assumes 'data/videos/example.mp4' exists
        example_path = "data/videos/example_exam.mp4"
        if not os.path.exists(example_path):
            QMessageBox.warning(self, "Missing File", f"Example video not found at: {example_path}\nPlease place a video there or use 'Load Video'.")
            return
        self._start_video_and_processing(example_path)

    def _start_video_and_processing(self, video_path):
        self.stop_all_processing()  # Ensure previous processes are stopped
        if self.video_player.load_video(video_path):
            self.play_pause_btn.setChecked(True)  # Set to play state
            self.play_pause_btn.setText("⏸️ Pause")

            self.processing_thread = VideoProcessingThread(video_path, self.config)
            self.processing_thread.cv_results_ready.connect(self.on_cv_results_ready)
            self.processing_thread.anomaly_detected.connect(self.on_anomaly_detected)
            self.processing_thread.processing_finished.connect(self.on_processing_finished)
            self.processing_thread.start()
            print(f"Started processing for: {video_path}")
        else:
            QMessageBox.critical(self, "Error Loading Video", f"Could not load video from {video_path}")

    def toggle_play_pause(self, checked):
        if checked:
            self.video_player.play()
            self.play_pause_btn.setText("⏸️ Pause")
        else:
            self.video_player.pause()
            self.play_pause_btn.setText("▶️ Play")

    def stop_all_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            self.processing_thread.wait()  # Wait for thread to finish
        self.video_player.stop()
        self.play_pause_btn.setChecked(False)
        self.play_pause_btn.setText("▶️ Play")
        # Clear alerts
        # self.alert_panel.clear_all_events() # Implement a clear method in AlertPanel if needed
        print("Stopped all video processing.")

    def on_frame_ready_for_cv_processing(self, frame_np_array, timestamp_sec):
        # This signal is mainly for the video player to display the frame.
        # CV processing runs in a separate thread.
        pass

    def on_anomaly_detected(self, event_data: dict):
        """Called when AnomalyDetector flags an event. Adds it to queue."""
        self.anomaly_event_queue.put(event_data)
        log_msg = (
            f"Anomaly detected and added to queue: "
            f"{event_data.get('type')} at {event_data.get('timestamp'):.2f}s"
        )
        print(log_msg)
        self.alert_panel.add_event(event_data)  # Display initial event in UI

    def on_vlm_analysis_complete(self, data: dict):
        """Called when VLM analysis is complete by the consumer thread."""
        event_data = data['event_data']
        llm_constraint = data['llm_constraint']
        vlm_result = data['vlm_result']

        # Update the UI with the full AI reasoning process
        self.alert_panel.display_ai_reasoning(
            event_data, llm_constraint, vlm_result
        )

    def on_feedback_provided(self, event_data: dict, feedback_type: str):
        """
        Handles user feedback, triggers the FeedbackLearner, and updates the system.
        """
        print(f"Received feedback: {feedback_type} for event at {event_data.get('timestamp_sec')}")

        # Get the actual frame sequence for saving
        video_path = event_data['video_path']
        timestamp_sec = event_data['timestamp_sec']

        # Extract a short sequence of frames around the event for VLM fine-tuning
        # This is a critical step for collecting data for RL
        frames_to_save = []
        cap_temp = cv2.VideoCapture(video_path)
        if cap_temp.isOpened():
            fps = cap_temp.get(cv2.CAP_PROP_FPS)
            # Go back a bit and grab a few seconds
            start_frame_idx = max(0, int((timestamp_sec - 2) * fps))  # 2 seconds before
            end_frame_idx = int((timestamp_sec + 2) * fps)  # 2 seconds after

            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

            for i in range(start_frame_idx, end_frame_idx):
                ret, frame = cap_temp.read()
                if not ret: break
                frames_to_save.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap_temp.release()

        if frames_to_save:
            self.feedback_learner.save_feedback(
                event_data=event_data,
                frame_sequence=frames_to_save,
                vlm_decision=event_data['vlm_decision_text'],  # Using text from the VLM display
                human_feedback=feedback_type,
                vlm_explanation=event_data['vlm_explanation_text']  # Using text from the VLM display
            )
        else:
            print("Could not retrieve frames for feedback saving.")

        # Optional: update the specific event in the alert panel to show it was addressed
        self.alert_panel.update_event_feedback_status(event_data['id'], feedback_type)

        print("Feedback processed and system updated (simulation).")

    def on_processing_finished(self):
        print("Video processing thread has finished.")
        self.play_pause_btn.setChecked(False)
        self.play_pause_btn.setText("▶️ Play")

    def closeEvent(self, event):
        """Ensure all threads are properly shut down on application close."""
        print("Closing application...")
        self.stop_all_processing()
        self.anomaly_consumer.stop_consuming()
        self.anomaly_consumer_thread.quit()
        self.anomaly_consumer_thread.wait()
        event.accept()

# New Thread to consume anomaly events and run LLM/VLM
class AnomalyConsumer(QThread):
    llm_constraint_generated = pyqtSignal(str)
    vlm_analysis_complete = pyqtSignal(dict)  # Contains event_data, llm_constraint, vlm_result

    def __init__(self, event_queue: queue.Queue, llm_generator: LLMConstraintGenerator,
                 vlm_analyzer: VLMAnalyzer):
        super().__init__()
        self.event_queue = event_queue
        self.llm_generator = llm_generator
        self.vlm_analyzer = vlm_analyzer
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                event_data = self.event_queue.get(timeout=1)  # Wait for 1 second
                print(f"Consumer: Processing event {event_data.get('type')} at {event_data.get('timestamp'):.2f}s")

                # Step 1: LLM generates constraints
                llm_constraint = self.llm_generator.generate_constraints(event_data)
                self.llm_constraint_generated.emit(llm_constraint)  # Update UI
                print(f"Consumer: LLM generated constraint: {llm_constraint}")

                # Step 2: VLM analyzes frames with constraints
                # Need to retrieve frames around the timestamp
                video_path = event_data['video_path']
                timestamp_sec = event_data['timestamp']

                frames_for_vlm = []
                cap_temp = cv2.VideoCapture(video_path)
                if cap_temp.isOpened():
                    fps = cap_temp.get(cv2.CAP_PROP_FPS)
                    start_frame_idx = max(0, int((timestamp_sec - 1) * fps))  # 1 sec before
                    end_frame_idx = int((timestamp_sec + 1) * fps)  # 1 sec after, total ~2 sec clip

                    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                    for i in range(start_frame_idx, end_frame_idx):
                        ret, frame = cap_temp.read()
                        if not ret: break
                        frames_for_vlm.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))  # Convert to PIL Image
                    cap_temp.release()
                else:
                    print(f"Consumer: Failed to open video {video_path} for VLM frame extraction.")

                vlm_result = self.vlm_analyzer.analyze_and_explain(frames_for_vlm, llm_constraint)
                print(f"Consumer: VLM analysis complete: {vlm_result}")

                self.vlm_analysis_complete.emit({'event_data': event_data, 'llm_constraint': llm_constraint, 'vlm_result': vlm_result})

                self.event_queue.task_done()  # Mark task as done
            except queue.Empty:
                continue  # Continue waiting for events
            except Exception as e:
                print(f"Consumer Thread Error: {e}")
                self.event_queue.task_done()  # Ensure task is marked done even on error

    def stop_consuming(self):
        self.stop_flag = True



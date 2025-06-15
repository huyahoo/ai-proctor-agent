import mediapipe as mp
import numpy as np
import cv2
from cv.base_detector import BaseDetector
from core.config import Config
from core.utils import create_blank_frame, draw_gaze
from core.logger import logger

class GazeTracker(BaseDetector):
    def __init__(self, config: Config):
        super().__init__(config)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2, # Detect up to 2 faces for interaction analysis
            min_detection_confidence=self.config.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=self.config.POSE_MIN_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 3D model points for head pose estimation (simplified based on a generic face model)
        # These points are indices of relevant landmarks in the MediaPipe Face Mesh model
        # Using a subset for stability and relevance to head pose.
        # This is a general approach, exact mapping to MediaPipe landmarks is crucial.
        # Common points: Nose tip (1), Chin (152), Left Eye (33), Right Eye (263), Left Mouth (61), Right Mouth (291)
        # Note: Mediapipe Face Mesh has more detailed landmarks, choosing a robust subset for PnP
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, -330.0, -65.0),        # Chin (index 152)
            (-225.0, 170.0, -135.0),     # Left eye corner (index 33, approx)
            (225.0, 170.0, -135.0),      # Right eye corner (index 263, approx)
            (-150.0, -150.0, -125.0),    # Left mouth corner (index 61, approx)
            (150.0, -150.0, -125.0)      # Right mouth corner (index 291, approx)
        ], dtype=np.double)

        # Landmark indices corresponding to model_points for MediaPipe Face Mesh
        # These are approximate and should be refined with actual landmark definitions if precision is critical
        self.mesh_points_indices = [1, 152, 33, 263, 61, 291]

        # Camera intrinsic parameters (will be updated per frame for dynamic resolution)
        self.focal_length_factor = 1.0 # Heuristic, will scale with frame width
        self.camera_matrix = np.array([
            [self.focal_length_factor, 0, 0],
            [0, self.focal_length_factor, 0],
            [0, 0, 1]
        ], dtype=np.double)
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion (k1, k2, p1, p2)

    def detect(self, frame: np.ndarray) -> list:
        """
        Detects faces and estimates head pose (for simplified gaze) in a frame.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'head_pose': [nose_x, nose_y, pitch_deg, yaw_deg, roll_deg]}, ...]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        gaze_data = []
        if results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Update camera intrinsics based on current frame size
            self.camera_matrix[0, 0] = self.focal_length_factor * w # fx = focal_length * image_width
            self.camera_matrix[1, 1] = self.focal_length_factor * w # fy = fx (assuming square pixels)
            self.camera_matrix[0, 2] = w / 2 # cx = image_width / 2
            self.camera_matrix[1, 2] = h / 2 # cy = image_height / 2

            for face_landmarks in results.multi_face_landmarks:
                image_points = []
                for idx in self.mesh_points_indices:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        image_points.append((lm.x * w, lm.y * h))
                    else:
                        # Fallback if landmark index is out of bounds (shouldn't happen with correct indices)
                        image_points.append((0,0))
                image_points = np.array(image_points, dtype=np.double)

                if len(image_points) != len(self.model_points):
                    logger.warning(f"Warning: Mismatch in model points ({len(self.model_points)}) and image points ({len(image_points)}). Skipping face.")
                    continue

                # Solve for pose
                try:
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(
                        self.model_points, image_points, self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    # Convert rotation vector to rotation matrix and then Euler angles
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat) # angles are (pitch, yaw, roll) in radians

                    pitch_deg = np.degrees(angles[0])
                    yaw_deg = np.degrees(angles[1])
                    roll_deg = np.degrees(angles[2])

                    # Get nose tip for drawing direction
                    nose_tip = (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h)

                    # Calculate bounding box for the face
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    if x_coords and y_coords:
                        x_min, y_min = min(x_coords), min(y_coords)
                        x_max, y_max = max(x_coords), max(y_coords)
                        bbox = [x_min, y_min, x_max, y_max]
                    else:
                        bbox = [0,0,0,0]

                    gaze_data.append({
                        'bbox': bbox,
                        'head_pose': [nose_tip[0], nose_tip[1], pitch_deg, yaw_deg, roll_deg] # Storing in degrees
                    })
                except cv2.error as e:
                    logger.error(f"Head pose estimation solvePnP error: {e}")
                    continue

        return gaze_data

    def draw_results(self, frame_shape: tuple, gaze_data: list) -> np.ndarray:
        """
        Draws gaze estimation (head pose vectors) on a blank frame.
        Args:
            frame_shape (tuple): (height, width, channels) of the original frame.
            gaze_data (list): Output from self.detect method.
        Returns:
            np.ndarray: A new frame with only gaze detections.
        """
        display_frame = create_blank_frame(frame_shape[1], frame_shape[0])
        for data in gaze_data:
            if data['head_pose']:
                draw_gaze(display_frame, data['head_pose'], color=(255, 0, 0)) # Red color for gaze
        return display_frame


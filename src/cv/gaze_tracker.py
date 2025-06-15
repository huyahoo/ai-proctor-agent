import mediapipe as mp
import numpy as np
import cv2
from cv.base_detector import BaseDetector
from core.config import Config

class GazeTracker(BaseDetector):
    def __init__(self, config: Config):
        super().__init__(config)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2, # Detect up to 2 faces for interaction
            min_detection_confidence=self.config.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=self.config.POSE_MIN_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL

        # 3D model points for head pose estimation (simplified)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.double)

        self.focal_length = 1 * frame.shape[1] if 'frame' in locals() else 1000 # Placeholder, calibrate this!
        self.center = (frame.shape[1]/2, frame.shape[0]/2) if 'frame' in locals() else (500,500) # Placeholder
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype=np.double)
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    def detect(self, frame):
        """
        Detects faces and estimates head pose (for simplified gaze) in a frame.
        Returns a list of dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'head_pose': [nose_x, nose_y, pitch, yaw, roll]}, ...]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        gaze_data = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = frame.shape

                # Extract image points for head pose estimation (simplified points from face_mesh)
                # You'd need specific landmark indices for precise eye/nose/mouth corners
                # For demo, let's use a few prominent landmarks
                image_points = np.array([
                    (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),       # Nose tip
                    (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),   # Chin
                    (face_landmarks.landmark[226].x * w, face_landmarks.landmark[226].y * h),   # Left eye corner (approx)
                    (face_landmarks.landmark[446].x * w, face_landmarks.landmark[446].y * h),   # Right eye corner (approx)
                    (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h),     # Left mouth corner (approx)
                    (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h)    # Right mouth corner (approx)
                ], dtype=np.double)

                # Update camera matrix and center based on current frame size
                self.focal_length = w * 1.0 # Simple approximation
                self.center = (w/2, h/2)
                self.camera_matrix = np.array([
                    [self.focal_length, 0, self.center[0]],
                    [0, self.focal_length, self.center[1]],
                    [0, 0, 1]
                ], dtype=np.double)


                # Solve for pose
                try:
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(
                        self.model_points, image_points, self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    # Get rotation matrix and then Euler angles
                    rmat, jac = cv2.Rodrigues(rotation_vector)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x_rot = angles[0] * 180 / np.pi # Pitch
                    y_rot = angles[1] * 180 / np.pi # Yaw
                    z_rot = angles[2] * 180 / np.pi # Roll

                    # Get nose tip for drawing direction
                    nose_tip = (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h)

                    gaze_data.append({
                        'bbox': [min(image_points[:,0]), min(image_points[:,1]), max(image_points[:,0]), max(image_points[:,1])],
                        'head_pose': [nose_tip[0], nose_tip[1], x_rot, y_rot, z_rot]
                    })
                except cv2.error as e:
                    print(f"Head pose estimation error: {e}")
                    continue # Skip this face if PnP fails

        return gaze_data


import cv2
import numpy as np
import os
import json
from PIL import Image # For VLM input processing
from core.logger import logger
from core.constants import COCO_PERSON_SKELETON_INDICES
import warnings
from torch.cuda.amp import autocast

def setup_warning_filters():
    # Filter out torch.cuda.amp.autocast deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message=".*torch.cuda.amp.autocast.*",
        category=FutureWarning
    )
    # Filter out pytree node registration warning
    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils._pytree._register_pytree_node*",
        category=FutureWarning
    )

def load_video_capture(video_path):
    """Loads a video file and returns a cv2.VideoCapture object."""
    if not os.path.exists(video_path):
        logger.error(f"Error: Video file not found at {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file at {video_path}")
        return None
    return cap

def create_blank_frame(width, height, color=(0, 0, 0)):
    """Creates a black frame of specified dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8) + np.array(color, dtype=np.uint8)

def draw_bbox(frame, bbox, label=None, color=(0, 255, 0)):
    """Draws a bounding box on the frame."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_keypoints(frame, keypoints, color=(0, 255, 0), connections=None):
    """
    Draws keypoints and connections on the frame using OpenPifPaf style.
    Args:
        frame: The frame to draw on
        keypoints: List of keypoints in format [[x, y, confidence], ...]
        color: Color for keypoints and connections (BGR format)
        connections: List of connections between keypoints. If None, uses COCO skeleton
    """
    if not keypoints:
        return frame

    keypoints = np.array(keypoints)
    
    # Use COCO skeleton by default if no connections provided
    if connections is None:
        connections = COCO_PERSON_SKELETON_INDICES
        # Use blue for keypoints and cyan for connections in OpenPifPaf style
        keypoint_color = (255, 0, 0)  # Blue
        connection_color = (0, 255, 255)  # Cyan
    else:
        keypoint_color = color
        connection_color = color

    # Draw keypoints
    for x, y, conf in keypoints:
        if conf > 0.2:  # Only draw keypoints with confidence > 0.2
            cv2.circle(frame, (int(x), int(y)), 3, keypoint_color, -1)

    # Draw connections
    for j1, j2 in connections:
        if (j1 < len(keypoints) and j2 < len(keypoints) and 
            keypoints[j1][2] > 0.2 and keypoints[j2][2] > 0.2):
            pt1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
            pt2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
            cv2.line(frame, pt1, pt2, connection_color, 2)

    return frame

COLORS = [
    (199, 21, 133), (0, 128, 0), (30, 144, 255),
    (220, 20, 60), (218, 165, 32), (47, 79, 79),
    (139, 69, 19), (128, 0, 128), (0, 128, 128)
]

def draw_gaze(
    image,
    head_bboxes,
    gaze_points,
    gaze_vecs,
    inouts,
    pids,
    gaze_heatmaps,
    heatmap_pid=None,
    frame_nb=None,
    colors=COLORS,
    filter_by_inout=False,
    alpha=0.5,
    io_thr=0.5,
    gaze_pt_size=10,
    gaze_vec_factor=0.8,
    head_center_size=10,
    thickness=4,
    fs=0.6,
):
    """Draw gaze predictions on the image."""
    img_h, img_w, _ = image.shape
    canvas = image.copy()

    scale = max(img_h, img_w) / 1920
    fs *= scale
    thickness = int(scale * thickness)
    gaze_pt_size = int(scale * gaze_pt_size)
    head_center_size = int(scale * head_center_size)

    # Draw heatmap if applicable
    if heatmap_pid is not None and len(gaze_heatmaps) > 0:
        mask = (pids == heatmap_pid)
        if mask.sum() == 1:
            gaze_heatmap = gaze_heatmaps[mask][0]
            heatmap = TF.resize(gaze_heatmap, (img_h, img_w), antialias=True).squeeze().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            heatmap = (cm.inferno(heatmap) * 255).astype(np.uint8)
            canvas = ((1 - alpha) * image + alpha * heatmap[..., :3]).astype(np.uint8)

            hm_pid_text = f"Heatmap PID: {heatmap_pid}"
            (w_text, h_text), _ = cv2.getTextSize(hm_pid_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ul = (img_w - w_text - 20, img_h - h_text - 15)
            br = (img_w, img_h)
            cv2.rectangle(canvas, ul, br, (0, 0, 0), -1)
            cv2.putText(canvas, hm_pid_text, (img_w - w_text - 10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw gaze and bboxes
    if len(head_bboxes) > 0:
        head_bboxes = np.array(head_bboxes)
        inouts = np.array(inouts)
        if head_bboxes.max() <= 1.0:
            head_bboxes = head_bboxes * np.array([img_w, img_h, img_w, img_h])
        head_bboxes = head_bboxes.astype(int)

        head_centers = np.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) // 2,
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) // 2
        ]).astype(int)

        gaze_available = len(gaze_points) > 0
        if gaze_available:
            gaze_points = np.array(gaze_points)
            if gaze_points.max() <= 1.0:
                gaze_points = gaze_points * np.array([img_w, img_h])
            gaze_points = gaze_points.astype(int)

        if gaze_vecs is not None:
            gaze_vecs = np.array(gaze_vecs)

        for i, head_bbox in enumerate(head_bboxes):
            pid = pids[i]
            if heatmap_pid is not None and heatmap_pid != pid:
                continue

            xmin, ymin, xmax, ymax = head_bbox
            head_radius = max(xmax - xmin, ymax - ymin) // 2
            color = colors[pid % len(colors)]

            head_center = head_centers[i]
            head_center_ul = head_center - (head_center_size // 2)
            head_center_br = head_center + (head_center_size // 2)
            cv2.rectangle(canvas, head_center_ul, head_center_br, color, -1)
            cv2.circle(canvas, head_center, head_radius, color, thickness)

            io = inouts[i]
            header_text = f"P{pid}: {io:.2f}"
            (w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            header_ul = (int(head_center[0] - w_text / 2), int(ymin - thickness / 2))
            header_br = (int(head_center[0] + w_text / 2), int(ymin + h_text + 5))
            cv2.rectangle(canvas, header_ul, header_br, color, -1)
            cv2.putText(canvas, header_text, (header_ul[0], int(ymin + h_text)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

            if gaze_available and (io > io_thr or not filter_by_inout):
                gp = gaze_points[i]
                vec = (gp - head_center)
                vec = vec / (np.linalg.norm(vec) + 1e-6)
                intersection = head_center + (vec * head_radius).astype(int)
                cv2.line(canvas, intersection, gp, color, thickness)
                cv2.circle(canvas, gp, gaze_pt_size, color, -1)

            if gaze_vecs is not None:
                gv = gaze_vecs[i]
                cv2.arrowedLine(canvas, head_center,
                                (head_center + gaze_vec_factor * head_radius * gv).astype(int),
                                color, thickness)

    if frame_nb is not None:
        frame_nb = str(frame_nb)
        (w_text, h_text), _ = cv2.getTextSize(frame_nb, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        nb_ul = (int((img_w - w_text) / 2), (img_h - h_text - 15))
        nb_br = (int((img_w + w_text) / 2), img_h)
        cv2.rectangle(canvas, nb_ul, nb_br, (0, 0, 0), -1)
        cv2.putText(canvas, frame_nb, (int((img_w - w_text) / 2), (img_h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

def cv2_to_pil(cv2_image):
    """Converts an OpenCV image (BGR) to a PIL Image (RGB)."""
    if cv2_image is None:
        return None
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    """Converts a PIL Image (RGB) to an OpenCV image (BGR)."""
    if pil_image is None:
        return None
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



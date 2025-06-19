import cv2
import numpy as np
import os
import json
from PIL import Image # For VLM input processing
from core.logger import logger
from core.constants import COCO_PERSON_SKELETON_INDICES, COCO_PERSON_SKELETON, COLORS, UNAUTHORIZED_CLASSES
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

def iou(boxA, boxB):
    """
    Compute IoU of two bboxes in [x1,y1,x2,y2] format.
    """
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB

    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h

    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter + 1e-6

    return inter / union

def bbox_from_kpts(kpts, conf_th=0.2):
    """
    Compute a tight [x1,y1,x2,y2] box over only those keypoints
    whose confidence > conf_th. Returns None if no keypoints qualify.
    
    Args:
        kpts (List[List[float]]): [[x,y,conf], ...]
        conf_th (float): confidence threshold
        
    Returns:
        List[int] or None: [x1, y1, x2, y2] or None if no valid points
    """
    # filter by confidence
    good = [(x, y) for x, y, c in kpts if c > conf_th]
    if not good:
        return None

    arr = np.array(good)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return [int(x1), int(y1), int(x2), int(y2)]

def assign_yolo_pids(yolo_dets, gaze_data, iou_threshold=0.0):
    """
    Attach person IDs from gaze tracking to YOLO detections by spatial overlap.

    For each detection in `yolo_dets`, this function computes the Intersection-over-Union
    (IoU) against every head bounding box in `gaze_data`. If the highest IoU meets or exceeds
    `iou_threshold`, the detection inherits the corresponding `pid`; otherwise its `pid` is set to -1.

    Args:
        yolo_dets (list of dict): Each dict must include:
            - 'bbox': [x1, y1, x2, y2] coordinates of the detection.
        gaze_data (list of dict): Each dict must include:
            - 'bbox': [x1, y1, x2, y2] of a tracked head.
            - 'pid' : The person ID for that head box.
        iou_threshold (float): Minimum IoU required to assign a `pid`. Defaults to 0.0
            (always pick the highest-overlap ID, even if overlap is tiny).

    Returns:
        list of dict: The same `yolo_dets` list, but each dict now also has a `'pid'` key
        set to the matched person ID or -1 if no match meets the threshold.
    """
    head_boxes = [g['bbox'] for g in gaze_data]
    head_pids  = [g['pid']  for g in gaze_data]

    for det in yolo_dets["person"]:
        best_iou = 0.0
        best_pid = -1
        for hb, pid in zip(head_boxes, head_pids):
            i = iou(det['bbox'], hb)
            if i > best_iou:
                best_iou = i
                best_pid = pid
        
        det['pid'] = best_pid if best_iou >= iou_threshold else -1

    for paper_exam in yolo_dets["exam_paper"]:
        best_paper_iou = 0.0
        best_paper_id = -1
        for det in yolo_dets["person"]:
            i = iou(paper_exam["bbox"], det["bbox"])
            if i > best_paper_iou:
                best_paper_iou = i
                best_paper_id = det["pid"]
        
        paper_exam["pid"] = best_paper_id if best_paper_iou >= iou_threshold else -1

    return yolo_dets

def assign_pose_pids(raw_pose_data, gaze_data, iou_threshold=0.0):
    """
    Aligns pose detections with gaze-tracked person IDs based on spatial overlap.

    For each set of keypoints in `raw_pose_data`, this function first computes a tight
    bounding box around all keypoints using `bbox_from_kpts`. It then measures the
    Intersection-over-Union (IoU) between that box and each head bounding box in `gaze_data`.
    The pose inherits the `pid` of the gaze box with the highest IoU, provided that IoU
    meets or exceeds `iou_threshold`; otherwise the pose’s `pid` is set to –1.

    Args:
        raw_pose_data (List[List[List[float]]]):
            A list of poses, each represented as a list of [x, y, confidence] keypoints.
        gaze_data (List[Dict]):
            A list of gaze detections, each a dict containing:
              - 'bbox': [x1, y1, x2, y2]  head bounding box coordinates
              - 'pid' : int                the person ID for that head
        iou_threshold (float):
            Minimum IoU required to transfer a `pid`. Defaults to 0.0
            (always pick the best match, even if overlap is minimal).

    Returns:
        List[Dict]: A list of dicts, one per input pose, each containing:
            - 'keypoints': the original list of [x, y, confidence] points
            - 'pid'      : the matched person ID or –1 if no match reaches the threshold
    """
    head_boxes = [g['bbox'] for g in gaze_data]
    head_pids  = [g['pid']  for g in gaze_data]

    out = []
    for kpts in raw_pose_data:
        if not kpts:
            out.append({'keypoints': [], 'pid': -1})
            continue

        box = bbox_from_kpts(kpts)
        best_iou, best_pid = 0.0, -1

        for hb, pid in zip(head_boxes, head_pids):
            i = iou(box, hb)
            if i > best_iou:
                best_iou, best_pid = i, pid

        pid_out = best_pid if best_iou >= iou_threshold else -1
        out.append({'keypoints': kpts, 'pid': pid_out})

    return out

def draw_bbox(
    frame,
    bbox,
    label,
    text,
    pid=None,
    colors=COLORS,
    thickness=5,
    font_scale=0.7
) -> None:
    """
    Draws a bounding box on `frame` and overlays `text` just above it.
    
    Args:
      frame:        BGR image.
      bbox:         [x1, y1, x2, y2].
      label:        raw class name (e.g. 'person') → used for color decision.
      text:         Optional text to draw (e.g. 'person: 0.88').
      pid:          Person ID if label=='person'.
      colors:       palette for person boxes.
      thickness:    box line thickness.
      font_scale:   scale for text.
    """
    # 1) pick color
    if label in ['person', 'paper'] and pid is not None:
        clr = colors[pid % len(colors)]
    elif label in UNAUTHORIZED_CLASSES:
        clr = (0, 0, 255) # red
    else:
        clr = (0, 255, 255) # yellow

    # 2) draw the rectangle
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), clr, thickness)

    # 3) overlay text if provided
    if text:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # position the filled background above the box
        bg_tl = (x1, y1 - h - 6)
        bg_br = (x1 + w + 2, y1)
        cv2.rectangle(frame, bg_tl, bg_br, clr, -1)
        # then put the text in white
        text_org = (x1 + 1, y1 - 4)
        cv2.putText(
            frame,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

def draw_keypoints(
    frame,
    keypoints,
    pid= None,
    color=None,
    connections=COCO_PERSON_SKELETON,
    conf_threshold=0.2,
    radius=5,
    thickness=5
) -> None:
    """
    Draws joints and skeleton lines on `frame`.

    Args:
        frame:         BGR image to draw on.
        keypoints:     List of [x, y, confidence] for each joint.
        pid:           Optional person ID; if set and `color` is None,
                       uses COLORS[pid % len(COLORS)].
        color:         Optional BGR tuple to override palette color.
        connections:   List of index‐pairs defining skeleton edges.
        conf_threshold: Minimum confidence to draw a joint.
        radius:        Radius of each joint circle.
        thickness:     Line thickness for skeleton edges.

    Behavior:
      1. If `color` is given, uses it for both joints and bones.
      2. Else if `pid` is given, selects palette[pid % len(palette)].
      3. Otherwise defaults to blue joints / cyan bones.
      4. Draws only joints with confidence > conf_threshold.
      5. Draws bones only where both endpoints exceed conf_threshold.
    """
    if not keypoints:
        return

    pts = np.array(keypoints)

    # Determine drawing color
    if color is not None:
        joint_color = bone_color = color
    elif pid is not None:
        joint_color = (255, 255, 255) # white
        bone_color = COLORS[pid % len(COLORS)]
    else:
        joint_color = (255, 0, 0)    # default blue
        bone_color  = (0, 255, 255)  # default cyan

    # Draw joints
    for x, y, conf in pts:
        if conf > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), radius, joint_color, -1)

    # Draw bones
    for i, j in connections:
        # convert 1-based COCO indices to 0-based
        idx1, idx2 = i-1, j-1
        if idx1 < len(pts) and idx2 < len(pts):
            if pts[idx1,2] > conf_threshold and pts[idx2,2] > conf_threshold:
                p1 = (int(pts[idx1,0]), int(pts[idx1,1]))
                p2 = (int(pts[idx2,0]), int(pts[idx2,1]))
                cv2.line(frame, p1, p2, bone_color, thickness)

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



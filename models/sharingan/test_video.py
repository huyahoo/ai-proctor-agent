"""
Test script to run Sharingan model on a video file to check if it works on new environment.
"""

import os
import sys
import shlex
import shutil
import argparse
import datetime as dt
from tqdm import tqdm
import subprocess as sp
from termcolor import colored

import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from boxmot import OCSORT

from sharingan import Sharingan
from common import spatial_argmax2d, square_bbox

# ================================ ARGS ================================ #
parser = argparse.ArgumentParser(description="Test Sharingan model on video")
parser.add_argument("--input", type=str, required=True, help="Path to input video file")
parser.add_argument("--output", type=str, default="data/videos/test_video.mp4", help="Path to output video file")
parser.add_argument("--heatmap-pid", type=int, default=-1, help="PID of person to draw heatmap for")
parser.add_argument("--show-gaze-vec", action="store_true", help="Show gaze vectors")
parser.add_argument("--filter-by-inout", action="store_true", help="Filter gaze points by inout score")

args = parser.parse_args()

# =============================== GLOBALS =============================== #
TERM_COLOR = "cyan"
COLORS = [(199, 21, 133), (0, 128, 0), (30, 144, 255), (220, 20, 60), (218, 165, 32), 
          (47, 79, 79), (139, 69, 19), (128, 0, 128), (0, 128, 128)]

DET_THR = 0.4  # head detection threshold
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# Model paths
CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints/videoattentiontarget.pt")
YOLO_PATH = os.path.join(os.path.dirname(__file__), "weights/yolov5m_crowdhuman.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(colored(f"Using device: {DEVICE}", TERM_COLOR))

def load_tracker():
    """Load and return the tracker model."""
    return OCSORT()

def load_head_detection_model(device):
    """Load and return the head detection model."""
    model = torch.hub.load("ultralytics/yolov5", "custom", path=YOLO_PATH, verbose=False)
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = [1]  # filter by class, i.e. = [1] for heads
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    model = model.to(device)
    model.eval()
    return model

def detect_heads(image, model):
    """Detect heads in the image using the provided model."""
    detections = model(image, size=640).pred[0].cpu().numpy()[:, :-1]
    return detections

def load_sharingan_model(ckpt_path, device):
    """Load and return the Sharingan model."""
    # Build model
    sharingan = Sharingan(
        patch_size=16,
        token_dim=768,
        image_size=224,
        gaze_feature_dim=512,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_global_tokens=0,
        encoder_mlp_ratio=4.0,
        encoder_use_qkv_bias=True,
        encoder_drop_rate=0.0,
        encoder_attn_drop_rate=0.0,
        encoder_drop_path_rate=0.0,
        decoder_feature_dim=128,
        decoder_hooks=[2, 5, 8, 11],
        decoder_hidden_dims=[48, 96, 192, 384],
        decoder_use_bn=True,
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint = {name.replace("model.", ""): value for name, value in checkpoint["state_dict"].items()}
    sharingan.load_state_dict(checkpoint, strict=True)
    sharingan.eval()
    sharingan.to(device)
    return sharingan

def predict_gaze(image, sharingan, head_detector, tracker=None):
    """Predict gaze for all detected heads in the image."""
    # 1. Convert image
    image_np = np.array(image)
    img_h, img_w, img_c = image_np.shape
 
    # 2. Detect heads
    raw_detections = detect_heads(image_np, head_detector)
    detections = []
    for k, raw_detection in enumerate(raw_detections):
        bbox, conf = raw_detection[:4], raw_detection[4]
        if conf > DET_THR:
            cls_ = np.array([0.])
            detection = np.concatenate([bbox, conf[None], cls_])
            detections.append(detection)
    detections = np.stack(detections) if detections else np.array([])
    
    # 3. Track heads
    tracks = tracker.update(detections, image_np) if len(detections) > 0 else np.array([])
    if len(tracks) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), np.array([])
    
    pids = (tracks[:, 4] - 1).astype(int)
    head_bboxes = torch.from_numpy(tracks[:, :4]).float()
    t_head_bboxes = square_bbox(head_bboxes, img_w, img_h)
    
    # 4. Extract and transform heads
    heads = []
    for bbox in t_head_bboxes:
        head = TF.resize(TF.to_tensor(image.crop(bbox.numpy())), (224, 224))
        heads.append(head)
    heads = torch.stack(heads)
    heads = TF.normalize(heads, mean=IMG_MEAN, std=IMG_STD)

    # 5. Transform image
    image = TF.to_tensor(image)
    image = TF.resize(image, (224, 224))
    image = TF.normalize(image, mean=IMG_MEAN, std=IMG_STD)

    # 6. Normalize head bboxes
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    t_head_bboxes /= scale

    # 7. Build input sample
    sample = {
        "image": image.unsqueeze(0).to(DEVICE),
        "heads": heads.unsqueeze(0).to(DEVICE),
        "head_bboxes": t_head_bboxes.unsqueeze(0).to(DEVICE)
    }

    # 8. Predict gaze
    with torch.no_grad():
        gaze_vecs, gaze_heatmaps, inouts = sharingan(sample)
        gaze_heatmaps = gaze_heatmaps.squeeze(0).cpu()
        gaze_vecs = gaze_vecs.squeeze(0).cpu()
        gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
        inouts = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()
  
    return gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids

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
    # Create canvas
    img_h, img_w, img_c = image.shape
    canvas = image.copy()
    
    # Scale drawing parameters
    scale = max(img_h, img_w) / 1920
    fs *= scale
    thickness = int(scale * thickness)
    gaze_pt_size = int(scale * gaze_pt_size)
    head_center_size = int(scale * head_center_size)
    
    # Draw heatmap if requested
    if heatmap_pid is not None and len(gaze_heatmaps) > 0:
        mask = (pids == heatmap_pid)
        if mask.sum() == 1:
            gaze_heatmap = gaze_heatmaps[mask]
            heatmap = TF.resize(gaze_heatmap, (img_h, img_w), antialias=True).squeeze().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = cm.inferno(heatmap) * 255 
            canvas = ((1 - alpha) * image + alpha * heatmap[..., :3]).astype(np.uint8)
            
            # Add heatmap PID text
            hm_pid_text = f"Heatmap PID: {heatmap_pid}"
            (w_text, h_text), _ = cv2.getTextSize(hm_pid_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ul = (img_w - w_text - 20, img_h - h_text - 15)
            br = (img_w, img_h)
            cv2.rectangle(canvas, ul, br, (0, 0, 0), -1)
            cv2.putText(canvas, hm_pid_text, (img_w - w_text - 10, img_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw head bboxes and gaze
    if len(head_bboxes) > 0:
        # Convert to numpy
        head_bboxes = head_bboxes.numpy() if isinstance(head_bboxes, torch.Tensor) else np.array(head_bboxes)
        inouts = inouts.numpy() if isinstance(inouts, torch.Tensor) else np.array(inouts)
        if head_bboxes.max() <= 1.0:
            head_bboxes = head_bboxes * np.array([img_w, img_h, img_w, img_h])
        head_bboxes = head_bboxes.astype(int)
        
        # Compute head centers
        head_centers = np.hstack([(head_bboxes[:,[0]] + head_bboxes[:,[2]]) / 2,
                                (head_bboxes[:,[1]] + head_bboxes[:,[3]]) / 2])
        head_centers = head_centers.astype(int)
        
        # Process gaze points if available
        gaze_available = len(gaze_points) > 0
        if gaze_available:
            gaze_points = gaze_points.numpy() if isinstance(gaze_points, torch.Tensor) else np.array(gaze_points)
            if gaze_points.max() <= 1.0:
                gaze_points = gaze_points * np.array([img_w, img_h])
            gaze_points = gaze_points.astype(int)
            
        if gaze_vecs is not None:
            gaze_vecs = gaze_vecs.numpy() if isinstance(gaze_vecs, torch.Tensor) else np.array(gaze_vecs)
        
        # Draw each head and its gaze
        for i, head_bbox in enumerate(head_bboxes):
            pid = pids[i]
            if heatmap_pid is not None and heatmap_pid != pid:
                continue
                
            xmin, ymin, xmax, ymax = head_bbox
            head_radius = max(xmax-xmin, ymax-ymin) // 2
            color = colors[pid % len(colors)]
            
            # Draw head center and circle
            head_center = head_centers[i]
            head_center_ul = head_center - (head_center_size // 2)
            head_center_br = head_center + (head_center_size // 2)
            cv2.rectangle(canvas, head_center_ul, head_center_br, color, -1)
            cv2.circle(canvas, head_center, head_radius, color, thickness)
            
            # Draw header with PID and inout score
            io = inouts[i] if inouts is not None else "-"
            header_text = f"P{pid}: {io:.2f}"
            (w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            header_ul = (int(head_center[0] - w_text / 2), int(ymin - thickness / 2))
            header_br = (int(head_center[0] + w_text / 2), int(ymin + h_text + 5))
            cv2.rectangle(canvas, header_ul, header_br, color, -1)
            cv2.putText(canvas, header_text, (header_ul[0], int(ymin + h_text)),
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw gaze point and vector if available
            if gaze_available and (io > io_thr or not filter_by_inout):
                gp = gaze_points[i]
                vec = (gp - head_center)
                vec = vec / (np.linalg.norm(vec) + 0.000001)
                intersection = head_center + (vec * head_radius).astype(int)
                cv2.line(canvas, intersection, gp, color, thickness)
                cv2.circle(canvas, gp, gaze_pt_size, color, -1)
                
            if gaze_vecs is not None:
                gv = gaze_vecs[i]
                cv2.arrowedLine(canvas, head_center,
                              (head_center + gaze_vec_factor * head_radius * gv).astype(int),
                              color, thickness)
    
    # Draw frame number if provided
    if frame_nb is not None:
        frame_nb = str(frame_nb)
        (w_text, h_text), _ = cv2.getTextSize(frame_nb, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        nb_ul = (int((img_w - w_text) / 2), (img_h - h_text - 15))
        nb_br = (int((img_w + w_text) / 2), img_h)
        cv2.rectangle(canvas, nb_ul, nb_br, (0, 0, 0), -1)
        cv2.putText(canvas, frame_nb, (int((img_w - w_text) / 2), (img_h - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

def main():
    start = dt.datetime.now()
    
    # Check input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video file not found: {args.input}")
    
    print(colored(f"Processing {args.input}", TERM_COLOR))

    # Load models
    tracker = load_tracker()
    head_detector = load_head_detection_model(DEVICE)
    sharingan = load_sharingan_model(CKPT_PATH, DEVICE)
    print(colored("Loaded tracker, head detector, and sharingan models.", TERM_COLOR))

    # Open video
    cap = cv2.VideoCapture(args.input)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read video file")
        
    img_h, img_w, _ = frame.shape
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    command = f"ffmpeg -loglevel error -y -s {img_w}x{img_h} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {args.output}"
    command = shlex.split(command)
    process = sp.Popen(command, stdin=sp.PIPE)
    
    # Process frames
    frame_nb = 0
    with tqdm(total=frame_count) as pbar:
        while ret:
            frame_nb += 1
            
            # Predict gaze
            frame_np = frame[..., ::-1]  # BGR to RGB
            frame_pil = Image.fromarray(frame_np)
            output = predict_gaze(frame_pil, sharingan, head_detector, tracker)
            gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids = output

            # Draw predictions
            heatmap_pid = args.heatmap_pid if args.heatmap_pid >= 0 else None
            num_people = len(head_bboxes)
            pids = np.arange(num_people) if len(pids) == 0 else pids
            
            frame = draw_gaze(
                frame_np,
                head_bboxes=head_bboxes,
                gaze_points=gaze_points,
                gaze_vecs=gaze_vecs if args.show_gaze_vec else None,
                inouts=inouts,
                pids=pids,
                gaze_heatmaps=gaze_heatmaps,
                heatmap_pid=heatmap_pid,
                frame_nb=None,
                colors=COLORS,
                filter_by_inout=args.filter_by_inout,
                alpha=0.6,
                gaze_pt_size=20,
                gaze_vec_factor=0.6,
                head_center_size=18,
                thickness=10,
                fs=0.8,
            )

            # Write frame
            process.stdin.write(frame.tobytes())
            
            # Read next frame
            ret, frame = cap.read()
            pbar.update(1)
    
    # Cleanup
    cap.release()
    process.stdin.close()
    process.wait()
    process.terminate()
    
    end = dt.datetime.now()
    print(colored(f"Finished. Processing took {end - start}.", TERM_COLOR))
    print(colored(f"Output saved to: {args.output}", TERM_COLOR))

if __name__ == "__main__":
    main() 
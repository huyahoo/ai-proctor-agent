import os
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
sys.path.insert(0, models_path)

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List, Dict, Optional, Tuple
from boxmot import OCSORT

from sharingan.sharingan import Sharingan
from sharingan.common import spatial_argmax2d, square_bbox
from core.utils import draw_gaze

# Constants
DET_THR = 0.4  # head detection threshold
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

class GazeTracker:
    """
    Advanced gaze tracking using Sharingan model.
    
    Features:
    - Multi-person gaze tracking
    - Real-time head detection and tracking
    - Accurate gaze point and vector prediction
    - Confidence scoring for predictions
    """
    
    def __init__(self, device=None):
        """Initialize the GazeTracker with Sharingan model."""
        # Device configuration
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GazeTracker using device: {self.device}")
        
        # Model configuration
        self._setup_model_config()
        
        # Initialize models
        self._initialize_models()
        
        print("GazeTracker initialized successfully")
    
    def _setup_model_config(self):
        """Setup model configuration parameters."""
        # Detection thresholds
        self.head_detection_threshold = DET_THR
        self.gaze_confidence_threshold = 0.5
        
        # Image normalization parameters
        self.image_mean = IMG_MEAN
        self.image_std = IMG_STD
        
        # Model paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../..'))
        self.checkpoint_path = os.path.join(project_root, "models/sharingan/checkpoints/videoattentiontarget.pt")
        self.weights_path = os.path.join(project_root, "models/sharingan/weights/yolov5m_crowdhuman.pt")
        
        # Validate paths
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Sharingan checkpoint not found: {self.checkpoint_path}")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Head detection weights not found: {self.weights_path}")
    
    def _initialize_models(self):
        """Initialize all required models."""
        # Initialize tracker
        self.tracker = OCSORT()
        
        # Initialize head detector
        self.head_detector = self._create_head_detector()
        
        # Initialize Sharingan model
        self.sharingan = self._create_sharingan_model()
    
    def _create_head_detector(self):
        """Create and configure the YOLOv5 head detection model."""
        model = torch.hub.load("ultralytics/yolov5", "custom", 
                             path=self.weights_path, 
                             verbose=False)
        
        # Configure detection parameters
        model.conf = 0.25  # NMS confidence threshold
        model.iou = 0.45   # NMS IoU threshold
        model.classes = [1]  # Filter for head class only
        model.amp = False  # Disable automatic mixed precision
        
        model = model.to(self.device)
        model.eval()
        
        print("Head detection model loaded successfully")
        return model
    
    def _create_sharingan_model(self):
        """Create and configure the Sharingan gaze prediction model."""
        # Model architecture configuration
        model_config = {
            'patch_size': 16,
            'token_dim': 768,
            'image_size': 224,
            'gaze_feature_dim': 512,
            'encoder_depth': 12,
            'encoder_num_heads': 12,
            'encoder_num_global_tokens': 0,
            'encoder_mlp_ratio': 4.0,
            'encoder_use_qkv_bias': True,
            'encoder_drop_rate': 0.0,
            'encoder_attn_drop_rate': 0.0,
            'encoder_drop_path_rate': 0.0,
            'decoder_feature_dim': 128,
            'decoder_hooks': [2, 5, 8, 11],
            'decoder_hidden_dims': [48, 96, 192, 384],
            'decoder_use_bn': True,
        }
        
        # Create model
        sharingan = Sharingan(**model_config)
        
        # Load pretrained weights
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        checkpoint = {name.replace("model.", ""): value for name, value in checkpoint["state_dict"].items()}
        sharingan.load_state_dict(checkpoint, strict=True)
        
        sharingan.eval()
        sharingan.to(self.device)
        
        print("Sharingan model loaded successfully")
        return sharingan
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces and predict gaze directions in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing gaze data for each detected person:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'gaze_point': [x, y],
                    'gaze_vector': [x, y],
                    'inout_score': float,
                    'pid': int
                },
                ...
            ]
        """
        # Predict gaze using Sharingan
        gaze_points, gaze_vectors, inout_scores, head_bboxes, _, person_ids = self._predict_gaze(frame)
        
        # Format results
        gaze_data = []
        for i in range(len(head_bboxes)):
            gaze_data.append({
                'bbox': head_bboxes[i].detach().numpy().tolist(),
                'gaze_point': gaze_points[i].detach().numpy().tolist() if len(gaze_points) > i else None,
                'gaze_vector': gaze_vectors[i].detach().numpy().tolist() if len(gaze_vectors) > i else None,
                'inout_score': inout_scores[i].item() if len(inout_scores) > i else 0.0,
                'pid': person_ids[i] if len(person_ids) > i else i
            })
        
        return gaze_data
    
    def _predict_gaze(self, image: np.ndarray) -> Tuple:
        """
        Core gaze prediction pipeline.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (gaze_points, gaze_vectors, inout_scores, head_bboxes, gaze_heatmaps, person_ids)
        """
        # Convert to PIL Image (RGB)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[..., ::-1])  # BGR to RGB
        
        image_np = np.array(image)
        img_height, img_width, _ = image_np.shape
        
        # Step 1: Detect heads
        head_detections = self._detect_heads(image_np)
        if len(head_detections) == 0:
            return [], [], [], [], [], []
        
        # Step 2: Track heads across frames
        tracks = self.tracker.update(head_detections, image_np)
        if len(tracks) == 0:
            return [], [], [], [], [], []
        
        # Step 3: Process tracked heads
        person_ids = (tracks[:, 4] - 1).astype(int)
        head_bboxes = torch.from_numpy(tracks[:, :4]).float()
        normalized_bboxes = square_bbox(head_bboxes, img_width, img_height)
        
        # Step 4: Extract and preprocess head crops
        head_crops = self._extract_head_crops(image, normalized_bboxes)
        
        # Step 5: Prepare model inputs
        model_inputs = self._prepare_model_inputs(image, head_crops, normalized_bboxes, img_width, img_height)
        
        # Step 6: Run Sharingan model
        with torch.no_grad():
            gaze_vectors, gaze_heatmaps, inout_logits = self.sharingan(model_inputs)
        
        # Step 7: Post-process outputs
        gaze_heatmaps = gaze_heatmaps.squeeze(0).cpu()
        gaze_vectors = gaze_vectors.squeeze(0).cpu()
        gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
        inout_scores = torch.sigmoid(inout_logits.squeeze(0)).flatten().cpu()
        
        return gaze_points, gaze_vectors, inout_scores, head_bboxes, gaze_heatmaps, person_ids
    
    def _detect_heads(self, image: np.ndarray) -> np.ndarray:
        """Detect heads in the image using YOLOv5."""
        with torch.no_grad():
            detections = self.head_detector(image, size=640).pred[0].cpu().numpy()[:, :-1]
        
        # Filter by confidence threshold
        filtered_detections = []
        for detection in detections:
            bbox, confidence = detection[:4], detection[4]
            if confidence > self.head_detection_threshold:
                class_id = np.array([0.])
                filtered_detection = np.concatenate([bbox, confidence[None], class_id])
                filtered_detections.append(filtered_detection)
        
        return np.stack(filtered_detections) if filtered_detections else np.array([])
    
    def _extract_head_crops(self, image: Image.Image, bboxes: torch.Tensor) -> torch.Tensor:
        """Extract and preprocess head crops from the image."""
        head_crops = []
        for bbox in bboxes:
            # Crop head region
            head_crop = TF.resize(TF.to_tensor(image.crop(bbox.numpy())), (224, 224))
            head_crops.append(head_crop)
        
        head_crops = torch.stack(head_crops)
        # Normalize
        head_crops = TF.normalize(head_crops, mean=self.image_mean, std=self.image_std)
        
        return head_crops
    
    def _prepare_model_inputs(self, image: Image.Image, head_crops: torch.Tensor, 
                            bboxes: torch.Tensor, img_width: int, img_height: int) -> Dict:
        """Prepare inputs for the Sharingan model."""
        # Process full image
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.resize(image_tensor, (224, 224))
        image_tensor = TF.normalize(image_tensor, mean=self.image_mean, std=self.image_std)
        
        # Normalize bounding boxes
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)
        normalized_bboxes = bboxes / scale
        
        # Build sample dictionary
        sample = {
            "image": image_tensor.unsqueeze(0).to(self.device),
            "heads": head_crops.unsqueeze(0).to(self.device),
            "head_bboxes": normalized_bboxes.unsqueeze(0).to(self.device)
        }
        
        return sample
    
    def draw_results(self, original_frame: np.ndarray, data, gaze_heatmaps=None, heatmap_pid=None, frame_nb=None) -> np.ndarray:
        """
        Draw gaze predictions on a copy of the original frame using the utility draw_gaze().
        """
        if not data:
            return original_frame.copy()

        head_bboxes = []
        gaze_points = []
        gaze_vecs = []
        inouts = []
        pids = []

        for entry in data:
            head_bboxes.append(entry['bbox'])
            gaze_points.append(entry['gaze_point'])
            gaze_vecs.append(entry['gaze_vector'])
            inouts.append(entry['inout_score'])
            pids.append(entry['pid'])

        return draw_gaze(
            image=original_frame,
            head_bboxes=head_bboxes,
            gaze_points=gaze_points,
            gaze_vecs=gaze_vecs,
            inouts=inouts,
            pids=np.array(pids),
            gaze_heatmaps=gaze_heatmaps or [],
            heatmap_pid=heatmap_pid,
            frame_nb=frame_nb
        )
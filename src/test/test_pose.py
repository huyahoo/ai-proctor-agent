import os
import sys
import time
from tqdm import tqdm
import json
from pathlib import Path

# Get the project root directory and src directory
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"

# Add src directory to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
from cv.pose_estimator import PoseEstimator
from core.config import Config
from core.logger import logger
from core.utils import assign_pose_pids

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def process_video(input_video_path: str, output_video_path: str):
    """
    Process a video file using PoseEstimator.
    Args:
        input_video_path (str): Path to input MP4 video file
        output_video_path (str): Path to save the annotated output video
    """
    # Create config
    config = Config()
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(config)
    
    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize timing variables
    total_start_time = time.time()
    frame_times = []
    
    try:
        # Use tqdm for better progress visualization
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Start timing this frame
                frame_start_time = time.time()
                
                # Process frame
                poses_data = pose_estimator.detect(frame)
                annotated_frame = pose_estimator.draw_results(frame, poses_data)
                
                # Calculate frame processing time
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                
                # Write frame
                out.write(annotated_frame)
                
                # Update progress bar with current FPS
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                pbar.set_postfix({'FPS': f'{current_fps:.1f}'})
                pbar.update(1)
                
    finally:
        # Calculate total processing time
        total_time = time.time() - total_start_time
        
        # Calculate statistics
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        min_fps = 1.0 / max(frame_times) if frame_times else 0
        max_fps = 1.0 / min(frame_times) if frame_times else 0
        
        # Clean up
        cap.release()
        out.release()
        
        # Log timing statistics
        logger.success(f"Video processing complete. Output saved to: {output_video_path}")
        logger.info(f"Total processing time: {format_time(total_time)}")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Min FPS: {min_fps:.1f}")
        logger.info(f"Max FPS: {max_fps:.1f}")
        logger.info(f"Average frame processing time: {avg_frame_time*1000:.1f}ms")

def test_pose_with_video():
    # Ensure the input video exists
    input_video = "data/videos/IMG_4723.mp4"  # Relative to src directory
    if not os.path.exists(input_video):
        logger.error(f"Input video not found at {input_video}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_video = "data/videos/IMG_4723_annotated.mp4"  # Relative to src directory
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    try:
        process_video(input_video, output_video)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        sys.exit(1)

def test_pose_with_image():
    # Ensure the input image exists
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(parent_dir, "src", "test", "gaze_estimations.json"), "r") as f:
        gaze_info = json.load(f)
    input_image = gaze_info["image_path"] # Relative to src directory
    
    if not os.path.exists(input_image):
        logger.error(f"Input image not found at {input_image}")
        sys.exit(1)
    
    # Create config
    config = Config()
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(config)
    
    # Read the image
    frame = cv2.imread(input_image)
    if frame is None:
        logger.error(f"Could not read image: {input_image}")
        sys.exit(1)
    
    # Process the image
    gaze_data = gaze_info["gaze_estimations"]
    poses_data = pose_estimator.detect(frame)
    pose_estimations = assign_pose_pids(poses_data, gaze_data)
    pose_info = {
        "image_path": input_image,
        "pose_estimations": convert_to_python_types(pose_estimations)
    }
    with open(os.path.join(parent_dir, "src", "test", "pose_estimations.json"), "w") as f:
        json.dump(pose_info, f, indent=4)
        print(f"Results saved to pose_estimations.json")
    annotated_frame = pose_estimator.draw_results(frame, pose_estimations)
    print(f"Detected {len(pose_estimations)} poses in the image: ", pose_estimations)
    
    # Save the annotated image
    output_image = os.path.join(parent_dir, "data", "output", "test_pose_output.jpg")
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    cv2.imwrite(output_image, annotated_frame)
    
    logger.success(f"Annotated image saved to: {output_image}")

# Convert NumPy types to native Python types for JSON serialization
def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
    
def main():
    # test_pose_with_video()
    test_pose_with_image()

if __name__ == "__main__":
    main() 
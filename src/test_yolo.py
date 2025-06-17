import os
import sys
import time
from tqdm import tqdm

# Add the parent directory (project root) to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import numpy as np
from cv.yolo_detector import YOLODetector
from core.config import Config
from core.logger import logger


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
    Process a video file using YOLODetector.
    Args:
        input_video_path (str): Path to input MP4 video file
        output_video_path (str): Path to save the annotated output video
    """
    # Create config and initialize detector
    config = Config()
    yolo_detector = YOLODetector(config)
    
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
        # Process each frame with progress bar
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame and time it
                frame_start_time = time.time()
                
                # Detect objects
                detections = yolo_detector.detect(frame)
                
                # Draw detections
                annotated_frame = yolo_detector.draw_results(frame, detections)
                
                # Calculate timing
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                
                # Write frame
                out.write(annotated_frame)
                
                # Update progress
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                pbar.set_postfix({'FPS': f'{current_fps:.1f}'})
                pbar.update(1)
                
    finally:
        # Calculate statistics
        total_time = time.time() - total_start_time
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        min_fps = 1.0 / max(frame_times) if frame_times else 0
        max_fps = 1.0 / min(frame_times) if frame_times else 0
        
        # Clean up
        cap.release()
        out.release()
        
        # Log results
        logger.success(f"Video processing complete. Output saved to: {output_video_path}")
        logger.info(f"Total processing time: {format_time(total_time)}")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Min FPS: {min_fps:.1f}")
        logger.info(f"Max FPS: {max_fps:.1f}")
        logger.info(f"Average frame processing time: {avg_frame_time*1000:.1f}ms")

def main():
    # Set up input/output paths
    input_video = "data/videos/IMG_4721.mp4"
    if not os.path.exists(input_video):
        logger.error(f"Input video not found at {input_video}")
        sys.exit(1)
    
    output_video = "data/videos/example_exam_yolo.mp4"
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    try:
        process_video(input_video, output_video)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
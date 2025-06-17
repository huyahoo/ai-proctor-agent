"""
Test script to run pose estimation on a video file.
"""

import os
import cv2
import argparse
from tqdm import tqdm

from cv.pose_estimator import PoseEstimator
from core.config import Config
from core.logger import logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test pose estimation on video")
    parser.add_argument("--input", type=str, default="data/videos/test_video.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/output/test_pose_output.mp4", help="Path to output video file")
    return parser.parse_args()

def main():
    """Main function to process video."""
    # Parse arguments
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Error: Input file '{args.input}' does not exist")
        return
        
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file '{args.input}'")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize pose estimator
    logger.info("Initializing PoseEstimator...")
    config = Config()
    estimator = PoseEstimator(config)
    
    # Process video
    logger.info(f"Processing video: {args.input}")
    logger.info(f"Output will be saved to: {args.output}")
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect poses
            results = estimator.detect(frame)
            
            # Draw results
            output_frame = estimator.draw_results(frame, results)
            
            # Write frame
            out.write(output_frame)
            
            # Update progress
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    logger.success("Done!")

if __name__ == "__main__":
    main() 
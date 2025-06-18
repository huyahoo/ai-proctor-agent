"""
Test script to run gaze tracking on a video file.
"""

import os
import cv2
import argparse
from tqdm import tqdm

from cv.gaze_tracker import GazeTracker
from core.config import Config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test gaze tracking on video")
    parser.add_argument("--input", type=str, default="data/videos/test_video.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/output/test_gaze_output.mp4", help="Path to output video file")
    return parser.parse_args()

def main():
    """Main function to process video."""
    # Parse arguments
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
        
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.input}'")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize gaze tracker
    tracker = GazeTracker(config=Config())
    
    # Process video
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect and track gaze
            results = tracker.detect(frame)
            
            # Draw results
            output_frame = tracker.draw_results(frame, results)
            
            # Write frame
            out.write(output_frame)
            
            # Update progress
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    main() 
"""
Test script to run gaze tracking on a video file.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import sys
import json
import numpy as np
from pathlib import Path

# Get the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_dir = project_root / "src"

# Add parent directory to sys.path
parent_dir = str(current_file.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cv.gaze_tracker import GazeTracker
from core.config import Config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test gaze tracking on video")
    parser.add_argument("--input", type=str, default="data/videos/test_video.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="data/output/test_gaze_output.mp4", help="Path to output video file")
    parser.add_argument("--input_image", type=str, default="data/images/missing_wrist.jpg", help="Path to input image file")
    return parser.parse_args()

def test_gaze_with_video():
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

def test_gaze_with_image():
    """Test gaze tracking on a single image."""
    args = parse_args()
    # Initialize gaze tracker
    tracker = GazeTracker(config=Config())
    # Load test image
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    test_image_path = args.input_image

    if not os.path.exists(test_image_path):
        print(f"Error: Test image '{test_image_path}' does not exist")
        return
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Error: Could not read image '{test_image_path}'")
        return
    # Detect and track gaze
    results = tracker.detect(frame)
    print(f"Detected {len(results)} gaze points in the image: ", results)
    gaze_info = {
        "image_path": test_image_path,
        "gaze_estimations": convert_to_python_types(results)
    }
    with open(os.path.join(parent_dir, "src", "test", "gaze_estimations.json"), "w") as f:
        json.dump(gaze_info, f, indent=4)
        print(f"Results saved to gaze_estimations.json")
    # Draw results
    output_frame = tracker.draw_results(frame, results)
    # Save output image
    output_image_path = os.path.join(parent_dir, "data", "output", "test_gaze_output.jpg")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, output_frame)
    print(f"Output image saved to: {output_image_path}")

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
    # test_gaze_with_video()
    test_gaze_with_image()

if __name__ == "__main__":
    main() 
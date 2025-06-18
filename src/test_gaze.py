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
    # Initialize gaze tracker
    tracker = GazeTracker(config=Config())
    # Load test image
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    test_image_path = os.path.join(parent_dir, "data", "images", "162.jpg")

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
    # Draw results
    output_frame = tracker.draw_results(frame, results)
    # Save output image
    output_image_path = "results/test_gaze_output.jpg"
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, output_frame)
    print(f"Output image saved to: {output_image_path}")


def main():
    # test_gaze_with_video()
    test_gaze_with_image()

if __name__ == "__main__":
    main() 
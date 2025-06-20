import os
import subprocess
import sys
from pathlib import Path

# Get the project root directory and src directory
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / "src"

# Add src directory to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import os
import argparse
import subprocess

def parse_args():
    """Parse command line arguments for batch runner."""
    parser = argparse.ArgumentParser(description="Run processing script on all .mp4 videos in a folder.")
    parser.add_argument('--video_dir', '-v', type=str, required=True, help="Path to the folder containing .mp4 videos.")
    parser.add_argument('--output_dir', '-o', type=str, default="data/output/examples", help="Output folder for JSON results.")
    parser.add_argument('--export_dir', '-e', type=str, default=None, help="Optional folder to export visualizations.")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.video_dir):
        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(args.video_dir, filename)
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}.json")

            cmd = [
                "python", "src/test/test_export_json.py",
                "--input", input_path,
                "--output", output_path
            ]

            if args.export_dir:
                cmd += ["--export_dir", args.export_dir]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

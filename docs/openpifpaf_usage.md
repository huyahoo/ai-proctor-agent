# 🤸‍♂️ Pose Estimation with OpenPifPaf

This document provides instructions for using the OpenPifPaf pose estimation model in the Proctor Agent system.

## 📥 Installation

The primary dependency for OpenPifPaf is included in the main [requirements.txt](/requirements.txt) file. However, depending on your system configuration, you may need to install or update the GNU C++ runtime library.

### Required Dependencies
``` bash
pip install openpifpaf=0.13.11
```

### System Dependencies (conda)
- Note that openpifpaf requires `torch==1.13.1, torchvision==0.14.1`
- If you encounter compilation errors during installation, run the following command:
```bash
conda install -c conda-forge libstdcxx-ng
```

## 🧪 Usage and Testing

### Video Processing Test
To verify the installation and see the model in action, run the included test script. This script processes a sample video and saves an annotated version.

```bash
python src/test_pose.py
```

The script will:
- Process video [test_video.mp4](data/videos/test_video.mp4)
- Video output [test_pose_output.mp4](`data/output/test_pose_output.mp4)
- Show real-time FPS and processing stats

### Basic Code Example
Here is a minimal example of how to use the `PoseEstimator` in your own code:
```python
from cv.pose_estimator import PoseEstimator
from core.config import Config
import cv2

# Initialize
config = Config()
pose_estimator = PoseEstimator(config)

# Process a frame
frame = cv2.imread('image.jpg')
poses_data = pose_estimator.detect(frame)
annotated_frame = pose_estimator.draw_results(frame, poses_data)
```

## 📊 Output Details

### Pose Data Format
The detector returns a list of pose arrays, with one array for each person detected. Each pose array contains 17 keypoints.

```python
poses_data = [
    # Person 1
    [
        [x1, y1, conf1],  # Keypoint 1 (Nose)
        [x2, y2, conf2],  # Keypoint 2 (Left Eye)
        # ... 17 keypoints total
    ],
    # Person 2
    [
        [x1, y1, conf1],
        [x2, y2, conf2],
        # ...
    ]
]
```
Each keypoint is a list `[x, y, confidence]`:
- `x, y`: Integer pixel coordinates.
- `confidence`: A float value between 0.0 and 1.0. Keypoints with confidence less than 0.2 are typically ignored.

### COCO Keypoint Map
The model uses the standard 17 COCO keypoints, indexed 0-16:
0. Nose
1. Left Eye
2. Right Eye
3. Left Ear
4. Right Ear
5. Left Shoulder
6. Right Shoulder
7. Left Elbow
8. Right Elbow
9. Left Wrist
10. Right Wrist
11. Left Hip
12. Right Hip
13. Left Knee
14. Right Knee
15. Left Ankle
16. Right Ankle

## 🎨 Visualization

The `draw_results` method annotates the frame with:
- **Keypoints**: Blue dots
- **Skeleton**: Cyan lines connecting the keypoints

The skeleton connections are based on the COCO format:
```python
COCO_PERSON_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13),  # Legs
    (12, 13), (6, 12), (7, 13),              # Hips and torso
    (6, 7), (6, 8), (7, 9),                  # Shoulders and arms
    (8, 10), (9, 11),                        # Forearms
    (2, 3), (1, 2), (1, 3),                  # Face
    (2, 4), (3, 5), (4, 6), (5, 7)           # Ears to shoulders
]
```

## 🚀 Performance Metrics

The `test_pose.py` script provides a summary of performance upon completion:
- Total processing time
- Average, minimum, and maximum Frames Per Second (FPS)
- Average time to process a single frame

Example output:
```
Processing video: 100%|██████████| 300/300 [00:15<00:00, 19.8it/s, FPS=20.1]
Total processing time: 15.23 seconds
Average FPS: 19.7
Min FPS: 15.2
Max FPS: 22.5
Average frame processing time: 50.8ms
```

## 📚 Additional Resources

- [OpenPifPaf Official GitHub Repository](https://github.com/openpifpaf/openpifpaf)
- [OpenPifPaf Documentation](https://openpifpaf.github.io/intro.html)
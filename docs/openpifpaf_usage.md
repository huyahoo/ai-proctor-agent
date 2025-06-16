# OpenPifPaf Pose Estimation

## Installation

If you encounter any errors, you may need to install/update the GNU C++ runtime library:
```bash
conda install -c conda-forge libstdcxx-ng
```

## Usage

### Basic Example

```python
from cv.pose_estimator import PoseEstimator
from core.config import Config

# Initialize
config = Config()
pose_estimator = PoseEstimator(config)

# Process a frame
frame = cv2.imread('image.jpg')
poses_data = pose_estimator.detect(frame)
annotated_frame = pose_estimator.draw_results(frame, poses_data)
```

### Video Processing

Use the test script to process videos:
```bash
python src/test_pose.py
```

The script will:
- Process video from `data/videos/IMG_4721.mp4`
- Save output to `data/videos/IMG_4721_annotated.mp4`
- Show real-time FPS and processing stats

## Output Format

The detector returns a list of keypoint arrays (one array per person):
```python
poses_data = [
    # Person 1
    [
        [x1, y1, conf1],  # Keypoint 1
        [x2, y2, conf2],  # Keypoint 2
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

Each keypoint is `[x, y, confidence]` where:
- `x, y`: Integer pixel coordinates
- `confidence`: Float between 0-1 (only keypoints with conf > 0.2 are included)

## Visualization

The visualization uses:
- Blue dots for keypoints
- Cyan lines for skeleton connections
- COCO format skeleton (17 keypoints)

## Performance Metrics

The test script provides:
- Real-time FPS display
- Total processing time
- Average/Min/Max FPS
- Average frame processing time

Example output:
```
Processing video: 100%|██████████| 300/300 [00:15<00:00, 19.8it/s, FPS=20.1]
Total processing time: 15.23 seconds
Average FPS: 19.7
Min FPS: 15.2
Max FPS: 22.5
Average frame processing time: 50.8ms
```

## COCO Keypoint Format

The 17 keypoints (0-based indices):
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

## Skeleton Connections

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
# protor-agent
This is repository for Hackathon "HACK U TOKYO 2025" organized by LINE YAHOO Corporation!

## Folder structure
```text
proctor_agent/
├── data/
│   ├── videos/             # Store input videos (e.g., example_exam.mp4)
│   └── feedback/           # Store human-reviewed cheating examples for VLM fine-tuning
│       ├── images/
│       └── annotations.jsonl
├── models/
│   ├── yolov8n.pt          # Placeholder for pre-trained YOLO model weights
│   └── llava/              # If you decide to download/host LLaVA models locally
├── src/
│   ├── core/
│   │   ├── config.py       # Configuration settings (API keys, thresholds)
│   │   ├── utils.py        # Helper functions (video loading, frame processing)
│   │   └── constants.py    # Global constants (labels, colors)
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── base_detector.py # Base class/interface for detectors
│   │   ├── yolo_detector.py # YOLO object detection module
│   │   ├── pose_estimator.py # MediaPipe BlazePose module
│   │   └── gaze_tracker.py  # Gaze tracking module
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py # Logic for combining CV outputs into events
│   │   ├── llm_constraint_generator.py # LLM interaction for constraints
│   │   ├── vlm_analyzer.py # VLM interaction for verification & explanation
│   │   └── feedback_learner.py # Handles feedback data & prepares for RL
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py  # Main PyQt6 application window
│   │   ├── video_player.py # Widget for video display and overlays
│   │   └── alert_panel.py  # Widget for displaying alerts and VLM reasoning
│   └── main.py             # Entry point for the application
├── videos/                 # Symlink or copy of videos if `data/videos` is too deep
├── README.md               # Project documentation
├── requirements.txt        # List of pip dependencies
└── .env                    # Environment variables (API keys)
```

## Installation

- Create conda environment. This project use python 3.10
``` bash
conda create -n proctor_agent python=3.10
```
- Activate conda environment
``` bash
conda activate proctor_agent
```
- Install Python Package
``` bash
pip install -r requirements.txt
```

-Create .env config file from .env.example, then add your gemini api key
``` bash
cp .env.example .env
```

- Install pretrained model for OpenPose
``` bash
cd models
wget -O pose_model_scratch.pth 'https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?e=1&dl=1'
```

## Run
``` bash
cd proctor-agent
python src/main.py
```
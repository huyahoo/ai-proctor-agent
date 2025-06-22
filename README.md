# AI Proctor Agent

AI Proctor Agent is an XAI system designed to bolster exam integrity in in-person settings. It uses a sophisticated multi-stage AI approach, combining cutting-edge Computer Vision (YOLOv8 for objects, OpenPifPaf for precise pose, Sharingan for Gaze Estimation) to comprehensively analyze both individual and collaborative suspicious behaviors. A AI reasoning pipeline, powered by LLMs for dynamic constraint generation and VLMs for contextual verification, provides highly accurate and explainable alerts. This project was developed for the "HACK U TOKYO 2025" hackathon organized by LINE YAHOO Corporation.

# 🏆 Hackathon - First Prize Winner 🏆
This repository proudly represents the project that secured **1st place** at the **Open Hack U 2025 TOKYO hackathon**!

## 🚀 Features

- **Real-time Monitoring**: Track multiple students simultaneously using computer vision
- **Gaze Analysis**: Detect suspicious eye movements and attention patterns
- **Pose Estimation**: Monitor body language and posture for potential cheating
- **AI-Powered Analysis**: Use LLMs and VLMs to verify and explain suspicious activities
- **Interactive UI**: User-friendly interface for monitoring and reviewing exam sessions
<!-- - **Feedback Learning**: System improves over time through human feedback -->

## 📋 Prerequisites

- Python 3.10
- CUDA-compatible GPU (recommended)
- NVIDIA drivers
- Conda package manager

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/huyahoo/proctor-agent.git
cd proctor-agent
```

2. **Create and activate conda environment**
- This command creates a new conda environment named `proctor-agent` and installs all necessary dependencies from the `environment.yml` file.
- Note that you should change CUDA toolkit version suitable for your environment.
```bash
conda env create -f environment.yml
conda activate proctor-agent
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

4. **Setup Gaze Estimation**
- In this demo, we use model [sharingan](https://github.com/idiap/sharingan/) for Gaze Estimation
- Check out [Gaze Estimation Usage](docs/gaze_usage.md).

5. **Setup Pose Estimation**
- In this demo, we use library [Openpifpaf](https://openpifpaf.github.io/intro.html) for Pose Estimation.
- Check out [Pose Estimation Usage](docs/openpifpaf_usage.md).

## 📁 Project Structure

```
proctor-agent/
├── data/                  # Data storage
│   ├── videos/            # Input video files
│   └── output/
│         └── examples/    # Output Example (JSON)
├── models/                # Model weights and checkpoints
├── src/                   # Source code
│   ├── core/              # Utils, Constant
│   ├── cv/                # Computer vision modules
│   ├── ai/                # Gen AI components
│   └── ui/                # User interface
│   └── test/              # Model Test
└── docs/                  # Documentation
```

## 🚀 Usage

1. **Start the application**
```bash
python src/main.py
```

2. **Load a video**
- Use the UI to select an exam recording
- The system will automatically begin analysis

3. **Monitor results**
- View real-time detections and alerts
- Review AI-generated explanations
- Export reports as needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LINE YAHOO Corporation for organizing HACK U TOKYO 2025
- All contributors and supporters of this project
- A part of demo were adapted from the repository [sharingan](https://github.com/idiap/sharingan/). We are thankful to the authors for their contribution.

# Proctor Agent

An intelligent proctoring system that uses computer vision and AI to detect potential academic misconduct during exams. This project was developed for the "HACK U TOKYO 2025" hackathon organized by LINE YAHOO Corporation.

## 🚀 Features

- **Real-time Monitoring**: Track multiple students simultaneously using computer vision
- **Gaze Analysis**: Detect suspicious eye movements and attention patterns
- **Pose Estimation**: Monitor body language and posture for potential cheating
- **AI-Powered Analysis**: Use LLMs and VLMs to verify and explain suspicious activities
- **Interactive UI**: User-friendly interface for monitoring and reviewing exam sessions
- **Feedback Learning**: System improves over time through human feedback

## 📋 Prerequisites

- Python 3.10
- CUDA-compatible GPU (recommended)
- NVIDIA drivers
- Conda package manager

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/proctor-agent.git
cd proctor-agent
```

2. **Create and activate conda environment**
```bash
conda create -n proctor-agent python=3.10
conda activate proctor-agent
```

3. **Install CUDA dependencies**
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge opencv
```

4. **Install Python packages**
```bash
pip install -r requirements.txt
```

5. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

## 📁 Project Structure

```
proctor-agent/
├── data/                   # Data storage
│   ├── videos/            # Input video files
│   └── feedback/          # Training data for VLM
├── models/                # Model weights and checkpoints
├── src/                   # Source code
│   ├── core/             # Core functionality
│   ├── cv/               # Computer vision modules
│   ├── ai/               # AI and ML components
│   └── ui/               # User interface
└── docs/                 # Documentation
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
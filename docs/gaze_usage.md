# Gaze Tracking with Sharingan

This document provides instructions for setting up and using the Sharingan gaze tracking model in the Proctor Agent system.

## 📥 Model Installation

### 1. Download Model Files

The Sharingan model requires two sets of files:
- Model weights for head detection
- Checkpoints for gaze prediction

Run these commands from your project root:

```bash
# Navigate to models directory
cd models

# Download model weights
wget "https://zenodo.org/records/14066123/files/sharingan_weights.tar.gz"
tar -xvzf sharingan_weights.tar.gz

# Download model checkpoints
wget "https://zenodo.org/records/14066123/files/sharingan_checkpoints.tar.gz"
tar -xvzf sharingan_checkpoints.tar.gz
```

### 2. Required Dependencies

#### Python Packages (pip)
```bash
pip install einops==0.7.0 \
            pytorch-lightning==2.1.2 \
            termcolor==2.4.0 \
            transformers==4.35.2 \
            wandb==0.16.1 \
            boxmot==10.0.46 \
            seaborn==0.13.0
```

#### System Dependencies (conda)
```bash
conda install -c conda-forge ffmpeg==4.2.2
```

## 🧪 Testing the Installation

The system includes two test scripts to verify the installation:

### 1. Model Test
```bash
python models/sharingan/test_video.py
```
This script verifies that the Sharingan model is properly installed and can run in your environment.

### 2. Integration Test
```bash
python src/test_gaze.py
```
This script tests the full gaze tracking integration with the Proctor Agent system.

## 📝 Notes

- Ensure all dependencies are installed before running the tests
- The model requires CUDA support for optimal performance
- Check the console output for any error messages during testing

## 🔍 Troubleshooting

If you encounter issues:

1. Verify CUDA installation:
```bash
nvidia-smi
```

2. Check PyTorch CUDA availability:
```python
import torch
print(torch.cuda.is_available())
```

3. Ensure all model files are in the correct locations:
```
models/
├── sharingan/
│   ├── weights/
│   │   └── yolov5m_crowdhuman.pt
│   └── checkpoints/
│       └── videoattentiontarget.pt
```

## 📚 Additional Resources

- [Sharingan Model Documentation](https://github.com/idiap/sharingan)
- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [BoxMOT Documentation](https://github.com/mikel-brostrom/boxmot)
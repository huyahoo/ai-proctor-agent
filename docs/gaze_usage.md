# Sharingan Model Files

This project includes pretrained Sharingan weights and checkpoints. Use the commands below to download and extract the necessary files inside the `models/` directory.

## Download and Extract

Run the following commands in your terminal **from the root of your project**:

```bash
# Go to models directory
cd models

# Download Sharingan Weights
wget "https://zenodo.org/records/14066123/files/sharingan_weights.tar.gz"

# Extract Weights
tar -xvzf sharingan_weights.tar.gz

# Download Sharingan Checkpoints
wget "https://zenodo.org/records/14066123/files/sharingan_checkpoints.tar.gz"

# Extract Checkpoints
tar -xvzf sharingan_checkpoints.tar.gz
```
## Install libraries and dependences
Using pip
```
    einops==0.7.0
    pytorch-lightning==2.1.2
    termcolor==2.4.0
    transformers==4.35.2
    wandb==0.16.1 
    boxmot==10.0.46
    seaborn==0.13.0
```
Using conda
```
    ffmpeg==4.2.2
```
## Testing
There are 2 different test_file.

The first one is `models/sharingan/test_video.py` to check if the model is running in the new conda env.


The second one is `src/test_gaze.py` to check if the gaze_tracker is working.
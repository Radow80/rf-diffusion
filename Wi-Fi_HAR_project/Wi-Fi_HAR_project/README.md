# GenFall: Self-Supervised Wireless Fall Detection via Generative Diffusion Model

Falls pose a significant public health concern, leading to numerous injuries and fatalities annually. Most existing fall detection methods often rely on supervised learning, necessitating extensive labeled datasets that are challenging to obtain and may not encompass the diverse nature of fall events. To address these limitations, we introduce GenFall, a pioneering self-supervised wireless fall detection system utilizing a generative diffusion model. GenFall employs an imputation-based spectrogram reconstruction framework, coupled with a dual-path signal diffusion model, to accurately reconstruct normal signal patterns while amplifying anomalies associated with falls. A multi-metric anomaly detection module then assesses the similarity between original and reconstructed spectrograms, effectively distinguishing fall events from typical daily activities. Extensive real-world experiments demonstrate that GenFall, operating without the need for labeled data, achieves an overall accuracy of 97.5%, surpassing both existing supervised and unsupervised methods.

## Environment Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU acceleration)
- Other dependencies (see requirements.txt):
  ```
  numpy
  scipy
  scikit-image
  sewar
  einops
  matplotlib
  lpips
  torchvision
  tqdm
  ```

## Quick Start

1. Prepare your environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset:
   - Download `data.zip` from [GitHub Releases](https://github.com/Guoxuan-Chi/GenFall/releases).
   - Extract `data.zip` into the project root so that the data directory structure is as follows:
     ```
     clean_version/
     ├── data/
     │   ├── train/
     │   │   ├── walk_silence/   # normal walking samples
     │   │   ├── walk_sit/       # samples of walking then sitting
     │   │   ├── squat/          # squatting samples
     │   │   ├── dance/          # dancing samples
     │   │   ├── Open_Close_door/ # samples of opening/closing door
     │   │   └── fall/           # fall samples
     │   ├── test_fall/          # fall test data
     └── test_unfall/            # non-fall test data
     ```

3. Download the pretrained model:
   - Download `models.zip` from [GitHub Releases](https://github.com/Guoxuan-Chi/GenFall/releases).
   - Extract `models.zip` into the project root so that the model directory structure is as follows:
     ```
     clean_version/
     ├── models/
     │   └── weights/
     │       └── weights.pt       # pretrained model file
     ```

4. Test the model:
   ```bash
   cd src
   # Test fall samples
   python test_fall_example.py
   
   # Test non-fall samples
   python test_unfall_example.py
   ```
   The test results (metrics such as SSIM, PSNR, UQI, etc.) are saved in the `logs` directory.

## Project Structure

```
.
├── data/                    # Dataset directory (download from GitHub Releases)
│   ├── train/              # Training data (200 samples)
│   │   ├── walk_silence/   # normal walking (20 samples)
│   │   ├── walk_sit/       # walking then sitting (20 samples)
│   │   ├── squat/          # squatting (20 samples)
│   │   ├── dance/          # dancing (20 samples)
│   │   ├── Open_Close_door/ # opening/closing door (20 samples)
│   │   └── fall/           # fall (100 samples)
│   ├── test_fall/          # Fall test data (50 samples)
│   └── test_unfall/        # Non-fall test data (50 samples)
├── models/                  # Model weights directory (download from GitHub Releases)
│   └── weights/            # Pretrained model
├── src/                    # Source code
│   ├── AD_model.py         # Model definition
│   ├── diffusion.py        # Diffusion model implementation
│   ├── learner.py          # Trainer
│   ├── params.py           # Parameter configuration
│   ├── dataset_*.py        # Data loading
│   ├── train.py            # Training script
│   └── test_*.py           # Testing scripts
└── logs/                   # Log and result directory (created automatically)
```

## Dataset Description

**Note:** Currently, only a small sample of the dataset is provided for preliminary feasibility testing.

The dataset used in this project consists of the following categories:
- Training data (200 samples):
  - Normal behavior (100 samples):
    - Normal walking (walk_silence): 20 samples
    - Walking then sitting (walk_sit): 20 samples
    - Squatting (squat): 20 samples
    - Dancing (dance): 20 samples
    - Opening/closing door (Open_Close_door): 20 samples
  - Fall behavior (100 samples):
    - Fall events (fall): 100 samples
- Test data (100 samples):
  - Fall events (50 samples)
  - Non-fall events (50 samples)

Each sample is a STFT feature of Doppler radar signals, saved in .mat format.

## Paths

This project uses relative paths, so all paths are relative to the project root. You can place the project anywhere as long as the directory structure is maintained. Key paths include:
- Data path: `./data/`
- Model path: `./models/weights/`
- Log path: `./logs/`

## Model Description

This project uses a Transformer-based diffusion model with the following characteristics:
- Input: Raw CSI then convert to STFT features.
- Output: Reconstructed signal features.
- Evaluation metrics: Reconstruction quality (SSIM, PSNR, etc.) is used to detect anomalies (Falls).

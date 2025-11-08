markdown
# MSFT-Net: Multimodal Sparse Fusion Transformer Network

## 📖 Description

MSFT-Net is a multimodal transformer-based model for video classification tasks.

## ✨ Features

- **Multimodal Support**: Processes three distinct modalities simultaneously
- **Attention Mechanisms**: Implements both temporal and spatial attention
- **Configurable Pipeline**: Flexible training and testing scripts
- **Comprehensive Evaluation**: Multiple metrics for performance assessment
- **Sparse Fusion**: Efficient fusion of multimodal features

## 📋 Requirements

- Python 3.8+
- PyTorch 2.3.1+
- Albumentations
- OpenCV
- NumPy
- scikit-learn
- pandas

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/msft-net.git
cd msft-net
Install dependencies:

bash
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
pip install torch torchvision albumentations opencv-python numpy scikit-learn pandas
📊 Dataset Preparation
Directory Structure
Organize your dataset as follows:

text
dataset/
    case1/
        b/          # Frames for modality b
            frame_001.jpg
            frame_002.jpg
            ...
        c/          # Frames for modality c  
            frame_001.jpg
            frame_002.jpg
            ...
        e/          # Frames for modality e
            frame_001.jpg
            frame_002.jpg
            ...
    case2/
        b/
        c/
        e/
    ...
CSV Files Format
Create CSV files with the following format:

train.csv/val.csv/test.csv:

csv
case_name,label
case1,0
case2,1
case3,0
...

## 🏃‍♂️ Training
To train the model, run the following command:
bash
python train.py --data_root path/to/dataset --train_csv path/to/train.csv --val_csv path/to/val.csv --epochs 50 --batch_size 8 --learning_rate 0.001

## 🧪 Testing

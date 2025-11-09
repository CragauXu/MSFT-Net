# MSFT-Net: Multimodal Sparse Fusion Transformer Network

## 📖 Description

MSFT-Net is a multimodal transformer-based model for video classification tasks.

![MSFT-Net Architecture](./imgs/msft_net_architecture.png)

## 👀 Data Preview

Below are sample video sequences from the three modalities processed by MSFT-Net:

<div align="center">

### US - Ultrasound Imaging
![US Sample](./imgs/US_sample.gif)  
*Captures structural and anatomical features from ultrasound*

### SMI - Superb Microvascular Imaging  
![SMI Sample](./imgs/SMI_sample.gif)  
*Represents microvascular blood flow characteristics*

### SE - Strain Elastography
![SE Sample](./imgs/SE_sample.gif)  
*Provides tissue stiffness and elasticity information*

</div>

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
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install manually:
```bash
pip install torch torchvision albumentations opencv-python numpy scikit-learn pandas
```

## 📊 Dataset Preparation

### Directory Structure
Organize your dataset as follows:

```text
dataset/
    case1/
        US/          # Frames for US modality
            frame_001.jpg
            frame_002.jpg
            ...
        SMI/         # Frames for SMI modality  
            frame_001.jpg
            frame_002.jpg
            ...
        SE/          # Frames for SE modality
            frame_001.jpg
            frame_002.jpg
            ...
    case2/
        US/
        SMI/
        SE/
    ...
```

### CSV Files Format
Create CSV files with the following format:

train.csv/val.csv/test.csv:
```csv
case_name,label
case1,0
case2,1
case3,0
...
```

## 🔍 Data Description

The dataset consists of three complementary modalities that capture different aspects of the video sequences:

<div align="center">

### US - Ultrasound Imaging
![US Sample](./imgs/US_sample.gif)  
*Standard B-mode ultrasound for anatomical structure visualization*

### SMI - Superb Microvascular Imaging  
![SMI Sample](./imgs/SMI_sample.gif)  
*Advanced Doppler technique for microvascular blood flow detection*

### SE - Strain Elastography
![SE Sample](./imgs/SE_sample.gif)  
*Quantitative assessment of tissue stiffness and elasticity*

</div>

## 🏋️ Training

Train the model using the train.py script:

```bash
python train.py --root_path ./dataset \
                --train_csv ./dataset/train.csv \
                --val_csv ./dataset/val.csv \
                --output_dir ./results \
                --batch_size 8 \
                --epochs 50 \
                --learning_rate 0.001
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--root_path` | Root directory of the dataset | `./dataset` |
| `--train_csv` | Path to training CSV file | Required |
| `--val_csv` | Path to validation CSV file | Required |
| `--output_dir` | Directory to save models and logs | `./results` |
| `--batch_size` | Batch size for training | `8` |
| `--epochs` | Number of training epochs | `50` |
| `--learning_rate` | Learning rate | `0.001` |

## 🧪 Testing

Evaluate the trained model using the test.py script:

```bash
python test.py --model_path ./results/best_model.pth \
               --test_csv ./dataset/test.csv \
               --root_path ./dataset \
               --batch_size 8
```

### Testing Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to trained model checkpoint | Required |
| `--test_csv` | Path to testing CSV file | Required |
| `--root_path` | Root directory of the dataset | `./dataset` |
| `--batch_size` | Batch size for testing | `8` |

## 📈 Evaluation Metrics

The model evaluation includes the following comprehensive metrics:

- **Accuracy (ACC)**: Overall classification accuracy
- **Sensitivity (SEN)**: True positive rate (recall)
- **Specificity (SPE)**: True negative rate
- **Precision (PRE)**: Positive predictive value
- **AUC**: Area under the ROC curve
- **F1-score**: Harmonic mean of precision and recall

## 🗂️ Project Structure

```text
msft-net/
├── imgs/                 # Images and GIFs for documentation
│   ├── msft_net_architecture.png
│   ├── US_sample.gif
│   ├── SMI_sample.gif
│   └── SE_sample.gif
├── train.py              # Training script
├── test.py               # Testing script
├── models/               # Model architecture definitions
│   └── msft_net.py       # MSFT-Net implementation
├── utils/                # Utility functions
│   ├── data_loader.py    # Data loading utilities
│   ├── metrics.py        # Evaluation metrics
│   └── transforms.py     # Data transformations
├── dataset/              # Dataset directory
├── results/              # Output directory for models and logs
└── requirements.txt      # Dependencies list
```

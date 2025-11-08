import os
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from glob import glob
import albumentations as A
import numpy as np
import random
from typing import Optional, Tuple

def temporal_clip(data: torch.Tensor, t: int, mode: str = "uniform") -> torch.Tensor:
    """
    Temporal sampling function for video frames.
    Args:
        data: Tensor, shape (C, T, H, W)
        t: int, target temporal length
        mode: str, ["random", "center", "uniform"]
              - random: Random cropping (T > t)
              - center: Center cropping (T > t)
              - uniform: Uniform sampling (T > t)
    Returns:
        Tensor, shape (C, t, H, W)
    """
    C, T, H, W = data.shape

    if T == t:
        return data

    if T > t:
        if mode == "random":
            start = random.randint(0, T - t)
            end = start + t
            return data[:, start:end, :, :]
        elif mode == "center":
            start = (T - t) // 2
            end = start + t
            return data[:, start:end, :, :]
        elif mode == "uniform":
            indices = np.linspace(0, T - 1, t, dtype=int)
            return data[:, indices, :, :]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # If T < t, pad symmetrically
    pad_len = t - T
    left = pad_len // 2
    right = pad_len - left
    pad_left = torch.zeros((C, left, H, W), dtype=data.dtype, device=data.device)
    pad_right = torch.zeros((C, right, H, W), dtype=data.dtype, device=data.device)
    return torch.cat([pad_left, data, pad_right], dim=1)

class Dataset_Multimodal(Dataset):
    def __init__(self, root_dir: str, num_frame: int, csv_path: str,
                 transforms: Optional[A.Compose] = None, clip_mode: str = "uniform"):
        """
        Multimodal dataset for loading video frames.
        Args:
            root_dir: Root directory of the dataset.
            num_frame: Number of frames to sample per video.
            csv_path: Path to the CSV file containing case names and labels.
            transforms: Albumentations transformations to apply to the frames.
            clip_mode: Temporal sampling mode ("random", "center", "uniform").
        """
        super(Dataset_Multimodal, self).__init__()
        self.root_dir = root_dir
        self.num_frame = num_frame
        self.transforms = transforms
        self.clip_mode = clip_mode

        # Read CSV file (assumes no header)
        df = pd.read_csv(csv_path, header=None, dtype=str)
        self.case_names = df.iloc[:, 0].tolist()
        self.labels = df.iloc[:, 1].astype(int).tolist()

    def images2tensor(self, image_dir: str) -> torch.Tensor:
        """
        Load images from a directory and convert them to a tensor.
        Args:
            image_dir: Directory containing image frames.
        Returns:
            Tensor of shape (C, T, H, W).
        """
        image_paths = glob(os.path.join(image_dir, '*.png'))
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        images = [cv2.imread(image_path) for image_path in image_paths]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        images = np.array(images)

        if self.transforms:
            augmented = self.transforms(image=images)
            images = augmented['image']
        else:
            transforms = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2()
            ])
            augmented = transforms(image=images)
            images = augmented['image']

        images = torch.tensor(images).permute(3, 0, 1, 2)  # (C, T, H, W)
        images = temporal_clip(images, self.num_frame, self.clip_mode)
        return images

    def __len__(self) -> int:
        return len(self.case_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single data sample.
        Args:
            idx: Index of the sample.
        Returns:
            Tuple of tensors for three modalities and the label.
        """
        case_name = self.case_names[idx]
        label = self.labels[idx]

        image_dir_modal1 = os.path.join(self.root_dir, case_name, 'b')
        image_dir_modal2 = os.path.join(self.root_dir, case_name, 'c')
        image_dir_modal3 = os.path.join(self.root_dir, case_name, 'e')

        tensor_modal1 = self.images2tensor(image_dir_modal1)
        tensor_modal2 = self.images2tensor(image_dir_modal2)
        tensor_modal3 = self.images2tensor(image_dir_modal3)

        return tensor_modal1, tensor_modal2, tensor_modal3, label
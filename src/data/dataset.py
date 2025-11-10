import torch
from torch.utils.data import Dataset
import tifffile
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PixelClassificationDataset(Dataset):
    """A dataset for pixel-wise classification from segmented images.

    Args:
        Dataset (torch.utils.data.Dataset): The parent class.
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Args:
            image_paths: List of paths to input images
            mask_paths: List of paths to segmentation masks
            transform: Albumentations transform pipeline
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        if not self.transform:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.75, interpolation=3), 
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0)

                # Color Augmentations
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                A.GaussNoise(p=0.2),

                ToTensorV2()
            ])
        
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = tifffile.imread(self.image_paths[idx])
        mask = tifffile.imread(self.mask_paths[idx])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()
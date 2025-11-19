import sys
from pathlib import Path

from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm
from omegaconf import DictConfig
from dotenv import load_dotenv

import torch
import hydra
import os
import numpy as np
import random

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
script_path = Path(__file__).resolve()
src_dir = script_path.parent
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.dataset import PixelClassificationDataset
from src.models.tissue_segmentation import TissueSegmentationModel
from src.utils.helpers import get_image_and_mask_paths, seed_everything


def get_train_transform():
    """Returns the training augmentation pipeline

    Returns:
        train_transform (albumentation): The albumentations augmentation pipeline
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.75, interpolation=3),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=0,
            ),
            # Color Augmentations
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ]
    )


def train(
    tissue_model: TissueSegmentationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    model_name: str,
    checkpoint_path: str = "checkpoints",
    gradient_clipping: bool = True,
    max_norm: float = 1.0,
):
    """Trains the tissue segmentation model.

    Args:
        tissue_model (TissueSegmentationModel): The tissue segmentation model.
        train_loader (DataLoader): The training data loader.
        test_loader (DataLoader): The test data loader.
        epochs (int): The number of epochs to train for.
        model_name (str): The name of the model.
        checkpoint_path (str, optional): The path to save the checkpoints. Defaults to "checkpoints".
        gradient_clipping (bool): Whether to apply gradient clipping. Defaults to False.
        max_norm (float): Maximum norm for gradient clipping. Defaults to 1.0.

    Returns:
        Tuple[float, int]: The best test F1 score and the best epoch.
    """

    optimizer = tissue_model.configure_optimizers()

    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    temp_checkpoint = Path(checkpoint_path) / f"temp_{model_name}.pth"
    best_model_path = Path(checkpoint_path) / f"{model_name}.pth"

    epoch_pbar = tqdm(range(epochs), desc="Training", position=0)
    best_test_f1 = 0.0
    best_epoch = 0

    for epoch in epoch_pbar:
        ### Training
        tissue_model.model.train()
        train_losses = []

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            position=1,
            leave=False,
        )
        for images, masks in train_pbar:
            images = images.to(tissue_model.device)
            masks = masks.to(tissue_model.device)

            optimizer.zero_grad()
            loss, loss_val = tissue_model.training_step((images, masks))
            loss.backward()

            if gradient_clipping:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        tissue_model.model.parameters(), max_norm=max_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        tissue_model.model.parameters(), max_norm=1.0
                    )

            optimizer.step()
            train_losses.append(loss_val)

            train_pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_metrics = tissue_model.training_epoch_end(avg_train_loss)

        ### Testing
        tissue_model.model.eval()
        test_losses = []

        test_pbar = tqdm(
            test_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Test]",
            position=1,
            leave=False,
        )
        with torch.no_grad():
            for images, masks in test_pbar:
                images = images.to(tissue_model.device)
                masks = masks.to(tissue_model.device)

                loss_val = tissue_model.test_step((images, masks))
                test_losses.append(loss_val)

                test_pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_test_loss = sum(test_losses) / len(test_losses)
        test_metrics = tissue_model.test_epoch_end(avg_test_loss)

        ### Logging
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(
            f"  Train - Loss: {train_metrics['train_loss']:.4f}, "
            f"Acc: {train_metrics['train_accuracy']:.4f}, "
            f"F1: {train_metrics['train_f1']:.4f}"
        )
        print(
            f"  Test  - Loss: {test_metrics['test_loss']:.4f}, "
            f"Acc: {test_metrics['test_accuracy']:.4f}, "
            f"F1: {test_metrics['test_f1']:.4f}"
        )

        ### Model checkpointing
        current_test_f1 = test_metrics["test_f1"]

        # Save as temp checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": tissue_model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "test_f1": current_test_f1,
            },
            temp_checkpoint,
        )

        # Check for best model and save if so
        if current_test_f1 > best_test_f1:
            best_test_f1 = current_test_f1
            best_epoch = epoch + 1

            # Copy temp checkpoint to best model
            torch.save(tissue_model.model.state_dict(), best_model_path)
            print(f"New best model saved! (F1: {best_test_f1:.4f})")

        epoch_pbar.set_postfix(
            {"best_f1": f"{best_test_f1:.4f}", "best_epoch": best_epoch}
        )

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Test F1: {best_test_f1:.4f}")
    print(f"  Best model saved to: {best_model_path}")

    # Delete temporary checkpoint
    if temp_checkpoint.exists():
        temp_checkpoint.unlink()
        print("Cleaned up temporary checkpoints")

    print(f"{'=' * 60}\n")

    tissue_model.finish()

    return best_test_f1, best_epoch


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object loaded from configs/config.yaml
    """
    load_dotenv()

    seed = cfg.get("seed", 42)
    seed_everything(seed)

    # Extract configuration values
    model_name = cfg.model.name
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    data_dir = cfg.data.data_dir
    mask_dir = cfg.data.mask_dir
    num_augmentations = cfg.data.num_augmentations
    checkpoint_path = cfg.checkpoint_path
    gradient_clipping = cfg.model.training.get("gradient_clipping", True)
    max_norm = cfg.model.training.get("max_norm", 1.0)

    # Initialize model
    base_model = hydra.utils.instantiate(cfg.model.params)

    encoder = cfg.model.params.get("encoder", None)
    weights = cfg.model.params.get("encoder_weights", None)

    preprocessor = None
    if encoder and weights:
        preprocessor = get_preprocessing_fn(encoder, pretrained=weights)

    tissue_model = TissueSegmentationModel(
        model=base_model, cfg=cfg, preprocessor=preprocessor
    )

    print(f"\nUsing device: {tissue_model.device}")
    print(f"Model name: {model_name}")

    train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = (
        get_image_and_mask_paths(data_dir, mask_dir)
    )

    binary = cfg.model.params.classes == 2

    train_dataset = PixelClassificationDataset(
        train_image_paths,
        train_mask_paths,
        transform=get_train_transform(),
        augmentations_per_image=num_augmentations,
        binary_mode=binary,
    )
    test_dataset = PixelClassificationDataset(
        test_image_paths, test_mask_paths, transform=None, binary_mode=binary
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Starting training...")
    best_f1, best_epoch = train(
        tissue_model=tissue_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        gradient_clipping=gradient_clipping,
        max_norm=max_norm,
    )


if __name__ == "__main__":
    main()

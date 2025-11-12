import sys
from pathlib import Path
# Add project root to path
script_path = Path(__file__).resolve()
scripts_dir = script_path.parent
src_dir = scripts_dir.parent
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.data.dataset import PixelClassificationDataset
from src.models.cnn_model import UNet
from src.utils.helpers import get_image_and_mask_paths
from tqdm import tqdm

CHECKPOINT_PATH = "checkpoints"
DATA_DIR = "data/cnn_training/resized_images"
MASK_DIR = "data/cnn_training/resized_masks"
NUM_AUGMENTATIONS_PER_IMAGE = 10
EPOCHS = 100

train_transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=90, p=0.75, interpolation=3), 
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),

                    # Color Augmentations
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                    A.GaussNoise(p=0.2),

                    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                    ToTensorV2()
                ])


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    epoch_pbar = tqdm(range(epochs), desc="Training", position=0)
    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                         position=1, leave=False)
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                         position=1, leave=False)
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch+1
            torch.save(model.state_dict(), CHECKPOINT_PATH + f"/best_model.pth")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, CHECKPOINT_PATH + f"/checkpoint_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    print(f"\nTraining completed!")
    print(f"Best model at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    best_model_state = torch.load(CHECKPOINT_PATH + "/best_model.pth", weights_only=True)
    torch.save(best_model_state, CHECKPOINT_PATH + "/final_model.pth")
    print("Saved final model successfully!")
    return best_val_loss, best_epoch



def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet().to(device)
    print("Loaded model successfully!")

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = get_image_and_mask_paths(DATA_DIR, MASK_DIR)
    train_dataset = PixelClassificationDataset(train_image_paths, train_mask_paths, transform=train_transform, augmentations_per_image=NUM_AUGMENTATIONS_PER_IMAGE)
    test_dataset = PixelClassificationDataset(test_image_paths, test_mask_paths, transform=None)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Calculate class weights for imbalanced data
    print("\nCalculating class weights...")
    class_counts = torch.zeros(3)
    for _, masks in tqdm(train_loader, desc="Analyzing classes"):
        for c in range(3):
            class_counts[c] += (masks == c).sum()
    
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (3 * class_counts)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Use weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Start training...")
    best_val_loss, best_epoch = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS)
    print("\n" + "="*60)
    print(f"Finished Training! Training Summary:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {CHECKPOINT_PATH}/final_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()
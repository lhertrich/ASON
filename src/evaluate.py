import sys
from pathlib import Path

# Add project root to path
src_dir = Path().resolve()
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json
import os
import argparse

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader
from datetime import datetime

from src.models.cnn_model import UNet
from src.data.dataset import PixelClassificationDataset
from src.utils.helpers import get_image_and_mask_paths, compare_two_images, clean_mask


def get_data(data_dir: str, mask_dir: str):
    DATA_DIR = str(project_root / data_dir)
    MASK_DIR = str(project_root / mask_dir)
    train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = get_image_and_mask_paths(DATA_DIR, MASK_DIR)
    test_dataset = PixelClassificationDataset(test_image_paths, test_mask_paths, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return test_loader


def load_model(checkpoint_path: Path, device: str):
    model_path = project_root / checkpoint_path
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def evaluate(
    checkpoint_path:str, 
    data_dir: str, 
    mask_dir: str,
    output_path: str = None
    ):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)
    test_loader = get_data(data_dir, mask_dir)

    model.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.cpu().numpy()

            outputs = model(images)

            pred_classes = torch.argmax(outputs, dim=1)  
            pred_classes = pred_classes.cpu().numpy()

            preds.append(pred_classes)
            gts.append(masks)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    preds_flat = preds.flatten()
    gt_flat = gts.flatten()

    accuracy = accuracy_score(gt_flat, preds_flat)
    f1_macro = f1_score(gt_flat, preds_flat, average='macro')
    f1_weighted = f1_score(gt_flat, preds_flat, average='weighted')
    f1_per_class = f1_score(gt_flat, preds_flat, average=None)

    # Confusion matrix
    cm = confusion_matrix(gt_flat, preds_flat)

    # Detailed classification report
    report = classification_report(gt_flat, preds_flat, 
                                target_names=['Background', 'Nucleus', 'Other'])  # Adjust class names

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score per class: {f1_per_class}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_path': checkpoint_path,
        'metrics': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': f1_per_class.tolist(),
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    Path("results").mkdir(parents=True, exist_ok=True) 
    
    # Save to file
    if not output_path:
        output_path = f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the trained model')
    
    parser.add_argument('--checkpoint_path', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='data/cnn_training/resized_images',
                       help='Directory containing test images')
    parser.add_argument('--mask_dir', type=str,
                       default='data/cnn_training/resized_masks',
                       help='Directory containing test masks')
    parser.add_argument('--output_path', type=str,
                       default=None,
                       help='Path to save evaluation results (JSON)')
    
    args = parser.parse_args()
    
    # Call evaluate with parsed arguments
    evaluate(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        output_path=args.output_path
    )
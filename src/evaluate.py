import sys
from pathlib import Path

import torch
import numpy as np
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Add project root to path
script_path = Path(__file__).resolve()
src_dir = script_path.parent
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.dataset import PixelClassificationDataset
from src.utils.helpers import get_image_and_mask_paths


def load_model(checkpoint_path: Path, cfg: DictConfig, device: torch.device):
    """
    Load a trained model from checkpoint using Hydra config.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        cfg: Hydra configuration object
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize model using Hydra instantiation (same as training)
    model = hydra.utils.instantiate(cfg.model.params)
    model = model.to(device)
    
    # Load the trained weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    return model


def get_preprocessing_fn_from_config(cfg: DictConfig):
    """
    Get preprocessing function from config (same as training).
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Preprocessing function or None
    """
    encoder = cfg.model.params.get("encoder_name", None)
    weights = cfg.model.params.get("encoder_weights", None)
    
    preprocessor = None
    if encoder and weights:
        try:
            preprocessor = get_preprocessing_fn(encoder, pretrained=weights)
            print(f"✓ Using preprocessor for {encoder} with {weights} weights")
        except Exception as e:
            print(f"⚠ Warning: Could not load preprocessor: {e}")
            preprocessor = None
    
    return preprocessor


def evaluate_model(model, test_loader, device, num_classes=2):
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        num_classes: Number of classes in the dataset
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    preds = []
    gts = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
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
    
    # Calculate metrics
    accuracy = accuracy_score(gt_flat, preds_flat)
    f1_macro = f1_score(gt_flat, preds_flat, average="macro")
    f1_weighted = f1_score(gt_flat, preds_flat, average="weighted")
    f1_per_class = f1_score(gt_flat, preds_flat, average=None)
    
    # For binary classification, also get binary F1 (tissue only)
    if num_classes == 2:
        f1_binary = f1_score(gt_flat, preds_flat, average="binary")
    else:
        f1_binary = None
    
    # Confusion matrix
    cm = confusion_matrix(gt_flat, preds_flat)
    
    # Class names based on number of classes
    if num_classes == 2:
        class_names = ["Background", "Tissue"]
    else:
        class_names = ["Background", "Tissue Type 1", "Tissue Type 2"]
    
    # Detailed classification report
    report = classification_report(gt_flat, preds_flat, target_names=class_names)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_binary": f1_binary,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": preds,
        "ground_truth": gts,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main evaluation function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object loaded from configs/config.yaml
    """
    # Load environment variables
    load_dotenv()
    
    print("="*60)
    print("Evaluation Configuration:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Setup device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Get model name and checkpoint path
    model_name = cfg.model.name
    checkpoint_path = Path(cfg.checkpoint_path) / f"{model_name}.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model using config
    model = load_model(checkpoint_path, cfg, device)
    print(f"✓ Model loaded successfully: {model_name}")
    
    # Load data
    data_dir = cfg.data.data_dir
    mask_dir = cfg.data.mask_dir
    batch_size = cfg.training.get("batch_size", 8)
    
    train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = (
        get_image_and_mask_paths(data_dir, mask_dir)
    )
    
    # Determine if binary mode
    num_classes = cfg.model.params.classes
    binary_mode = num_classes == 2
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Binary mode: {binary_mode}")
    
    # Create test dataset (no augmentation, same as training)
    test_dataset = PixelClassificationDataset(
        test_image_paths,
        test_mask_paths,
        transform=None,
        binary_mode=binary_mode
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, num_classes=num_classes)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
    if results['f1_binary'] is not None:
        print(f"F1 Score (Binary - Tissue only): {results['f1_binary']:.4f}")
    print(f"F1 Score per class: {results['f1_per_class']}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])
    print("="*60)
    
    # Save results to JSON
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": {
            "accuracy": float(results['accuracy']),
            "f1_macro": float(results['f1_macro']),
            "f1_weighted": float(results['f1_weighted']),
            "f1_binary": float(results['f1_binary']) if results['f1_binary'] is not None else None,
            "f1_per_class": results['f1_per_class'].tolist(),
        },
        "confusion_matrix": results['confusion_matrix'].tolist(),
        "classification_report": results['classification_report'],
    }
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to file with model name and timestamp
    output_path = results_dir / f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

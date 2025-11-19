import sys
import wandb
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig
from dotenv import load_dotenv

import hydra

# Add project root to path
script_path = Path(__file__).resolve()
src_dir = script_path.parent
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.random_forest import RandomForestWrapper
from src.utils.helpers import seed_everything

@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object loaded from configs/config.yaml
    """
    load_dotenv()

    seed = cfg.get("seed", 42)
    seed_everything(seed)

    use_wandb = cfg.wandb.get("use", False)
    if use_wandb:
        wandb_project = cfg.get("wandb", {}).get("project", "rp-tissue_segmentation")
        
        wandb.init(
            project=wandb_project,
            name=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["random_forest", "traditional_ml"]
        )
        print(f"WandB initialized: {wandb_project}")
    else:
        print("WandB logging disabled")

    # Extract configuration values
    model_name = cfg.model.name
    checkpoint_path = cfg.checkpoint_path
    sample_ratio = cfg.model.get("sample_ratio", 1.0)

    print(f"\nModel name: {model_name}")
    print(f"Sample ratio: {sample_ratio} ({sample_ratio*100:.0f}% of pixels)")

    print("\n" + "=" * 60)
    print("Initializing Random Forest Classifier")
    print("=" * 60)
    rf_classifier = RandomForestWrapper(cfg)

    print("\n" + "=" * 60)
    print("Preparing Data")
    print("=" * 60)
    rf_classifier.prepare_data(subsample_rate=sample_ratio)

    if use_wandb:
        wandb.log({
            "data/train_samples": len(rf_classifier.X_train),
            "data/test_samples": len(rf_classifier.X_test),
            "data/num_features": rf_classifier.X_train.shape[1],
            "data/sample_ratio": sample_ratio
        })

    # Train the model
    print("\n" + "=" * 60)
    print("Training Random Forest")
    print("=" * 60)
    
    import time
    start_time = time.time()
    rf_classifier.fit()
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")

    if use_wandb:
        wandb.log({"training/time_seconds": training_time})

    # Evaluate on training data
    print("\n" + "=" * 60)
    print("Evaluating on Training Data")
    print("=" * 60)
    train_results = rf_classifier.evaluate(
        mode="train"
    )

    # Evaluate on test data
    print("\n" + "=" * 60)
    print("Evaluating on Test Data")
    print("=" * 60)
    test_results = rf_classifier.evaluate()

    if use_wandb:
        # Training metrics
        wandb.log({
            "train/accuracy": train_results["accuracy"],
            "train/f1": train_results["f1"]
        })
        
        if train_results["f1_binary"] is not None:
            wandb.log({"train/f1_binary": train_results["f1_binary"]})

        # Test metrics
        wandb.log({
            "test/accuracy": test_results["accuracy"],
            "test/f1": test_results["f1"],
        })
        
        if test_results["f1_binary"] is not None:
            wandb.log({"test/f1_binary": test_results["f1_binary"]})

    # Save the model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    save_path = rf_classifier.save(Path(checkpoint_path), name=model_name)

    # Print final summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nTraining Results:")
    print(f"Accuracy: {train_results['accuracy']:.4f}")
    print(f"F1: {train_results['f1']:.4f}")
    if train_results['f1_binary'] is not None:
        print(f"  F1 (Binary): {train_results['f1_binary']:.4f}")
    print("\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1: {test_results['f1']:.4f}")
    if test_results['f1_binary'] is not None:
        print(f"  F1 (Binary): {test_results['f1_binary']:.4f}")
    print(f"\nModel saved to: {save_path}")
    print("=" * 60)

    # Finish WandB run
    if use_wandb:
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    main()
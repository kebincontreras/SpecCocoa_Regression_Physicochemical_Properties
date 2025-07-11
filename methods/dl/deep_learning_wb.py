import sys
import os
import argparse
import torch
import json
import wandb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure warnings only if not already configured globally
try:
    import warnings_config
except ImportError:
    import warnings
    warnings.filterwarnings('ignore')

from methods.resources.dataloader import prepare_data
from methods.dl import regress, build_regressor

# Silent W&B configuration
os.environ["WANDB_SILENT"] = "true"

config_path = 'methods/dl_models.json'

def main(modality, config_file, wb_project):
    """ Execute Deep Learning for the indicated modality with configurations from JSON. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load configurations from JSON
    with open(config_file, "r") as f:
        experiments = json.load(f)

    print(f"Executing Deep Learning on {modality} with {len(experiments)} configurations...\n")

    for exp in experiments:
        classifier_name = exp["name"]
        batch_size = exp["batch_size"]
        epochs = exp["epochs"]
        lr = exp["lr"]
        weight_decay = exp["weight_decay"]

        print(f"Executing model: {classifier_name} (epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay})")

        # Initialize W&B
        wandb.init(
        project=wb_project,
        name=f"{classifier_name}_{modality}_experiment",
        config=exp
        )

        
        # Prepare data
        train_loader, test_loader, num_bands, num_outputs, _ = prepare_data(
            "cocoa_regression", modality, dl=True,
            dataset_params=dict(batch_size=batch_size, num_workers=4)
        )

        # Build model
        model_dict = build_regressor(classifier_name, {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay
        }, num_bands, num_outputs, device=device)

        # Train and evaluate
        dict_metrics = regress(model_dict, train_loader, test_loader, classifier_name, modality)

        # Log metrics
        wandb.log(dict_metrics)
        wandb.log({"Total Parameters": sum(p.numel() for p in model_dict['model'].parameters())})
        wandb.finish()

    print(f"Deep Learning completed on {modality}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Spectral Regression")
    parser.add_argument("--modality", type=str, required=True, choices=["NIR", "VIS"])
    parser.add_argument("--config", type=str, required=True, help="JSON file with experiments")
    parser.add_argument("--wb_project", type=str, required=True, help="W&B project name")

    args = parser.parse_args()
    main(args.modality, args.config, args.wb_project)
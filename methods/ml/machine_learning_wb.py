import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure warnings only if not already configured globally
try:
    import methods.warnings_config as warnings_config
except ImportError:
    import warnings
    warnings.filterwarnings('ignore')

from methods.resources.dataloader import prepare_data
import wandb
import argparse
import json
import numpy as np
from methods.ml import regress, build_regressor

os.environ["WANDB_SILENT"] = "true"

def main(modality, config_file, wb_project):
    """ Execute Machine Learning for the indicated modality with configurations from JSON. """

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load configurations from JSON
    with open(config_file, "r") as f:
        hyperparams = json.load(f)

    print(f"\nExecuting Machine Learning on {modality}...\n")

    # Prepare data
    train_dataset, test_dataset, _, _, _ = prepare_data("cocoa_regression", modality, dl=False)
    
    best_configs = {}

    for regressor_name, params_list in hyperparams.items():
        best_score = float("-inf")
        best_params = None
        
        for i, params in enumerate(params_list):
            wandb.init(
                project=wb_project,
                name=f"{regressor_name}_{modality}_config_{i}",
                config=params
            )

            print(f"\nTraining {regressor_name} with {params} on {modality}")
            regressor = build_regressor(regressor_name, params)
            _, _, dict_metrics = regress(regressor, train_dataset, test_dataset, modality, save_name=f"{regressor_name}_{i}")

            # Log metrics to W&B
            for dataset_name in ["train", "test"]:
                for metric in ["mse", "r2", "mae"]:
                    for property_name, value in dict_metrics[dataset_name][metric].items():
                        wandb.log({f"{dataset_name}/{metric}/{property_name}": value})

            # Evaluate metric of interest (example: average R² on test)
            r2_mean_test = np.mean(list(dict_metrics["test"]["r2"].values()))

            if r2_mean_test > best_score:
                best_score = r2_mean_test
                best_params = params

            wandb.finish()

        best_configs[regressor_name] = best_params

    print(f"\nMachine Learning completed on {modality}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Spectral Regression")
    parser.add_argument("--modality", type=str, required=True, choices=["NIR", "VIS"])
    parser.add_argument("--config", type=str, required=True, help="JSON file with hyperparameters")
    parser.add_argument("--wb_project", type=str, required=True, help="W&B project name")

    args = parser.parse_args()
    main(args.modality, args.config, args.wb_project)

config_path = 'methods/ml_hyperparams.json'
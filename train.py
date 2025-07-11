import warnings_config  # Configure warnings before any other imports
import os
import subprocess
import json
import torch
from methods.dl.deep_learning_wb import *

# Path to the experiments JSON file in resources/
EXPERIMENTS_FILE = "methods/resources/experiments.json"

# Ensure 'configs' directory exists
os.makedirs("configs", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print device information
print("=" * 50)
print("DEVICE CONFIGURATION")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Training Device: GPU (CUDA)")
else:
    print(f"Training Device: CPU")
print("=" * 50)
print()

# Read experiment configurations
with open(EXPERIMENTS_FILE, "r") as f:
    experiments = json.load(f)

# Extract ML and DL experiment configs
ML_HYPERPARAMS = experiments["ml_hyperparams"]
DL_MODELS = experiments["dl_models"]

# Modalities to run
MODALITIES = ["VIS", "NIR"]

# Flags to execute ML and DL
EXECUTE_ML = False
EXECUTE_DL = True

# WandB project names
WB_PROJECT_ML = "ML_Cocoa_Regressionn"
WB_PROJECT_DL = "DL_Cocoa_Regressionn"

def run_experiments():
    print("\n===== EXPERIMENT CONFIGURATION =====")
    print(f"Execute Machine Learning: {'Yes' if EXECUTE_ML else 'No'}")
    print(f"Execute Deep Learning: {'Yes' if EXECUTE_DL else 'No'}")
    print(f"W&B Project (ML): {WB_PROJECT_ML}")
    print(f"W&B Project (DL): {WB_PROJECT_DL}")
    print(f"Training Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print("=" * 40)

    # Save JSON config files inside configs/
    ml_config_path = "configs/ml_hyperparams.json"
    dl_config_path = "configs/dl_models.json"

    with open(ml_config_path, "w") as ml_file:
        json.dump(ML_HYPERPARAMS, ml_file, indent=4)

    with open(dl_config_path, "w") as dl_file:
        json.dump(DL_MODELS, dl_file, indent=4)

    # Run Machine Learning experiments
    if EXECUTE_ML:
        for mod in MODALITIES:
            print(f"Running Machine Learning for {mod}...\n")
            subprocess.run([
                'python', '-W', 'ignore', 'methods/ml/machine_learning_wb.py', '--modality', mod, '--config', 'configs/ml_hyperparams.json', '--wb_project', WB_PROJECT_ML
            ], check=True, env={**os.environ, 'PYTHONWARNINGS': 'ignore'})

    # Run Deep Learning experiments
    if EXECUTE_DL:
        for mod in MODALITIES:
            print(f"Running Deep Learning for {mod}...\n")
            subprocess.run([
                'python', '-W', 'ignore', 'methods/dl/deep_learning_wb.py', '--modality', mod, '--config', 'configs/dl_models.json', '--wb_project', WB_PROJECT_DL
            ], check=True, env={**os.environ, 'PYTHONWARNINGS': 'ignore'})

    print("All experiments have been successfully executed.")

if __name__ == "__main__":
    run_experiments()

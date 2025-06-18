import subprocess
import json
import os
import torch
from methods.ml.machine_learning_wb import *
from methods.dl.deep_learning_wb import *

# Path to the experiments JSON file in resources/
EXPERIMENTS_FILE = "methods/resources/experiments.json"

# Ensure 'configs' directory exists
os.makedirs("configs", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Read experiment configurations
with open(EXPERIMENTS_FILE, "r") as f:
    experiments = json.load(f)

# Extract ML and DL experiment configs
ML_HYPERPARAMS = experiments["ml_hyperparams"]
DL_MODELS = experiments["dl_models"]

# Modalities to run
#MODALITIES = ["VIS"]
# MODALITIES = ["NIR"]
MODALITIES = ["VIS", "NIR"]

# Flags to execute ML and DL
EXECUTE_ML = True
EXECUTE_DL = True


# WandB project names
WB_PROJECT_ML = "ML_Cocoa_Regressionn"
WB_PROJECT_DL = "4kebin_DL_Cocoa_Regressionn"

def run_experiments():
    print("\n===== EXPERIMENT CONFIGURATION =====")
    print(f"🔹 Execute Machine Learning: {'Yes' if EXECUTE_ML else 'No'}")
    print(f"🔹 Execute Deep Learning: {'Yes' if EXECUTE_DL else 'No'}")
    print(f"🔹 W&B Project (ML): {WB_PROJECT_ML}")
    print(f"🔹 W&B Project (DL): {WB_PROJECT_DL}")

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
            print(f"\n🚀 Running Machine Learning for {mod}...\n")
            subprocess.run([
                'python', 'methods/ml/machine_learning_wb.py', '--modality', mod, '--config', 'configs/ml_hyperparams.json', '--wb_project', WB_PROJECT_ML
            ], check=True)

    # Run Deep Learning experiments
    if EXECUTE_DL:
        for mod in MODALITIES:
            print(f"\n🚀 Running Deep Learning for {mod}...\n")
            subprocess.run([
                'python', 'methods/dl/deep_learning_wb.py', '--modality', mod, '--config', 'configs/dl_models.json', '--wb_project', WB_PROJECT_DL
            ], check=True)

    print("\n✅ All experiments have been successfully executed.")

if __name__ == "__main__":
    run_experiments()

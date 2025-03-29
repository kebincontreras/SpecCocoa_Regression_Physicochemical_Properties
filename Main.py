import subprocess
import json
import os
import torch

# 🔹 Path to the JSON experiment file
EXPERIMENTS_FILE = "methods/experiments.json"

# 🔹 Ensure the `methods` directory exists
os.makedirs("methods", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 🔹 Read the experiment configurations
with open(EXPERIMENTS_FILE, "r") as f:
    experiments = json.load(f)

# 🔹 Get Machine Learning and Deep Learning experiments
ML_HYPERPARAMS = experiments["ml_hyperparams"]
DL_MODELS = experiments["dl_models"]

# 🔹 Define modalities (VIS first, then NIR)
#MODALITIES = ["VIS"]
#MODALITIES = [ "NIR" ]
MODALITIES = ["VIS", "NIR"]


# 🔹 Execute ML and DL (set False if not needed)
EXECUTE_ML = False
EXECUTE_DL = True

# 🔹 W&B Project Names
WB_PROJECT_ML = "ML_Cocoa_Regressionn"
WB_PROJECT_DL = "emma_DL_Cocoa_Regressionn "

def run_experiments():
    """ Run ML and DL experiments based on the configuration in experiments.json """

    print("\n===== EXPERIMENT CONFIGURATION =====")
    print(f"🔹 Execute Machine Learning: {'Yes' if EXECUTE_ML else 'No'}")
    print(f"🔹 Execute Deep Learning: {'Yes' if EXECUTE_DL else 'No'}")
    print(f"🔹 W&B Project (ML): {WB_PROJECT_ML}")
    print(f"🔹 W&B Project (DL): {WB_PROJECT_DL}")

    # 🔹 Save JSON files inside `methods/`
    ml_config_path = "methods/ml_hyperparams.json"
    dl_config_path = "methods/dl_models.json"

    with open(ml_config_path, "w") as ml_file:
        json.dump(ML_HYPERPARAMS, ml_file, indent=4)

    with open(dl_config_path, "w") as dl_file:
        json.dump(DL_MODELS, dl_file, indent=4)

    # 🔹 Execute Machine Learning first
    if EXECUTE_ML:
        for mod in MODALITIES:
            print(f"\n🚀 Running Machine Learning for {mod}...\n")
            subprocess.run(["python", "machine_learning_wb.py", "--modality", mod, "--config", ml_config_path, "--wb_project", WB_PROJECT_ML], check=True)

    # 🔹 Execute Deep Learning next
    if EXECUTE_DL:
        for mod in MODALITIES:
            print(f"\n🚀 Running Deep Learning for {mod}...\n")
            subprocess.run(["python", "deep_learning_wb.py", "--modality", mod, "--config", dl_config_path, "--wb_project", WB_PROJECT_DL], check=True)

    print("\n✅ All experiments have been successfully executed.")

if __name__ == "__main__":
    run_experiments()


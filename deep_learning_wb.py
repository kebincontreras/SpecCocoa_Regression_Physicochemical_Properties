import os
import torch
import argparse
import wandb
import json
from methods.dataloader import prepare_data
from methods.dl import build_regressor, regress

# Configuración silenciosa de W&B
os.environ["WANDB_SILENT"] = "true"

def main(modality, config_file, wb_project):
    """ Ejecuta Deep Learning para la modalidad indicada con configuraciones desde JSON. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🔹 No agregamos "methods/" porque Main.py ya pasa la ruta correcta
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"❌ No se encontró el archivo de configuración: {config_file}")

    # 🔹 Cargar configuraciones desde JSON
    with open(config_file, "r") as f:
        experiments = json.load(f)

    print(f"\n🚀 Ejecutando Deep Learning en {modality} con {len(experiments)} configuraciones...\n")

    for exp in experiments:
        classifier_name = exp["name"]
        batch_size = exp["batch_size"]
        epochs = exp["epochs"]
        lr = exp["lr"]
        weight_decay = exp["weight_decay"]

        # 🔹 Inicializar W&B
        wandb.init(
            project=wb_project,
            name=f"{classifier_name}_{modality}_experiment",
            config=exp
        )

        # 🔹 Preparar datos
        train_loader, test_loader, num_bands, num_outputs, _ = prepare_data(
            "cocoa_regression", modality, dl=True,
            dataset_params=dict(batch_size=batch_size, num_workers=4)
        )

        # 🔹 Construir modelo
        model_dict = build_regressor(classifier_name, {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay
        }, num_bands, num_outputs, device=device)

        # 🔹 Entrenar y evaluar
        dict_metrics = regress(model_dict, train_loader, test_loader, classifier_name, modality)

        # 🔹 Registrar métricas
        wandb.log(dict_metrics)
        wandb.log({"Total Parameters": sum(p.numel() for p in model_dict['model'].parameters())})
        wandb.finish()

    print(f"\n✅ Deep Learning completado en {modality}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Spectral Regression")
    parser.add_argument("--modality", type=str, required=True, choices=["NIR", "VIS"])
    parser.add_argument("--config", type=str, required=True, help="Archivo JSON con experimentos")
    parser.add_argument("--wb_project", type=str, required=True, help="Nombre del proyecto en W&B")

    args = parser.parse_args()
    main(args.modality, args.config, args.wb_project)

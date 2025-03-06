import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter
from methods.dataloader import prepare_data
from methods.ml import regress, build_regressor
from methods.metrics_2 import print_results
import os
os.environ["WANDB_SILENT"] = "true"  

def init_parser():
    parser = argparse.ArgumentParser(description='Machine Learning Spectral Regression')
    parser.add_argument("--save-name", default="exp", type=str, help="Path to save specific experiment")

    parser.add_argument("--dataset", type=str, default="cocoa_regression",
                    choices=['cocoa_public', 'cocoa_regression'], help="Dataset name")

    parser.add_argument('--regressor', type=str, default='none', help='Regressor name',
                        choices=['svr', 'rfr', 'mlp', 'knnr'])
    return parser

def main(regressor_name):
    parser = init_parser()
    args = parser.parse_args()

    args.regressor = regressor_name
    save_name = args.save_name
    dataset_name = args.dataset

    # 🔹 Iniciar un experimento en Weights & Biases
    wandb.init(project="spectral_regression_Machine_Learning", name=f"{regressor_name}_{dataset_name}", config={
        "regressor": regressor_name,
        "dataset": dataset_name,
        "save_name": save_name
    })

    # Preparar datos y construir el modelo
    train_dataset, test_dataset, _, _ = prepare_data(dataset_name, dl=False)
    regressor = build_regressor(regressor_name)

    save_name += f"{args.dataset}_{regressor_name}"
    train_dataset, test_dataset, dict_metrics = regress(regressor, train_dataset, test_dataset, save_name=save_name)

    # 🔹 Registrar métricas en Weights & Biases
    for phase in ["train", "test"]:
        for metric, values in dict_metrics[phase].items():
            for var_name, value in values.items():
                wandb.log({f"{phase}/{metric}_{var_name}": value})

    print_results(regressor_name, dataset_name, dict_metrics)

    wandb.finish()  # 🔹 Finalizar el experimento en wandb

    return train_dataset, test_dataset, dict_metrics  # 🔹 Devuelve datasets y métricas

if __name__ == '__main__':
    wandb.login()  # 🔹 Asegurar que está logueado en wandb

    regressors = ['mlp', 'knnr', 'svr', 'rfr']
    results = {}

    for regressor_name in regressors:
        train_dataset, test_dataset, metrics = main(regressor_name)
        results[regressor_name] = {
            "train": train_dataset,
            "test": test_dataset,
            "metrics": metrics
        }

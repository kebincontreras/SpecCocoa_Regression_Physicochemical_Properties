import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter
from methods.dataloader import prepare_data
from methods.ml import regress, build_regressor
from methods.metrics_2 import print_results
import os
import numpy as np  

os.environ["WANDB_SILENT"] = "true"  

hyperparams = {
    "svr": [
        {"C": 1e3, "gamma": 0.1}, 
        {"C": 1e4, "gamma": 0.01}, 
        {"C": 1e5, "gamma": 1}
    ],
    "rfr": [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 500, "max_depth": 15},
        {"n_estimators": 1000, "max_depth": 20}
    ],
    "mlp": [
        {"solver": "adam", "max_iter": 500, "alpha": 1e-4},
        {"solver": "lbfgs", "max_iter": 1000, "alpha": 1e-3},
        {"solver": "sgd", "max_iter": 1500, "alpha": 1e-2}
    ],
    "knnr": [
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 10}
    ]
}


def init_parser():
    parser = argparse.ArgumentParser(description='Machine Learning Spectral Regression')
    parser.add_argument("--save-name", default="exp", type=str, help="Path to save specific experiment")

    parser.add_argument("--dataset", type=str, default="cocoa_regression",
                    choices=['cocoa_public', 'cocoa_regression'], help="Dataset name")

    parser.add_argument('--regressor', type=str, default='none', help='Regressor name',
                        choices=['svr', 'rfr', 'mlp', 'knnr'])
    return parser


def main(regressor_name, hyperparams_list):
    parser = init_parser()
    args = parser.parse_args()

    args.regressor = regressor_name
    save_name = args.save_name
    dataset_name = args.dataset

    train_dataset, test_dataset, _, _ = prepare_data(dataset_name, dl=False)

    best_score = float("-inf")
    best_params = None

    for i, params in enumerate(hyperparams_list):
        wandb.init(project="2spectral_regression_Machine_Learning", name=f"{regressor_name}_{dataset_name}_config_{i}", config=params)
        
        print(f"\n🔹 Entrenando {regressor_name} con {params}")
        regressor = build_regressor(regressor_name, params)
        _, _, dict_metrics = regress(regressor, train_dataset, test_dataset, save_name=f"{save_name}_{i}")

        # 🔹 Guardar métricas en W&B correctamente
        for dataset_name in ["train", "test"]:
            for metric in ["mse", "r2", "mae"]:
                for property_name, value in dict_metrics[dataset_name][metric].items():
                    wandb.log({f"{dataset_name}/{metric}/{property_name}": value})
        
        # Evaluar métrica de interés (ejemplo: R² promedio en test)
        r2_mean_test = np.mean([val for val in dict_metrics["test"]["r2"].values()])

        if r2_mean_test > best_score:
            best_score = r2_mean_test
            best_params = params

        wandb.finish()
    
    print(f"\n🔹 Mejor configuración para {regressor_name}: {best_params} con R²={best_score:.4f}")

    return best_params


if __name__ == '__main__':
    wandb.login()

    regressors = ['mlp', 'knnr', 'svr', 'rfr']
    best_configs = {}

    for regressor_name in regressors:
        best_params = main(regressor_name, hyperparams[regressor_name])
        best_configs[regressor_name] = best_params

    print("\n===== Mejor Configuración por Modelo =====")
    for model, params in best_configs.items():
        print(f"{model}: {params}")

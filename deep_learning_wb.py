import torch
import argparse
import numpy as np
import wandb
from methods.dataloader import prepare_data
from methods.dl import build_regressor, regress
from methods.models.config import (  
    config_spectralnet, config_lstm, config_cnn, 
    config_TSTransformer, config_spectralformer
)
import os
os.environ["WANDB_SILENT"] = "true"  


def init_parser():
    parser = argparse.ArgumentParser(description="Deep Learning Spectral Regression")
    parser.add_argument("--save-name", default="exp", type=str, help="Path to save specific experiment")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="cocoa_regression",
                        choices=['indian_pines', 'cocoa_public', 'cocoa_regression'],
                        help="Dataset name")
    
    # Model args
    parser.add_argument("--regressor", type=str, default="none", help="Regressor name",
                        choices=["spectralnet", "transformer", "lstm", "cnn", "spectralformer", "karen"])

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step_lr"],
                        help="Learning rate scheduler")
    parser.add_argument("--step-size", type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma for learning rate scheduler")

    # GPU config
    parser.add_argument("-j", "--num-workers", default=4, type=int, metavar="N",
                        help="Number of data loading workers")

    # Model Configurations (🔹 Se agregan como argumentos opcionales)
    parser.add_argument("--spectralnet_architecture", type=int, nargs="+", default=config_spectralnet["architecture"],
                        help="Architecture for SpectralNet")

    parser.add_argument("--lstm_n_layers", type=int, default=config_lstm["n_layers"],
                        help="Number of LSTM layers")
    parser.add_argument("--lstm_dropout_rate", type=float, default=config_lstm["dropout_rate"],
                        help="LSTM dropout rate")

    parser.add_argument("--cnn_conv_layers", type=int, nargs="+", default=config_cnn["conv_layers"],
                        help="CNN convolutional layers")
    parser.add_argument("--cnn_kernel_size", type=int, default=config_cnn["kernel_size"],
                        help="CNN kernel size")
    parser.add_argument("--cnn_pool_size", type=int, default=config_cnn["pool_size"],
                        help="CNN pool size")
    parser.add_argument("--cnn_dropout_rate", type=float, default=config_cnn["dropout_rate"],
                        help="CNN dropout rate")

    parser.add_argument("--transformer_d_model", type=int, default=config_TSTransformer["d_model"],
                        help="Transformer model dimension")
    parser.add_argument("--transformer_n_heads", type=int, default=config_TSTransformer["n_heads"],
                        help="Transformer number of heads")
    parser.add_argument("--transformer_num_layers", type=int, default=config_TSTransformer["num_layers"],
                        help="Transformer number of layers")
    parser.add_argument("--transformer_dropout", type=float, default=config_TSTransformer["dropout"],
                        help="Transformer dropout rate")

    parser.add_argument("--spectralformer_dim", type=int, default=config_spectralformer["dim"],
                        help="SpectralFormer dimension")
    parser.add_argument("--spectralformer_depth", type=int, default=config_spectralformer["depth"],
                        help="SpectralFormer depth")
    parser.add_argument("--spectralformer_heads", type=int, default=config_spectralformer["heads"],
                        help="SpectralFormer heads")
    parser.add_argument("--spectralformer_dropout", type=float, default=config_spectralformer["dropout"],
                        help="SpectralFormer dropout rate")

    return parser

def main(classifier_name, batch_size, epochs, lr, weight_decay):
    # set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # parse arguments
    parser = init_parser()
    args = parser.parse_args()

    args.classifier = classifier_name
    args.batch_size = batch_size
    args.epochs = epochs
    args.lr = lr
    args.weight_decay = weight_decay

    # Iniciar W&B
    wandb.init(
        project="2cocoa_regression_Deep_Learning",
        entity="kebincontreras", 
        name=f"{classifier_name}_experiment",
        config=vars(args)  # 🔹 Guarda toda la configuración en W&B
    )

    # Preparar datos y modelo
    train_loader, test_loader, num_bands, num_outputs = prepare_data(args.dataset, dl=True, dataset_params=dict(batch_size=args.batch_size, num_workers=args.num_workers))

    regressor = build_regressor(classifier_name, dict(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                                      scheduler=args.scheduler, step_size=args.step_size, gamma=args.gamma), 
                                num_bands, num_outputs, device=device)

    dict_metrics = regress(regressor, train_loader, test_loader, save_name=args.save_name)

    print(f"Tipo de dict_metrics: {type(dict_metrics)}")  # Depuración

    # Contar parámetros
    total_params = sum(p.numel() for p in regressor["model"].parameters())  

    # Registrar métricas en W&B
    wandb.log(dict_metrics)
    wandb.log({"Total Parameters": total_params})

    # Guardar y mostrar resultados
    #print_results(classifier_name, args.dataset, dict_metrics)
    print(f"Total parameters: {total_params}")

    # Finalizar W&B
    wandb.finish()

    return dict_metrics


if __name__ == "__main__":
    Regression_1 = [
    dict(name='spectralnet', batch_size=128, epochs=30, lr=1e-4, weight_decay=1e-4),
    dict(name='spectralnet', batch_size=256, epochs=40, lr=5e-4, weight_decay=1e-5),
    
    dict(name='lstm', batch_size=32, epochs=30, lr=1e-4, weight_decay=1e-8),
    dict(name='lstm', batch_size=64, epochs=35, lr=5e-4, weight_decay=1e-6),
    
    dict(name='cnn', batch_size=512, epochs=30, lr=1e-3, weight_decay=0),
    dict(name='cnn', batch_size=256, epochs=40, lr=1e-4, weight_decay=1e-5),
    
    dict(name='transformer', batch_size=30, epochs=30, lr=1e-4, weight_decay=1e-8),
    dict(name='transformer', batch_size=64, epochs=50, lr=3e-4, weight_decay=1e-6),
    
    dict(name='spectralformer', batch_size=32, epochs=30, lr=1e-3, weight_decay=5e-4),
    dict(name='spectralformer', batch_size=128, epochs=45, lr=2e-4, weight_decay=1e-5)
]

    for classifier in Regression_1:
        main(classifier['name'], classifier['batch_size'], classifier['epochs'], classifier['lr'], classifier['weight_decay'])

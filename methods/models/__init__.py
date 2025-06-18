import torch
import torchmetrics
from torch import nn
from tqdm import tqdm

from .spectralnet import SpectralNet
from .transformer import TSTransformerEncoderClassiregressor as TSTransformerEncoder
from .lstm import Lstm
from methods.models import Lstm as LSTMModel
from .cnn import CNN
from .spectralformer import SpectralFormer


class ClassifierNet:
    def __init__(self, backbone: nn.Module, config: dict, hyperparameters: dict, device='cpu'):
        super(ClassifierNet, self).__init__()
        lr = hyperparameters['lr']
        weight_decay = hyperparameters['weight_decay']
        scheduler = hyperparameters['scheduler']
        step_size = hyperparameters['step_size']
        gamma = hyperparameters['gamma']

        self.epochs = hyperparameters['epochs']
        self.device = device

        self.model = backbone(**config)
        self.model.to(device)

        self.model_name = self.model.__class__.__name__.lower()

        self.criterion = nn.CrossEntropyLoss()
        self.acc = torchmetrics.functional.accuracy

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler == 'step_lr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            self.scheduler = None

    def fit(self, dataset):
        self.model.train()

        for epoch in range(self.epochs):
            data_loop = tqdm(enumerate(dataset), total=len(dataset))
            for _, data in data_loop:
                X, y = data

                X = X.to(self.device)
                y = y.to(self.device)

                if 'spectralformer' not in self.model_name:
                    X = X.squeeze()

                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                lr = format(self.optimizer.param_groups[0]['lr'], '.1e')
                data_loop.set_description(f'Train - Epoch {epoch + 1}/{self.epochs}, lr: {lr}')
                data_loop.set_postfix(loss=loss.item())

            if self.scheduler is not None:
                self.scheduler.step()

    def predict(self, dataset_name, dataset):
        self.model.eval()

        y_list = []
        y_pred_list = []
        with torch.no_grad():
            data_loop = tqdm(enumerate(dataset), total=len(dataset))
            for _, data in data_loop:
                X, y = data

                X = X.to(self.device)
                y = y.to(self.device)

                if self.model_name not in 'spectralformer':
                    X = X.squeeze()

                y_pred = self.model(X)
                y_pred = torch.argmax(y_pred, dim=1)

                y_list.append(y)
                y_pred_list.append(y_pred)

                data_loop.set_description(f'Prediction - {dataset_name}')

        y_list = torch.cat(y_list).detach().cpu().numpy()
        y_pred_list = torch.cat(y_pred_list).detach().cpu().numpy()

        return y_list, y_pred_list

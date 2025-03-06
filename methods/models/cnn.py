import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes, conv_layers, **kwargs):
        super(CNN, self).__init__()
        kernel_size = kwargs["kernel_size"]
        fc1_out_features = input_dim
        dropout_rate = kwargs["dropout_rate"]
        pool_size = kwargs["pool_size"]

        self.conv_layers = nn.ModuleList()
        in_channels = 1  
        for out_channels in conv_layers:

            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=False),
                nn.BatchNorm1d(out_channels),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding="same", bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )

            self.conv_layers.append(conv_block)
            in_channels = out_channels  # Actualizar in_channels para la próxima capa
        
        # self.pool = nn.MaxPool1d(pool_size)
        self.pool = nn.AvgPool1d(pool_size)
        self.fc1_size = conv_layers[-1] * (input_dim // pool_size ** len(conv_layers))
        
        self.fc1 = nn.Linear(self.fc1_size, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Asegurar que x tenga la forma correcta
        
        for conv in self.conv_layers:
            x = self.pool(conv(x))
        
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.relu(self.fc1(x))
        if self.dropout.p > 0:  
            x = self.dropout(x)
        x = self.fc2(x)
        return x

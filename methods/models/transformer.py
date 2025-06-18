import torch
import torch.nn as nn
import math

class TSTransformerEncoderClassiregressor(nn.Module):
    def __init__(self, feat_dim, input_dim, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()
        self.max_len = input_dim
        self.d_model = d_model
        self.n_heads = n_heads

        # Proyecto las características de entrada al d_model deseado
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = nn.Identity()  # Considera reemplazar esto con una implementación de codificación posicional si es necesario
        
        # Añadir LayerNorm después de la proyección
        self.norm_after_project = nn.LayerNorm(d_model)
        
        # Configurar el encoder del Transformer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation),
            num_layers)
        
        # Aplicar dropout después de la codificación posicional
        self.dropout_after_pos_enc = nn.Dropout(dropout)
        
        # Dropout antes de la capa de salida
        self.dropout1 = nn.Dropout(dropout)
        
        # Aplicar BatchNorm1d o LayerNorm antes de la capa de salida
        if norm == 'BatchNorm':
            self.norm_before_output = nn.BatchNorm1d(d_model * input_dim)
        elif norm == 'LayerNorm':
            self.norm_before_output = nn.LayerNorm(d_model * input_dim)
        else:
            raise ValueError("Unsupported norm type. Choose 'BatchNorm' or 'LayerNorm'.")

        # Capa de salida
        self.output_layer = nn.Linear(d_model * input_dim, num_classes)

    def forward(self, X):
        X = X[..., None]  # Agrega una dimensión extra al final si es necesario
        padding_masks = torch.ones(X.shape[0], X.shape[1], dtype=torch.bool, device=X.device)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.norm_after_project(inp)  # Aplica LayerNorm después de la proyección
        inp = self.pos_enc(inp)
        inp = self.dropout_after_pos_enc(inp)  # Aplica dropout después de la codificación posicional
        
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)  # Dropout antes de la capa de salida
        
        # Aplanar la salida antes de aplicar la normalización
        output = output.reshape(output.shape[0], -1)
        output = self.norm_before_output(output)  # Normalización antes de la capa de salida
        
        output = self.output_layer(output)
        return output

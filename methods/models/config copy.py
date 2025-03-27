config_spectralnet = dict(
    architecture=[8, 12, 32],
)

config_spectralnet_0 = dict(
    architecture=[64, 512, 256, 128],
)


config_lstm = dict(
    n_layers=2,
    dropout_rate=0.15,
)

config_cnn = dict(
    conv_layers=[64],  # [64, 128, 256, 256, 256],
    kernel_size=3,
    pool_size=2,
    dropout_rate=0.15,
)

config_TSTransformer_0 = dict(
    feat_dim=1,
    d_model=64,
    n_heads=8,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    pos_encoding='fixed',
    activation='gelu',
    norm='BatchNorm',
    freeze=False
)

config_TSTransformer = dict(
    feat_dim=1,
    d_model=256,  # Aumentado de 64 a 256
    n_heads=16,   # Aumentado de 8 a 16
    num_layers=6,  # Aumentado de 2 a 6
    dim_feedforward=1024,  # Aumentado de 256 a 1024
    dropout=0.1,
    pos_encoding='fixed',
    activation='gelu',
    norm='BatchNorm',
    freeze=False
)


config_spectralformer0 = dict(
    image_size=1,
    near_band=1,
    dim=64,
    depth=5,
    heads=4,
    mlp_dim=8,
    #dropout=0.1,
    emb_dropout=0.1,
    mode='CAF'
)

config_spectralformer = dict(
    image_size=1,
    near_band=1,
    dim=500,  # Aumentamos la dimensión de embeddings
    depth=15,  # Más capas en el Transformer
    heads=12,  # Más cabezas de atención
    mlp_dim=512,  # Aumentamos la dimensión del feedforward
    dropout=0.3,  # Mayor regularización para evitar overfitting
    emb_dropout=0.3,
    mode='CAF'
)

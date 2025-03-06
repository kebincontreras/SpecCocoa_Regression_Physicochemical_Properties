config_spectralnet = dict(
    architecture=[512, 512, 256, 128],
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

config_TSTransformer = dict(
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

config_spectralformer = dict(
    image_size=1,
    near_band=1,
    dim=64,
    depth=5,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
    mode='CAF'
)

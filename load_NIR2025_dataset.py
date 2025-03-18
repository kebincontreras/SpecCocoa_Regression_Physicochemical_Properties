import h5py
import numpy as np

import matplotlib.pyplot as plt

num_samples = 16

np.random.seed(0)
with h5py.File('train_nir_cocoa_dataset.h5', 'r') as f:
    sample_indices = sorted(np.random.randint(0, len(f['spec']), num_samples))
    x_train = f['spec'][sample_indices, :]
    ferm_train = f['fermentation_level'][sample_indices, :]
    mois_train = f['moisture'][sample_indices, :]
    cad_train = f['cadmium'][sample_indices, :]
    poly_train = f['polyphenols'][sample_indices, :]


# plot the first 16 samples

fig, axs = plt.subplots(4, 4, figsize=(10, 10))

for i in range(num_samples):
    ax = axs[i // 4, i % 4]
    ax.plot(x_train[i])
    ax.set_title(f"Sample {i} - Reflectance")
    ax.set_ylabel(f"Ferm {ferm_train[i][0]:.2f} - Mois {mois_train[i][0]:.2f}")
    ax.set_xlabel(f"Cad {cad_train[i][0]:.2f} - Poly {poly_train[i][0]:.2f}")
    ax.set_ylim(0, 1)
    ax.grid()

plt.tight_layout()
plt.show()
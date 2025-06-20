# SpecCocoa Regression - Physicochemical Properties

A regression analysis project to predict the physicochemical properties of cocoa using NIR and VIS spectroscopy.

## 📋 Description

This project uses machine learning and deep learning techniques to predict physicochemical properties of cocoa beans based on spectral data from NIR (Near Infrared) and VIS (Visible) ranges. The target properties include:

- **Fermentation level** 
- **Moisture**
- **Cadmium** 
- **Polyphenols** 

## 🚀 Quick Installation and Execution

### ✅ Recommended: Install and run with a single command

```bash
setup.bat
```


This script automatically performs:

- 🔧 Virtual environment creation
- 📦 Dependency installation (from `requirements.txt`)
- ⬇️ Dataset download and extraction from HuggingFace
- 🏗️ Dataset creation (train/test for VIS and NIR)
- 📊 Automatic normalization of all datasets
- 🧠 Model training and testing

## 📦 Dependencies

Automatically installed by `setup.bat`:

- Python ≥ 3.8
- numpy, pandas, matplotlib, seaborn, scikit-learn
- h5py, tables, openpyxl, xlrd, tqdm, rarfile
- Requires `UnRAR.exe` at `C:\Program Files\WinRAR`

## 📁 Project Structure

```
SpecCocoa_Regression_Physicochemical_Properties/
├── data/
│   ├── create_dataset/
│   ├── raw_dataset/
│   │   ├── *.h5 (original and normalized)
│   │   ├── *.csv
│   │   └── *.xlsx
├── model/
│   ├── Deep_Learning/
│   │   ├── NIR/ → SpectralNet models trained on NIR
│   │   └── VIS/ → SpectralNet models trained on VIS
│   └── Machine_Learning/
│       ├── NIR/ → SVR, KNN models trained on NIR
│       └── VIS/ → SVR, KNN models trained on VIS
├── configs/ → hyperparameter configurations
├── methods/, utils/, resources/ → support scripts
├── Regression_cocoa/ (virtual environment)
├── setup.bat
├── Train.py
├── requirements.txt
└── README.md
```

## 🧪 Workflow

After running `setup.bat`, the following will be generated automatically:

### Original files (`.h5`)
- `train_NIR_cocoa_dataset.h5`
- `train_VIS_cocoa_dataset.h5`
- `test_NIR_cocoa_dataset.h5`
- `test_VIS_cocoa_dataset.h5`
- and their TEST_* versions

### Normalized files (`*_normalized.h5`)
- `train_NIR_cocoa_dataset_normalized.h5`
- `train_VIS_cocoa_dataset_normalized.h5`
- `test_NIR_cocoa_dataset_normalized.h5`
- `test_VIS_cocoa_dataset_normalized.h5`
- and their TEST_* versions

## 🛠️ Normalization

A fixed scaling factor is applied to each property:

```python
NORM_FACTORS = {
    'fermentation_level': 100,
    'moisture': 10,
    'cadmium': 5.6,
    'polyphenols': 50
}
```

The resulting files are stored in `.h5` format with GZIP compression.

## 🧠 Model Training and Evaluation

The `setup.bat` script also runs model training and evaluation (based on `Train.py`).

Generated models are automatically saved based on type and modality under:

├── model/
│   ├── Deep_Learning/
│   │   ├── NIR/ → SpectralNet models trained on NIR
│   │   └── VIS/ → SpectralNet models trained on VIS
│   └── Machine_Learning/
│       ├── NIR/ → SVR, KNN models trained on NIR
│       └── VIS/ → SVR, KNN models trained on VIS

Models include `.pth` (DL), `.pkl` (ML), and `.json` metric files.

## 📄 License

MIT License – see the `LICENSE` file for details.
# SpecCocoa Regression - Physicochemical Properties

---

A regression analysis project to predict the physicochemical properties of cocoa using NIR and VIS spectroscopy.

## 📋 Description

This project uses machine learning and deep learning techniques to predict physicochemical properties of cocoa beans based on spectral data from NIR (Near Infrared) and VIS (Visible) ranges. The target properties include:

- **Fermentation level** 
- **Moisture**
- **Cadmium** 
- **Polyphenols** 

---

## 🚀 Quick Start

Follow these steps to get started with this repository:

### 1. 🔧 Install the Environment

```bash
conda create -n Regression_cocoa python=3.10 -y
conda run -n Regression_cocoa pip install -r requirements.txt
```

You must wait until the environment is fully set up. This may take a few minutes depending on your internet speed and system performance. Finally, you must activate the environment:

```bash
conda activate Regression_cocoa
```

### 2. ⬇️ Build the Datasets

This step involves downloading the base dataset, generating specific training/test sets, and applying normalization. You can run only the parts you need.

#### 📥 Step 2.1 – Download the Base Dataset

```bash
python data/create_dataset/download_cocoa_dataset.py
```
This script downloads and extracts the raw dataset into `data/raw_dataset`.

#### 🏗️ Step 2.2 – Generate Training and Testing Datasets

```bash
python data/create_dataset/create_NIR2025_dataset.py
python data/create_dataset/create_VIS2025_dataset.py
```
These scripts generate the training and testing datasets for the NIR and VIS spectrums.

#### 📊 Step 2.3 – Normalize the Datasets (Required)

```bash
python data/create_dataset/normalize_datasets.py
```
This script automatically normalizes all datasets that were generated in the previous steps.

💡 **Tip:** You don’t need to run every script, just the ones relevant to your experiment. However, the base dataset download is required for any further processing.


### 3. Train the Model

```bash
python train.py
```

---

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

Generated models are automatically saved based on type and modality under:

```
├── model/
│   ├── Deep_Learning/
│   │   ├── NIR/ → SpectralNet models trained on NIR
│   │   └── VIS/ → SpectralNet models trained on VIS
│   └── Machine_Learning/
│       ├── NIR/ → SVR, KNN models trained on NIR
│       └── VIS/ → SVR, KNN models trained on VIS
```

Models include `.pth` (DL), `.pkl` (ML), and `.json` metric files.

## 📄 License

MIT License – see the `LICENSE` file for details.
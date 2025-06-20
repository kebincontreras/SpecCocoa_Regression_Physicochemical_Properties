# SpecCocoa_Regression_Physicochemical_Properties

## 🚀 Configuración Rápida (Un Solo Clic)

### Opción 1: Configuración Automática Completa
```batch
# Doble clic en:
START_HERE.bat
```
Este script configurará todo automáticamente:
- ✅ Entorno virtual Python
- ✅ Instalación de dependencias
- ✅ Creación de todos los datasets
- ✅ Normalización de datasets

### Opción 2: Configuración Paso a Paso

#### Paso 1: Configurar Entorno
```batch
setup_environment.bat
```

#### Paso 2: Crear y Normalizar Datasets
```batch
create_datasets.bat
```

## 📊 Scripts Disponibles

### Scripts de Configuración
- **`START_HERE.bat`** - 🚀 Configuración completa en un clic
- **`setup_complete.bat`** - Configuración detallada paso a paso  
- **`setup_environment.bat`** - Solo configuración del entorno Python
- **`create_datasets.bat`** - Solo creación y normalización de datasets
- **`normalize_datasets.bat`** - Solo normalización de datasets existentes

### Scripts de Ejecución
- **`Train.py`** - Entrenar modelos de Machine Learning y Deep Learning
- **`Test_industrial.py`** - Ejecutar pruebas industriales

## 📁 Estructura del Proyecto

```
SpecCocoa_Regression_Physicochemical_Properties/
├── START_HERE.bat                 # 🚀 Punto de entrada principal
├── setup_environment.bat          # Configuración del entorno
├── create_datasets.bat           # Creación de datasets
├── Train.py                      # Script de entrenamiento
├── Test_industrial.py            # Script de pruebas
├── data/
│   ├── create_dataset/           # Scripts de creación de datasets
│   │   ├── normalize_datasets.py       # Normalización interactiva
│   │   ├── normalize_datasets_auto.py  # Normalización automática
│   │   └── create_*_dataset.py         # Creación de datasets
│   ├── raw_dataset/              # Datos originales
│   ├── *.h5                      # Datasets creados
│   └── *_normalized.h5           # Datasets normalizados
├── methods/                      # Algoritmos ML y DL
├── model/                        # Modelos entrenados
└── configs/                      # Configuraciones
```

## 🔧 Requisitos

### Software Necesario
- Python 3.8 o superior
- Git (opcional)

### Dependencias (se instalan automáticamente)
- numpy, pandas, scikit-learn
- tensorflow, keras
- h5py, openpyxl
- wandb (Weights & Biases)
- matplotlib, seaborn

## 🎯 Uso Rápido

### 1. Primera Vez (Configuración Completa)
```batch
# Ejecutar UNA VEZ:
START_HERE.bat
```

### 2. Entrenar Modelos
```batch
# Activar entorno y entrenar:
call Regression_cocoa\Scripts\activate.bat
python Train.py
```

### 3. Ejecutar Pruebas
```batch
# Ejecutar pruebas industriales:
call Regression_cocoa\Scripts\activate.bat
python Test_industrial.py
```

## 📊 Datasets Generados

El proyecto genera automáticamente:

### Datasets Originales
- `train_nir_cocoa_dataset.h5` - Entrenamiento NIR
- `train_vis_cocoa_dataset.h5` - Entrenamiento VIS  
- `test_nir_cocoa_dataset.h5` - Prueba NIR
- `test_vis_cocoa_dataset.h5` - Prueba VIS
- `TEST_*.h5` - Datasets de prueba industrial

### Datasets Normalizados
- `*_normalized.h5` - Versiones normalizadas de todos los datasets

### Factores de Normalización
- **Fermentación**: ÷ 100
- **Humedad**: ÷ 10
- **Cadmio**: ÷ 5.6
- **Polifenoles**: ÷ 50

## 🛠️ Solución de Problemas

### Error: "Archivo no encontrado"
```batch
# Ejecutar normalización:
normalize_datasets.bat
```

### Error: "Entorno virtual no encontrado"
```batch
# Reconfigurar entorno:
setup_environment.bat
```

### Error: "Dependencias faltantes"
```batch
# Reinstalar dependencias:
call Regression_cocoa\Scripts\activate.bat
pip install -r requirements.txt
```

## 🔄 Flujo de Trabajo Recomendado

1. **Primera configuración**: `START_HERE.bat`
2. **Entrenar modelos**: `python Train.py`
3. **Evaluar resultados**: Revisar carpeta `model/`
4. **Pruebas industriales**: `python Test_industrial.py`
5. **Re-normalizar si es necesario**: `normalize_datasets.bat`

## 📈 Monitoreo con Weights & Biases

El proyecto usa W&B para tracking:
- **ML Models**: Proyecto `ML_Cocoa_Regressionn`
- **DL Models**: Proyecto `4kebin_DL_Cocoa_Regressionn`
setup_environment.bat
```

This script will:
- Create a virtual environment named "Regression_cocoa"
- Activate the virtual environment
- Install all required dependencies from requirements.txt

### Manual Activation
To activate the environment manually in the future:
```bash
Regression_cocoa\Scripts\activate.bat
```

### Project Structure
```
SpecCocoa_Regression_Physicochemical_Properties/
├── setup_environment.bat    # Environment setup script
├── requirements.txt        # Project dependencies
├── Train.py               # Main training script
└── ...
```

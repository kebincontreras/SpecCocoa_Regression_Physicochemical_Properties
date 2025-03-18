import wandb
import json
import numpy as np

# Conectar a W&B
api = wandb.Api()

# Nombre del proyecto en W&B (debe coincidir con el usado en deep_learning_wb.py)
PROJECT_NAME = "2emmacocoa_regression_Deep_Learning"

# Propiedades físico-químicas a analizar
properties = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

# Modelos de Deep Learning a evaluar
models = ["spectralnet", "lstm", "cnn", "transformer", "spectralformer"]

# Inicializar diccionario para almacenar la mejor configuración de cada modelo
best_configs = {
    model: {
        "config": None,
        "avg_r2": -float("inf"),
        "r2_values": {prop: None for prop in properties}
    }
    for model in models
}

# Obtener todos los runs del proyecto en W&B
runs = api.runs(PROJECT_NAME)

# Verificar si hay runs disponibles
if not runs:
    print("\n❌ No se encontraron ejecuciones (runs) en W&B para este proyecto.")
    exit()

print("\n🔹 Runs encontrados en W&B:")
for run in runs:
    print(f"   - Run: {run.name}")

# Recorrer cada ejecución en W&B
for run in runs:
    config = run.config  # Configuración del modelo
    metrics = run.summary  # Métricas finales

    # Determinar el modelo en base al nombre del run en W&B
    model_type = None
    for model in models:
        if model in run.name.lower():  # Detecta "spectralnet" en "spectralnet_experiment"
            model_type = model
            break

    if model_type:
        # Obtener valores de R² en Test
        r2_values = {prop: metrics.get(f"Test/R²/{prop}", None) for prop in properties}

        # Si las métricas están en otro formato, intenta con "test/R²/{prop}"
        if all(v is None for v in r2_values.values()):
            r2_values = {prop: metrics.get(f"test/R²/{prop}", None) for prop in properties}

        # Obtener solo los valores válidos
        valid_r2_values = [r2 for r2 in r2_values.values() if r2 is not None]

        if valid_r2_values:
            avg_r2 = np.mean(valid_r2_values)  # Calcular el promedio solo con valores disponibles

            # Si este modelo tiene mejor promedio de R² dentro de su tipo, lo actualizamos
            if avg_r2 > best_configs[model_type]["avg_r2"]:
                best_configs[model_type] = {
                    "config": config,
                    "avg_r2": avg_r2,
                    "r2_values": r2_values
                }

# Imprimir los mejores modelos para cada arquitectura
print("\n===== 📌 Mejores Modelos de Deep Learning por Arquitectura =====")
for model, data in best_configs.items():
    if data["config"] is not None:
        print(f"\n🔹 Modelo: {model.upper()}")
        print(f"   - Promedio R²: {data['avg_r2']:.4f}")
        print(f"   - Configuración del modelo: {json.dumps(data['config'], indent=4)}")

        print("\n   - R² por Propiedad en Test:")
        for prop, r2_value in data["r2_values"].items():
            if r2_value is not None:
                print(f"     {prop}: {r2_value:.4f}")
            else:
                print(f"     {prop}: No disponible")

print("\n✅ Finalizado: Se imprimieron las mejores configuraciones para cada modelo de Deep Learning junto con R² por propiedad.")

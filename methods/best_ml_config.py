import wandb
import json
import numpy as np

# Conectar a W&B
api = wandb.Api()

# Nombre del proyecto en W&B (debe coincidir con el usado en machine_learning_wb.py)
PROJECT_NAME = "2emmaspectral_regression_Machine_Learning"

# Propiedades físico-químicas a analizar
properties = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]


# Modelos a evaluar
models = ["svr", "mlp", "rfr", "knnr"]

# Inicializar diccionario para almacenar la mejor configuración de cada modelo
best_configs = {model: {"config": None, "avg_r2": -float("inf"), "r2_values": {prop: None for prop in properties}} for model in models}

# Obtener todos los runs del proyecto en W&B
runs = api.runs(PROJECT_NAME)

# Recorrer cada ejecución en W&B
for run in runs:
    config = run.config  # Configuración del modelo
    metrics = run.summary  # Métricas finales

    # Determinar el modelo en base al nombre del run
    model_type = None
    for model in models:
        if model in run.name.lower():
            model_type = model
            break

    if model_type:
        r2_values = {prop: metrics.get(f"test/r2/{prop}", None) for prop in properties}
        valid_r2_values = [r2 for r2 in r2_values.values() if r2 is not None]

        if valid_r2_values:
            avg_r2 = np.mean(valid_r2_values)  # Calcular el promedio solo con valores disponibles

            # Si este modelo tiene mejor promedio de R², lo actualizamos
            if avg_r2 > best_configs[model_type]["avg_r2"]:
                best_configs[model_type] = {
                    "config": config,
                    "avg_r2": avg_r2,
                    "r2_values": r2_values
                }

# Imprimir resultados
print("\n===== Mejores Configuraciones por Modelo Basado en R² Promedio =====")
for model, data in best_configs.items():
    if data["config"] is not None:
        print(f"\n🔹 Modelo: {model.upper()}")
        print(f"   - Promedio R²: {data['avg_r2']:.4f}")
        print(f"   - Configuración: {json.dumps(data['config'], indent=4)}")
        
        print("\n   - R² por Propiedad:")
        for prop, r2_value in data["r2_values"].items():
            if r2_value is not None:
                print(f"     {prop}: {r2_value:.4f}")
            else:
                print(f"     {prop}: No disponible")


print("\n✅ Finalizado: Se imprimieron las mejores configuraciones para cada modelo junto con R² por propiedad.")

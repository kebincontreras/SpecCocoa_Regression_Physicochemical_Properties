import numpy as np
import torch
import matplotlib.pyplot as plt
from methods.dataloader import prepare_data

def check_normalization(modality, dataset_name="cocoa_regression"):
    """ Verifica la normalización de las firmas espectrales en el dataset. """
    result = prepare_data(dataset_name, modality, dl=False)
    
    # 🔹 Verificar qué devuelve exactamente prepare_data()
    if not isinstance(result, tuple) or len(result) < 2:
        print("❌ Error: `prepare_data()` no está devolviendo el formato esperado.")
        return
    
    train_data, test_data, num_bands, _, _ = result
    print(f"🔍 `prepare_data()` devolvió correctamente datos para {modality}.")
    
    if not isinstance(train_data, dict) or "X" not in train_data:
        print("❌ Error: `train_data` no tiene el formato esperado. Se esperaba un diccionario con clave 'X'.")
        return
    
    # Extraer datos de entrenamiento
    all_data = train_data["X"]  # Matriz con firmas espectrales
    
    # Calcular estadísticas
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    mean_val = np.mean(all_data)
    std_val = np.std(all_data)
    
    print(f"\n🔍 Estadísticas de las firmas espectrales en {modality}:")
    print(f"   ➤ Mínimo: {min_val:.4f}")
    print(f"   ➤ Máximo: {max_val:.4f}")
    print(f"   ➤ Media: {mean_val:.4f}")
    print(f"   ➤ Desviación estándar: {std_val:.4f}")
    
    # Graficar histograma de valores
    plt.figure(figsize=(8, 5))
    plt.hist(all_data.flatten(), bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"Distribución de valores en {modality}")
    plt.xlabel("Valor de la firma espectral")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()
    
    # Determinar si está normalizado
    if min_val >= 0 and max_val <= 1:
        print("✅ Las firmas parecen estar normalizadas en el rango [0,1].")
    elif min_val >= -1 and max_val <= 1:
        print("✅ Las firmas parecen estar normalizadas en el rango [-1,1].")
    else:
        print("⚠️ Las firmas NO están normalizadas en un rango típico. Considera aplicar normalización.")

# Ejecutar para ambas modalidades
check_normalization("VIS")
check_normalization("NIR")

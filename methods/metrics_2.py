import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metric_params(matrix):
    shape = np.shape(matrix)
    number = 0
    add = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        add += np.sum(matrix[i, :]) * np.sum(matrix[:, i])

    return AA, add, number

def overall_accuracy(matrix, number):
    return number / np.sum(matrix)

def average_accuracy(AA):
    return np.mean(AA)

def kappa(OA, matrix, add):
    pe = add / (np.sum(matrix) ** 2)
    return (OA - pe) / (1 - pe)

def print_results(model_name, dataset_name, dict_metrics):
    print('#' * 60)
    print(f'Model: {model_name}, Dataset: {dataset_name}')
    
    for phase in ["train", "test"]:  # 🔹 Imprimir métricas para entrenamiento y prueba
        print(f"\n🔹 {phase.upper()} METRICS:")
        for metric, values in dict_metrics[phase].items():
            out_print = f'  {metric.upper()} -> '
            for var_name, value in values.items():
                out_print += f'{var_name}: {value:.5f}  '  # 🔹 Mostrar 5 cifras decimales
            print(out_print)

    print('#' * 60)


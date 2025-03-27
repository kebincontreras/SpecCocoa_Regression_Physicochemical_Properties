import torch
print(torch.cuda.is_available())  # Debe devolver True si la GPU está disponible
print(torch.cuda.device_count())  # Debe devolver el número de GPUs detectadas
print(torch.cuda.get_device_name(0))  # Verifica el nombre de la GPU

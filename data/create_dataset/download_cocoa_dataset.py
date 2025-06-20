#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm

RAR_URL = "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"

RAR_DIR = os.path.join("data", "raw_dataset")
RAR_FILENAME = os.path.join(RAR_DIR, "Spectral_signatures_of_cocoa_beans.rar")
EXTRACT_DIR = RAR_DIR

def download_file(url, dest):
    """Descarga un archivo grande con barra de progreso."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"❌ Error al descargar archivo: {response.status_code}")
    total = int(response.headers.get('content-length', 0))

    with open(dest, 'wb') as file:
        bar = tqdm(
            desc=f"Descargando {os.path.basename(dest)}",
            total=total,
            unit='B',
            unit_scale=True
        )
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
        bar.close()

def extract_rar(rar_path, extract_to):
    """Extrae un archivo .rar al directorio especificado."""
    import rarfile
    if not rarfile.is_rarfile(rar_path):
        print(f"❌ El archivo no es un RAR válido: {rar_path}")
        return
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(extract_to)
    print(f"✅ Extracción completada en: {extract_to}")

def main():
    print("🚀 DESCARGA DE BASE DE DATOS DE CACAO")
    print("=" * 50)

    os.makedirs(RAR_DIR, exist_ok=True)

    if not os.path.exists(RAR_FILENAME):
        print(f"Descargando archivo desde:\n{RAR_URL}")
        download_file(RAR_URL, RAR_FILENAME)
    else:
        print(f"✅ El archivo ya existe: {RAR_FILENAME}")

    print(f"Extrayendo {RAR_FILENAME} en {EXTRACT_DIR} ...")
    extract_rar(RAR_FILENAME, EXTRACT_DIR)

    if os.path.exists(RAR_FILENAME):
        os.remove(RAR_FILENAME)
        print(f"🗑️ Archivo .rar eliminado: {RAR_FILENAME}")

    print("🎉 Proceso completado.")

if __name__ == "__main__":
    try:
        import rarfile
    except ImportError:
        print("Instalando dependencia 'rarfile'...")
        os.system("pip install rarfile")
        import rarfile

    main()

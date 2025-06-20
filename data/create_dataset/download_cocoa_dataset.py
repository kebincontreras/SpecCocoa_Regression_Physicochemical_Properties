#!/usr/bin/env python3
import os
import requests
import shutil
from tqdm import tqdm

# URL del .rar
RAR_URL = "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"

# Rutas
RAR_DIR = os.path.join("data", "raw_dataset")
RAR_FILENAME = os.path.join(RAR_DIR, "Spectral_signatures_of_cocoa_beans.rar")
EXTRACT_DIR = RAR_DIR
SUBFOLDER_TO_FLATTEN = "Spectral_signatures_of_cocoa_beans"

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
    """Extrae un archivo .rar usando una ruta explícita a UnRAR.exe."""
    import rarfile
    rarfile.UNRAR_TOOL = r"C:\Program Files\WinRAR\UnRAR.exe"

    try:
        if not rarfile.is_rarfile(rar_path):
            print(f"❌ El archivo no es un RAR válido o no se puede leer: {rar_path}")
            return

        with rarfile.RarFile(rar_path) as rf:
            print("📦 Archivos contenidos:")
            for f in rf.infolist():
                print("   └─", f.filename)
            rf.extractall(extract_to)

        print(f"✅ Extracción completada en: {extract_to}")

    except rarfile.RarCannotExec as e:
        print("❌ ERROR: No se pudo ejecutar UnRAR. Verifica la ruta.")
        print("🔧 Ruta usada:", rarfile.UNRAR_TOOL)
        print("🛠️ Detalles:", e)

    except Exception as e:
        print("❌ Error inesperado durante la extracción:")
        print(e)

def flatten_extracted_folder(parent_dir, subfolder_name):
    """Mueve el contenido de una subcarpeta al directorio padre y elimina la subcarpeta."""
    source_dir = os.path.join(parent_dir, subfolder_name)
    if not os.path.exists(source_dir):
        print(f"⚠️ Carpeta {source_dir} no existe. No se movió nada.")
        return

    for item in os.listdir(source_dir):
        src = os.path.join(source_dir, item)
        dst = os.path.join(parent_dir, item)
        if os.path.isdir(src):
            shutil.move(src, dst)
        else:
            shutil.move(src, dst)
    os.rmdir(source_dir)
    print(f"✅ Contenido movido a {parent_dir} y carpeta eliminada: {subfolder_name}")

def main():
    print("🚀 DESCARGA DE BASE DE DATOS DE CACAO")
    print("=" * 50)

    os.makedirs(RAR_DIR, exist_ok=True)

    if not os.path.exists(RAR_FILENAME):
        print(f"Descargando archivo desde:\n{RAR_URL}")
        download_file(RAR_URL, RAR_FILENAME)
    else:
        print(f"✅ El archivo ya existe: {RAR_FILENAME}")

    print(f"📂 Extrayendo {RAR_FILENAME} en {EXTRACT_DIR} ...")
    extract_rar(RAR_FILENAME, EXTRACT_DIR)

    print(f"📦 Moviendo archivos desde '{SUBFOLDER_TO_FLATTEN}' a '{RAR_DIR}' ...")
    flatten_extracted_folder(RAR_DIR, SUBFOLDER_TO_FLATTEN)

    # if os.path.exists(RAR_FILENAME):
    #     os.remove(RAR_FILENAME)
    #     print(f"🗑️ Archivo .rar eliminado: {RAR_FILENAME}")

    print("🎉 Proceso completado.")

if __name__ == "__main__":
    try:
        import rarfile
    except ImportError:
        print("Instalando dependencia 'rarfile'...")
        os.system("pip install rarfile")
        import rarfile

    main()

import os
import shutil
from pathlib import Path
from gradio_client import Client, handle_file

uploads_dir = Path("storage/uploads")
archivos = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.jpeg")) + list(uploads_dir.glob("*.png"))

if not archivos:
    raise FileNotFoundError("Coloca una imagen en storage/uploads/ para probar.")

imagen_prueba = archivos[0]
print(f"Probando restauración remota con: {imagen_prueba.name}...")

# 1. Probamos con el Space público de Real-ESRGAN
space_name = "akhaliq/Real-ESRGAN"
print(f"Conectando a Hugging Face Space: {space_name}...")

try:
    client = Client(space_name)
    
    # Imprime la estructura exacta de endpoints disponibles en el Space
    print("\n--- Endpoints descubiertos ---")
    client.view_api()
    print("------------------------------\n")

    print("Enviando imagen a la GPU remota...")
    # akhaliq/Real-ESRGAN recibe la imagen y el tipo de modelo
    resultado_temp = client.predict(
        handle_file(str(imagen_prueba)),
        "base",  # o "anime" según el modelo
        api_name="/predict"
    )

    salidas_dir = Path("storage/outputs")
    salidas_dir.mkdir(parents=True, exist_ok=True)
    destino = salidas_dir / f"restaurada_{imagen_prueba.name}"

    shutil.copy(resultado_temp, destino)
    print("\n====================================")
    print("¡Super-resolución remota completada con éxito!")
    print(f"Imagen guardada en: {destino}")
    print("====================================\n")

except Exception as e:
    print(f"\nDetalle: {e}")
import os
import shutil
from pathlib import Path
from gradio_client import Client, handle_file

print("Conectando con el motor gratuito de Real-ESRGAN en Hugging Face...")

# Nos conectamos a un Space público que ejecuta Real-ESRGAN con GPU en la nube
client = Client("doevent/Face-Real-ESRGAN")

# Imagen local de prueba que ya tienes
input_image_path = "storage/uploads/images2.jpeg"

print("Enviando imagen y procesando en servidor remoto...")
try:
    # Llamada directa a la API remota
    result = client.predict(
        image=handle_file(input_image_path),
        model_name="RealESRGAN_x4plus",
        denoise_strength=0.5,
        api_name="/predict"
    )

    print("\n====================================")
    print("¡Procesamiento exitoso sin usar GPU local!")
    print(f"Ruta temporal descargada: {result}")
    
    # Copiamos la salida a tu carpeta de storage
    Path("storage/outputs").mkdir(parents=True, exist_ok=True)
    destino = "storage/outputs/test_cloud_sr.png"
    shutil.copy(result, destino)
    print(f"Imagen guardada en: {destino}")
    print("====================================\n")

except Exception as e:
    print(f"Ocurrió un detalle: {e}")
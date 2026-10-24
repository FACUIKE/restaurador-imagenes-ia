import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY en el .env.")

client = genai.Client(api_key=API_KEY)

uploads_dir = Path("storage/uploads")
archivos = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.jpeg")) + list(uploads_dir.glob("*.png"))

if not archivos:
    raise FileNotFoundError("Coloca una imagen en storage/uploads/")

imagen_path = archivos[0]
print(f"Enviando imagen a Gemini para restauración: {imagen_path.name}...")

img_input = Image.open(imagen_path)

prompt = """
Task: High fidelity image restoration and enhancement.
Instructions:
- Remove noise, blur, artifacts and scratches.
- Sharpen details, letters, text edges and facial features without altering identity or structure.
- Output the restored image in high resolution.
"""

try:
    # Usamos el modelo multimodal de generación y edición de imágenes de Gemini
    response = client.models.generate_content(
        model="Gemini-3.8-Flash",
        contents=[img_input, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        )
    )

    salidas_dir = Path("storage/outputs")
    salidas_dir.mkdir(parents=True, exist_ok=True)
    destino = salidas_dir / f"restaurada_gemini_{imagen_path.name}"

    # Guardar la imagen devuelta por la API
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            img_bytes = part.inline_data.data
            img_restaurada = Image.open(BytesIO(img_bytes))
            img_restaurada.save(destino)
            print("\n====================================")
            print("¡Imagen restaurada con éxito por Gemini!")
            print(f"Guardada en: {destino}")
            print("====================================\n")
            break

except Exception as e:
    print(f"\nDetalle de la prueba: {e}")
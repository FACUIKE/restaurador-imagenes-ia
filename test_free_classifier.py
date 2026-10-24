import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PIL import Image

# 1. Cargar API Key desde .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY. Verifica que tu archivo .env la contenga.")

client = genai.Client(api_key=API_KEY)

def clasificar_imagen_gratis(ruta_imagen: str) -> str:
    img = Image.open(ruta_imagen)
    
    prompt = """
    Analiza esta imagen y clasifícala estrictamente en una de estas 3 opciones:
    - rostro
    - texto
    - paisaje
    
    Responde ÚNICAMENTE con una sola palabra en minúsculas: 'rostro', 'texto' o 'paisaje'.
    """
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[img, prompt]
    )
    return response.text.strip().lower()

if __name__ == "__main__":
    uploads_dir = Path("storage/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Buscar cualquier imagen existente
    archivos = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.jpeg")) + list(uploads_dir.glob("*.png"))

    if archivos:
        imagen_prueba = archivos[0]
        print(f"Usando imagen existente: {imagen_prueba.name}")
    else:
        # Si no hay ninguna, creamos una de prueba para no fallar
        imagen_prueba = uploads_dir / "muestra_temporal.png"
        Image.new("RGB", (200, 200), color=(80, 120, 200)).save(imagen_prueba)
        print(f"Carpeta vacía. Se creó una muestra de prueba en: {imagen_prueba.name}")

    print("Enviando imagen a la API de Gemini...")
    categoria = clasificar_imagen_gratis(str(imagen_prueba))
    print(f"\n====================================")
    print(f"Resultado de la clasificación: {categoria}")
    print(f"====================================\n")
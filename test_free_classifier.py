import os
import time
from dotenv import load_dotenv
from google import genai
from PIL import Image

# 1. Carga las variables definidas en el archivo .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY en el archivo .env ni en el sistema.")

# Si GEMINI_API_KEY está en el entorno, genai.Client() la toma automáticamente
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
    
    # Modelo ultrarrápido y gratuito elegido
    modelo = "gemini-3.5-flash-lite"
    
    try:
        response = client.models.generate_content(
            model=modelo,
            contents=[img, prompt]
        )
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error al clasificar con {modelo}: {e}")
        raise e

if __name__ == "__main__":
    print("Enviando imagen a la API gratuita de Google (Gemini 3.5 Flash Lite)...")
    categoria = clasificar_imagen_gratis("storage/uploads/images2.jpeg")
    print(f"\n====================================")
    print(f"Resultado de la API gratuita: {categoria}")
    print(f"====================================\n")
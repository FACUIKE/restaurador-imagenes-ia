import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Configuración del Agente Decisor
AGENT_PROVIDER = os.getenv("AGENT_PROVIDER", "ollama").lower()  # 'ollama' o 'gemini'

# Ollama Local (Optimizado para GPUs de 6GB como RTX 3050)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "minicpm-v")

# Google Gemini (Alternativa en la nube)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Plataforma de Herramientas / Inferencia (Local o Replicate)
TOOLS_BACKEND = os.getenv("TOOLS_BACKEND", "local").lower()  # 'local' o 'replicate'
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

# Rutas de Almacenamiento Local
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
OUTPUTS_DIR = STORAGE_DIR / "outputs"

# Crear directorios si no existen
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

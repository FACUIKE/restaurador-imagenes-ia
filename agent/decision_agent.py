import json
import logging
from pathlib import Path
from typing import Union
from PIL import Image
import ollama

from config import settings
from config.prompts import SYSTEM_PROMPT_RESTORATION, USER_PROMPT_ANALYZE
from agent.schemas import RestorationDecision

logger = logging.getLogger(__name__)

class DecisionAgent:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or settings.AGENT_PROVIDER
        self.model = model or (settings.OLLAMA_MODEL if self.provider == "ollama" else settings.GEMINI_MODEL)

    def _prepare_image_for_vlm(self, image_path: Path, max_dim: int = 1024) -> Path:
        """
        Genera una versión optimizada en resolución para evitar Out of Memory en Ollama.
        """
        temp_path = settings.STORAGE_DIR / f"temp_vlm_{image_path.name}"
        with Image.open(image_path) as img:
            # Convertir a RGB por si viene en RGBA o formatos raros
            img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            img.save(temp_path, "JPEG", quality=85)
        return temp_path

    def analyze_image(self, image_path: Union[str, Path]) -> RestorationDecision:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")

        if self.provider == "ollama":
            return self._analyze_with_ollama(path)
        elif self.provider == "gemini":
            return self._analyze_with_gemini(path)
        else:
            raise ValueError(f"Proveedor no soportado: {self.provider}")

    def _analyze_with_ollama(self, image_path: Path) -> RestorationDecision:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        
        # Redimensionar la imagen para no desbordar VRAM
        vlm_image_path = self._prepare_image_for_vlm(image_path)
        
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_RESTORATION
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT_ANALYZE,
                        "images": [str(vlm_image_path.resolve())]
                    }
                ],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_ctx": 2048  # Limitar el contexto evita que se reserve memoria excesiva
                }
            )
        finally:
            # Eliminar imagen temporal
            if vlm_image_path.exists():
                vlm_image_path.unlink()

        content = response["message"]["content"]
        raw_json = json.loads(content)
        return RestorationDecision(**raw_json)